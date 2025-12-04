"""
NoobAI XL V-Pred 1.0 - State Management

This module contains state management classes including performance monitoring,
generation state tracking, resource pooling, queue management, and gallery management.
"""

import time
import threading
import contextlib
import gc
import uuid
import traceback
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, List, Optional, Tuple
from config import logger, QUEUE_CONFIG

# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

class PerformanceMonitor:
    """Optional performance monitoring for debugging."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.timings = {}
        self._lock = threading.Lock()

    @contextlib.contextmanager
    def time_section(self, name: str):
        """Context manager for timing code sections."""
        if not self.enabled:
            yield
            return

        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            with self._lock:
                if name not in self.timings:
                    self.timings[name] = []
                self.timings[name].append(elapsed)
            logger.debug(f"{name} took {elapsed:.3f}s")

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get timing summary statistics."""
        # Copy data under lock to prevent race conditions
        with self._lock:
            timings_copy = {name: list(times) for name, times in self.timings.items()}

        # Process outside lock
        summary = {}
        for name, times in timings_copy.items():
            if times:
                summary[name] = {
                    'count': len(times),
                    'total': sum(times),
                    'average': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times)
                }
        return summary

# Global performance monitor (disabled by default)
perf_monitor = PerformanceMonitor(enabled=False)

# ============================================================================
# THREAD-SAFE STATE MANAGEMENT
# ============================================================================

class GenerationState(Enum):
    """Generation state enumeration."""
    IDLE = "idle"
    GENERATING = "generating"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    ERROR = "error"

class StateManager:
    """State management for generation."""

    def __init__(self):
        self._lock = threading.Lock()
        self._state = GenerationState.IDLE

    def set_state(self, state: GenerationState) -> None:
        """Set the current application state."""
        with self._lock:
            self._state = state

    def get_state(self) -> GenerationState:
        """Get the current application state."""
        with self._lock:
            return self._state

    def is_generating(self) -> bool:
        """Check if currently generating."""
        with self._lock:
            return self._state == GenerationState.GENERATING

    def is_interrupted(self) -> bool:
        """Check if generation was interrupted."""
        with self._lock:
            return self._state == GenerationState.INTERRUPTED

    def request_interrupt(self) -> None:
        """Request generation interruption."""
        with self._lock:
            if self._state == GenerationState.GENERATING:
                self._state = GenerationState.INTERRUPTED

    def try_start_generation(self) -> bool:
        """Attempt to start generation."""
        with self._lock:
            if self._state == GenerationState.IDLE:
                self._state = GenerationState.GENERATING
                return True
            return False

    def try_complete_generation(self) -> bool:
        """Attempt to mark generation as completed."""
        with self._lock:
            if self._state == GenerationState.GENERATING:
                self._state = GenerationState.COMPLETED
                return True
            return False

    def finish_generation(self) -> None:
        """Finish generation and return to IDLE state."""
        with self._lock:
            # Set IDLE if we're in any terminal or active state
            # This ensures state is reset regardless of how generation ended
            if self._state in [GenerationState.GENERATING, GenerationState.COMPLETED,
                              GenerationState.INTERRUPTED, GenerationState.ERROR]:
                self._state = GenerationState.IDLE

# Global state manager instance
state_manager = StateManager()

# ============================================================================
# RESOURCE MANAGEMENT
# ============================================================================

class ResourcePool:
    """Manage pooled resources for better performance."""

    def __init__(self):
        self._lock = threading.Lock()
        self._resources = {}
        self._failed_cleanups = {}  # Track metadata for failed cleanups (not resource objects)

    def get_or_create(self, key: str, creator_func: Callable) -> Any:
        """Get existing resource or create new one."""
        with self._lock:
            if key not in self._resources:
                self._resources[key] = creator_func()
            return self._resources[key]

    def clear(self):
        """Clear resource pool with retry for failed cleanups."""
        with self._lock:
            if self._failed_cleanups:
                self._clear_stale_cleanup_metadata_internal()

            successfully_cleaned = []
            failed_to_clean = {}

            for key, resource in list(self._resources.items()):
                try:
                    if hasattr(resource, 'close'):
                        resource.close()
                    elif hasattr(resource, 'cleanup'):
                        resource.cleanup()
                    # Only add to success list if cleanup succeeded
                    successfully_cleaned.append(key)
                except Exception as e:
                    logger.warning(f"Error cleaning up resource '{key}': {e}")
                    # Store metadata including traceback for debugging
                    failed_to_clean[key] = {
                        'error': str(e),
                        'type': type(resource).__name__,
                        'traceback': traceback.format_exc()
                    }

            for key in successfully_cleaned:
                del self._resources[key]

            # Retry failed cleanups once with gc.collect() in between
            if failed_to_clean:
                gc.collect()
                retry_succeeded = []
                for key in list(failed_to_clean.keys()):
                    if key in self._resources:
                        try:
                            resource = self._resources[key]
                            if hasattr(resource, 'close'):
                                resource.close()
                            elif hasattr(resource, 'cleanup'):
                                resource.cleanup()
                            retry_succeeded.append(key)
                            del self._resources[key]
                            logger.debug(f"Retry cleanup succeeded for resource '{key}'")
                        except Exception as retry_e:
                            logger.debug(f"Retry cleanup also failed for '{key}': {retry_e}")

                # Remove retried successes from failed list
                for key in retry_succeeded:
                    del failed_to_clean[key]

            if failed_to_clean:
                for key, info in failed_to_clean.items():
                    self._failed_cleanups[key] = {
                        'error': info['error'],
                        'timestamp': time.time(),
                        'type': info['type'],
                        'traceback': info['traceback']
                    }
                logger.error(f"Failed cleanup for {len(failed_to_clean)} resource(s): {list(failed_to_clean.keys())}")

            gc.collect()
            if failed_to_clean:
                logger.warning(f"Resource pool cleared with {len(failed_to_clean)} failures")

    def get_failed_cleanups(self) -> Dict[str, Dict[str, Any]]:
        """Get resources that failed to clean up."""
        with self._lock:
            return self._failed_cleanups.copy()

    def _clear_stale_cleanup_metadata_internal(self) -> int:
        """Clear stale cleanup metadata older than 1 hour."""
        if not self._failed_cleanups:
            return 0

        current_time = time.time()
        stale_entries = []

        for key, info in list(self._failed_cleanups.items()):
            if current_time - info['timestamp'] > 3600:
                stale_entries.append(key)

        for key in stale_entries:
            del self._failed_cleanups[key]

        if stale_entries:
            gc.collect()

        return len(stale_entries)

    def clear_stale_cleanup_metadata(self) -> int:
        """Clear stale cleanup metadata."""
        with self._lock:
            return self._clear_stale_cleanup_metadata_internal()

# Global resource pool
resource_pool = ResourcePool()

# ============================================================================
# QUEUE AND GALLERY DATA STRUCTURES
# ============================================================================

@dataclass
class QueueItem:
    """Represents a queued generation request."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)

    # Generation parameters
    prompt: str = ""
    negative_prompt: str = ""
    resolution: str = ""
    cfg_scale: float = 4.2
    steps: int = 34
    rescale_cfg: float = 0.55
    seed: str = ""
    use_custom_resolution: bool = False
    custom_width: int = 1216
    custom_height: int = 832
    auto_randomize_seed: bool = True
    adapter_strength: float = 1.0
    enable_dora: bool = False
    dora_start_step: int = 1
    dora_toggle_mode: Optional[str] = None
    dora_manual_schedule: str = ""

    def get_prompt_snippet(self, max_length: int = 50) -> str:
        """Return truncated prompt for display."""
        if len(self.prompt) <= max_length:
            return self.prompt
        return self.prompt[:max_length-3] + "..."

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for generation."""
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'prompt': self.prompt,
            'negative_prompt': self.negative_prompt,
            'resolution': self.resolution,
            'cfg_scale': self.cfg_scale,
            'steps': self.steps,
            'rescale_cfg': self.rescale_cfg,
            'seed': self.seed,
            'use_custom_resolution': self.use_custom_resolution,
            'custom_width': self.custom_width,
            'custom_height': self.custom_height,
            'auto_randomize_seed': self.auto_randomize_seed,
            'adapter_strength': self.adapter_strength,
            'enable_dora': self.enable_dora,
            'dora_start_step': self.dora_start_step,
            'dora_toggle_mode': self.dora_toggle_mode,
            'dora_manual_schedule': self.dora_manual_schedule
        }


@dataclass
class GalleryItem:
    """Represents an image in the gallery."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    image_path: str = ""
    seed: int = 0
    prompt: str = ""
    generation_info: str = ""

    def get_prompt_snippet(self, max_length: int = 50) -> str:
        """Return truncated prompt for tooltip."""
        if len(self.prompt) <= max_length:
            return self.prompt
        return self.prompt[:max_length-3] + "..."


# ============================================================================
# QUEUE MANAGER
# ============================================================================

class QueueManager:
    """Thread-safe queue management for generation requests."""

    def __init__(self, max_size: int = 10):
        self._lock = threading.Lock()
        self._queue: List[QueueItem] = []
        self._max_size = max_size
        self._auto_process = True

    def add(self, item: QueueItem) -> Tuple[bool, str]:
        """Add item to queue. Returns (success, message)."""
        with self._lock:
            if len(self._queue) >= self._max_size:
                return False, f"Queue full (max {self._max_size} items)"
            self._queue.append(item)
            return True, f"Added to queue (position {len(self._queue)})"

    def remove(self, item_id: str) -> bool:
        """Remove item from queue by ID."""
        with self._lock:
            for i, item in enumerate(self._queue):
                if item.id == item_id:
                    self._queue.pop(i)
                    return True
            return False

    def pop_next(self) -> Optional[QueueItem]:
        """Get and remove next item from queue."""
        with self._lock:
            if self._queue:
                return self._queue.pop(0)
            return None

    def peek_next(self) -> Optional[QueueItem]:
        """View next item without removing."""
        with self._lock:
            if self._queue:
                return self._queue[0]
            return None

    def clear(self) -> int:
        """Clear all items from queue. Returns count removed."""
        with self._lock:
            count = len(self._queue)
            self._queue.clear()
            return count

    def get_all(self) -> List[QueueItem]:
        """Get copy of all queue items."""
        with self._lock:
            return list(self._queue)

    def size(self) -> int:
        """Get current queue size."""
        with self._lock:
            return len(self._queue)

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        with self._lock:
            return len(self._queue) == 0

    def set_auto_process(self, enabled: bool) -> None:
        """Enable/disable auto-processing."""
        with self._lock:
            self._auto_process = enabled

    def is_auto_process_enabled(self) -> bool:
        """Check if auto-processing is enabled."""
        with self._lock:
            return self._auto_process

    def get_max_size(self) -> int:
        """Get maximum queue size."""
        return self._max_size


# ============================================================================
# GALLERY MANAGER
# ============================================================================

class GalleryManager:
    """Thread-safe gallery management for session images."""

    def __init__(self, max_size: int = 10):
        self._lock = threading.Lock()
        self._items: List[GalleryItem] = []
        self._max_size = max_size

    def add(self, item: GalleryItem) -> None:
        """Add item to gallery, removing oldest if at capacity."""
        with self._lock:
            if len(self._items) >= self._max_size:
                self._items.pop(0)  # Remove oldest
            self._items.append(item)

    def get_all(self) -> List[GalleryItem]:
        """Get copy of all gallery items."""
        with self._lock:
            return list(self._items)

    def get_by_id(self, item_id: str) -> Optional[GalleryItem]:
        """Get gallery item by ID."""
        with self._lock:
            for item in self._items:
                if item.id == item_id:
                    return item
            return None

    def get_by_index(self, index: int) -> Optional[GalleryItem]:
        """Get gallery item by index."""
        with self._lock:
            if 0 <= index < len(self._items):
                return self._items[index]
            return None

    def get_paths(self) -> List[str]:
        """Get list of all image paths."""
        with self._lock:
            return [item.image_path for item in self._items if item.image_path]

    def clear(self) -> int:
        """Clear gallery. Returns count removed."""
        with self._lock:
            count = len(self._items)
            self._items.clear()
            return count

    def size(self) -> int:
        """Get current gallery size."""
        with self._lock:
            return len(self._items)

    def get_max_size(self) -> int:
        """Get maximum gallery size."""
        return self._max_size


# Global queue and gallery manager instances
queue_manager = QueueManager(max_size=QUEUE_CONFIG.MAX_QUEUE_SIZE)
gallery_manager = GalleryManager(max_size=QUEUE_CONFIG.MAX_GALLERY_SIZE)
