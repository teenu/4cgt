"""
NoobAI XL V-Pred 1.0 - State Management

This module contains state management classes including performance monitoring,
generation state tracking, and resource pooling.
"""

import time
import threading
import contextlib
import gc
from enum import Enum
from typing import Dict, Any, Callable
from config import logger

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
        with self._lock:
            summary = {}
            for name, times in self.timings.items():
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
    """Thread-safe state management for generation with atomic operations."""

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
        """Atomically attempt to start generation."""
        with self._lock:
            if self._state == GenerationState.IDLE:
                self._state = GenerationState.GENERATING
                return True
            return False

    def try_complete_generation(self) -> bool:
        """Atomically attempt to mark generation as completed."""
        with self._lock:
            if self._state == GenerationState.GENERATING:
                self._state = GenerationState.COMPLETED
                return True
            return False

    def finish_generation(self) -> None:
        """Finish generation and return to IDLE state."""
        with self._lock:
            if self._state != GenerationState.GENERATING:
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
        """Clear resource pool."""
        with self._lock:
            if self._failed_cleanups:
                cleared_count = self._clear_stale_cleanup_metadata_internal()
                if cleared_count > 0:
                    logger.info(f"Cleared {cleared_count} stale cleanup metadata entries")

            successfully_cleaned = []
            failed_to_clean = {}

            for key, resource in list(self._resources.items()):
                try:
                    if hasattr(resource, 'close'):
                        resource.close()
                    elif hasattr(resource, 'cleanup'):
                        resource.cleanup()
                except Exception as e:
                    logger.warning(f"Error cleaning up resource '{key}': {e}")
                    failed_to_clean[key] = {'resource': resource, 'error': str(e)}

                successfully_cleaned.append(key)

            for key in successfully_cleaned:
                del self._resources[key]

            if failed_to_clean:
                for key, info in failed_to_clean.items():
                    self._failed_cleanups[key] = {
                        'error': info['error'],
                        'timestamp': time.time(),
                        'type': type(info['resource']).__name__
                    }
                logger.error(f"Failed cleanup for {len(failed_to_clean)} resource(s): {list(failed_to_clean.keys())}")

            collected = gc.collect()
            logger.info(f"Resource pool cleared: {len(successfully_cleaned)} cleaned, {len(failed_to_clean)} failed, {collected} objects freed")

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
