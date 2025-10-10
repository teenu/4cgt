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
    """Thread-safe state management for generation."""

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

    def get_or_create(self, key: str, creator_func: Callable) -> Any:
        """Get existing resource or create new one."""
        with self._lock:
            if key not in self._resources:
                self._resources[key] = creator_func()
            return self._resources[key]

    def clear(self):
        """Enhanced resource pool clearing with proper resource cleanup."""
        with self._lock:
            # Properly close/cleanup each resource before clearing
            for key, resource in self._resources.items():
                try:
                    # Handle different types of resources that need cleanup
                    if hasattr(resource, 'close'):
                        resource.close()
                    elif hasattr(resource, 'cleanup'):
                        resource.cleanup()
                    elif hasattr(resource, '__del__'):
                        # Force cleanup if resource has destructor
                        del resource
                except Exception as e:
                    logger.warning(f"Error cleaning up resource '{key}': {e}")

            # Clear the dictionary
            self._resources.clear()

            # Force garbage collection to clean up resources
            gc.collect()

            logger.info(f"Resource pool cleared and cleaned up")

# Global resource pool
resource_pool = ResourcePool()
