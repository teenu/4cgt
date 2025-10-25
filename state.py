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
        self._failed_cleanups = {}  # Track resources that failed to clean up

    def get_or_create(self, key: str, creator_func: Callable) -> Any:
        """Get existing resource or create new one."""
        with self._lock:
            if key not in self._resources:
                self._resources[key] = creator_func()
            return self._resources[key]

    def clear(self):
        """Enhanced resource pool clearing with proper resource cleanup and leak prevention."""
        with self._lock:
            successfully_cleaned = []
            failed_to_clean = {}

            # Attempt to close/cleanup each resource
            for key, resource in list(self._resources.items()):
                cleanup_attempted = False
                cleanup_succeeded = False

                try:
                    # Handle different types of resources that need cleanup
                    if hasattr(resource, 'close'):
                        resource.close()
                        cleanup_attempted = True
                        cleanup_succeeded = True
                    elif hasattr(resource, 'cleanup'):
                        resource.cleanup()
                        cleanup_attempted = True
                        cleanup_succeeded = True
                    else:
                        # Resources without explicit cleanup methods
                        # Will be handled by gc.collect() - consider as successful
                        cleanup_succeeded = True

                except Exception as e:
                    logger.warning(f"Error cleaning up resource '{key}': {e}")
                    failed_to_clean[key] = {'resource': resource, 'error': str(e)}
                    cleanup_succeeded = False

                # Only remove from active resources if cleanup succeeded
                if cleanup_succeeded:
                    successfully_cleaned.append(key)

            # Remove successfully cleaned resources from the pool
            for key in successfully_cleaned:
                del self._resources[key]

            # Track failed cleanups for potential retry or debugging
            if failed_to_clean:
                self._failed_cleanups.update(failed_to_clean)
                logger.error(f"Resource pool has {len(failed_to_clean)} resource(s) that failed cleanup: {list(failed_to_clean.keys())}")
                logger.error("These resources remain in memory to prevent leaks. Manual cleanup may be required.")

            # Force garbage collection to clean up successfully released resources
            collected = gc.collect()

            logger.info(f"Resource pool cleared: {len(successfully_cleaned)} cleaned, {len(failed_to_clean)} failed, {collected} objects freed")

    def get_failed_cleanups(self) -> Dict[str, Dict[str, Any]]:
        """Get resources that failed to clean up."""
        with self._lock:
            return self._failed_cleanups.copy()

    def retry_failed_cleanups(self) -> int:
        """Retry cleanup of previously failed resources. Returns number of resources successfully cleaned."""
        with self._lock:
            if not self._failed_cleanups:
                return 0

            successfully_cleaned = []

            for key, info in list(self._failed_cleanups.items()):
                resource = info['resource']
                try:
                    if hasattr(resource, 'close'):
                        resource.close()
                    elif hasattr(resource, 'cleanup'):
                        resource.cleanup()
                    successfully_cleaned.append(key)
                    logger.info(f"Successfully cleaned previously failed resource '{key}' on retry")
                except Exception as e:
                    logger.warning(f"Retry cleanup failed for resource '{key}': {e}")

            # Remove successfully cleaned resources from failed list
            for key in successfully_cleaned:
                del self._failed_cleanups[key]

            if successfully_cleaned:
                gc.collect()

            return len(successfully_cleaned)

# Global resource pool
resource_pool = ResourcePool()
