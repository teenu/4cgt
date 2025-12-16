"""DoRA adapter discovery and management utilities."""

import os
import glob
import time
from typing import List, Dict, Any, Optional
from config import logger, DORA_SEARCH_DIRECTORIES
from utils.validation import validate_dora_path
from utils.formatting import format_file_size

# Cache for discover_dora_adapters to avoid redundant filesystem scans
_adapters_cache: List[Dict[str, Any]] = []
_adapters_cache_time: float = 0.0
_CACHE_TTL_SECONDS: float = 5.0


def detect_adapter_precision(adapter_path: str) -> str:
    """Detect adapter precision from filename heuristic (display only)."""
    filename_lower = os.path.basename(adapter_path).lower()
    if "_fp16" in filename_lower:
        return "fp16"
    elif "_bf16" in filename_lower:
        return "bfloat16"
    elif "_fp32" in filename_lower:
        return "fp32"
    return "unknown"


def discover_dora_adapters(force_refresh: bool = False) -> List[Dict[str, Any]]:
    """Discover all DoRA adapter files in search directories.

    Results are cached for 5 seconds to avoid redundant filesystem scans.

    Args:
        force_refresh: If True, bypass the cache and rescan directories.
    """
    global _adapters_cache, _adapters_cache_time

    current_time = time.time()
    if not force_refresh and _adapters_cache and (current_time - _adapters_cache_time) < _CACHE_TTL_SECONDS:
        return _adapters_cache.copy()

    adapters = []
    seen_names = set()

    for search_dir in DORA_SEARCH_DIRECTORIES:
        if not os.path.exists(search_dir):
            continue

        try:
            search_pattern = os.path.join(search_dir, "*.safetensors")
            for adapter_path in glob.glob(search_pattern):
                if not os.path.isfile(adapter_path):
                    continue

                is_valid, validated_path = validate_dora_path(adapter_path)
                if not is_valid:
                    continue

                adapter_name = os.path.basename(validated_path)

                if adapter_name in seen_names:
                    continue
                seen_names.add(adapter_name)

                file_size = os.path.getsize(validated_path)
                precision = detect_adapter_precision(validated_path)

                adapters.append({
                    'name': adapter_name,
                    'path': validated_path,
                    'size': file_size,
                    'size_formatted': format_file_size(file_size),
                    'precision': precision,
                    'display_name': f"{adapter_name} ({format_file_size(file_size)}, {precision})"
                })

        except (IOError, OSError, PermissionError) as e:
            logger.warning(f"Error accessing directory {search_dir}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error scanning directory {search_dir}: {e}")

    adapters.sort(key=lambda x: x['name'])

    # Update cache
    _adapters_cache = adapters.copy()
    _adapters_cache_time = time.time()

    return adapters


def find_dora_path() -> Optional[str]:
    """Search for DoRA adapter file in common locations (backward compatibility)."""
    adapters = discover_dora_adapters()
    if adapters:
        return adapters[0]['path']
    return None


def get_dora_adapter_by_name(adapter_name: str) -> Optional[Dict[str, Any]]:
    """Get adapter info by filename."""
    adapters = discover_dora_adapters()
    for adapter in adapters:
        if adapter['name'] == adapter_name:
            return adapter
    return None


def clear_adapters_cache() -> None:
    """Clear the adapters cache to force a fresh scan on next discovery."""
    global _adapters_cache, _adapters_cache_time
    _adapters_cache = []
    _adapters_cache_time = 0.0
