"""
NoobAI XL V-Pred 1.0 - Utility Functions

This module contains utility functions for validation, file operations,
DoRA adapter discovery, and other helper functions.
"""

import os
import glob
import json
import struct
import hashlib
import unicodedata
import torch
from typing import Tuple, Dict, Any, List, Optional
from config import (
    logger, MODEL_CONFIG, GEN_CONFIG, USER_FRIENDLY_ERRORS,
    DORA_SEARCH_DIRECTORIES, DTYPE_MAP, SAFETENSORS_AVAILABLE
)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_user_friendly_error(error: Exception) -> str:
    """Convert technical errors to user-friendly messages."""
    error_str = str(error).lower()
    for key, message in USER_FRIENDLY_ERRORS.items():
        if key.lower() in error_str:
            return message
    return str(error)

def validate_model_path(path: str) -> Tuple[bool, str]:
    """Validate model path with comprehensive checks."""
    if not path.strip():
        return False, "Please provide a model path"

    try:
        # Normalize and validate path
        normalized_path = os.path.normpath(os.path.abspath(path))

        # Check for path traversal attempts
        if '..' in os.path.relpath(normalized_path):
            return False, "Invalid path format"

        if not os.path.exists(normalized_path):
            return False, f"Model file not found: {normalized_path}"

        if not os.path.isfile(normalized_path):
            return False, "Path must point to a file, not a directory"

        # Check file extension
        if not any(normalized_path.lower().endswith(fmt) for fmt in MODEL_CONFIG.SUPPORTED_FORMATS):
            return False, f"Unsupported format. Expected: {', '.join(MODEL_CONFIG.SUPPORTED_FORMATS)}"

        # Check file size
        file_size = os.path.getsize(normalized_path)
        if file_size < MODEL_CONFIG.MIN_FILE_SIZE_MB * 1024 * 1024:
            return False, f"File too small ({format_file_size(file_size)}). Expected > {MODEL_CONFIG.MIN_FILE_SIZE_MB}MB"

        if file_size > MODEL_CONFIG.MAX_FILE_SIZE_GB * 1024 * 1024 * 1024:
            return False, f"File too large ({format_file_size(file_size)}). Expected < {MODEL_CONFIG.MAX_FILE_SIZE_GB}GB"

        return True, normalized_path

    except Exception as e:
        return False, f"Path validation error: {str(e)}"

def get_safe_csv_paths() -> Dict[str, str]:
    """Get validated CSV file paths with security checks."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    style_dir = os.path.join(script_dir, "style")

    if not os.path.exists(style_dir):
        logger.warning(f"Style directory not found: {style_dir}")
        return {}

    style_dir = os.path.normpath(style_dir)
    if not style_dir.startswith(os.path.normpath(script_dir)):
        logger.warning("Style directory outside of script directory - security risk")
        return {}

    csv_files = {
        'danbooru_character': "danbooru_character_webui.csv",
        'e621_character': "e621_character_webui.csv",
        'danbooru_artist': "danbooru_artist_webui.csv",
        'e621_artist': "e621_artist_webui.csv"
    }

    validated_paths = {}
    for key, filename in csv_files.items():
        if '..' in filename or '/' in filename or '\\' in filename:
            logger.warning(f"Invalid filename detected: {filename}")
            continue

        full_path = os.path.join(style_dir, filename)
        full_path = os.path.normpath(full_path)

        if full_path.startswith(style_dir) and os.path.isfile(full_path):
            validated_paths[key] = full_path
        else:
            logger.warning(f"CSV file not found or outside safe directory: {filename}")

    return validated_paths

def normalize_text(text: str) -> str:
    """Normalize Unicode text and strip whitespace."""
    if not text:
        return ""
    return unicodedata.normalize('NFC', text.strip())

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1000.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1000.0
    return f"{size_bytes:.2f} TB"

def calculate_image_hash(file_path: str) -> str:
    """Calculate MD5 hash of an image file."""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def validate_dora_path(path: str) -> Tuple[bool, str]:
    """Validate DoRA adapter path with comprehensive checks."""
    if not path or not path.strip():
        return False, "Please provide a DoRA adapter path"

    try:
        # Normalize and validate path
        normalized_path = os.path.normpath(os.path.abspath(path))

        # Check for path traversal attempts
        if '..' in os.path.relpath(normalized_path):
            return False, "Invalid path format"

        if not os.path.exists(normalized_path):
            return False, f"DoRA file not found: {normalized_path}"

        if not os.path.isfile(normalized_path):
            return False, "Path must point to a file, not a directory"

        # Check file extension
        if not normalized_path.lower().endswith('.safetensors'):
            return False, "DoRA file must be in .safetensors format"

        # Check file size
        file_size = os.path.getsize(normalized_path)
        min_size = MODEL_CONFIG.DORA_MIN_FILE_SIZE_MB * 1024 * 1024
        max_size = MODEL_CONFIG.DORA_MAX_FILE_SIZE_MB * 1024 * 1024

        if file_size < min_size:
            return False, f"DoRA file too small ({format_file_size(file_size)}). Expected > {MODEL_CONFIG.DORA_MIN_FILE_SIZE_MB}MB"

        if file_size > max_size:
            return False, f"DoRA file too large ({format_file_size(file_size)}). Expected < {MODEL_CONFIG.DORA_MAX_FILE_SIZE_MB}MB"

        return True, normalized_path

    except Exception as e:
        return False, f"DoRA path validation error: {str(e)}"

def detect_base_model_precision(model_path: str) -> torch.dtype:
    """Detect the native precision using lightweight header analysis (400x faster)."""
    try:
        # Read only the safetensors header (tiny compared to full model)
        with open(model_path, 'rb') as f:
            # Read header size (8 bytes)
            header_size = struct.unpack('<Q', f.read(8))[0]
            # Read header JSON (typically ~350KB vs 6.6GB full model)
            header_data = json.loads(f.read(header_size).decode('utf-8'))

        # Find first UNet tensor dtype from header metadata
        unet_tensors = {k: v for k, v in header_data.items()
                       if k != '__metadata__' and 'model.diffusion_model' in k}

        if unet_tensors:
            # Get dtype from first UNet tensor
            dtype_str = list(unet_tensors.values())[0]['dtype']
            detected_dtype = DTYPE_MAP.get(dtype_str, torch.bfloat16)
            logger.info(f"Detected base model native precision: {detected_dtype} (from header)")
            return detected_dtype

        # Fallback for modern SDXL models
        logger.info("Using BF16 as default for SDXL model")
        return torch.bfloat16

    except Exception as e:
        logger.warning(f"Could not detect base model precision from header: {e}")
        return torch.bfloat16

def detect_adapter_precision(adapter_path: str) -> str:
    """Detect the precision of a DoRA adapter file using filename heuristic."""
    # Use filename heuristic first as it's most reliable and fast
    filename_lower = os.path.basename(adapter_path).lower()
    if "_fp16" in filename_lower:
        return "fp16"
    elif "_bf16" in filename_lower:
        return "bfloat16"
    elif "_fp32" in filename_lower:
        return "fp32"

    # If no precision in filename, assume fp16 for DoRA adapters
    # (most common format for NoobAI adapters)
    return "fp16"

def discover_dora_adapters() -> List[Dict[str, Any]]:
    """Discover all DoRA adapter files in search directories."""
    adapters = []
    seen_names = set()

    for search_dir in DORA_SEARCH_DIRECTORIES:
        if not os.path.exists(search_dir):
            continue

        try:
            # Find all .safetensors files in directory
            search_pattern = os.path.join(search_dir, "*.safetensors")
            for adapter_path in glob.glob(search_pattern):
                if not os.path.isfile(adapter_path):
                    continue

                # Validate adapter file
                is_valid, validated_path = validate_dora_path(adapter_path)
                if not is_valid:
                    continue

                adapter_name = os.path.basename(validated_path)

                # Avoid duplicates (same filename from different directories)
                if adapter_name in seen_names:
                    continue
                seen_names.add(adapter_name)

                # Get file info
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

        except Exception as e:
            logger.warning(f"Error scanning directory {search_dir}: {e}")

    # Sort by name for consistent ordering
    adapters.sort(key=lambda x: x['name'])
    return adapters

def find_dora_path() -> Optional[str]:
    """Search for DoRA adapter file in common locations (backward compatibility)."""
    adapters = discover_dora_adapters()
    if adapters:
        # Return first (alphabetically) adapter for backward compatibility
        return adapters[0]['path']
    return None

def get_dora_adapter_by_name(adapter_name: str) -> Optional[Dict[str, Any]]:
    """Get adapter info by filename."""
    adapters = discover_dora_adapters()
    for adapter in adapters:
        if adapter['name'] == adapter_name:
            return adapter
    return None

# Initialize CSV paths
CSV_PATHS = get_safe_csv_paths()
