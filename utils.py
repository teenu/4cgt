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
from functools import lru_cache
from typing import Tuple, Dict, Any, List, Optional, Sequence
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

@lru_cache(maxsize=1)
def _get_allowed_directories() -> tuple:
    """Get list of allowed directories for model/adapter files.

    Returns tuple for hashability (required by lru_cache).
    Cached as directory list rarely changes during runtime.
    """
    allowed = [
        os.getcwd(),  # Current working directory
        os.path.expanduser("~"),  # User home directory
        os.path.expanduser("~/Downloads"),  # User Downloads
        os.path.expanduser("~/Documents"),  # User Documents
        os.path.expanduser("~/Models"),  # Common models directory
        "/tmp",  # Temporary directory (for user convenience, but logged)
    ]

    # Normalize all paths and resolve symlinks
    # Return as tuple for lru_cache hashability
    return tuple(os.path.realpath(os.path.normpath(d)) for d in allowed if os.path.exists(d))

def _is_path_in_allowed_directory(path: str, allowed_dirs: Sequence[str]) -> Tuple[bool, Optional[str]]:
    """
    Check if path is within allowed directories using robust path validation.

    Args:
        path: Path to validate
        allowed_dirs: Sequence of allowed directory paths (list or tuple)

    Returns:
        Tuple of (is_allowed, reason)
        - is_allowed: True if path is in allowed directory
        - reason: If not allowed, reason why; if allowed and in /tmp, warning message
    """
    # Resolve to real path (follows symlinks) and normalize
    try:
        real_path = os.path.realpath(os.path.normpath(path))
    except Exception as e:
        return False, f"Cannot resolve path: {e}"

    # Additional security: detect potential path traversal attempts
    if '..' in path or path != os.path.normpath(path):
        logger.warning(f"Path traversal attempt detected: {path}")

    # Check if path is within any allowed directory using robust method
    for allowed_dir in allowed_dirs:
        try:
            # Normalize both paths for consistent comparison
            allowed_dir_real = os.path.realpath(os.path.normpath(allowed_dir))

            # Method 1: Check if real_path starts with allowed_dir
            # Must check with os.sep to prevent partial matches (e.g., /home vs /home2)
            if real_path == allowed_dir_real:
                # Exact match to allowed directory
                if allowed_dir_real == os.path.realpath("/tmp"):
                    return True, f"⚠️ File in temporary directory: {real_path}"
                return True, None
            elif real_path.startswith(allowed_dir_real + os.sep):
                # Subdirectory of allowed directory
                # Additional check: ensure no path components escape the allowed dir
                try:
                    # Verify path doesn't escape via symlinks
                    common_path = os.path.commonpath([real_path, allowed_dir_real])
                    if common_path != allowed_dir_real:
                        continue  # Path escapes, check next allowed dir
                except (ValueError, TypeError):
                    continue  # Invalid comparison, check next allowed dir

                if allowed_dir_real == os.path.realpath("/tmp"):
                    return True, f"⚠️ File in temporary directory: {real_path}"
                return True, None

        except Exception as e:
            logger.debug(f"Error checking path against {allowed_dir}: {e}")
            continue

    # Not in any allowed directory
    return False, f"File must be in an allowed directory (current directory, home, Downloads, Documents, or Models)"

def _validate_file_path(
    path: str,
    file_type: str,
    allowed_extensions: Tuple[str, ...],
    min_size_mb: int,
    max_size_mb: int
) -> Tuple[bool, str]:
    """
    Common file path validation logic.

    Args:
        path: Path to validate
        file_type: Display name for error messages (e.g., "Model", "DoRA")
        allowed_extensions: Tuple of allowed file extensions
        min_size_mb: Minimum file size in MB
        max_size_mb: Maximum file size in MB

    Returns:
        Tuple of (is_valid, result)
        - is_valid: True if valid, False otherwise
        - result: If valid, normalized path; if invalid, error message
    """
    if not path.strip():
        return False, f"Please provide a {file_type.lower()} path"

    try:
        # Normalize and validate path (resolves all '..' and makes absolute)
        normalized_path = os.path.normpath(os.path.abspath(path))

        if not os.path.exists(normalized_path):
            return False, f"{file_type} file not found: {normalized_path}"

        if not os.path.isfile(normalized_path):
            return False, "Path must point to a file, not a directory"

        # Check file extension
        if not any(normalized_path.lower().endswith(ext) for ext in allowed_extensions):
            return False, f"Unsupported format. Expected: {', '.join(allowed_extensions)}"

        # Check file size
        file_size = os.path.getsize(normalized_path)
        min_size_bytes = min_size_mb * 1024 * 1024
        max_size_bytes = max_size_mb * 1024 * 1024

        if file_size < min_size_bytes:
            return False, f"File too small ({format_file_size(file_size)}). Expected > {min_size_mb}MB"

        if file_size > max_size_bytes:
            return False, f"File too large ({format_file_size(file_size)}). Expected < {max_size_mb}MB"

        # Security: Check directory containment
        allowed_dirs = _get_allowed_directories()
        is_allowed, reason = _is_path_in_allowed_directory(normalized_path, allowed_dirs)

        if not is_allowed:
            return False, f"Security: {reason}"

        # If allowed but has warning (e.g., /tmp), log it
        if reason:
            logger.warning(f"{file_type} path security warning: {reason}")

        return True, normalized_path

    except (IOError, OSError) as e:
        return False, f"Path access error: {str(e)}"
    except (ValueError, TypeError) as e:
        return False, f"Path validation error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error validating path: {str(e)}"


def validate_model_path(path: str) -> Tuple[bool, str]:
    """Validate model path (supports both single files and diffusers directories)."""
    if not path.strip():
        return False, "Please provide a model path"

    try:
        # Normalize and validate path
        normalized_path = os.path.normpath(os.path.abspath(path))

        if not os.path.exists(normalized_path):
            return False, f"Model not found: {normalized_path}"

        # Support both files (.safetensors) and directories (diffusers format)
        if os.path.isdir(normalized_path):
            # Diffusers directory format - check for required subdirectories
            unet_path = os.path.join(normalized_path, "unet")
            vae_path = os.path.join(normalized_path, "vae")

            if not os.path.isdir(unet_path):
                return False, f"Invalid diffusers directory: missing 'unet' subdirectory"

            if not os.path.isdir(vae_path):
                return False, f"Invalid diffusers directory: missing 'vae' subdirectory"

            # Security: Check directory containment
            allowed_dirs = _get_allowed_directories()
            is_allowed, reason = _is_path_in_allowed_directory(normalized_path, allowed_dirs)

            if not is_allowed:
                return False, f"Security: {reason}"

            if reason:
                logger.warning(f"Model path security warning: {reason}")

            return True, normalized_path

        else:
            # Single file format - use original validation
            return _validate_file_path(
                path=path,
                file_type="Model",
                allowed_extensions=MODEL_CONFIG.SUPPORTED_FORMATS,
                min_size_mb=MODEL_CONFIG.MIN_FILE_SIZE_MB,
                max_size_mb=MODEL_CONFIG.MAX_FILE_SIZE_GB * 1024  # Convert GB to MB
            )

    except Exception as e:
        return False, f"Path validation error: {str(e)}"

def get_safe_csv_paths() -> Dict[str, str]:
    """Get validated CSV file paths with enhanced security checks."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    style_dir = os.path.join(script_dir, "style")

    if not os.path.exists(style_dir):
        logger.warning(f"Style directory not found: {style_dir}")
        return {}

    # Normalize and verify style directory is within script directory
    style_dir_real = os.path.realpath(os.path.normpath(style_dir))
    script_dir_real = os.path.realpath(os.path.normpath(script_dir))

    if not style_dir_real.startswith(script_dir_real + os.sep):
        logger.error("Style directory outside of script directory - security violation")
        return {}

    csv_files = {
        'danbooru_character': "danbooru_character_webui.csv",
        'e621_character': "e621_character_webui.csv",
        'danbooru_artist': "danbooru_artist_webui.csv",
        'e621_artist': "e621_artist_webui.csv"
    }

    validated_paths = {}
    for key, filename in csv_files.items():
        # Use basename to strip any path components (security)
        safe_filename = os.path.basename(filename)

        # Additional check: verify filename hasn't been manipulated
        if safe_filename != filename:
            logger.warning(f"Filename contains path components, using basename only: {filename} -> {safe_filename}")

        # Verify no path traversal characters
        if '..' in safe_filename or os.sep in safe_filename:
            logger.warning(f"Invalid filename detected: {safe_filename}")
            continue

        full_path = os.path.join(style_dir, safe_filename)
        full_path_real = os.path.realpath(os.path.normpath(full_path))

        # Verify the resolved path is still within style directory
        if full_path_real.startswith(style_dir_real + os.sep) and os.path.isfile(full_path_real):
            validated_paths[key] = full_path_real
        else:
            logger.debug(f"CSV file not found or outside safe directory: {safe_filename}")

    return validated_paths

def normalize_text(text: str) -> str:
    """Normalize Unicode text and strip whitespace."""
    if not text:
        return ""
    return unicodedata.normalize('NFC', text.strip())

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable binary format (1 KiB = 1024 bytes)."""
    for unit in ['B', 'KiB', 'MiB', 'GiB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TiB"

def calculate_image_hash(file_path: str) -> str:
    """Calculate MD5 hash of an image file using memory-efficient chunked reading."""
    hash_md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b""):  # 64KB chunks
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def validate_dora_path(path: str) -> Tuple[bool, str]:
    """Validate DoRA adapter path with comprehensive checks including directory containment."""
    return _validate_file_path(
        path=path,
        file_type="DoRA",
        allowed_extensions=('.safetensors',),
        min_size_mb=MODEL_CONFIG.DORA_MIN_FILE_SIZE_MB,
        max_size_mb=MODEL_CONFIG.DORA_MAX_FILE_SIZE_MB
    )

def detect_base_model_precision(model_path: str) -> torch.dtype:
    """Detect and validate model precision (BF16 or pre-converted FP32 supported)."""
    try:
        # Check if model_path is a directory (diffusers format) or file (safetensors)
        if os.path.isdir(model_path):
            # Diffusers directory format - check UNet config for precision
            logger.info(f"Detecting precision from diffusers directory: {model_path}")

            # Check for UNet safetensors file
            unet_path = os.path.join(model_path, "unet", "diffusion_pytorch_model.safetensors")
            if not os.path.exists(unet_path):
                # Try sharded format
                unet_path = os.path.join(model_path, "unet", "diffusion_pytorch_model.fp32.safetensors")

            if os.path.exists(unet_path):
                # Read safetensors header from UNet
                with open(unet_path, 'rb') as f:
                    header_size = struct.unpack('<Q', f.read(8))[0]
                    header_data = json.loads(f.read(header_size).decode('utf-8'))

                # Get dtype from first tensor
                for key, value in header_data.items():
                    if key != '__metadata__' and isinstance(value, dict) and 'dtype' in value:
                        dtype_str = value['dtype']
                        detected_dtype = DTYPE_MAP.get(dtype_str)

                        if detected_dtype == torch.float32:
                            logger.info(f"FP32 pre-converted model detected (lossless from BF16)")
                            return detected_dtype
                        elif detected_dtype == torch.bfloat16:
                            logger.info(f"BF16 model detected (canonical)")
                            return detected_dtype
                        elif detected_dtype == torch.float16:
                            raise ValueError(
                                f"FP16 model detected. FP16 models are NOT supported. "
                                f"Use BF16 model or FP32 pre-converted for lossless quality."
                            )
                        break

            # Fallback: assume FP32 if it's a known pre-converted directory
            if "FP32" in os.path.basename(model_path):
                logger.info(f"FP32 pre-converted model assumed from directory name")
                return torch.float32

            raise ValueError(f"Could not detect precision from directory: {model_path}")

        else:
            # Single file format - read safetensors header
            with open(model_path, 'rb') as f:
                # Read header size (8 bytes)
                header_size = struct.unpack('<Q', f.read(8))[0]
                # Read header JSON (typically ~350KB vs 7.0GB full model)
                header_data = json.loads(f.read(header_size).decode('utf-8'))

            # Find first UNet tensor dtype from header metadata
            unet_tensors = {k: v for k, v in header_data.items()
                           if k != '__metadata__' and 'model.diffusion_model' in k}

            if unet_tensors:
                # Get dtype from first UNet tensor
                dtype_str = list(unet_tensors.values())[0]['dtype']
                detected_dtype = DTYPE_MAP.get(dtype_str)

                if detected_dtype is None:
                    raise ValueError(
                        f"Unsupported model precision: {dtype_str}. "
                        f"Only BF16 model (NoobAI-XL-Vpred-v1.0.safetensors) is supported. "
                        f"FP16 models are NOT supported due to lossy quantization."
                    )

                if detected_dtype == torch.float16:
                    raise ValueError(
                        f"FP16 model detected. FP16 models are NOT supported. "
                        f"Use BF16 model (NoobAI-XL-Vpred-v1.0.safetensors) for lossless quality "
                        f"and cross-platform parity."
                    )

                logger.info(f"Model precision: {detected_dtype} (validated)")
                return detected_dtype

        # Default to BF16 (canonical format)
        logger.info("Using BF16 as default for SDXL model")
        return torch.bfloat16

    except ValueError:
        # Re-raise validation errors
        raise
    except (IOError, OSError) as e:
        logger.error(f"Could not read model file: {e}")
        raise
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Could not parse model header: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error detecting model precision: {e}")
        raise

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

        except (IOError, OSError, PermissionError) as e:
            logger.warning(f"Error accessing directory {search_dir}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error scanning directory {search_dir}: {e}")

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

def parse_manual_dora_schedule(schedule_input: Optional[str], num_steps: int) -> Tuple[Optional[List[int]], Optional[str]]:
    """
    Parse and validate manual DoRA schedule from CSV string or list.

    Args:
        schedule_input: CSV string (e.g., "1, 0, 0, 1") or None
        num_steps: Total number of diffusion steps

    Returns:
        Tuple of (normalized_schedule, warning_message)
        - normalized_schedule: List of 0/1 values with length=num_steps, or None if invalid
        - warning_message: Warning string if issues found, None otherwise

    Rules:
        - Format: comma+space-separated 0/1 values (e.g., "1, 0, 0, 1")
        - If entries < steps: missing positions are 0 (OFF)
        - If entries > steps: extras are ignored
        - Any non-0/1 token is treated as 0
        - If malformed: return None schedule with warning
    """
    if not schedule_input or not schedule_input.strip():
        return None, None

    try:
        # Split by comma and strip whitespace
        parts = [p.strip() for p in schedule_input.split(',')]

        # Parse each part to 0 or 1
        schedule = []
        had_invalid_tokens = False

        for part in parts:
            if part == '1':
                schedule.append(1)
            elif part == '0':
                schedule.append(0)
            else:
                # Invalid token - treat as 0
                schedule.append(0)
                had_invalid_tokens = True

        # Check if we have any valid data
        if not schedule:
            return None, "Manual DoRA schedule is empty or malformed - DoRA will be OFF for all steps"

        # Build warning message
        warning_parts = []
        if had_invalid_tokens:
            warning_parts.append("some entries were invalid (non-0/1) and treated as 0")

        # Adjust to num_steps
        if len(schedule) < num_steps:
            # Pad with zeros
            diff = num_steps - len(schedule)
            schedule.extend([0] * diff)
            warning_parts.append(f"{diff} missing step(s) set to OFF")
        elif len(schedule) > num_steps:
            # Truncate
            diff = len(schedule) - num_steps
            schedule = schedule[:num_steps]
            warning_parts.append(f"{diff} extra step(s) ignored")

        warning = f"Manual DoRA schedule: {', '.join(warning_parts)}" if warning_parts else None
        return schedule, warning

    except Exception as e:
        logger.warning(f"Failed to parse manual DoRA schedule: {e}")
        return None, f"Manual DoRA schedule is malformed ({str(e)}) - DoRA will be OFF for all steps"

def generate_standard_schedule(num_steps: int) -> List[int]:
    """
    Generate standard toggle schedule: ON,OFF,ON,OFF throughout all steps.

    Args:
        num_steps: Total number of diffusion steps

    Returns:
        List of 0/1 values (even indices=1/ON, odd indices=0/OFF)
    """
    return [1 if i % 2 == 0 else 0 for i in range(num_steps)]

def generate_smart_schedule(num_steps: int) -> List[int]:
    """
    Generate smart toggle schedule: ON,OFF through step 20, then ON for remainder.

    Args:
        num_steps: Total number of diffusion steps

    Returns:
        List of 0/1 values
    """
    schedule = []
    for i in range(num_steps):
        if i <= 19:
            # Alternating phase (indices 0-19)
            schedule.append(1 if i % 2 == 0 else 0)
        else:
            # Always ON phase (index 20+)
            schedule.append(1)
    return schedule

# Initialize CSV paths
CSV_PATHS = get_safe_csv_paths()
