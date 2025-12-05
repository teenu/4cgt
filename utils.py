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
import tempfile
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
    """Get list of allowed directories for model/adapter files."""
    allowed = [
        os.getcwd(),
        os.path.expanduser("~"),
        os.path.expanduser("~/Downloads"),
        os.path.expanduser("~/Documents"),
        os.path.expanduser("~/Models"),
        tempfile.gettempdir(),  # Cross-platform temp directory
    ]

    return tuple(os.path.realpath(os.path.normpath(d)) for d in allowed if os.path.exists(d))

def _is_path_in_allowed_directory(path: str, allowed_dirs: Sequence[str]) -> Tuple[bool, Optional[str]]:
    """Check if path is within allowed directories."""
    try:
        real_path = os.path.realpath(os.path.normpath(path))
    except Exception as e:
        return False, f"Cannot resolve path: {e}"

    # Reject path traversal attempts immediately
    if '..' in path or path != os.path.normpath(path):
        logger.warning(f"Path traversal attempt detected and rejected: {path}")
        return False, "Path traversal attempt rejected for security"

    for allowed_dir in allowed_dirs:
        try:
            allowed_dir_real = os.path.realpath(os.path.normpath(allowed_dir))

            temp_dir_real = os.path.realpath(tempfile.gettempdir())
            if real_path == allowed_dir_real:
                if allowed_dir_real == temp_dir_real:
                    return True, f"⚠️ File in temporary directory: {real_path}"
                return True, None
            elif real_path.startswith(allowed_dir_real + os.sep):
                try:
                    common_path = os.path.commonpath([real_path, allowed_dir_real])
                    if common_path != allowed_dir_real:
                        continue
                except (ValueError, TypeError):
                    continue

                if allowed_dir_real == temp_dir_real:
                    return True, f"⚠️ File in temporary directory: {real_path}"
                return True, None

        except Exception:
            continue

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

        # Windows path length validation (applies to all file types)
        if os.name == 'nt':
            # Check actual byte length of path for accurate validation
            try:
                path_bytes = os.fsencode(normalized_path)
                path_length = len(path_bytes)
            except (UnicodeEncodeError, AttributeError):
                # Fallback to character count if encoding fails
                path_length = len(normalized_path)

            if path_length > 260 and not normalized_path.startswith('\\\\?\\'):
                return False, (
                    f"{file_type} path too long for Windows "
                    f"({path_length} characters, limit 260).\n"
                    f"Solutions:\n"
                    f"1. Move {file_type.lower()} to shorter path (recommended)\n"
                    f"2. Enable long paths: https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation\n"
                    f"3. Use extended-length syntax: \\\\?\\{normalized_path}"
                )

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
        normalized_path = os.path.normpath(os.path.abspath(path))

        # Windows path length validation (for both files and directories)
        if os.name == 'nt':
            # Check actual byte length of path for accurate validation
            try:
                path_bytes = os.fsencode(normalized_path)
                path_length = len(path_bytes)
            except (UnicodeEncodeError, AttributeError):
                # Fallback to character count if encoding fails
                path_length = len(normalized_path)

            if path_length > 260 and not normalized_path.startswith('\\\\?\\'):
                return False, (
                    f"Model path too long for Windows "
                    f"({path_length} characters, limit 260).\n"
                    f"Solutions:\n"
                    f"1. Move model to shorter path (recommended)\n"
                    f"2. Enable long paths: https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation\n"
                    f"3. Use extended-length syntax: \\\\?\\{normalized_path}"
                )

        if not os.path.exists(normalized_path):
            return False, f"Model not found: {normalized_path}"

        if os.path.isdir(normalized_path):
            unet_path = os.path.join(normalized_path, "unet")
            vae_path = os.path.join(normalized_path, "vae")

            if not os.path.isdir(unet_path):
                return False, f"Invalid diffusers directory: missing 'unet' subdirectory"

            if not os.path.isdir(vae_path):
                return False, f"Invalid diffusers directory: missing 'vae' subdirectory"

            allowed_dirs = _get_allowed_directories()
            is_allowed, reason = _is_path_in_allowed_directory(normalized_path, allowed_dirs)

            if not is_allowed:
                return False, f"Security: {reason}"

            if reason:
                logger.warning(f"Model path security warning: {reason}")

            return True, normalized_path

        else:
            return _validate_file_path(
                path=path,
                file_type="Model",
                allowed_extensions=MODEL_CONFIG.SUPPORTED_FORMATS,
                min_size_mb=MODEL_CONFIG.MIN_FILE_SIZE_MB,
                max_size_mb=MODEL_CONFIG.MAX_FILE_SIZE_GB * 1024
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
        # Check for path traversal BEFORE basename (security)
        if '..' in filename or '/' in filename or '\\' in filename:
            logger.warning(f"Filename contains path components: {filename}")
            continue

        # Use basename to strip any remaining path components
        safe_filename = os.path.basename(filename)

        # Verify basename didn't change filename (additional safety check)
        if safe_filename != filename:
            logger.warning(f"Filename was modified by basename: {filename} -> {safe_filename}")
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

def calculate_image_hash(file_path: str) -> Optional[str]:
    """Calculate MD5 hash of an image file.

    Returns:
        MD5 hash string on success, None on failure.
    """
    try:
        hash_md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except (IOError, OSError) as e:
        logger.error(f"Failed to calculate hash for {file_path}: {e}")
        return None

def validate_dora_path(path: str) -> Tuple[bool, str]:
    """Validate DoRA adapter path."""
    return _validate_file_path(
        path=path,
        file_type="DoRA",
        allowed_extensions=('.safetensors',),
        min_size_mb=MODEL_CONFIG.DORA_MIN_FILE_SIZE_MB,
        max_size_mb=MODEL_CONFIG.DORA_MAX_FILE_SIZE_MB
    )

def detect_base_model_precision(model_path: str) -> torch.dtype:
    """Detect and validate model precision.

    Returns the detected dtype or raises ValueError for unsupported/invalid models.
    FP16 models are explicitly rejected with a clear error message.

    Supports:
    - Single safetensors files (diffusion_pytorch_model.safetensors)
    - Sharded safetensors files (diffusion_pytorch_model-00001-of-00002.safetensors)
    - FP32 variant files (diffusion_pytorch_model.fp32.safetensors)
    """
    try:
        if os.path.isdir(model_path):
            logger.info(f"Detecting precision from diffusers directory: {model_path}")

            unet_dir = os.path.join(model_path, "unet")

            if not os.path.isdir(unet_dir):
                raise ValueError(
                    f"Invalid diffusers directory: expected UNet weights under {unet_dir}. "
                    "Re-download the FP32 directory and ensure unet/ contains diffusion_pytorch_model*.safetensors files."
                )

            unet_path = None

            # Priority order for finding safetensors file:
            # 1. Single file: diffusion_pytorch_model.safetensors
            # 2. FP32 variant: diffusion_pytorch_model.fp32.safetensors
            # 3. Sharded files: diffusion_pytorch_model-00001-of-*.safetensors (via index.json)

            single_file = os.path.join(unet_dir, "diffusion_pytorch_model.safetensors")
            fp32_file = os.path.join(unet_dir, "diffusion_pytorch_model.fp32.safetensors")
            index_file = os.path.join(unet_dir, "diffusion_pytorch_model.safetensors.index.json")

            if os.path.exists(single_file):
                unet_path = single_file
                logger.debug(f"Using single safetensors file: {unet_path}")
            elif os.path.exists(fp32_file):
                unet_path = fp32_file
                logger.debug(f"Using FP32 safetensors file: {unet_path}")
            elif os.path.exists(index_file):
                # Handle sharded safetensors: read index.json to find first shard
                try:
                    with open(index_file, 'r') as f:
                        index_data = json.load(f)

                    # Index file has format: {"metadata": {...}, "weight_map": {"layer_name": "shard_file.safetensors"}}
                    weight_map = index_data.get('weight_map', {})
                    if weight_map:
                        shard_files = set(weight_map.values())
                        sorted_shards = sorted(shard_files)
                        if sorted_shards:
                            first_shard = sorted_shards[0]
                            unet_path = os.path.join(unet_dir, first_shard)
                            logger.debug(f"Using sharded safetensors (first shard): {unet_path}")
                        else:
                            raise ValueError(
                                f"Sharded UNet index found at {index_file} but no shard filenames listed. "
                                "Re-download the FP32 directory to restore missing diffusion_pytorch_model-*-of-*.safetensors files."
                            )
                    else:
                        raise ValueError(
                            f"Sharded UNet index found at {index_file} without a weight_map. "
                            "Re-download the FP32 directory to restore missing diffusion_pytorch_model-*-of-*.safetensors files."
                        )
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    raise ValueError(
                        f"Could not parse {index_file}: {e}. "
                        "Ensure unet/ contains valid diffusion_pytorch_model.safetensors shards."
                    )

            # Fallback: look for any sharded safetensors file pattern
            if unet_path is None or not os.path.exists(unet_path):
                import glob as glob_module
                shard_pattern = os.path.join(unet_dir, "diffusion_pytorch_model-*-of-*.safetensors")
                shard_files = sorted(glob_module.glob(shard_pattern))
                if shard_files:
                    unet_path = shard_files[0]  # Use first shard
                    logger.debug(f"Using sharded safetensors (glob fallback): {unet_path}")

            if unet_path and os.path.exists(unet_path):
                with open(unet_path, 'rb') as f:
                    header_size = struct.unpack('<Q', f.read(8))[0]
                    header_data = json.loads(f.read(header_size).decode('utf-8'))

                dtype_found = False
                for key, value in header_data.items():
                    if key != '__metadata__' and isinstance(value, dict) and 'dtype' in value:
                        dtype_found = True
                        dtype_str = value['dtype']
                        detected_dtype = DTYPE_MAP.get(dtype_str)

                        # Check for FP16 first with clear rejection message
                        if detected_dtype == torch.float16:
                            raise ValueError(
                                "FP16 model detected. FP16 models are NOT supported due to lossy quantization. "
                                "Please use the BF16 (.safetensors) or FP32 (directory) model format."
                            )
                        elif detected_dtype == torch.float32:
                            logger.info("FP32 model detected from safetensors header")
                            return detected_dtype
                        elif detected_dtype == torch.bfloat16:
                            logger.info("BF16 model detected from safetensors header")
                            return detected_dtype
                        elif detected_dtype is None:
                            raise ValueError(
                                f"Unsupported model precision: {dtype_str}. "
                                "Re-download the model to ensure diffusion_pytorch_model*.safetensors is valid."
                            )
                        break

                if not dtype_found:
                    raise ValueError(
                        f"No dtype metadata found in {unet_path}. "
                        "Ensure diffusion_pytorch_model*.safetensors is valid or re-download the FP32 directory."
                    )

            raise ValueError(
                "Could not detect precision from directory. Expected one of:\n"
                f"- {single_file}\n"
                f"- {fp32_file}\n"
                f"- {index_file} with diffusion_pytorch_model-00001-of-*.safetensors shards\n"
                "Re-download the FP32 directory; ensure unet/ contains diffusion_pytorch_model*.safetensors."
            )

        else:
            with open(model_path, 'rb') as f:
                header_size = struct.unpack('<Q', f.read(8))[0]
                header_data = json.loads(f.read(header_size).decode('utf-8'))

            unet_tensors = {k: v for k, v in header_data.items()
                           if k != '__metadata__' and 'model.diffusion_model' in k}

            if unet_tensors:
                dtype_str = list(unet_tensors.values())[0]['dtype']
                detected_dtype = DTYPE_MAP.get(dtype_str)

                if detected_dtype is None:
                    raise ValueError(f"Unsupported model precision: {dtype_str}")

                # Check for FP16 with clear rejection message
                if detected_dtype == torch.float16:
                    raise ValueError(
                        "FP16 model detected. FP16 models are NOT supported due to lossy quantization. "
                        "Please use the BF16 (.safetensors) or FP32 (directory) model format."
                    )

                logger.info(f"Model precision: {detected_dtype}")
                return detected_dtype

        # If we reach this point, precision could not be determined safely
        raise ValueError(
            "Could not detect model precision. Ensure the model is a supported "
            "BF16 .safetensors file or FP32 diffusers directory."
        )

    except ValueError:
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

    # Validate input length to prevent memory exhaustion
    if len(schedule_input) > 10000:
        return None, "Manual DoRA schedule too long (max 10000 characters) - DoRA will be OFF for all steps"

    # Validate num_steps
    if num_steps <= 0:
        return None, "Invalid number of steps"

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
    if num_steps <= 0:
        return []
    return [1 if i % 2 == 0 else 0 for i in range(num_steps)]

def generate_smart_schedule(num_steps: int) -> List[int]:
    """
    Generate smart toggle schedule: ON,OFF through step 20, then ON for remainder.

    Args:
        num_steps: Total number of diffusion steps

    Returns:
        List of 0/1 values
    """
    if num_steps <= 0:
        return []
    schedule = []
    for i in range(num_steps):
        if i <= 19:
            # Alternating phase (indices 0-19)
            schedule.append(1 if i % 2 == 0 else 0)
        else:
            # Always ON phase (index 20+)
            schedule.append(1)
    return schedule


def generate_optimized_schedule(num_steps: int) -> List[int]:
    """
    Generate optimized DoRA schedule based on empirically-tuned 34-step pattern.

    This pattern was optimized for NoobAI V-Pred with the stabilizer DoRA adapter.
    It achieves ~44% DoRA activation with strategic phase alignment:
    - OFF during high-noise structure formation
    - Selective pulses during mid-noise detail emergence
    - ON continuously during low-noise refinement

    Args:
        num_steps: Total number of diffusion steps

    Returns:
        List of 0/1 values scaled to the requested step count
    """
    from config import OPTIMIZED_DORA_SCHEDULE

    if num_steps <= 0:
        return []

    base_schedule = OPTIMIZED_DORA_SCHEDULE
    base_len = len(base_schedule)  # 34

    if num_steps == base_len:
        return list(base_schedule)

    # Scale the schedule to match num_steps
    # Use linear interpolation to preserve phase proportions
    result = []
    for i in range(num_steps):
        # Map current step to base schedule index
        base_idx = int(i * base_len / num_steps)
        base_idx = min(base_idx, base_len - 1)
        result.append(base_schedule[base_idx])

    return result


# Initialize CSV paths
CSV_PATHS = get_safe_csv_paths()
