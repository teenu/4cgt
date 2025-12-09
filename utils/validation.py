"""Path and file validation utilities."""

import os
import json
import struct
import torch
from typing import Tuple, Optional, Sequence, Dict
from config import logger, MODEL_CONFIG, DTYPE_MAP
from utils.formatting import format_file_size


def _get_allowed_directories() -> tuple:
    """Get list of allowed directories for model/adapter files."""
    allowed = [
        os.getcwd(),
        os.path.expanduser("~"),
        os.path.expanduser("~/Downloads"),
        os.path.expanduser("~/Documents"),
        os.path.expanduser("~/Models"),
    ]
    return tuple(os.path.realpath(os.path.normpath(d)) for d in allowed if os.path.exists(d))


def _is_path_in_allowed_directory(path: str, allowed_dirs: Sequence[str]) -> Tuple[bool, Optional[str]]:
    """Check if path is within allowed directories."""
    try:
        real_path = os.path.realpath(os.path.normpath(path))
    except Exception as e:
        return False, f"Cannot resolve path: {e}"

    if '..' in path or path != os.path.normpath(path):
        logger.warning(f"Path traversal attempt detected and rejected: {path}")
        return False, "Path traversal attempt rejected for security"

    for allowed_dir in allowed_dirs:
        try:
            allowed_dir_real = os.path.realpath(os.path.normpath(allowed_dir))

            if real_path == allowed_dir_real:
                return True, None
            elif real_path.startswith(allowed_dir_real + os.sep):
                try:
                    common_path = os.path.commonpath([real_path, allowed_dir_real])
                    if common_path != allowed_dir_real:
                        continue
                except (ValueError, TypeError):
                    continue

                return True, None

        except (ValueError, TypeError, OSError):
            continue

    return False, f"File must be in an allowed directory (current directory, home, Downloads, Documents, or Models)"


def _validate_file_path(
    path: str,
    file_type: str,
    allowed_extensions: Tuple[str, ...],
    min_size_mb: int,
    max_size_mb: int
) -> Tuple[bool, str]:
    """Common file path validation logic."""
    if not path.strip():
        return False, f"Please provide a {file_type.lower()} path"

    try:
        normalized_path = os.path.normpath(os.path.abspath(path))

        if os.name == 'nt':
            try:
                path_bytes = os.fsencode(normalized_path)
                path_length = len(path_bytes)
            except (UnicodeEncodeError, AttributeError):
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

        if not any(normalized_path.lower().endswith(ext) for ext in allowed_extensions):
            return False, f"Unsupported format. Expected: {', '.join(allowed_extensions)}"

        file_size = os.path.getsize(normalized_path)
        min_size_bytes = min_size_mb * 1024 * 1024
        max_size_bytes = max_size_mb * 1024 * 1024

        if file_size < min_size_bytes:
            return False, f"File too small ({format_file_size(file_size)}). Expected > {min_size_mb}MB"

        if file_size > max_size_bytes:
            return False, f"File too large ({format_file_size(file_size)}). Expected < {max_size_mb}MB"

        allowed_dirs = _get_allowed_directories()
        is_allowed, reason = _is_path_in_allowed_directory(normalized_path, allowed_dirs)

        if not is_allowed:
            return False, f"Security: {reason}"

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

        if os.name == 'nt':
            try:
                path_bytes = os.fsencode(normalized_path)
                path_length = len(path_bytes)
            except (UnicodeEncodeError, AttributeError):
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


def validate_dora_path(path: str) -> Tuple[bool, str]:
    """Validate DoRA adapter path."""
    return _validate_file_path(
        path=path,
        file_type="DoRA",
        allowed_extensions=('.safetensors',),
        min_size_mb=MODEL_CONFIG.DORA_MIN_FILE_SIZE_MB,
        max_size_mb=MODEL_CONFIG.DORA_MAX_FILE_SIZE_MB
    )


def get_safe_csv_paths() -> Dict[str, str]:
    """Get validated CSV file paths with enhanced security checks."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    style_dir = os.path.join(parent_dir, "style")

    if not os.path.exists(style_dir):
        logger.warning(f"Style directory not found: {style_dir}")
        return {}

    style_dir_real = os.path.realpath(os.path.normpath(style_dir))
    parent_dir_real = os.path.realpath(os.path.normpath(parent_dir))

    if not style_dir_real.startswith(parent_dir_real + os.sep):
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
        if '..' in filename or '/' in filename or '\\' in filename:
            logger.warning(f"Filename contains path components: {filename}")
            continue

        safe_filename = os.path.basename(filename)

        if safe_filename != filename:
            logger.warning(f"Filename was modified by basename: {filename} -> {safe_filename}")
            continue

        full_path = os.path.join(style_dir, safe_filename)
        full_path_real = os.path.realpath(os.path.normpath(full_path))

        if full_path_real.startswith(style_dir_real + os.sep) and os.path.isfile(full_path_real):
            validated_paths[key] = full_path_real
        else:
            logger.debug(f"CSV file not found or outside safe directory: {safe_filename}")

    return validated_paths


def detect_base_model_precision(model_path: str) -> torch.dtype:
    """Detect and validate model precision."""
    try:
        if os.path.isdir(model_path):
            logger.info(f"Detecting precision from diffusers directory: {model_path}")

            unet_path = os.path.join(model_path, "unet", "diffusion_pytorch_model.safetensors")
            if not os.path.exists(unet_path):
                unet_path = os.path.join(model_path, "unet", "diffusion_pytorch_model.fp32.safetensors")

            if os.path.exists(unet_path):
                with open(unet_path, 'rb') as f:
                    header_size = struct.unpack('<Q', f.read(8))[0]
                    header_data = json.loads(f.read(header_size).decode('utf-8'))

                for key, value in header_data.items():
                    if key != '__metadata__' and isinstance(value, dict) and 'dtype' in value:
                        dtype_str = value['dtype']
                        detected_dtype = DTYPE_MAP.get(dtype_str)

                        if detected_dtype == torch.float16:
                            raise ValueError(
                                "FP16 model detected. FP16 models are NOT supported due to lossy quantization. "
                                "Please use the BF16 (.safetensors) or FP32 (directory) model format."
                            )
                        elif detected_dtype == torch.float32:
                            logger.info("FP32 model detected")
                            return detected_dtype
                        elif detected_dtype == torch.bfloat16:
                            logger.info("BF16 model detected")
                            return detected_dtype
                        elif detected_dtype is None:
                            raise ValueError(f"Unsupported model precision: {dtype_str}")
                        break

            if "FP32" in os.path.basename(model_path):
                logger.info("FP32 model assumed from directory name")
                return torch.float32

            raise ValueError(f"Could not detect precision from directory: {model_path}")

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

                if detected_dtype == torch.float16:
                    raise ValueError(
                        "FP16 model detected. FP16 models are NOT supported due to lossy quantization. "
                        "Please use the BF16 (.safetensors) or FP32 (directory) model format."
                    )

                logger.info(f"Model precision: {detected_dtype}")
                return detected_dtype

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
