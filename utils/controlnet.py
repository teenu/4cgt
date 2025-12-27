"""ControlNet discovery and management utilities."""

import os
import glob
import time
from typing import List, Dict, Any, Optional
from PIL import Image
from config import logger, CONTROLNET_SEARCH_DIRECTORIES, CONTROLNET_CONFIG
from utils.formatting import format_file_size

# Cache for discover_controlnet_models to avoid redundant filesystem scans
_models_cache: List[Dict[str, Any]] = []
_models_cache_time: float = 0.0
_CACHE_TTL_SECONDS: float = 5.0


def detect_controlnet_precision(model_path: str) -> str:
    """Detect ControlNet model precision from filename heuristic.

    Returns precision string for display purposes.
    """
    filename_lower = os.path.basename(model_path).lower()
    if "_fp16" in filename_lower or "fp16" in filename_lower:
        return "fp16"
    elif "_bf16" in filename_lower or "bf16" in filename_lower:
        return "bfloat16"
    elif "_fp32" in filename_lower or "fp32" in filename_lower:
        return "fp32"
    return "unknown"


def detect_controlnet_type(model_path: str) -> str:
    """Detect ControlNet type from filename.

    Returns type string (e.g., 'openpose', 'canny', 'depth').
    """
    filename_lower = os.path.basename(model_path).lower()

    type_keywords = [
        'openpose', 'pose',
        'canny', 'edge',
        'depth', 'midas',
        'normal', 'normals',
        'seg', 'segmentation',
        'lineart', 'line',
        'scribble', 'sketch',
        'softedge', 'hed',
        'mlsd', 'line_segment',
        'shuffle', 'ip2p',
        'inpaint', 'tile'
    ]

    for keyword in type_keywords:
        if keyword in filename_lower:
            return keyword

    return "unknown"


def validate_controlnet_path(path: str) -> tuple[bool, str]:
    """Validate ControlNet model path.

    Args:
        path: Path to ControlNet safetensors file

    Returns:
        Tuple of (is_valid, message/path)
    """
    if not path.strip():
        return False, "Please provide a ControlNet path"

    try:
        normalized_path = os.path.normpath(os.path.abspath(path))

        if not os.path.exists(normalized_path):
            return False, f"ControlNet file not found: {normalized_path}"

        if not os.path.isfile(normalized_path):
            return False, "Path must point to a file, not a directory"

        if not normalized_path.lower().endswith('.safetensors'):
            return False, "Unsupported format. Expected: .safetensors"

        file_size = os.path.getsize(normalized_path)
        min_size_bytes = CONTROLNET_CONFIG.MIN_FILE_SIZE_MB * 1024 * 1024
        max_size_bytes = CONTROLNET_CONFIG.MAX_FILE_SIZE_GB * 1024 * 1024 * 1024

        if file_size < min_size_bytes:
            return False, f"File too small ({format_file_size(file_size)}). Expected > {CONTROLNET_CONFIG.MIN_FILE_SIZE_MB}MB"

        if file_size > max_size_bytes:
            return False, f"File too large ({format_file_size(file_size)}). Expected < {CONTROLNET_CONFIG.MAX_FILE_SIZE_GB}GB"

        return True, normalized_path

    except (IOError, OSError) as e:
        return False, f"Path access error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error validating path: {str(e)}"


def discover_controlnet_models(force_refresh: bool = False) -> List[Dict[str, Any]]:
    """Discover all ControlNet model files in search directories.

    Results are cached for 5 seconds to avoid redundant filesystem scans.

    Args:
        force_refresh: If True, bypass the cache and rescan directories.

    Returns:
        List of dictionaries containing model information.
    """
    global _models_cache, _models_cache_time

    current_time = time.time()
    if not force_refresh and _models_cache and (current_time - _models_cache_time) < _CACHE_TTL_SECONDS:
        return _models_cache.copy()

    models = []
    seen_names = set()

    for search_dir in CONTROLNET_SEARCH_DIRECTORIES:
        if not os.path.exists(search_dir):
            continue

        try:
            search_pattern = os.path.join(search_dir, "*.safetensors")
            for model_path in glob.glob(search_pattern):
                if not os.path.isfile(model_path):
                    continue

                is_valid, validated_path = validate_controlnet_path(model_path)
                if not is_valid:
                    continue

                model_name = os.path.basename(validated_path)

                if model_name in seen_names:
                    continue
                seen_names.add(model_name)

                file_size = os.path.getsize(validated_path)
                precision = detect_controlnet_precision(validated_path)
                model_type = detect_controlnet_type(validated_path)

                models.append({
                    'name': model_name,
                    'path': validated_path,
                    'size': file_size,
                    'size_formatted': format_file_size(file_size),
                    'precision': precision,
                    'type': model_type,
                    'display_name': f"{model_name} ({format_file_size(file_size)}, {precision})"
                })

        except (IOError, OSError, PermissionError) as e:
            logger.warning(f"Error accessing directory {search_dir}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error scanning directory {search_dir}: {e}")

    models.sort(key=lambda x: x['name'])

    # Update cache
    _models_cache = models.copy()
    _models_cache_time = time.time()

    return models


def find_controlnet_path(model_type: str = "openpose") -> Optional[str]:
    """Find a ControlNet model by type.

    Args:
        model_type: Type of ControlNet to find (e.g., 'openpose', 'canny')

    Returns:
        Path to the first matching model, or None if not found.
    """
    models = discover_controlnet_models()

    # First, try exact type match
    for model in models:
        if model['type'] == model_type:
            return model['path']

    # Fall back to checking if type is in filename
    model_type_lower = model_type.lower()
    for model in models:
        if model_type_lower in model['name'].lower():
            return model['path']

    return None


def get_controlnet_by_name(model_name: str) -> Optional[Dict[str, Any]]:
    """Get ControlNet info by filename.

    Args:
        model_name: Filename of the ControlNet model

    Returns:
        Dictionary with model information, or None if not found.
    """
    models = discover_controlnet_models()
    for model in models:
        if model['name'] == model_name:
            return model
    return None


def validate_pose_image(image: Image.Image) -> tuple[bool, str]:
    """Validate a pose image for ControlNet.

    Args:
        image: PIL Image to validate

    Returns:
        Tuple of (is_valid, message)
    """
    if image is None:
        return False, "No image provided"

    try:
        width, height = image.size

        if width < CONTROLNET_CONFIG.MIN_POSE_DIMENSION:
            return False, f"Image too small: width {width}px < {CONTROLNET_CONFIG.MIN_POSE_DIMENSION}px minimum"

        if height < CONTROLNET_CONFIG.MIN_POSE_DIMENSION:
            return False, f"Image too small: height {height}px < {CONTROLNET_CONFIG.MIN_POSE_DIMENSION}px minimum"

        if width > CONTROLNET_CONFIG.MAX_POSE_DIMENSION:
            return False, f"Image too large: width {width}px > {CONTROLNET_CONFIG.MAX_POSE_DIMENSION}px maximum"

        if height > CONTROLNET_CONFIG.MAX_POSE_DIMENSION:
            return False, f"Image too large: height {height}px > {CONTROLNET_CONFIG.MAX_POSE_DIMENSION}px maximum"

        # Ensure image is in RGB mode
        if image.mode not in ('RGB', 'RGBA', 'L'):
            return False, f"Unsupported image mode: {image.mode}. Expected RGB, RGBA, or L"

        return True, f"Valid pose image: {width}x{height}"

    except Exception as e:
        return False, f"Error validating pose image: {str(e)}"


def preprocess_pose_image(
    image: Image.Image,
    target_width: int,
    target_height: int
) -> Image.Image:
    """Preprocess a pose image for ControlNet.

    Resizes the image to match the target generation resolution while
    preserving the aspect ratio and using appropriate resampling.

    Args:
        image: Input pose image
        target_width: Target width for generation
        target_height: Target height for generation

    Returns:
        Preprocessed PIL Image in RGB format.
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize to match target dimensions
    if image.size != (target_width, target_height):
        image = image.resize(
            (target_width, target_height),
            resample=Image.Resampling.LANCZOS
        )

    return image


def clear_models_cache() -> None:
    """Clear the models cache to force a fresh scan on next discovery."""
    global _models_cache, _models_cache_time
    _models_cache = []
    _models_cache_time = 0.0
