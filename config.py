#!/usr/bin/env python3
"""
NoobAI XL V-Pred 1.0 - Configuration and Constants

This module contains all configuration constants, dataclasses, and custom exceptions
for the NoobAI application.
"""

import os
from dataclasses import dataclass
from typing import Tuple
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import safetensors.torch as safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

# DTYPE_MAP includes F16 for proper rejection message when FP16 models are detected
DTYPE_MAP = {
    'F32': torch.float32,
    'BF16': torch.bfloat16,
    'F16': torch.float16,  # Added for proper FP16 rejection message
    'FLOAT': torch.float32,
}

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

@dataclass
class ModelConfig:
    """Model configuration constants."""
    MIN_FILE_SIZE_MB: int = 100
    MAX_FILE_SIZE_GB: int = 50
    SUPPORTED_FORMATS: Tuple[str, ...] = ('.safetensors',)

    # DoRA adapter configuration
    DORA_MIN_FILE_SIZE_MB: int = 1
    DORA_MAX_FILE_SIZE_MB: int = 500
    MIN_ADAPTER_STRENGTH: float = 0.0
    MAX_ADAPTER_STRENGTH: float = 2.0
    DEFAULT_ADAPTER_STRENGTH: float = 1.0
    MIN_DORA_START_STEP: int = 0  # 0-based indexing
    MAX_DORA_START_STEP: int = 100
    DEFAULT_DORA_START_STEP: int = 0  # 0-based indexing

@dataclass
class GenerationConfig:
    """Generation configuration constants."""
    MAX_PROMPT_LENGTH: int = 5000  # Increased from 1000 for long prompts
    MIN_RESOLUTION: int = 256
    MAX_RESOLUTION: int = 2048
    MIN_STEPS: int = 1
    MAX_STEPS: int = 100
    MIN_CFG_SCALE: float = 1.0
    MAX_CFG_SCALE: float = 20.0
    MIN_RESCALE_CFG: float = 0.0
    MAX_RESCALE_CFG: float = 1.0

@dataclass
class SearchConfig:
    """Search configuration constants."""
    MIN_QUERY_LENGTH: int = 2
    MAX_QUERY_LENGTH: int = 100
    MAX_RESULTS: int = 15
    MAX_RESULTS_PER_SOURCE: int = 50
    INDEX_PREFIX_LENGTH: int = 3

class SearchScoring:
    """Constants for search result scoring."""
    EXACT_MATCH: int = 3
    PREFIX_MATCH: int = 2
    CONTAINS_MATCH: int = 1

@dataclass
class PromptConfig:
    """Prompt processing configuration for long prompt support."""
    MAX_CHUNKS: int = 8       # Maximum ~600 tokens (8 * 75 + overhead)
    WARN_CHUNKS: int = 4      # Warn above ~300 tokens
    TOKEN_LIMIT: int = 77     # CLIP token limit per encoder
    CHUNK_SIZE: int = 75      # Usable tokens per chunk (77 - BOS/EOS)

@dataclass
class ControlNetConfig:
    """ControlNet configuration constants."""
    MIN_FILE_SIZE_MB: int = 100
    MAX_FILE_SIZE_GB: int = 10
    SUPPORTED_FORMATS: Tuple[str, ...] = ('.safetensors',)

    # Conditioning scale configuration
    # V-prediction models require higher ControlNet scales (~2.0) for proper pose adherence
    MIN_CONDITIONING_SCALE: float = 0.0
    MAX_CONDITIONING_SCALE: float = 3.0
    DEFAULT_CONDITIONING_SCALE: float = 2.0

    # Pose image configuration
    MIN_POSE_DIMENSION: int = 64
    MAX_POSE_DIMENSION: int = 4096

# Create configuration instances
MODEL_CONFIG = ModelConfig()
GEN_CONFIG = GenerationConfig()
SEARCH_CONFIG = SearchConfig()
PROMPT_CONFIG = PromptConfig()
CONTROLNET_CONFIG = ControlNetConfig()

# Output directory for generated images
# Use 'outputs' subdirectory in script location for persistent storage
_script_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_script_dir, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# User-friendly error messages
USER_FRIENDLY_ERRORS = {
    "CUDA out of memory": "GPU memory full. Try: 1) Reduce resolution, 2) Restart the app, or 3) Close other GPU applications.",
    "MPS backend out of memory": "Mac GPU memory full. Try reducing resolution or restarting the app.",
    "Expected all tensors to be on the same device": "Device mismatch error. Please restart the application.",
    "cannot allocate memory": "System out of memory. Close other applications and try again.",
    "no space left on device": "Disk full. Free up space and try again.",
    "RuntimeError: CUDA error": "GPU error. Try restarting the application or your computer.",
}

# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class NoobAIError(Exception):
    """Base exception for NoobAI application."""
    pass

class ModelNotFoundError(NoobAIError):
    """Raised when the NoobAI model file cannot be found."""
    pass

class EngineNotInitializedError(NoobAIError):
    """Raised when trying to generate with uninitialized engine."""
    pass

class InvalidParameterError(NoobAIError):
    """Raised when invalid parameters are provided."""
    pass

class GenerationInterruptedError(NoobAIError):
    """Raised when generation is interrupted by user."""
    pass

# Check pandas availability
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# ============================================================================
# NOOBAI CONFIGURATION
# ============================================================================

# Official NoobAI supported resolutions
OFFICIAL_RESOLUTIONS = [
    (768, 1344), (832, 1216), (896, 1152), (1024, 1024),
    (1152, 896), (1216, 832), (1344, 768)
]

# Recommended resolutions (highest quality)
RECOMMENDED_RESOLUTIONS = [(832, 1216), (1216, 832)]

# Optimal settings
OPTIMAL_SETTINGS = {
    'steps': 35,
    'cfg_scale': 4.5,
    'rescale_cfg': 0.7,
    'width': 1216,
    'height': 832,
    'adapter_strength': 1.0,
    'dora_start_step': 0,  # 0-based indexing
}

# Optimized DoRA mode settings (locked parameters)
OPTIMIZED_DORA_SETTINGS = {
    'steps': 34,
    'cfg_scale': 4.2,
    'rescale_cfg': 0.55,
    'adapter_strength': 1.0,
    'schedule': [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
}
OPTIMIZED_DORA_SCHEDULE_CSV = ", ".join(str(x) for x in OPTIMIZED_DORA_SETTINGS['schedule'])

# DoRA None mode settings (default when DoRA enabled, toggle mode is None)
# These settings provide optimal results for anatomical accuracy with v-prediction models.
# Mathematical basis:
# - CFG 5.5261: Lower CFG optimal for v-prediction due to zero terminal SNR interaction
# - Rescale 0.6092: Compensates for overexposure while preserving prompt adherence
# - DoRA start step 3: Preserves semantic phase (steps 0-2) for structural formation
# - Steps 42: Ensures full convergence with adequate adapter influence window
DORA_NONE_MODE_SETTINGS = {
    'steps': 42,
    'cfg_scale': 5.5261,
    'rescale_cfg': 0.6092,
    'adapter_strength': 1.0,
    'dora_start_step': 3,
}

# Optimal parameter ranges for generation info warnings
# Extended ranges to accommodate both standard and DoRA None mode settings
OPTIMAL_STEPS_RANGE = (32, 45)
OPTIMAL_CFG_RANGE = (3.5, 6.0)

# Default prompts
DEFAULT_NEGATIVE_PROMPT = "worst aesthetic, worst quality, lowres, scan artifacts, ai-generated, old, 4koma, multiple views, furry, anthro, watermark, logo, signature, artist name, bad hands, extra digits, fewer digits"
DEFAULT_POSITIVE_PREFIX = "very awa, masterpiece, best quality, year 2024, newest, highres, absurdres"

_model_filenames = [
    "NoobAI-XL-Vpred-v1.0.safetensors",
]

_model_directories = [
    "NoobAI-XL-Vpred-v1.0-FP32",
]

_search_directories = [
    _script_dir,
    os.path.join(_script_dir, "models"),
    os.path.join(os.path.expanduser("~"), "Downloads"),
    os.path.join(os.path.expanduser("~"), "Models"),
]

if 'NOOBAI_MODEL_PATH' in os.environ:
    custom_path = os.environ['NOOBAI_MODEL_PATH']
    if os.path.isdir(custom_path):
        _search_directories.append(custom_path)

MODEL_SEARCH_PATHS = [
    os.path.join(directory, dirname)
    for directory in _search_directories
    for dirname in _model_directories
] + [
    os.path.join(directory, filename)
    for directory in _search_directories
    for filename in _model_filenames
]

DORA_SEARCH_DIRECTORIES = [
    os.path.join(_script_dir, "dora"),
    _script_dir,
    os.path.join(os.path.expanduser("~"), "Downloads", "dora")
]

CONTROLNET_SEARCH_DIRECTORIES = [
    os.path.join(_script_dir, "controlnet"),
    _script_dir,
    os.path.join(os.path.expanduser("~"), "Models", "controlnet"),
    os.path.join(os.path.expanduser("~"), "Downloads", "controlnet")
]
