#!/usr/bin/env python3
"""
NoobAI XL V-Pred 1.0 - Professional GUI (Hash Consistency Edition)

This version fixes the hash mismatch between CLI and GUI modes by ensuring
consistent image saving across both interfaces.

Key improvements in this edition:
- Fixed hash consistency between CLI and GUI modes
- Standardized PNG saving with consistent compression
- Enhanced performance with search indexing
- Improved error handling with user-friendly messages
- Better resource management and cleanup
- Optional performance monitoring
"""

import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from PIL import Image
from PIL import PngImagePlugin
import gradio as gr
import random
import gc
import os
import re
import csv
import time
import threading
import atexit
import contextlib
import argparse
import sys
import tempfile
import shutil
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Dict, Any, List, Callable
import unicodedata
import logging
import hashlib
import glob
import json
import struct

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import safetensors.torch as safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    logger.warning("safetensors not available. Adapter precision detection will be limited.")

# Dtype mapping for efficient header-based precision detection
DTYPE_MAP = {
    'F32': torch.float32,
    'F16': torch.float16,
    'BF16': torch.bfloat16,
    'FLOAT': torch.float32,
    'HALF': torch.float16
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
    MIN_DORA_START_STEP: int = 1
    MAX_DORA_START_STEP: int = 100
    DEFAULT_DORA_START_STEP: int = 1

@dataclass
class GenerationConfig:
    """Generation configuration constants."""
    MAX_PROMPT_LENGTH: int = 1000
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

# Create configuration instances
MODEL_CONFIG = ModelConfig()
GEN_CONFIG = GenerationConfig()
SEARCH_CONFIG = SearchConfig()

# Output directory for generated images
OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "noobai_outputs")
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
    logger.warning("pandas not available. CSV functionality will be limited.")

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
    'dora_start_step': 1,
}

# Default prompts
DEFAULT_NEGATIVE_PROMPT = "worst aesthetic, worst quality, lowres, scan artifacts, ai-generated, old, 4koma, multiple views, furry, anthro, watermark, logo, signature, artist name, bad hands, extra digits, fewer digits"
DEFAULT_POSITIVE_PREFIX = "very awa, masterpiece, best quality, year 2024, newest, highres, absurdres"

# Model search paths
MODEL_SEARCH_PATHS = [
    "./NoobAI-XL-Vpred-v1.0.safetensors",
    "./models/NoobAI-XL-Vpred-v1.0.safetensors",
    os.path.join(os.path.expanduser("~"), "Downloads", "NoobAI-XL-Vpred-v1.0.safetensors"),
    os.path.join(os.path.expanduser("~"), "Models", "NoobAI-XL-Vpred-v1.0.safetensors"),
    os.path.join(os.getcwd(), "NoobAI-XL-Vpred-v1.0.safetensors")
]

# DoRA adapter search directories
DORA_SEARCH_DIRECTORIES = [
    "./dora/",
    "./",
    os.path.join(os.getcwd(), "dora"),
    os.path.join(os.path.expanduser("~"), "Downloads", "dora")
]

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

CSV_PATHS = get_safe_csv_paths()

# ============================================================================
# NOOBAI ENGINE
# ============================================================================

class NoobAIEngine:
    """Clean, modular NoobAI engine with optimal configuration."""
    
    def __init__(self, model_path: str, enable_dora: bool = False, dora_path: Optional[str] = None, adapter_strength: float = MODEL_CONFIG.DEFAULT_ADAPTER_STRENGTH, dora_start_step: int = MODEL_CONFIG.DEFAULT_DORA_START_STEP):
        self.model_path = model_path
        self.enable_dora = enable_dora
        self.dora_path = dora_path
        self.adapter_strength = adapter_strength
        self.dora_start_step = dora_start_step
        self.dora_loaded = False
        self.pipe = None
        self.is_initialized = False
        self._device = None
        self._initialize()

    def _initialize(self):
        """Initialize the diffusion pipeline."""
        try:
            with perf_monitor.time_section("engine_initialization"):
                logger.info(f"Initializing NoobAI engine with model: {self.model_path}")
                
                # Detect device
                if torch.backends.mps.is_available():
                    self._device = "mps"
                elif torch.cuda.is_available():
                    self._device = "cuda"
                else:
                    self._device = "cpu"
                    
                logger.info(f"Using device: {self._device.upper()}")

                # Detect base model precision and use consistently
                base_precision = detect_base_model_precision(self.model_path)
                inference_dtype = base_precision if self._device != "cpu" else torch.float32
                
                logger.info(f"Using {inference_dtype} precision on {self._device.upper()}")

                # Load pipeline with detected precision
                self.pipe = StableDiffusionXLPipeline.from_single_file(
                    self.model_path,
                    torch_dtype=inference_dtype,
                    use_safetensors=True,
                )
                
                # Configure scheduler
                self.pipe.scheduler = EulerDiscreteScheduler.from_config(
                    self.pipe.scheduler.config,
                    prediction_type="v_prediction",
                    rescale_betas_zero_snr=True,
                    timestep_spacing="trailing"
                )

                # Move to device and enable optimizations
                self.pipe = self.pipe.to(self._device)
                if self._device != "cpu":
                    self.pipe.enable_vae_slicing()
                    if self._device == "cuda":
                        self.pipe.enable_attention_slicing()
                
                # Validate precision consistency
                pipeline_dtype = next(self.pipe.unet.parameters()).dtype
                logger.info(f"Pipeline initialized with {pipeline_dtype} precision")
                
                self.is_initialized = True
                logger.info("NoobAI engine initialized successfully")
                
                # Load DoRA adapter if enabled
                if self.enable_dora:
                    self._load_dora_adapter()
                
        except Exception as e:
            self.is_initialized = False
            logger.error(f"Failed to initialize engine: {e}")
            raise

    def _load_dora_adapter(self):
        """Load DoRA adapter if available and valid with precision detection."""
        try:
            dora_path = self.dora_path
            if not dora_path:
                # Try to auto-detect DoRA path
                dora_path = find_dora_path()
                if not dora_path:
                    logger.warning("DoRA enabled but no valid DoRA file found")
                    return
            
            # Validate DoRA path
            is_valid, validated_path = validate_dora_path(dora_path)
            if not is_valid:
                logger.warning(f"DoRA validation failed: {validated_path}")
                return
            
            # Log precision information
            adapter_precision = detect_adapter_precision(validated_path)
            pipeline_dtype = next(self.pipe.unet.parameters()).dtype
            
            logger.info(f"Loading DoRA adapter: {validated_path}")
            logger.info(f"Adapter stored as: {adapter_precision}, Pipeline using: {pipeline_dtype}")
            
            if adapter_precision == "fp16" and pipeline_dtype == torch.bfloat16:
                logger.info("DoRA adapter will be automatically converted from FP16 to BF16")
            elif adapter_precision == "fp16" and pipeline_dtype == torch.float32:
                logger.info("DoRA adapter will be automatically converted from FP16 to FP32")
            
            # Set path early to ensure it's available for error reporting
            self.dora_path = validated_path
            
            # Load DoRA adapter using the LoRA loading mechanism
            # The diffusers library will handle precision conversion automatically
            self.pipe.load_lora_weights(
                os.path.dirname(validated_path),
                weight_name=os.path.basename(validated_path),
                adapter_name="noobai_dora"
            )
            
            # Set adapter scale
            self.pipe.set_adapters(["noobai_dora"], adapter_weights=[self.adapter_strength])
            
            self.dora_loaded = True
            logger.info(f"DoRA adapter loaded successfully with {pipeline_dtype} precision")
            
        except Exception as e:
            logger.error(f"Failed to load DoRA adapter: {e}")
            self.dora_loaded = False

    def unload_dora_adapter(self):
        """Completely unload DoRA adapter with full memory cleanup."""
        try:
            if self.dora_loaded and self.pipe is not None:
                logger.info("Completely unloading DoRA adapter")
                
                # 1. First disable adapter by setting weight to 0
                try:
                    self.pipe.set_adapters(["noobai_dora"], adapter_weights=[0.0])
                except Exception as e:
                    logger.warning(f"Could not set adapter weights to 0: {e}")
                
                # 2. Completely remove LoRA weights from memory
                try:
                    self.pipe.unload_lora_weights()
                    logger.info("LoRA weights completely unloaded from memory")
                except Exception as e:
                    logger.warning(f"Error unloading LoRA weights: {e}")
                
                # 3. Delete adapter references if supported
                try:
                    if hasattr(self.pipe, 'delete_adapters'):
                        self.pipe.delete_adapters(["noobai_dora"])
                        logger.info("Adapter references deleted")
                except Exception as e:
                    logger.warning(f"Error deleting adapter references: {e}")
                
                # 4. Clear memory caches to ensure cleanup
                self.clear_memory()
                
                logger.info("DoRA adapter completely unloaded")
                
            self.dora_loaded = False
            self.dora_path = None
            
        except Exception as e:
            logger.error(f"Error completely unloading DoRA adapter: {e}")
            self.dora_loaded = False
            self.dora_path = None

    def switch_dora_adapter(self, new_adapter_path: str) -> bool:
        """Switch DoRA adapters with complete cleanup and fresh loading."""
        try:
            if not new_adapter_path:
                logger.error("Cannot switch to empty adapter path")
                return False
                
            # Validate new adapter path
            is_valid, validated_path = validate_dora_path(new_adapter_path)
            if not is_valid:
                logger.error(f"Invalid new adapter path: {validated_path}")
                return False
                
            logger.info(f"Switching DoRA adapter to: {validated_path}")
            
            # 1. Complete unload of current adapter
            if self.dora_loaded:
                self.unload_dora_adapter()
                
            # 2. Brief pause to ensure complete cleanup
            time.sleep(0.1)
            
            # 3. Update adapter path
            self.dora_path = validated_path
            
            # 4. Load new adapter with fresh state
            self._load_dora_adapter()
            
            if self.dora_loaded:
                logger.info(f"Successfully switched to DoRA adapter: {os.path.basename(validated_path)}")
                return True
            else:
                logger.error("Failed to load new DoRA adapter")
                return False
                
        except Exception as e:
            logger.error(f"Error switching DoRA adapter: {e}")
            self.dora_loaded = False
            self.dora_path = None
            return False

    def set_adapter_strength(self, strength: float):
        """Set DoRA adapter strength."""
        try:
            # Validate strength is in bounds
            if not (MODEL_CONFIG.MIN_ADAPTER_STRENGTH <= strength <= MODEL_CONFIG.MAX_ADAPTER_STRENGTH):
                logger.warning(f"Adapter strength {strength} out of bounds [{MODEL_CONFIG.MIN_ADAPTER_STRENGTH}-{MODEL_CONFIG.MAX_ADAPTER_STRENGTH}], clamping")
                strength = max(MODEL_CONFIG.MIN_ADAPTER_STRENGTH, min(strength, MODEL_CONFIG.MAX_ADAPTER_STRENGTH))
            
            if self.dora_loaded and self.pipe is not None:
                self.adapter_strength = strength
                # Only apply if DoRA is currently enabled
                if self.enable_dora:
                    self.pipe.set_adapters(["noobai_dora"], adapter_weights=[strength])
                    logger.info(f"Adapter strength set to {strength}")
                else:
                    logger.info(f"Adapter strength set to {strength} (will apply when DoRA is enabled)")
            else:
                # Store the strength even if DoRA is not loaded yet
                self.adapter_strength = strength
                logger.info(f"Adapter strength stored as {strength} (DoRA not loaded)")
        except Exception as e:
            logger.warning(f"Error setting adapter strength: {e}")

    def set_dora_start_step(self, start_step: int):
        """Set DoRA adapter start step."""
        try:
            # Validate start step is in bounds
            if not (MODEL_CONFIG.MIN_DORA_START_STEP <= start_step <= MODEL_CONFIG.MAX_DORA_START_STEP):
                logger.warning(f"DoRA start step {start_step} out of bounds [{MODEL_CONFIG.MIN_DORA_START_STEP}-{MODEL_CONFIG.MAX_DORA_START_STEP}], clamping")
                start_step = max(MODEL_CONFIG.MIN_DORA_START_STEP, min(start_step, MODEL_CONFIG.MAX_DORA_START_STEP))
            
            self.dora_start_step = start_step
            logger.info(f"DoRA start step set to {start_step}")
            
        except Exception as e:
            logger.warning(f"Error setting DoRA start step: {e}")

    def set_dora_enabled(self, enabled: bool):
        """Dynamically enable/disable DoRA adapter."""
        try:
            if enabled:
                # Validate DoRA is available before enabling
                if not self.dora_loaded:
                    logger.warning("Cannot enable DoRA: adapter not loaded")
                    self.enable_dora = False
                    return
                
                if self.pipe is None:
                    logger.warning("Cannot enable DoRA: pipeline not initialized")
                    self.enable_dora = False
                    return
                
                # Validate adapter strength is in bounds
                if not (MODEL_CONFIG.MIN_ADAPTER_STRENGTH <= self.adapter_strength <= MODEL_CONFIG.MAX_ADAPTER_STRENGTH):
                    logger.warning(f"Invalid adapter strength {self.adapter_strength}, using default")
                    self.adapter_strength = MODEL_CONFIG.DEFAULT_ADAPTER_STRENGTH
                
                # Re-enable with current strength
                self.pipe.set_adapters(["noobai_dora"], adapter_weights=[self.adapter_strength])
                self.enable_dora = True
                logger.info(f"DoRA adapter enabled (strength: {self.adapter_strength})")
            else:
                # Disable by setting weight to 0
                if self.dora_loaded and self.pipe is not None:
                    self.pipe.set_adapters(["noobai_dora"], adapter_weights=[0.0])
                self.enable_dora = False
                logger.info("DoRA adapter disabled")
        except Exception as e:
            logger.warning(f"Error setting DoRA enabled state: {e}")
            self.enable_dora = False

    def get_dora_info(self) -> Dict[str, Any]:
        """Get DoRA adapter information."""
        return {
            'enabled': self.enable_dora,
            'loaded': self.dora_loaded,
            'path': self.dora_path,
            'strength': self.adapter_strength if self.dora_loaded else 0.0,
            'start_step': self.dora_start_step
        }

    def save_image_standardized(self, image: Image.Image, output_path: str, 
                               include_metadata: bool = True) -> str:
        """Save image with standardized settings for consistent hashing."""
        # Create a copy to avoid modifying the original
        img_copy = image.copy()
        
        # Prepare PNG metadata
        pnginfo = None
        if include_metadata and hasattr(image, 'info') and image.info:
            pnginfo = PngImagePlugin.PngInfo()
            
            # Add metadata in a consistent order (sorted keys)
            for key in sorted(image.info.keys()):
                pnginfo.add_text(key, str(image.info[key]))
        
        # Save with consistent parameters
        img_copy.save(
            output_path,
            format='PNG',
            pnginfo=pnginfo,
            compress_level=6,  # Standard compression level
            optimize=False  # Disable optimization for consistency
        )
        
        return output_path

    def generate(
        self,
        prompt: str,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        width: int = OPTIMAL_SETTINGS['width'],
        height: int = OPTIMAL_SETTINGS['height'],
        steps: int = OPTIMAL_SETTINGS['steps'],
        cfg_scale: float = OPTIMAL_SETTINGS['cfg_scale'],
        rescale_cfg: float = OPTIMAL_SETTINGS['rescale_cfg'],
        seed: Optional[int] = None,
        adapter_strength: Optional[float] = None,
        enable_dora: Optional[bool] = None,
        dora_start_step: Optional[int] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[Image.Image, int, str]:
        """Generate an image with the specified parameters."""
        if not self.is_initialized:
            raise EngineNotInitializedError("NoobAI engine is not initialized")

        # Handle dynamic DoRA enable/disable state
        if enable_dora is not None:
            self.set_dora_enabled(enable_dora)

        # Apply adapter strength if DoRA is enabled and strength is provided
        if adapter_strength is not None and self.enable_dora and self.dora_loaded:
            self.set_adapter_strength(adapter_strength)
        
        # Apply DoRA start step if provided
        if dora_start_step is not None:
            self.set_dora_start_step(dora_start_step)

        with perf_monitor.time_section("image_generation"):
            # Build info message
            info_parts = []
            if not (32 <= steps <= 40):
                info_parts.append(f"⚠️ Steps {steps} outside optimal range 32-40")
            if not (3.5 <= cfg_scale <= 5.5):
                info_parts.append(f"⚠️ CFG {cfg_scale} outside optimal range 3.5-5.5")
            
            current_res = (height, width)
            if current_res in RECOMMENDED_RESOLUTIONS:
                info_parts.append(f"✅ Optimal resolution: {width}x{height}")
            elif current_res in OFFICIAL_RESOLUTIONS:
                info_parts.append(f"✅ Official resolution: {width}x{height}")
            else:
                info_parts.append(f"⚠️ Non-official resolution: {width}x{height}")

            # Set up seed
            if seed is None:
                seed = random.randint(0, 2**32 - 1)
            generator = torch.Generator(self._device).manual_seed(seed)
            
            # Generation callback
            start_time = time.time()
            def callback_on_step_end(pipe, step_index, timestep, callback_kwargs):
                if state_manager.is_interrupted():
                    raise GenerationInterruptedError()
                
                current_step = step_index + 1
                progress = current_step / steps
                elapsed = time.time() - start_time
                eta = (elapsed / current_step) * (steps - current_step) if current_step > 0 else 0
                
                # Handle DoRA start step control
                if self.enable_dora and self.dora_loaded and self.pipe is not None:
                    if current_step == 1 and self.dora_start_step > 1:
                        # Deactivate DoRA at the beginning if start step is later
                        self.pipe.set_adapters(["noobai_dora"], adapter_weights=[0.0])
                        desc = f"Step {current_step}/{steps} (DoRA starts at step {self.dora_start_step}, ETA: {eta:.1f}s)"
                    elif current_step == self.dora_start_step:
                        # Activate DoRA adapter at the specified start step
                        self.pipe.set_adapters(["noobai_dora"], adapter_weights=[self.adapter_strength])
                        desc = f"Step {current_step}/{steps} (DoRA activated, ETA: {eta:.1f}s)"
                    elif current_step < self.dora_start_step:
                        desc = f"Step {current_step}/{steps} (DoRA starts at step {self.dora_start_step}, ETA: {eta:.1f}s)"
                    else:
                        desc = f"Step {current_step}/{steps} (DoRA active, ETA: {eta:.1f}s)"
                else:
                    desc = f"Step {current_step}/{steps} (ETA: {eta:.1f}s)"
                
                if progress_callback:
                    progress_callback(progress, desc)
                return callback_kwargs

            try:
                # Generate image
                with torch.no_grad():
                    result = self.pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        num_inference_steps=steps,
                        guidance_scale=cfg_scale,
                        guidance_rescale=rescale_cfg,
                        generator=generator,
                        output_type="pil",
                        return_dict=True,
                        callback_on_step_end=callback_on_step_end,
                        callback_on_step_end_tensor_inputs=["latents"]
                    )
                
                info_parts.append(f"🌱 Generated with seed: {seed}")
                
                # Add DoRA information to info
                if self.dora_loaded:
                    dora_name = os.path.basename(self.dora_path) if self.dora_path else "DoRA"
                    if self.enable_dora:
                        info_parts.append(f"🎯 DoRA: {dora_name} (strength: {self.adapter_strength})")
                    else:
                        info_parts.append(f"⚪ DoRA: {dora_name} (disabled)")
                elif self.dora_path:  # DoRA file exists but not loaded
                    info_parts.append("⚠️ DoRA: Available but not loaded")
                
                image = result.images[0]
                
                # Add metadata
                metadata = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "seed": str(seed),
                    "width": str(width),
                    "height": str(height),
                    "steps": str(steps),
                    "cfg_scale": str(cfg_scale),
                    "rescale_cfg": str(rescale_cfg),
                    "model": "NoobAI-XL-Vpred-v1.0",
                    "scheduler": "EulerDiscreteScheduler"
                }
                
                # Add DoRA information to metadata
                if self.dora_loaded:
                    metadata["dora_enabled"] = str(self.enable_dora).lower()
                    metadata["dora_path"] = os.path.basename(self.dora_path) if self.dora_path else "unknown"
                    if self.enable_dora:
                        metadata["adapter_strength"] = str(self.adapter_strength)
                    else:
                        metadata["adapter_strength"] = "0.0"
                
                image.info = metadata
                
                return image, seed, "\n".join(info_parts)
                
            finally:
                self.clear_memory()

    def teardown_engine(self):
        """Comprehensive engine teardown with full resource cleanup."""
        try:
            logger.info("Performing comprehensive engine teardown")
            
            # 1. Unload any DoRA adapters completely
            if self.pipe and self.dora_loaded:
                try:
                    self.pipe.unload_lora_weights()  # Complete removal vs set_adapters(0)
                    if hasattr(self.pipe, 'delete_adapters'):
                        self.pipe.delete_adapters(["noobai_dora"])  # Remove adapter references
                    logger.info("DoRA adapters completely unloaded")
                except Exception as e:
                    logger.warning(f"Error unloading DoRA adapters: {e}")
            
            # 2. Clear pipeline components
            if self.pipe:
                try:
                    # Move to CPU to free GPU/MPS memory
                    self.pipe = self.pipe.to("cpu")
                    
                    # Delete individual components to ensure cleanup
                    components_to_delete = ['unet', 'vae', 'text_encoder', 'text_encoder_2', 'scheduler']
                    for component_name in components_to_delete:
                        if hasattr(self.pipe, component_name):
                            component = getattr(self.pipe, component_name)
                            if component is not None:
                                del component
                                setattr(self.pipe, component_name, None)
                    
                    logger.info("Pipeline components cleaned up")
                except Exception as e:
                    logger.warning(f"Error cleaning pipeline components: {e}")
                
                # 3. Delete entire pipeline
                try:
                    del self.pipe
                    self.pipe = None
                    logger.info("Pipeline object deleted")
                except Exception as e:
                    logger.warning(f"Error deleting pipeline: {e}")
            
            # 4. Clear device caches with synchronization
            try:
                if self._device == "mps":
                    torch.mps.empty_cache()
                    torch.mps.synchronize()  # Wait for MPS operations to complete
                elif self._device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # Wait for CUDA operations
                    if hasattr(torch.cuda, 'ipc_collect'):
                        torch.cuda.ipc_collect()  # Clear inter-process memory
                logger.info(f"Device caches cleared for {self._device}")
            except Exception as e:
                logger.warning(f"Error clearing device caches: {e}")
            
            # 5. Force garbage collection multiple times for thorough cleanup
            for i in range(3):
                collected = gc.collect()
                if i == 0:
                    logger.info(f"Garbage collection freed {collected} objects")
            
            # 6. Reset all state variables
            self.dora_loaded = False
            self.dora_path = None
            self.is_initialized = False
            self._device = None
            
            logger.info("Engine teardown completed successfully")
            
        except Exception as e:
            logger.error(f"Error during comprehensive engine teardown: {e}")
            # Ensure critical state is reset even if teardown fails partially
            self.pipe = None
            self.dora_loaded = False
            self.is_initialized = False

    def clear_memory(self):
        """Clear GPU/memory caches."""
        try:
            if self._device == "mps":
                torch.mps.empty_cache()
            elif self._device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            logger.warning(f"Could not clear memory cache: {e}")

# ============================================================================
# INDEXED PROMPT FORMATTER
# ============================================================================

class IndexedPromptFormatterData:
    """Enhanced prompt formatter with search indexing for better performance."""
    
    def __init__(self):
        self.character_data = {'danbooru': [], 'e621': []}
        self.artist_data = {'danbooru': [], 'e621': []}
        self.char_index = {'danbooru': {}, 'e621': {}}
        self.artist_index = {'danbooru': {}, 'e621': {}}
        self.is_loaded = False
        self.load_data()
    
    def load_data(self):
        """Load CSV data."""
        if not CSV_PATHS:
            return
            
        try:
            with perf_monitor.time_section("csv_loading"):
                if PANDAS_AVAILABLE:
                    self._load_with_pandas()
                else:
                    self._load_with_csv()
                    
                self._build_indices()
                self.is_loaded = True
                logger.info("CSV data loaded and indexed successfully.")
                
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            self.is_loaded = False

    def _load_with_pandas(self):
        """Load CSV data using pandas."""
        for source in ['danbooru', 'e621']:
            # Character data
            char_path = CSV_PATHS.get(f'{source}_character')
            if char_path:
                df = pd.read_csv(char_path)
                for _, row in df.iterrows():
                    character_entry = {
                        'trigger': str(row.get('trigger', '')),
                        'source': source,
                        'character': str(row.get('character', '')),
                        'copyright': str(row.get('copyright', '')),
                        'core_tags': str(row.get('core_tags', '')) if source == 'danbooru' else ''
                    }
                    self.character_data[source].append(character_entry)
            
            # Artist data
            artist_path = CSV_PATHS.get(f'{source}_artist')
            if artist_path:
                df = pd.read_csv(artist_path)
                for _, row in df.iterrows():
                    artist_entry = {
                        'trigger': str(row.get('trigger', '')),
                        'source': source,
                        'artist': str(row.get('artist', ''))
                    }
                    self.artist_data[source].append(artist_entry)

    def _load_with_csv(self):
        """Load CSV data without pandas."""
        for source in ['danbooru', 'e621']:
            # Character data
            char_path = CSV_PATHS.get(f'{source}_character')
            if char_path:
                with open(char_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        character_entry = {
                            'trigger': str(row.get('trigger', '')),
                            'source': source,
                            'character': str(row.get('character', '')),
                            'copyright': str(row.get('copyright', '')),
                            'core_tags': str(row.get('core_tags', '')) if source == 'danbooru' else ''
                        }
                        self.character_data[source].append(character_entry)
            
            # Artist data
            artist_path = CSV_PATHS.get(f'{source}_artist')
            if artist_path:
                with open(artist_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        artist_entry = {
                            'trigger': str(row.get('trigger', '')),
                            'source': source,
                            'artist': str(row.get('artist', ''))
                        }
                        self.artist_data[source].append(artist_entry)

    def _build_indices(self):
        """Build search indices for faster lookups."""
        with perf_monitor.time_section("index_building"):
            # Build character indices
            for source in ['danbooru', 'e621']:
                for item in self.character_data[source]:
                    trigger_lower = item['trigger'].lower()
                    # Index by prefixes for faster prefix matching
                    for i in range(SEARCH_CONFIG.MIN_QUERY_LENGTH, 
                                   min(SEARCH_CONFIG.INDEX_PREFIX_LENGTH + 1, len(trigger_lower) + 1)):
                        prefix = trigger_lower[:i]
                        if prefix not in self.char_index[source]:
                            self.char_index[source][prefix] = []
                        self.char_index[source][prefix].append(item)
                
                # Build artist indices
                for item in self.artist_data[source]:
                    trigger_lower = item['trigger'].lower()
                    for i in range(SEARCH_CONFIG.MIN_QUERY_LENGTH,
                                   min(SEARCH_CONFIG.INDEX_PREFIX_LENGTH + 1, len(trigger_lower) + 1)):
                        prefix = trigger_lower[:i]
                        if prefix not in self.artist_index[source]:
                            self.artist_index[source][prefix] = []
                        self.artist_index[source][prefix].append(item)

    def _calculate_search_score(self, query_lower: str, trigger_lower: str) -> int:
        """Calculate search relevance score."""
        if trigger_lower == query_lower:
            return SearchScoring.EXACT_MATCH
        elif trigger_lower.startswith(query_lower):
            return SearchScoring.PREFIX_MATCH
        else:
            return SearchScoring.CONTAINS_MATCH

    def search(self, query: str, data_type: str, limit: int = 10) -> List[Dict]:
        """Search for entries with optimized indexing."""
        if not self.is_loaded or not query.strip():
            return []
            
        with perf_monitor.time_section(f"search_{data_type}"):
            query = query.strip()[:SEARCH_CONFIG.MAX_QUERY_LENGTH]
            limit = min(limit, SEARCH_CONFIG.MAX_RESULTS)
            
            if len(query) < SEARCH_CONFIG.MIN_QUERY_LENGTH:
                return []
            
            query_lower = query.lower()
            
            # Select appropriate index and data
            if data_type == 'character':
                index = self.char_index
                full_data = self.character_data
            else:
                index = self.artist_index
                full_data = self.artist_data
            
            all_results = []
            seen_triggers = set()
            
            # First, check indexed entries for prefix matches
            for source in ['danbooru', 'e621']:
                # Use index for fast prefix lookup
                prefix_key = query_lower[:min(len(query_lower), SEARCH_CONFIG.INDEX_PREFIX_LENGTH)]
                indexed_items = index[source].get(prefix_key, [])
                
                for item in indexed_items:
                    trigger_lower = item['trigger'].lower()
                    if trigger_lower in seen_triggers:
                        continue
                        
                    if query_lower in trigger_lower:
                        score = self._calculate_search_score(query_lower, trigger_lower)
                        result = {
                            'display': item['trigger'],
                            'source': source,
                            'score': score
                        }
                        
                        # Add value based on type and source
                        if data_type == 'character' and source == 'danbooru' and item['core_tags']:
                            result['value'] = f"{item['core_tags']}, {item['trigger']}"
                        else:
                            result['value'] = item['trigger']
                            
                        all_results.append(result)
                        seen_triggers.add(trigger_lower)
                
                # Fallback to full search if not enough results
                if len(all_results) < limit:
                    for item in full_data[source]:
                        trigger_lower = item['trigger'].lower()
                        if trigger_lower in seen_triggers:
                            continue
                            
                        if query_lower in trigger_lower:
                            score = self._calculate_search_score(query_lower, trigger_lower)
                            result = {
                                'display': item['trigger'],
                                'source': source,
                                'score': score
                            }
                            
                            if data_type == 'character' and source == 'danbooru' and item['core_tags']:
                                result['value'] = f"{item['core_tags']}, {item['trigger']}"
                            else:
                                result['value'] = item['trigger']
                                
                            all_results.append(result)
                            seen_triggers.add(trigger_lower)
            
            # Sort by score and source
            all_results.sort(key=lambda x: (x['score'], x['source']), reverse=True)
            
            # Balance results between sources
            final_results = []
            danbooru_count = 0
            e621_count = 0
            max_per_source = (limit + 1) // 2
            
            for result in all_results:
                if result['source'] == 'danbooru' and danbooru_count < max_per_source:
                    final_results.append(result)
                    danbooru_count += 1
                elif result['source'] == 'e621' and e621_count < max_per_source:
                    final_results.append(result)
                    e621_count += 1
                    
                if len(final_results) >= limit:
                    break
            
            # Remove score from final results
            return [{k: v for k, v in r.items() if k != 'score'} for r in final_results]

# Create global instance
prompt_formatter_data: Optional[IndexedPromptFormatterData] = None

def get_prompt_data() -> IndexedPromptFormatterData:
    """Get or create prompt formatter data instance."""
    global prompt_formatter_data
    if prompt_formatter_data is None:
        prompt_formatter_data = IndexedPromptFormatterData()
    return prompt_formatter_data

# ============================================================================
# UI HELPER FUNCTIONS
# ============================================================================

def search_for_autocomplete(query: str, data_type: str) -> gr.update:
    """Handle autocomplete search."""
    try:
        if not query or len(query.strip()) < SEARCH_CONFIG.MIN_QUERY_LENGTH:
            return gr.update(choices=[], value=None)
            
        results = get_prompt_data().search(query, data_type, limit=SEARCH_CONFIG.MAX_RESULTS)
        choices = [f"{'🔴' if r['source'] == 'danbooru' else '🔵'} {r['display']}" for r in results]
        return gr.update(choices=choices, value=choices[0] if choices else None)
        
    except Exception as e:
        logger.error(f"Error in {data_type} search: {e}")
        return gr.update(choices=[], value=None)

def select_from_dropdown(search_query: str, selected_choice: str, data_type: str) -> str:
    """Handle dropdown selection."""
    try:
        if not selected_choice or not selected_choice.strip():
            return ""
            
        # Remove emoji prefix
        clean_trigger = selected_choice[2:].strip()
        
        data = get_prompt_data()
        if not data.is_loaded or not search_query:
            return clean_trigger
            
        results = data.search(search_query, data_type, limit=SEARCH_CONFIG.MAX_RESULTS_PER_SOURCE)
        for result in results:
            if result['display'] == clean_trigger:
                return result['value']
                
        return clean_trigger
        
    except Exception as e:
        logger.error(f"Error in {data_type} selection: {e}")
        return selected_choice or ""

def compose_final_prompt(prefix: str, character: str, artist: str, custom: str) -> str:
    """Compose final prompt from components."""
    return ", ".join(filter(None, map(normalize_text, [prefix, character, artist, custom])))

def parse_resolution_string(res_str: str) -> Tuple[int, int]:
    """Parse resolution string to width and height."""
    try:
        w, h = map(int, re.findall(r'\d+', res_str)[:2])
        return w, h
    except Exception:
        return OPTIMAL_SETTINGS['width'], OPTIMAL_SETTINGS['height']

def validate_parameters(w: int, h: int, s: int, c: float, r: float, a: Optional[float] = None, ds: Optional[int] = None) -> Optional[str]:
    """Validate generation parameters."""
    errors = []
    
    if not (GEN_CONFIG.MIN_RESOLUTION <= w <= GEN_CONFIG.MAX_RESOLUTION):
        errors.append(f"Width must be {GEN_CONFIG.MIN_RESOLUTION}-{GEN_CONFIG.MAX_RESOLUTION}")
        
    if not (GEN_CONFIG.MIN_RESOLUTION <= h <= GEN_CONFIG.MAX_RESOLUTION):
        errors.append(f"Height must be {GEN_CONFIG.MIN_RESOLUTION}-{GEN_CONFIG.MAX_RESOLUTION}")
        
    if not (GEN_CONFIG.MIN_STEPS <= s <= GEN_CONFIG.MAX_STEPS):
        errors.append(f"Steps must be {GEN_CONFIG.MIN_STEPS}-{GEN_CONFIG.MAX_STEPS}")
        
    if not (GEN_CONFIG.MIN_CFG_SCALE <= c <= GEN_CONFIG.MAX_CFG_SCALE):
        errors.append(f"CFG must be {GEN_CONFIG.MIN_CFG_SCALE}-{GEN_CONFIG.MAX_CFG_SCALE}")
        
    if not (GEN_CONFIG.MIN_RESCALE_CFG <= r <= GEN_CONFIG.MAX_RESCALE_CFG):
        errors.append(f"Rescale must be {GEN_CONFIG.MIN_RESCALE_CFG}-{GEN_CONFIG.MAX_RESCALE_CFG}")
    
    if a is not None and not (MODEL_CONFIG.MIN_ADAPTER_STRENGTH <= a <= MODEL_CONFIG.MAX_ADAPTER_STRENGTH):
        errors.append(f"Adapter strength must be {MODEL_CONFIG.MIN_ADAPTER_STRENGTH}-{MODEL_CONFIG.MAX_ADAPTER_STRENGTH}")
    
    if ds is not None:
        if not isinstance(ds, int):
            errors.append("DoRA start step must be an integer")
        elif not (MODEL_CONFIG.MIN_DORA_START_STEP <= ds <= MODEL_CONFIG.MAX_DORA_START_STEP):
            errors.append(f"DoRA start step must be {MODEL_CONFIG.MIN_DORA_START_STEP}-{MODEL_CONFIG.MAX_DORA_START_STEP}")
        elif ds > s:
            errors.append(f"DoRA start step ({ds}) cannot be greater than total steps ({s})")
        
    return "❌ " + "\n❌ ".join(errors) if errors else None

# UI Handler factories
def create_clear_handler(component_type: str):
    """Create a clear handler for different component types."""
    def clear_search():
        return "", "", gr.update(choices=[], value=None)

    def clear_text():
        return ""

    handlers = {
        'character': clear_search,
        'artist': clear_search,
        'text': clear_text
    }
    return handlers.get(component_type, clear_text)

def create_status_updater(param_type: str):
    """Create a status update function for parameters."""
    def update_cfg_status(value):
        if 3.5 <= value <= 5.5:
            return '<div style="color: green;">✅ Optimal range (3.5-5.5)</div>'
        else:
            return '<div style="color: orange;">⚠️ Outside optimal range (3.5-5.5)</div>'
    
    def update_steps_status(value):
        if 32 <= value <= 40:
            return '<div style="color: green;">✅ Optimal range (32-40)</div>'
        elif value >= 10:
            return '<div style="color: orange;">⚠️ Below optimal range (32-40)</div>'
        else:
            return '<div style="color: red;">❌ Too low for quality results</div>'
    
    def update_rescale_status(value):
        if abs(value - 0.7) < 0.1:
            return '<div style="color: green;">✅ Optimal (around 0.7)</div>'
        else:
            return '<div style="color: blue;">📊 Valid</div>'
    
    def update_adapter_status(value):
        if 0.8 <= value <= 1.2:
            return '<div style="color: green;">✅ Optimal range (0.8-1.2)</div>'
        elif value == 0.0:
            return '<div style="color: gray;">⚪ Disabled</div>'
        elif value > 1.2:
            return '<div style="color: orange;">⚠️ High strength (amplified)</div>'
        else:
            return '<div style="color: blue;">📊 Valid</div>'
    
    def update_dora_start_step_status(value):
        if value == 1:
            return '<div style="color: green;">✅ Start at step 1</div>'
        elif value <= 5:
            return '<div style="color: blue;">🚀 Early activation</div>'
        elif value <= 15:
            return '<div style="color: orange;">⏰ Mid activation</div>'
        else:
            return '<div style="color: purple;">🔄 Late activation</div>'
    
    updaters = {
        'cfg': update_cfg_status,
        'steps': update_steps_status,
        'rescale': update_rescale_status,
        'adapter': update_adapter_status,
        'dora_start_step': update_dora_start_step_status
    }
    
    return updaters.get(param_type, lambda x: "")

def create_search_ui(label: str, number: int) -> Tuple[gr.Textbox, gr.Dropdown, gr.Textbox, gr.Button]:
    """Creates the UI for a search segment."""
    with gr.Group(elem_classes=["segment-container"]):
        gr.HTML(f'<div class="segment-header">{number}️⃣ {label}</div>')
        search_box = gr.Textbox(placeholder=f"Search {label.lower()}s...", lines=1)
        dropdown = gr.Dropdown(choices=[], interactive=True, allow_custom_value=True)
        text_output = gr.Textbox(lines=2, interactive=False)
        clear_btn = gr.Button("🧹 Clear", size="sm")
    return search_box, dropdown, text_output, clear_btn

def connect_search_events(
    data_type: str,
    search_box: gr.Textbox,
    dropdown: gr.Dropdown,
    text_output: gr.Textbox,
    clear_btn: gr.Button,
):
    """Connects event handlers for a search segment."""
    search_box.change(
        lambda q: search_for_autocomplete(q, data_type),
        inputs=[search_box],
        outputs=[dropdown],
        show_progress=False,
    )
    dropdown.change(
        lambda q, c: select_from_dropdown(q, c, data_type),
        inputs=[search_box, dropdown],
        outputs=[text_output],
        show_progress=False,
    )
    clear_btn.click(
        create_clear_handler(data_type),
        outputs=[search_box, text_output, dropdown],
        show_progress=False,
    )

# Global engine instance
engine: Optional[NoobAIEngine] = None

def initialize_engine(model_path: str, enable_dora: bool = False, dora_path: str = "", dora_selection: str = "") -> str:
    """Initialize the NoobAI engine with comprehensive teardown of previous instance."""
    global engine
    try:
        # COMPREHENSIVE TEARDOWN of existing engine before fresh initialization
        if engine is not None:
            logger.info("Performing comprehensive teardown of previous engine instance")
            
            try:
                # Full engine teardown with all cleanup
                engine.teardown_engine()
                
                # Explicit deletion and nullification
                del engine
                engine = None
                
                # Additional global cleanup
                resource_pool.clear()
                
                # Force garbage collection after teardown
                gc.collect()
                
                # Brief pause to ensure complete cleanup
                time.sleep(0.1)
                
                logger.info("Previous engine instance completely torn down")
                
            except Exception as e:
                logger.error(f"Error during engine teardown: {e}")
                # Force reset even if teardown fails
                engine = None
                resource_pool.clear()
                gc.collect()
        # Validate model path
        is_valid, validated_model_path = validate_model_path(model_path)
        if not is_valid:
            return f"❌ {validated_model_path}"
        
        # Handle DoRA path if enabled
        dora_path_to_use = None
        dora_status = ""
        
        if enable_dora:
            if dora_selection and dora_selection != "None":
                # Use selected adapter from dropdown
                adapter_info = get_dora_adapter_by_name(dora_selection)
                if adapter_info:
                    dora_path_to_use = adapter_info['path']
                    dora_status = f"\n🎯 DoRA: {adapter_info['display_name']}"
                else:
                    dora_status = f"\n⚠️ DoRA: Selected adapter '{dora_selection}' not found"
            elif dora_path.strip():
                # Validate provided DoRA path (manual override)
                dora_valid, dora_result = validate_dora_path(dora_path)
                if dora_valid:
                    dora_path_to_use = dora_result
                    precision = detect_adapter_precision(dora_result)
                    dora_status = f"\n🎯 DoRA: {os.path.basename(dora_result)} ({precision})"
                else:
                    dora_status = f"\n⚠️ DoRA Error: {dora_result}"
            else:
                # Auto-detect DoRA path
                auto_dora_path = find_dora_path()
                if auto_dora_path:
                    dora_path_to_use = auto_dora_path
                    precision = detect_adapter_precision(auto_dora_path)
                    dora_status = f"\n🎯 DoRA: {os.path.basename(auto_dora_path)} ({precision}, auto-detected)"
                else:
                    dora_status = "\n⚠️ DoRA: Enabled but no valid DoRA file found"
        
        # Initialize engine
        engine = NoobAIEngine(
            model_path=validated_model_path,
            enable_dora=enable_dora,
            dora_path=dora_path_to_use,
            dora_start_step=OPTIMAL_SETTINGS['dora_start_step']
        )
        
        model_size = os.path.getsize(validated_model_path)
        status_msg = f"✅ Engine initialized!\n📊 Model: {format_file_size(model_size)}{dora_status}"
        
        return status_msg
        
    except Exception as e:
        engine = None
        error_msg = get_user_friendly_error(e)
        logger.error(f"Engine initialization failed: {e}")
        return f"❌ Initialization failed: {error_msg}"

def find_model_path() -> Optional[str]:
    """Search for the model file in common locations."""
    for path in MODEL_SEARCH_PATHS:
        if os.path.exists(path):
            try:
                file_size = os.path.getsize(path)
                if file_size > MODEL_CONFIG.MIN_FILE_SIZE_MB * 1024 * 1024:
                    return path
            except Exception:
                continue
    return None

def get_adapter_choices() -> List[str]:
    """Get list of adapter choices - no 'None' when adapters exist."""
    adapters = discover_dora_adapters()
    if adapters:
        # Only adapter names when adapters exist (no "None" option)
        return [adapter['name'] for adapter in adapters]
    else:
        # Only "None" when no adapters available
        return ["None"]

def get_default_adapter_selection() -> str:
    """Get default adapter selection based on availability."""
    adapters = discover_dora_adapters()
    if adapters:
        return adapters[0]['name']  # First available adapter
    return "None"  # Only when no adapters exist

def get_dora_ui_state() -> dict:
    """Get DoRA UI state based on adapter availability."""
    adapters = discover_dora_adapters()
    has_adapters = len(adapters) > 0
    
    return {
        'enable_dora_interactive': has_adapters,
        'enable_dora_value': has_adapters,  # Auto-enable when adapters exist
        'dropdown_choices': get_adapter_choices(),
        'dropdown_value': get_default_adapter_selection(),
        'dropdown_interactive': has_adapters,
        'info_message': get_dora_info_message(has_adapters),
        'checkbox_info': get_dora_checkbox_info(has_adapters)
    }

def get_dora_info_message(has_adapters: bool) -> str:
    """Get appropriate info message for DoRA dropdown."""
    if has_adapters:
        return "Select DoRA adapter from /dora directory"
    else:
        return "No adapters found in /dora directory"

def get_dora_checkbox_info(has_adapters: bool) -> str:
    """Get appropriate info message for DoRA checkbox."""
    if has_adapters:
        return "Load DoRA adapter for enhanced generation"
    else:
        return "No adapters available - install adapters in /dora directory"

def refresh_adapter_choices() -> gr.update:
    """Refresh adapter choices in dropdown."""
    choices = get_adapter_choices()
    default_value = get_default_adapter_selection()
    return gr.update(choices=choices, value=default_value)

def auto_initialize() -> Tuple[str, str, bool, str, str]:
    """Enhanced auto-initialize with smart DoRA defaults."""
    model_path = find_model_path()
    
    # Get smart DoRA state
    dora_ui_state = get_dora_ui_state()
    enable_dora = dora_ui_state['enable_dora_value']
    default_adapter = dora_ui_state['dropdown_value']
    
    if model_path:
        status = initialize_engine(model_path, enable_dora=enable_dora, dora_selection=default_adapter)
        return status, model_path, enable_dora, "", default_adapter
                
    return ("⚠️ No model found. Please specify path manually.", 
            os.path.join(os.getcwd(), "NoobAI-XL-Vpred-v1.0.safetensors"),
            enable_dora, "", default_adapter)

# Generation handlers
def start_generation() -> Tuple[gr.update, gr.update]:
    """Start generation UI update."""
    state_manager.set_state(GenerationState.GENERATING)
    return gr.update(visible=True, interactive=True), gr.update(value="🔄 Generating...", interactive=False)

def generate_image_with_progress(
    prompt: str, negative_prompt: str, resolution: str, cfg_scale: float, steps: int,
    rescale_cfg: float, seed: str, use_custom_resolution: bool, custom_width: int,
    custom_height: int, auto_randomize_seed: bool, adapter_strength: float, enable_dora: bool, dora_start_step: int, progress=gr.Progress()
) -> Tuple[Optional[str], str, str]:
    """Generate image with progress tracking and return file path for hash consistency."""
    try:
        # Check engine
        if not (engine and engine.is_initialized):
            state_manager.set_state(GenerationState.ERROR)
            return None, "❌ Engine not initialized", seed

        # Validate prompt
        if not prompt.strip():
            state_manager.set_state(GenerationState.ERROR)
            return None, "❌ Please enter a prompt", seed
        
        # Parse resolution
        if use_custom_resolution:
            width, height = custom_width, custom_height
        else:
            width, height = parse_resolution_string(resolution)
            
        # Validate parameters
        param_error = validate_parameters(width, height, steps, cfg_scale, rescale_cfg, adapter_strength, dora_start_step)
        if param_error:
            state_manager.set_state(GenerationState.ERROR)
            return None, param_error, seed

        # Handle seed
        used_seed = None
        if not auto_randomize_seed:
            try:
                seed_val = int(seed.strip())
                if not (0 <= seed_val < 2**32):
                    raise InvalidParameterError(f"Seed must be between 0 and {2**32-1}")
                used_seed = seed_val
            except (ValueError, InvalidParameterError) as e:
                state_manager.set_state(GenerationState.ERROR)
                return None, f"❌ Invalid seed: {str(e)}", seed

        # Generate
        progress(0, desc="Starting generation...")
        
        image, final_seed, info = engine.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            cfg_scale=cfg_scale,
            steps=steps,
            rescale_cfg=rescale_cfg,
            seed=used_seed,
            adapter_strength=adapter_strength if enable_dora else None,
            enable_dora=enable_dora,
            dora_start_step=dora_start_step if enable_dora else None,
            progress_callback=lambda p, d: progress(p, desc=d)
        )
        
        # Save image with standardized settings
        output_path = os.path.join(OUTPUT_DIR, f"noobai_{final_seed}.png")
        engine.save_image_standardized(image, output_path)
        
        # Add hash info to the generation info
        image_hash = calculate_image_hash(output_path)
        info += f"\n📄 MD5 Hash: {image_hash}"
        
        progress(1.0, desc="Complete!")
        state_manager.set_state(GenerationState.COMPLETED)
        
        return output_path, info, str(final_seed)

    except GenerationInterruptedError:
        state_manager.set_state(GenerationState.INTERRUPTED)
        return None, "⚠️ Generation interrupted", seed
    except Exception as e:
        state_manager.set_state(GenerationState.ERROR)
        error_msg = get_user_friendly_error(e)
        logger.error(f"Generation error: {e}")
        return None, f"❌ Generation failed: {error_msg}", seed

def finish_generation() -> Tuple[gr.update, gr.update]:
    """Finish generation UI update."""
    return gr.update(visible=False), gr.update(value="🎨 Generate Image", interactive=True)

def interrupt_generation() -> Tuple[gr.update, gr.update]:
    """Interrupt generation."""
    state_manager.request_interrupt()
    return gr.update(visible=False), gr.update(value="🔄 Interrupting...", interactive=False)

# ============================================================================
# CLEANUP
# ============================================================================

def cleanup_resources():
    """Clean up resources on application exit."""
    global engine
    try:
        if engine:
            engine.clear_memory()
        resource_pool.clear()
        # Clean up old temporary files
        if os.path.exists(OUTPUT_DIR):
            try:
                # Remove files older than 1 day
                current_time = time.time()
                for filename in os.listdir(OUTPUT_DIR):
                    file_path = os.path.join(OUTPUT_DIR, filename)
                    if os.path.isfile(file_path):
                        file_age = current_time - os.path.getmtime(file_path)
                        if file_age > 86400:  # 1 day in seconds
                            os.remove(file_path)
            except Exception as e:
                logger.warning(f"Could not clean up old files: {e}")
        logger.info("Resources cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

# Register cleanup
atexit.register(cleanup_resources)

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_interface() -> gr.Blocks:
    """Create the Gradio interface."""
    init_status, default_model_path, default_enable_dora, default_dora_path, default_adapter_selection = auto_initialize()
    is_ready = engine is not None and engine.is_initialized
    
    # Get smart DoRA UI state
    dora_ui_state = get_dora_ui_state()
    
    resolution_options = [
        f"{w}x{h}{' (Optimal)' if (h, w) in RECOMMENDED_RESOLUTIONS else ''}" 
        for h, w in OFFICIAL_RESOLUTIONS
    ]

    with gr.Blocks(
        title="NoobAI XL V-Pred 1.0 (Hash Consistency Edition)",
        theme=gr.themes.Soft(),
        css="""
        .title-text { 
            text-align: center; 
            font-size: 24px; 
            font-weight: bold; 
            margin-bottom: 20px;
        }
        .status-success { 
            background: rgba(34, 197, 94, 0.1) !important; 
            color: rgb(34, 197, 94) !important; 
            border: 1px solid rgba(34, 197, 94, 0.3) !important;
        }
        .status-error { 
            background: rgba(239, 68, 68, 0.1) !important; 
            color: rgb(239, 68, 68) !important; 
            border: 1px solid rgba(239, 68, 68, 0.3) !important;
        }
        .segment-container { 
            border: 1px solid var(--border-color-primary); 
            border-radius: 6px; 
            padding: 10px; 
            margin-bottom: 10px;
        }
        .segment-header { 
            font-weight: bold; 
            margin-bottom: 8px; 
        }
        .final-prompt-container { 
            background: var(--background-fill-secondary); 
            border: 2px solid var(--border-color-accent); 
            border-radius: 8px; 
            padding: 15px; 
        }
        """
    ) as demo:
        
        gr.HTML('<div class="title-text">🎯 NoobAI XL V-Pred 1.0 - Hash Consistency Edition</div>')
        
        with gr.Row():
            with gr.Column(scale=2):
                # Engine initialization
                with gr.Group():
                    gr.HTML("<h3>🚀 Engine Initialization</h3>")
                    model_path = gr.Textbox(
                        label="Model Path",
                        value=default_model_path
                    )
                    
                    # DoRA controls with smart UI state
                    with gr.Row():
                        enable_dora = gr.Checkbox(
                            label="🎯 Enable DoRA Adapter",
                            value=dora_ui_state['enable_dora_value'],
                            interactive=dora_ui_state['enable_dora_interactive'],
                            info=dora_ui_state['checkbox_info']
                        )
                        dora_refresh_btn = gr.Button("🔄 Refresh Adapters", size="sm")
                    
                    dora_selection = gr.Dropdown(
                        label="DoRA Adapter Selection",
                        choices=dora_ui_state['dropdown_choices'],
                        value=dora_ui_state['dropdown_value'],
                        interactive=dora_ui_state['dropdown_interactive'],
                        info=dora_ui_state['info_message']
                    )
                    
                    dora_path = gr.Textbox(
                        label="Manual DoRA Path (Override)",
                        value=default_dora_path,
                        interactive=True,
                        placeholder="Optional: Specify custom path to override adapter selection above"
                    )
                    
                    with gr.Row():
                        init_btn = gr.Button("Initialize Engine", variant="primary")
                        status_indicator = gr.Button(
                            "✅ Ready" if is_ready else "❌ Not Ready",
                            variant="secondary" if is_ready else "stop",
                            interactive=False
                        )
                    init_status_display = gr.Textbox(
                        label="Status",
                        value=init_status,
                        interactive=False,
                        elem_classes=["status-success" if is_ready else "status-error"]
                    )
                
                # Positive prompt formatter
                with gr.Group():
                    gr.HTML(
                        '<h3>🎨 Positive Prompt Formatter</h3>'
                        '<div style="margin-bottom: 10px; color: #666;">🔴 Danbooru | 🔵 E621</div>'
                    )
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Group(elem_classes=["segment-container"]):
                                gr.HTML('<div class="segment-header">1️⃣ Quality Tags</div>')
                                prefix_text = gr.Textbox(
                                    value=DEFAULT_POSITIVE_PREFIX,
                                    lines=2
                                )
                                prefix_reset_btn = gr.Button("🔄 Reset", size="sm")
                                
                        with gr.Column(scale=1):
                            character_search, character_dropdown, character_text, character_clear_btn = create_search_ui("Character", 2)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            artist_search, artist_dropdown, artist_text, artist_clear_btn = create_search_ui("Artist", 3)
                                
                        with gr.Column(scale=1):
                            with gr.Group(elem_classes=["segment-container"]):
                                gr.HTML('<div class="segment-header">4️⃣ Custom Tags</div>')
                                custom_text = gr.Textbox(
                                    placeholder="Additional tags...",
                                    lines=4
                                )
                                custom_clear_btn = gr.Button("🧹 Clear", size="sm")
                    
                    with gr.Group(elem_classes=["final-prompt-container"]):
                        gr.HTML('<div class="segment-header">🎯 Final Prompt</div>')
                        final_prompt = gr.Textbox(label="Positive Prompt", lines=4)
                        with gr.Row():
                            compose_btn = gr.Button("🔄 Compose", variant="primary")
                            clear_all_btn = gr.Button("🧹 Clear All", variant="secondary")
                
                # Negative prompt
                with gr.Group():
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        value=DEFAULT_NEGATIVE_PROMPT,
                        lines=3
                    )
                    negative_reset_btn = gr.Button("🔄 Reset Negative", size="sm")
                
                # Resolution settings
                with gr.Group():
                    gr.HTML("<h4>📐 Resolution</h4>")
                    use_custom_resolution = gr.Checkbox(
                        label="Use Custom Resolution",
                        value=False
                    )
                    resolution = gr.Dropdown(
                        label="Preset",
                        choices=resolution_options,
                        value="1216x832 (Optimal)",
                        visible=True
                    )
                    with gr.Row(visible=False) as custom_res_row:
                        custom_width = gr.Number(
                            label="Width",
                            value=OPTIMAL_SETTINGS['width'],
                            minimum=GEN_CONFIG.MIN_RESOLUTION,
                            maximum=GEN_CONFIG.MAX_RESOLUTION,
                            step=64
                        )
                        custom_height = gr.Number(
                            label="Height",
                            value=OPTIMAL_SETTINGS['height'],
                            minimum=GEN_CONFIG.MIN_RESOLUTION,
                            maximum=GEN_CONFIG.MAX_RESOLUTION,
                            step=64
                        )
                
                # Generation parameters
                with gr.Group():
                    gr.HTML("<h4>⚙️ Parameters</h4>")
                    with gr.Row():
                        with gr.Column():
                            cfg_scale = gr.Slider(
                                label="CFG Scale",
                                minimum=GEN_CONFIG.MIN_CFG_SCALE,
                                maximum=GEN_CONFIG.MAX_CFG_SCALE,
                                step=0.1,
                                value=OPTIMAL_SETTINGS['cfg_scale']
                            )
                            cfg_status = gr.HTML('<div style="color: green;">✅ Optimal</div>')
                            
                        with gr.Column():
                            rescale_cfg = gr.Slider(
                                label="Rescale CFG",
                                minimum=GEN_CONFIG.MIN_RESCALE_CFG,
                                maximum=GEN_CONFIG.MAX_RESCALE_CFG,
                                step=0.05,
                                value=OPTIMAL_SETTINGS['rescale_cfg']
                            )
                            rescale_status = gr.HTML('<div style="color: green;">✅ Optimal</div>')
                        
                        with gr.Column():
                            adapter_strength = gr.Slider(
                                label="🎯 Adapter Strength",
                                minimum=MODEL_CONFIG.MIN_ADAPTER_STRENGTH,
                                maximum=MODEL_CONFIG.MAX_ADAPTER_STRENGTH,
                                step=0.1,
                                value=OPTIMAL_SETTINGS['adapter_strength'],
                                visible=default_enable_dora,
                                info="Control DoRA adapter influence"
                            )
                            adapter_status = gr.HTML(
                                '<div style="color: green;">✅ Optimal</div>',
                                visible=default_enable_dora
                            )
                            
                            dora_start_step = gr.Slider(
                                label="🚀 DoRA Start Step",
                                minimum=MODEL_CONFIG.MIN_DORA_START_STEP,
                                maximum=min(OPTIMAL_SETTINGS['steps'], MODEL_CONFIG.MAX_DORA_START_STEP),
                                step=1,
                                value=OPTIMAL_SETTINGS['dora_start_step'],
                                visible=default_enable_dora,
                                info="Step at which DoRA adapter activates"
                            )
                            dora_start_step_status = gr.HTML(
                                '<div style="color: green;">✅ Start at step 1</div>',
                                visible=default_enable_dora
                            )
                            
                    steps = gr.Slider(
                        label="Steps",
                        minimum=GEN_CONFIG.MIN_STEPS,
                        maximum=GEN_CONFIG.MAX_STEPS,
                        step=1,
                        value=OPTIMAL_SETTINGS['steps']
                    )
                    steps_status = gr.HTML('<div style="color: green;">✅ Optimal</div>')
                
                # Seed settings
                with gr.Group():
                    gr.HTML("<h4>🌱 Seed</h4>")
                    with gr.Row():
                        with gr.Column(scale=3):
                            seed = gr.Textbox(
                                value=str(random.randint(0, 2**32-1)),
                                label="Seed"
                            )
                            auto_randomize_seed = gr.Checkbox(
                                label="🔄 Ignore seed box and use random for next run",
                                value=True
                            )
                        with gr.Column(scale=1):
                            random_seed_btn = gr.Button("🎲 New Random Seed", size="lg")
                
                # Generate button
                generate_btn = gr.Button(
                    "🎨 Generate Image" if is_ready else "❌ Initialize Engine First",
                    variant="primary" if is_ready else "stop",
                    size="lg",
                    interactive=is_ready
                )
                interrupt_btn = gr.Button(
                    "⏹️ Interrupt",
                    variant="stop",
                    size="sm",
                    visible=False
                )
            
            # Output column
            with gr.Column(scale=2):
                with gr.Group():
                    gr.HTML("<h3>🖼️ Result</h3>")
                    output_image = gr.Image(
                        type="filepath",  # Changed from "pil" for hash consistency
                        interactive=False,
                        height=400,
                        format="png"
                    )
                    generation_info = gr.Textbox(
                        label="Generation Info",
                        lines=9,  # Increased to show hash
                        interactive=False
                    )
        
        with gr.Row():
            reset_btn = gr.Button("🔄 Reset to Optimal", variant="secondary", size="sm")

        # === Event Handlers ===
        
        # DoRA refresh handler with comprehensive UI updates
        def refresh_dora_adapters():
            """Enhanced refresh with conditional UI updates for both checkbox and dropdown."""
            global engine
            
            # Get fresh adapter state
            dora_ui_state = get_dora_ui_state()
            
            # Get current engine settings if engine exists
            current_settings = None
            if engine is not None and engine.is_initialized:
                current_settings = {
                    'model_path': engine.model_path,
                    'enable_dora': engine.enable_dora,
                    'adapter_strength': engine.adapter_strength
                }
            
            # Add re-initialization suggestion if engine was using DoRA
            suggestion_msg = ""
            if current_settings and current_settings['enable_dora']:
                suggestion_msg = " (Re-initialize engine if switching adapters)"
            
            # Update both checkbox and dropdown based on adapter availability
            return (
                # Update checkbox state
                gr.update(
                    interactive=dora_ui_state['enable_dora_interactive'],
                    value=dora_ui_state['enable_dora_value'] if not current_settings else current_settings['enable_dora'],
                    info=dora_ui_state['checkbox_info']
                ),
                # Update dropdown state  
                gr.update(
                    choices=dora_ui_state['dropdown_choices'],
                    value=dora_ui_state['dropdown_value'],
                    interactive=dora_ui_state['dropdown_interactive'],
                    info=dora_ui_state['info_message'] + suggestion_msg
                )
            )
        
        # Component lists for event handling
        all_prompt_inputs = [prefix_text, character_text, artist_text, custom_text]
        all_prompt_components = [
            prefix_text, character_search, character_text, character_dropdown,
            artist_search, artist_text, artist_dropdown, custom_text, final_prompt
        ]
        gen_inputs = [
            final_prompt, negative_prompt, resolution, cfg_scale, steps,
            rescale_cfg, seed, use_custom_resolution, custom_width,
            custom_height, auto_randomize_seed, adapter_strength, enable_dora, dora_start_step
        ]
        gen_outputs = [output_image, generation_info, seed]
        
        # Engine initialization
        def init_and_update(path, enable_dora_val, dora_path_val, dora_selection_val):
            """Enhanced initialization with teardown feedback."""
            global engine
            
            # Provide teardown feedback if engine exists
            if engine is not None:
                # Show teardown progress
                teardown_status = "🔄 Performing comprehensive engine teardown..."
                yield (
                    teardown_status,
                    gr.update(value="🔄 Tearing Down", variant="stop"),
                    gr.update(value="🔄 Cleaning up...", variant="stop", interactive=False),
                    gr.update(elem_classes=["status-warning"])
                )
            
            # Perform initialization with comprehensive teardown
            status = initialize_engine(path, enable_dora_val, dora_path_val, dora_selection_val)
            ready = engine is not None and engine.is_initialized
            
            # Final status update
            final_status = (
                status,
                gr.update(
                    value="✅ Ready" if ready else "❌ Not Ready",
                    variant="secondary" if ready else "stop"
                ),
                gr.update(
                    value="🎨 Generate Image" if ready else "❌ Initialize...",
                    variant="primary" if ready else "stop",
                    interactive=ready
                ),
                gr.update(elem_classes=["status-success" if ready else "status-error"])
            )
            
            if engine is not None:
                yield final_status
            else:
                return final_status
            
        init_btn.click(
            init_and_update,
            inputs=[model_path, enable_dora, dora_path, dora_selection],
            outputs=[init_status_display, status_indicator, generate_btn, init_status_display]
        )
        
        # DoRA refresh handler with multiple outputs
        dora_refresh_btn.click(
            refresh_dora_adapters,
            outputs=[enable_dora, dora_selection]
        )
        
        # DoRA visibility toggle with feedback
        def toggle_dora_visibility(enabled):
            """Handle DoRA toggle with immediate feedback."""
            # Update adapter strength slider visibility
            adapter_strength_update = gr.update(visible=enabled)
            
            # Update DoRA start step slider visibility
            dora_start_step_update = gr.update(visible=enabled)
            dora_start_step_status_update = gr.update(visible=enabled)
            
            # Provide status feedback with visibility and message
            if enabled:
                status_msg = '<div style="color: green;">🎯 DoRA will be enabled for next generation</div>'
            else:
                status_msg = '<div style="color: gray;">⚪ DoRA will be disabled for next generation</div>'
            
            adapter_status_update = gr.update(visible=enabled, value=status_msg)
            
            return adapter_strength_update, adapter_status_update, dora_start_step_update, dora_start_step_status_update
        
        enable_dora.change(
            toggle_dora_visibility,
            inputs=[enable_dora],
            outputs=[adapter_strength, adapter_status, dora_start_step, dora_start_step_status]
        )

        # Search handlers
        connect_search_events(
            "character",
            character_search,
            character_dropdown,
            character_text,
            character_clear_btn,
        )
        connect_search_events(
            "artist", artist_search, artist_dropdown, artist_text, artist_clear_btn
        )
        
        # Prompt composition
        compose_btn.click(
            compose_final_prompt,
            inputs=all_prompt_inputs,
            outputs=[final_prompt],
            show_progress=False
        )
        
        # Reset/clear handlers
        prefix_reset_btn.click(
            lambda: DEFAULT_POSITIVE_PREFIX,
            outputs=[prefix_text],
            show_progress=False
        )
        negative_reset_btn.click(
            lambda: DEFAULT_NEGATIVE_PROMPT,
            outputs=[negative_prompt],
            show_progress=False
        )
        custom_clear_btn.click(
            create_clear_handler('text'),
            outputs=[custom_text],
            show_progress=False
        )
        
        clear_all_btn.click(
            lambda: (
                DEFAULT_POSITIVE_PREFIX, "", "", gr.update(choices=[], value=None),
                "", "", gr.update(choices=[], value=None), "", ""
            ),
            outputs=all_prompt_components,
            show_progress=False
        )

        # Generation handlers
        generate_btn.click(
            start_generation,
            outputs=[interrupt_btn, generate_btn]
        ).then(
            generate_image_with_progress,
            inputs=gen_inputs,
            outputs=gen_outputs
        ).then(
            finish_generation,
            outputs=[interrupt_btn, generate_btn]
        )
        
        interrupt_btn.click(
            interrupt_generation,
            outputs=[interrupt_btn, generate_btn]
        )
        
        # Seed management
        random_seed_btn.click(
            lambda: str(random.randint(0, 2**32-1)),
            outputs=[seed]
        )
        
        # Resolution toggle
        use_custom_resolution.change(
            lambda x: [gr.update(visible=not x), gr.update(visible=x)],
            inputs=[use_custom_resolution],
            outputs=[resolution, custom_res_row]
        )
        
        # Parameter status updates
        cfg_scale.change(
            create_status_updater('cfg'),
            inputs=[cfg_scale],
            outputs=[cfg_status]
        )
        steps.change(
            create_status_updater('steps'),
            inputs=[steps],
            outputs=[steps_status]
        )
        
        # Update DoRA start step maximum when steps change
        def update_dora_start_step_max(steps_value):
            return gr.update(maximum=steps_value)
        
        steps.change(
            update_dora_start_step_max,
            inputs=[steps],
            outputs=[dora_start_step]
        )
        rescale_cfg.change(
            create_status_updater('rescale'),
            inputs=[rescale_cfg],
            outputs=[rescale_status]
        )
        adapter_strength.change(
            create_status_updater('adapter'),
            inputs=[adapter_strength],
            outputs=[adapter_status]
        )
        dora_start_step.change(
            create_status_updater('dora_start_step'),
            inputs=[dora_start_step],
            outputs=[dora_start_step_status]
        )

        # Reset to optimal
        def reset_to_optimal():
            cfg_updater = create_status_updater('cfg')
            steps_updater = create_status_updater('steps')
            rescale_updater = create_status_updater('rescale')
            adapter_updater = create_status_updater('adapter')
            dora_start_step_updater = create_status_updater('dora_start_step')
            
            return (
                OPTIMAL_SETTINGS['cfg_scale'],
                OPTIMAL_SETTINGS['steps'],
                OPTIMAL_SETTINGS['rescale_cfg'],
                OPTIMAL_SETTINGS['adapter_strength'],
                OPTIMAL_SETTINGS['dora_start_step'],
                "1216x832 (Optimal)",
                False,
                OPTIMAL_SETTINGS['width'],
                OPTIMAL_SETTINGS['height'],
                cfg_updater(OPTIMAL_SETTINGS['cfg_scale']),
                steps_updater(OPTIMAL_SETTINGS['steps']),
                rescale_updater(OPTIMAL_SETTINGS['rescale_cfg']),
                adapter_updater(OPTIMAL_SETTINGS['adapter_strength']),
                dora_start_step_updater(OPTIMAL_SETTINGS['dora_start_step'])
            )
            
        reset_btn.click(
            reset_to_optimal,
            outputs=[
                cfg_scale, steps, rescale_cfg, adapter_strength, dora_start_step, resolution,
                use_custom_resolution, custom_width, custom_height,
                cfg_status, steps_status, rescale_status, adapter_status, dora_start_step_status
            ]
        )

        return demo

# ============================================================================
# CLI FUNCTIONS
# ============================================================================

def cli_list_adapters():
    """List all discovered DoRA adapters."""
    print("🎯 Discovered DoRA Adapters:")
    adapters = discover_dora_adapters()
    
    if not adapters:
        print("   No DoRA adapters found in search directories.")
        print("   Search directories:")
        for directory in DORA_SEARCH_DIRECTORIES:
            if os.path.exists(directory):
                print(f"     ✓ {directory}")
            else:
                print(f"     ✗ {directory} (not found)")
        return
    
    for i, adapter in enumerate(adapters):
        print(f"   [{i}] {adapter['display_name']}")
        print(f"       Path: {adapter['path']}")
    print()

def cli_generate(args):
    """Generate image in CLI mode."""
    try:
        # Handle list-adapters option
        if hasattr(args, 'list_dora_adapters') and args.list_dora_adapters:
            cli_list_adapters()
            return 0
        
        # Initialize engine
        if not args.model_path:
            args.model_path = find_model_path()
            if not args.model_path:
                print("❌ No model found. Please specify --model-path")
                return 1
        
        # Validate model path
        is_valid, path_or_error = validate_model_path(args.model_path)
        if not is_valid:
            print(f"❌ {path_or_error}")
            return 1
        
        print(f"🚀 Initializing engine with model: {path_or_error}")
        
        # Handle DoRA if enabled
        dora_path_to_use = None
        if args.enable_dora:
            if hasattr(args, 'dora_adapter') and args.dora_adapter is not None:
                # Select by index
                adapters = discover_dora_adapters()
                if 0 <= args.dora_adapter < len(adapters):
                    adapter_info = adapters[args.dora_adapter]
                    dora_path_to_use = adapter_info['path']
                    print(f"🎯 DoRA adapter: {adapter_info['display_name']}")
                else:
                    print(f"❌ Invalid adapter index {args.dora_adapter}. Available: 0-{len(adapters)-1}")
                    return 1
            elif hasattr(args, 'dora_name') and args.dora_name:
                # Select by name
                adapter_info = get_dora_adapter_by_name(args.dora_name)
                if adapter_info:
                    dora_path_to_use = adapter_info['path']
                    print(f"🎯 DoRA adapter: {adapter_info['display_name']}")
                else:
                    print(f"❌ DoRA adapter '{args.dora_name}' not found")
                    print("Available adapters:")
                    cli_list_adapters()
                    return 1
            elif args.dora_path:
                # Manual path specification
                dora_valid, dora_result = validate_dora_path(args.dora_path)
                if dora_valid:
                    dora_path_to_use = dora_result
                    precision = detect_adapter_precision(dora_result)
                    print(f"🎯 DoRA adapter: {os.path.basename(dora_result)} ({precision})")
                else:
                    print(f"⚠️ DoRA validation failed: {dora_result}")
                    return 1
            else:
                # Auto-detect DoRA
                auto_dora_path = find_dora_path()
                if auto_dora_path:
                    dora_path_to_use = auto_dora_path
                    precision = detect_adapter_precision(auto_dora_path)
                    print(f"🎯 DoRA adapter: {os.path.basename(auto_dora_path)} ({precision}, auto-detected)")
                else:
                    print("⚠️ DoRA enabled but no valid DoRA file found")
                    print("Use --list-dora-adapters to see available adapters")
        
        global engine
        engine = NoobAIEngine(
            model_path=path_or_error, 
            enable_dora=args.enable_dora,
            dora_path=dora_path_to_use,
            adapter_strength=args.adapter_strength,
            dora_start_step=args.dora_start_step
        )
        
        # Parse resolution
        width = args.width or OPTIMAL_SETTINGS['width']
        height = args.height or OPTIMAL_SETTINGS['height']
        
        # Validate parameters
        param_error = validate_parameters(
            width, height, args.steps, args.cfg_scale, args.rescale_cfg, args.adapter_strength, args.dora_start_step
        )
        if param_error:
            print(param_error)
            return 1
        
        # Prepare prompt
        prompt = args.prompt
        if not prompt.strip():
            print("❌ Please provide a prompt")
            return 1
        
        # Generate image
        print(f"🎨 Generating image...")
        print(f"   Prompt: {prompt}")
        print(f"   Resolution: {width}x{height}")
        print(f"   Steps: {args.steps}")
        print(f"   CFG Scale: {args.cfg_scale}")
        
        def progress_callback(progress, desc):
            print(f"   {desc}")
        
        image, seed, info = engine.generate(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            width=width,
            height=height,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            rescale_cfg=args.rescale_cfg,
            seed=args.seed,
            adapter_strength=args.adapter_strength if args.enable_dora else None,
            enable_dora=args.enable_dora,
            dora_start_step=args.dora_start_step if args.enable_dora else None,
            progress_callback=progress_callback
        )
        
        # Save image with standardized settings
        output_path = args.output or f"noobai_output_{seed}.png"
        engine.save_image_standardized(image, output_path)
        
        # Calculate and display hash
        image_hash = calculate_image_hash(output_path)
        
        print(f"✅ Image saved to: {output_path}")
        print(f"🌱 Seed: {seed}")
        print(f"📄 MD5 Hash: {image_hash}")
        
        # Show DoRA info if enabled
        if engine.dora_loaded:
            print(f"🎯 DoRA: {os.path.basename(engine.dora_path)} (strength: {engine.adapter_strength})")
        
        if args.verbose:
            print("\nGeneration Info:")
            print(info)
        
        return 0
        
    except Exception as e:
        error_msg = get_user_friendly_error(e)
        print(f"❌ Generation failed: {error_msg}")
        logger.error(f"CLI generation error: {e}")
        return 1

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="NoobAI XL V-Pred 1.0 - Professional AI Image Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --gui                                    # Launch GUI (default)
  %(prog)s --cli --prompt "cat girl, anime"        # CLI generation
  %(prog)s --cli --prompt "dragon" --steps 40      # CLI with custom steps
  %(prog)s --cli --prompt "landscape" --width 1024 --height 768  # Custom resolution
  %(prog)s --list-dora-adapters                     # List available DoRA adapters
  %(prog)s --cli --prompt "anime girl" --enable-dora  # CLI with DoRA adapter (auto-detect)
  %(prog)s --cli --prompt "fantasy" --enable-dora --dora-adapter 0  # Select by index
  %(prog)s --cli --prompt "portrait" --enable-dora --dora-name "noobai_vp10_stabilizer_v0.271_fp16.safetensors"  # Select by name
  %(prog)s --cli --prompt "portrait" --enable-dora --dora-path /path/to/dora.safetensors --adapter-strength 0.8 --dora-start-step 10  # DoRA activates at step 10/35
  %(prog)s --cli --prompt "landscape" --enable-dora --dora-start-step 1   # DoRA active from first step (default)
  %(prog)s --cli --prompt "abstract art" --enable-dora --dora-start-step 25 --steps 40  # Late DoRA activation for subtle effects
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--gui",
        action="store_true",
        default=True,
        help="Launch GUI mode (default)"
    )
    mode_group.add_argument(
        "--cli",
        action="store_true",
        help="Use CLI mode for batch generation"
    )
    
    # Model configuration
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to NoobAI model file (.safetensors)"
    )
    
    # DoRA adapter configuration
    parser.add_argument(
        "--enable-dora",
        action="store_true",
        help="Enable DoRA (Weight-Decomposed Low-Rank Adaptation) adapter"
    )
    parser.add_argument(
        "--list-dora-adapters",
        action="store_true",
        help="List all discovered DoRA adapters and exit"
    )
    parser.add_argument(
        "--dora-adapter",
        type=int,
        help="Select DoRA adapter by index (use --list-dora-adapters to see options)"
    )
    parser.add_argument(
        "--dora-name",
        type=str,
        help="Select DoRA adapter by filename (e.g., 'noobai_vp10_stabilizer_v0.271_fp16.safetensors')"
    )
    parser.add_argument(
        "--dora-path",
        type=str,
        help="Manual path to DoRA adapter file (.safetensors). Overrides --dora-adapter and --dora-name."
    )
    
    # CLI-specific options
    cli_group = parser.add_argument_group("CLI Generation Options")
    cli_group.add_argument(
        "--prompt",
        type=str,
        required="--cli" in sys.argv,
        help="Positive prompt for image generation"
    )
    cli_group.add_argument(
        "--negative-prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt (default: built-in negative prompt)"
    )
    cli_group.add_argument(
        "--width",
        type=int,
        help=f"Image width (default: {OPTIMAL_SETTINGS['width']})"
    )
    cli_group.add_argument(
        "--height",
        type=int,
        help=f"Image height (default: {OPTIMAL_SETTINGS['height']})"
    )
    cli_group.add_argument(
        "--steps",
        type=int,
        default=OPTIMAL_SETTINGS['steps'],
        help=f"Number of inference steps (default: {OPTIMAL_SETTINGS['steps']})"
    )
    cli_group.add_argument(
        "--cfg-scale",
        type=float,
        default=OPTIMAL_SETTINGS['cfg_scale'],
        help=f"CFG scale (default: {OPTIMAL_SETTINGS['cfg_scale']})"
    )
    cli_group.add_argument(
        "--rescale-cfg",
        type=float,
        default=OPTIMAL_SETTINGS['rescale_cfg'],
        help=f"Rescale CFG (default: {OPTIMAL_SETTINGS['rescale_cfg']})"
    )
    cli_group.add_argument(
        "--seed",
        type=int,
        help="Seed for generation (random if not specified)"
    )
    cli_group.add_argument(
        "--output",
        type=str,
        help="Output file path (default: noobai_output_<seed>.png)"
    )
    cli_group.add_argument(
        "--adapter-strength",
        type=float,
        default=OPTIMAL_SETTINGS['adapter_strength'],
        help=f"DoRA adapter strength when enabled (default: {OPTIMAL_SETTINGS['adapter_strength']}, range: {MODEL_CONFIG.MIN_ADAPTER_STRENGTH}-{MODEL_CONFIG.MAX_ADAPTER_STRENGTH})"
    )
    cli_group.add_argument(
        "--dora-start-step",
        type=int,
        default=OPTIMAL_SETTINGS['dora_start_step'],
        help=f"Step at which DoRA adapter activates (default: {OPTIMAL_SETTINGS['dora_start_step']}, range: {MODEL_CONFIG.MIN_DORA_START_STEP}-{MODEL_CONFIG.MAX_DORA_START_STEP})"
    )
    cli_group.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed generation information"
    )
    
    # GUI options
    gui_group = parser.add_argument_group("GUI Options")
    gui_group.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for GUI server (default: 127.0.0.1)"
    )
    gui_group.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for GUI server (default: 7860)"
    )
    gui_group.add_argument(
        "--share",
        action="store_true",
        help="Create public Gradio link"
    )
    gui_group.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically"
    )
    
    return parser.parse_args()

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main application entry point with CLI support."""
    try:
        args = parse_args()
        
        # Handle list-dora-adapters (can be called without CLI flag)
        if hasattr(args, 'list_dora_adapters') and args.list_dora_adapters:
            cli_list_adapters()
            return 0
        
        # Handle CLI mode
        if args.cli:
            return cli_generate(args)
        
        # GUI mode (default)
        logger.info("Starting NoobAI XL V-Pred 1.0 - Hash Consistency Edition")
        logger.info(f"Performance monitoring: {'Enabled' if perf_monitor.enabled else 'Disabled'}")
        logger.info(f"Output directory: {OUTPUT_DIR}")
        
        # Pre-load CSV data
        logger.info("Loading CSV data for prompt formatter...")
        get_prompt_data()
        
        # Create and launch interface
        demo = create_interface()
        demo.launch(
            share=args.share,
            inbrowser=not args.no_browser,
            show_error=True,
            server_name=args.host,
            server_port=args.port
        )
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Application error: {e}")
        if 'args' in locals() and (args.cli or (hasattr(args, 'list_dora_adapters') and args.list_dora_adapters)):
            print(f"❌ Error: {e}")
            return 1
        raise
    finally:
        if 'perf_monitor' in globals() and perf_monitor.enabled:
            logger.info("Performance summary:")
            for section, stats in perf_monitor.get_summary().items():
                logger.info(f"  {section}: {stats}")

if __name__ == "__main__":
    sys.exit(main())