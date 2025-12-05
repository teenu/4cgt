"""
NoobAI XL V-Pred 1.0 - Engine

This module contains the NoobAIEngine class which handles model loading,
image generation, and DoRA adapter management.
"""

import os
import time
import random
import gc
import threading

# ============================================================================
# DETERMINISM BOOTSTRAP (MUST RUN BEFORE ANY TORCH IMPORTS)
# ============================================================================
# When the engine is imported directly (e.g., unit tests or embedding), main.py
# is bypassed and CUBLAS_WORKSPACE_CONFIG may not be set. CUDA requires this
# variable to enforce deterministic algorithms; if it is missing or incorrect,
# PyTorch will silently fall back to non-deterministic kernels. Configure it
# defensively here so deterministic mode is guaranteed regardless of entrypoint.
if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
else:
    _cublas_value = os.environ["CUBLAS_WORKSPACE_CONFIG"]
    if _cublas_value not in (":4096:8", ":16:8"):
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be ':4096:8' or ':16:8' for deterministic CUDA execution. "
            f"Current value: '{_cublas_value}'."
        )

import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler, AutoencoderKL
from PIL import Image
from PIL import PngImagePlugin
from typing import Optional, Tuple, Dict, Any, Callable, List
from config import (
    logger, MODEL_CONFIG, GEN_CONFIG, DEFAULT_NEGATIVE_PROMPT, OPTIMAL_SETTINGS,
    OFFICIAL_RESOLUTIONS, RECOMMENDED_RESOLUTIONS,
    EngineNotInitializedError, GenerationInterruptedError, InvalidParameterError
)
from state import perf_monitor, state_manager, GenerationState
from utils import (
    detect_base_model_precision, detect_adapter_precision,
    validate_dora_path, find_dora_path, format_file_size,
    parse_manual_dora_schedule, generate_optimized_schedule
)

# ============================================================================
# DETERMINISM ENFORCEMENT
# ============================================================================
# Enable deterministic algorithms for reproducible outputs across platforms
torch.use_deterministic_algorithms(True, warn_only=False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Disable TF32 everywhere to avoid backend-specific variance (quality priority)
try:
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
except Exception as tf32_error:
    logger.warning(f"Could not disable TF32; deterministic parity may be affected: {tf32_error}")

def _collect_determinism_issues(device: str) -> Tuple[List[str], List[str]]:
    """Return blocking determinism issues and non-blocking notices for the platform."""

    blocking_issues: List[str] = []
    warnings: List[str] = []

    if device == "cuda":
        cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if cublas_config not in [":4096:8", ":16:8"]:
            blocking_issues.append(
                "Set CUBLAS_WORKSPACE_CONFIG to ':4096:8' or ':16:8' before launching to enable deterministic CUDA kernels."
            )

    if device == "mps":
        warnings.append(
            "The PyTorch MPS backend does not currently guarantee full determinism. Use CUDA or CPU for strict reproducibility."
        )

    return blocking_issues, warnings


# Platform-specific determinism settings
if torch.cuda.is_available():
    # CUBLAS_WORKSPACE_CONFIG is set in main.py before torch import
    torch.cuda.manual_seed_all(0)

if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

logger.info("Deterministic mode enabled for reproducible generation across platforms")

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
        self._cpu_offload_enabled = False  # Initialize here to ensure it always exists
        self._using_device_map = False
        self._toggle_lock = threading.Lock()  # Lock for DoRA toggle operations
        self._initialize()

    def _initialize(self):
        """Initialize the diffusion pipeline."""
        try:
            with perf_monitor.time_section("engine_initialization"):
                logger.info(f"Initializing NoobAI engine with model: {self.model_path}")

                # Detect device
                mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                if mps_available:
                    self._device = "mps"
                elif torch.cuda.is_available():
                    self._device = "cuda"
                else:
                    self._device = "cpu"

                logger.info(f"Using device: {self._device.upper()}")

                # Validate platform determinism requirements before heavy initialization
                determinism_issues, determinism_warnings = _collect_determinism_issues(self._device)
                if determinism_issues:
                    formatted_issues = "\n- " + "\n- ".join(determinism_issues)
                    raise RuntimeError(
                        "Strict determinism is required but cannot be guaranteed on this platform:" + formatted_issues
                    )
                if determinism_warnings:
                    formatted_warnings = "\n- " + "\n- ".join(determinism_warnings)
                    logger.warning(
                        "Determinism safeguards are enabled where supported, but limitations exist on this platform:" + formatted_warnings
                    )

                # Detect and validate model precision
                base_precision = detect_base_model_precision(self.model_path)
                logger.info(f"Base model dtype successfully parsed: {base_precision}")
                is_directory = os.path.isdir(self.model_path)

                if base_precision not in [torch.bfloat16, torch.float32]:
                    raise ValueError(
                        f"Model precision validation failed: got {base_precision}. "
                        f"Only BF16 model or FP32 pre-converted models are supported."
                    )

                # Check platform BF16 support
                bf16_supported = False
                if self._device == "cuda":
                    try:
                        compute_capability = torch.cuda.get_device_capability(0)
                        bf16_supported = compute_capability[0] >= 8
                    except (AttributeError, RuntimeError):
                        pass
                elif self._device == "mps":
                    bf16_supported = True
                elif self._device == "cpu":
                    bf16_supported = False

                # Select inference precision based on model format and platform support
                if base_precision == torch.bfloat16 and bf16_supported:
                    inference_dtype = torch.bfloat16
                else:
                    inference_dtype = torch.float32

                # Decide module placement strategy based on platform capabilities
                device_map = None
                max_memory = None
                cpu_offload_planned = False

                # Track whether Accelerate device_map placement is in use to avoid invalid .to() calls
                # on partially-offloaded models ("You can't move a model that has some modules offloaded").
                using_device_map = False

                if self._device == "cuda":
                    try:
                        total_vram_bytes = torch.cuda.get_device_properties(0).total_memory
                        total_vram_gb = total_vram_bytes / (1024 ** 3)

                        if total_vram_gb < 8.0:
                            device_map = "balanced"
                            max_memory = {0: f"{int(total_vram_gb)}GiB", "cpu": "32GiB"}
                            cpu_offload_planned = True
                            using_device_map = True
                            logger.info(
                                "Low VRAM detected (%.1fGB); using CUDA device_map 'balanced' with max_memory %s and planning sequential CPU offload",
                                total_vram_gb,
                                max_memory,
                            )
                        else:
                            device_map = "balanced"
                            using_device_map = True
                            logger.info("Using CUDA device_map 'balanced' for module placement")
                    except Exception as vram_detect_error:
                        device_map = "balanced"
                        using_device_map = True
                        logger.debug(
                            "Could not detect VRAM size, defaulting to balanced device_map: %s",
                            vram_detect_error,
                        )
                elif self._device == "mps":
                    device_map = "mps"
                    using_device_map = True
                    logger.info("Using MPS device_map 'mps' for module placement")
                else:
                    device_map = "cpu"
                    using_device_map = True
                    logger.info("Using CPU device_map for module placement")

                # Load FP32 pre-converted model or BF16 model with precision selection
                if base_precision == torch.float32 and is_directory:
                    # CRITICAL FIX: Explicitly load VAE as FP32 for lossless decode
                    # This ensures consistent decode quality across all model formats
                    vae_path = os.path.join(self.model_path, "vae")
                    if os.path.isdir(vae_path):
                        vae = AutoencoderKL.from_pretrained(
                            vae_path,
                            torch_dtype=torch.float32,
                        )
                        logger.info("VAE loaded as FP32 from directory for lossless decode")
                    else:
                        # Fallback: load from parent and hope for the best
                        logger.warning("VAE subdirectory not found, loading from parent directory")
                        vae = None

                    self.pipe = StableDiffusionXLPipeline.from_pretrained(
                        self.model_path,
                        vae=vae,
                        torch_dtype=inference_dtype,
                        device_map=device_map,
                        max_memory=max_memory,
                        # NOTE: torch_dtype parameter may be ignored for some components
                        # but we explicitly set VAE above to guarantee FP32 decode
                    )

                    # Ensure UNet/Text Encoder use the selected inference dtype when not using Accelerate device_map
                    if not using_device_map:
                        self.pipe.unet.to(dtype=inference_dtype)
                        if hasattr(self.pipe, "text_encoder"):
                            self.pipe.text_encoder.to(dtype=inference_dtype)
                        if hasattr(self.pipe, "text_encoder_2"):
                            self.pipe.text_encoder_2.to(dtype=inference_dtype)

                    # Validate actual loaded precision matches expected settings
                    actual_unet_dtype = next(self.pipe.unet.parameters()).dtype
                    actual_vae_dtype = next(self.pipe.vae.parameters()).dtype

                    if not using_device_map and actual_unet_dtype != inference_dtype:
                        raise ValueError(
                            f"Expected {inference_dtype} model but UNet loaded with {actual_unet_dtype}. "
                            f"Directory may contain wrong precision weights."
                        )

                    if not using_device_map and hasattr(self.pipe, "text_encoder"):
                        actual_te_dtype = next(self.pipe.text_encoder.parameters()).dtype
                        if actual_te_dtype != inference_dtype:
                            raise ValueError(
                                f"Expected {inference_dtype} model but TextEncoder loaded with {actual_te_dtype}. "
                                f"Directory may contain wrong precision weights."
                            )

                    if not using_device_map and hasattr(self.pipe, "text_encoder_2"):
                        actual_te2_dtype = next(self.pipe.text_encoder_2.parameters()).dtype
                        if actual_te2_dtype != inference_dtype:
                            raise ValueError(
                                f"Expected {inference_dtype} model but TextEncoder 2 loaded with {actual_te2_dtype}. "
                                f"Directory may contain wrong precision weights."
                            )

                    if actual_vae_dtype != torch.float32:
                        raise ValueError(
                            f"Expected FP32 VAE but loaded with {actual_vae_dtype}. "
                            f"This could cause decode quality issues."
                        )

                    logger.info(
                        "FP32 directory model loaded with all components validated "
                        f"(UNet/Text: {inference_dtype}, VAE: torch.float32)"
                    )

                else:
                    # Single file model (BF16) - load pipeline, then force VAE to FP32
                    # AutoencoderKL.from_single_file is not supported in older diffusers versions;
                    # load the full pipeline first, then upcast the VAE for lossless decode.
                    self.pipe = StableDiffusionXLPipeline.from_single_file(
                        self.model_path,
                        torch_dtype=inference_dtype,
                        use_safetensors=True,
                        device_map=device_map,
                        max_memory=max_memory,
                    )

                    # Upcast VAE to FP32 for consistent decoding regardless of inference dtype
                    try:
                        self.pipe.vae.to(dtype=torch.float32)
                        logger.info("Single file model loaded; VAE upcast to FP32 for lossless decode")
                    except RuntimeError as e:
                        error_str = str(e).lower()
                        if "out of memory" in error_str or "cuda" in error_str:
                            raise MemoryError(f"Insufficient GPU memory for FP32 VAE upcast: {e}")
                        raise ValueError(f"Failed to upcast VAE to FP32: {e}")
                    except MemoryError:
                        raise
                    except Exception as e:
                        logger.error(f"Unexpected error during VAE upcast: {e}")
                        raise ValueError(f"Failed to upcast VAE to FP32: {e}")

                    # Ensure UNet/Text Encoder use the selected inference dtype when not using Accelerate device_map
                    if not using_device_map:
                        self.pipe.unet.to(dtype=inference_dtype)
                        if hasattr(self.pipe, "text_encoder"):
                            self.pipe.text_encoder.to(dtype=inference_dtype)
                        if hasattr(self.pipe, "text_encoder_2"):
                            self.pipe.text_encoder_2.to(dtype=inference_dtype)

                    # Validate component dtypes after adjustments
                    actual_unet_dtype = next(self.pipe.unet.parameters()).dtype
                    actual_vae_dtype = next(self.pipe.vae.parameters()).dtype

                    if not using_device_map and actual_unet_dtype != inference_dtype:
                        raise ValueError(
                            f"Expected {inference_dtype} model but UNet loaded with {actual_unet_dtype}. "
                            "Single-file weights may be in an unexpected precision."
                        )

                    if not using_device_map and hasattr(self.pipe, "text_encoder"):
                        actual_te_dtype = next(self.pipe.text_encoder.parameters()).dtype
                        if actual_te_dtype != inference_dtype:
                            raise ValueError(
                                f"Expected {inference_dtype} model but TextEncoder loaded with {actual_te_dtype}. "
                                "Single-file weights may be in an unexpected precision."
                            )

                    if not using_device_map and hasattr(self.pipe, "text_encoder_2"):
                        actual_te2_dtype = next(self.pipe.text_encoder_2.parameters()).dtype
                        if actual_te2_dtype != inference_dtype:
                            raise ValueError(
                                f"Expected {inference_dtype} model but TextEncoder 2 loaded with {actual_te2_dtype}. "
                                "Single-file weights may be in an unexpected precision."
                            )

                    if actual_vae_dtype != torch.float32:
                        raise ValueError(
                            f"Expected FP32 VAE but loaded with {actual_vae_dtype}. "
                            "This could cause decode quality issues."
                        )

                logger.info(
                    "Pipeline created with inference dtype %s (UNet/Text) and VAE dtype torch.float32",
                    inference_dtype,
                )

                # Persist device_map usage for other methods
                self._using_device_map = using_device_map

                # Configure CPU offload immediately after loading when planned
                if cpu_offload_planned:
                    self.pipe.enable_sequential_cpu_offload()
                    self._cpu_offload_enabled = True
                    logger.info("Sequential CPU offload enabled during initialization")

                # Configure scheduler
                self.pipe.scheduler = EulerDiscreteScheduler.from_config(
                    self.pipe.scheduler.config,
                    prediction_type="v_prediction",
                    rescale_betas_zero_snr=True,
                    timestep_spacing="trailing"
                )

                # Move to device and configure memory management
                if not self._cpu_offload_enabled and not self._using_device_map:
                    if self._device == "cuda":
                        try:
                            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                            if vram_gb < 8.0:
                                self.pipe.enable_sequential_cpu_offload()
                                self._cpu_offload_enabled = True
                                logger.info(f"CPU offloading enabled ({vram_gb:.1f}GB VRAM) after device move")
                            else:
                                self.pipe = self.pipe.to(self._device)
                        except Exception as vram_check_error:
                            logger.debug(f"Could not detect VRAM size, loading to device directly: {vram_check_error}")
                            self.pipe = self.pipe.to(self._device)
                    else:
                        self.pipe = self.pipe.to(self._device)

                # Enable memory optimizations
                self.pipe.enable_vae_slicing()

                self.is_initialized = True
                logger.info("Engine initialized")

                # Load DoRA adapter if enabled
                if self.enable_dora:
                    self._load_dora_adapter()

        except (IOError, OSError, RuntimeError, ValueError, MemoryError) as e:
            self.is_initialized = False
            if self.pipe is not None:
                try:
                    if self._device in ["cuda", "mps"]:
                        self.clear_memory()
                except Exception as cleanup_error:
                    logger.debug(f"Non-critical cleanup error during init failure: {cleanup_error}")
                finally:
                    self.pipe = None
            logger.error(f"Failed to initialize engine: {e}")
            deterministic_error = "deterministic" in str(e).lower()
            if deterministic_error:
                raise RuntimeError(
                    f"Deterministic initialization failed: {e}. "
                    "Ensure required environment variables are set and run on a backend with deterministic support."
                )
            raise
        except Exception as e:
            self.is_initialized = False
            if self.pipe is not None:
                try:
                    if self._device in ["cuda", "mps"]:
                        self.clear_memory()
                except Exception:
                    pass
                finally:
                    self.pipe = None
            logger.error(f"Unexpected error during engine initialization: {e}")
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

            # Load DoRA adapter with precision detection
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="It seems like you are using a DoRA checkpoint")
                warnings.filterwarnings("ignore", message="No LoRA keys associated to CLIPTextModel")
                warnings.filterwarnings("ignore", message="No LoRA keys associated to CLIPTextModelWithProjection")
                self.pipe.load_lora_weights(
                    os.path.dirname(validated_path),
                    weight_name=os.path.basename(validated_path),
                    adapter_name="noobai_dora"
                )

            # Set adapter scale
            self.pipe.set_adapters(["noobai_dora"], adapter_weights=[self.adapter_strength])

            # Set path only after successful load
            self.dora_path = validated_path
            self.dora_loaded = True
            logger.info(f"DoRA adapter loaded: {os.path.basename(validated_path)}")

        except (IOError, OSError) as e:
            logger.error(f"Failed to load DoRA adapter (file error): {e}")
            self.dora_loaded = False
        except (RuntimeError, ValueError) as e:
            logger.error(f"Failed to load DoRA adapter (runtime/validation error): {e}")
            # Attempt to unload partial state to prevent GPU memory leak
            try:
                self.pipe.unload_lora_weights()
            except Exception:
                pass
            # Force memory cleanup after failed load
            self.clear_memory()
            self.dora_loaded = False
        except Exception as e:
            logger.error(f"Unexpected error loading DoRA adapter: {e}")
            # Attempt to unload partial state to prevent GPU memory leak
            try:
                self.pipe.unload_lora_weights()
            except Exception:
                pass
            # Force memory cleanup after failed load
            self.clear_memory()
            self.dora_loaded = False
        finally:
            # Ensure path is reset on failure (all error paths set dora_loaded=False)
            if not self.dora_loaded:
                self.dora_path = None

    def unload_dora_adapter(self) -> None:
        """Completely unload DoRA adapter with full memory cleanup."""
        try:
            if self.dora_loaded and self.pipe is not None:
                # Disable adapter by setting weight to 0
                try:
                    self.pipe.set_adapters(["noobai_dora"], adapter_weights=[0.0])
                except Exception as e:
                    logger.warning(f"Could not set adapter weights to 0: {e}")

                # Completely remove LoRA weights from memory
                try:
                    self.pipe.unload_lora_weights()
                except Exception as e:
                    logger.warning(f"Error unloading LoRA weights: {e}")

                # Delete adapter references if supported
                try:
                    if hasattr(self.pipe, 'delete_adapters'):
                        self.pipe.delete_adapters(["noobai_dora"])
                except Exception as e:
                    logger.warning(f"Error deleting adapter references: {e}")

                # Clear memory caches
                self.clear_memory()

                logger.info("DoRA adapter unloaded")

            self.dora_loaded = False
            self.dora_path = None

        except Exception as e:
            logger.error(f"Error unloading DoRA adapter: {e}")
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
            old_dora_path = self.dora_path
            if self.dora_loaded:
                self.unload_dora_adapter()
                # GPU synchronization now handled by clear_memory()

            # 2. Update adapter path and attempt to load
            self.dora_path = validated_path
            self._load_dora_adapter()

            if self.dora_loaded:
                logger.info(f"Successfully switched to DoRA adapter: {os.path.basename(validated_path)}")
                return True
            else:
                # Restore old path on failure to maintain state consistency
                self.dora_path = old_dora_path
                logger.error("Failed to load new DoRA adapter")
                return False

        except Exception as e:
            logger.error(f"Error switching DoRA adapter: {e}")
            self.dora_loaded = False
            self.dora_path = None
            return False

    def set_adapter_strength(self, strength: float) -> float:
        """Set DoRA adapter strength with clamping.

        Args:
            strength: Desired adapter strength

        Returns:
            Actual strength value used (after clamping if necessary)
        """
        try:
            original_strength = strength
            # Validate strength is in bounds and clamp if necessary
            if not (MODEL_CONFIG.MIN_ADAPTER_STRENGTH <= strength <= MODEL_CONFIG.MAX_ADAPTER_STRENGTH):
                strength = max(MODEL_CONFIG.MIN_ADAPTER_STRENGTH, min(strength, MODEL_CONFIG.MAX_ADAPTER_STRENGTH))
                logger.warning(
                    f"Adapter strength {original_strength} out of bounds "
                    f"[{MODEL_CONFIG.MIN_ADAPTER_STRENGTH}-{MODEL_CONFIG.MAX_ADAPTER_STRENGTH}], "
                    f"clamped to {strength}"
                )

            self.adapter_strength = strength

            # Apply to pipeline if loaded and enabled
            if self.dora_loaded and self.pipe is not None and self.enable_dora:
                self.pipe.set_adapters(["noobai_dora"], adapter_weights=[strength])

            return strength

        except Exception as e:
            logger.warning(f"Error setting adapter strength: {e}")
            # Return current strength on error
            return self.adapter_strength

    def set_dora_start_step(self, start_step: int) -> int:
        """Set DoRA adapter start step with clamping.

        Args:
            start_step: Desired start step

        Returns:
            Actual start step value used (after clamping if necessary)
        """
        try:
            # Coerce to int first to handle floats from UI
            if not isinstance(start_step, int):
                try:
                    start_step = int(start_step)
                except (TypeError, ValueError):
                    logger.warning(f"Invalid DoRA start step type: {type(start_step)}, using default")
                    start_step = MODEL_CONFIG.DEFAULT_DORA_START_STEP

            original_start_step = start_step
            # Validate start step is in bounds and clamp if necessary
            if not (MODEL_CONFIG.MIN_DORA_START_STEP <= start_step <= MODEL_CONFIG.MAX_DORA_START_STEP):
                start_step = max(MODEL_CONFIG.MIN_DORA_START_STEP, min(start_step, MODEL_CONFIG.MAX_DORA_START_STEP))
                logger.warning(
                    f"DoRA start step {original_start_step} out of bounds "
                    f"[{MODEL_CONFIG.MIN_DORA_START_STEP}-{MODEL_CONFIG.MAX_DORA_START_STEP}], "
                    f"clamped to {start_step}"
                )

            self.dora_start_step = start_step
            return start_step

        except Exception as e:
            logger.warning(f"Error setting DoRA start step: {e}")
            # Return current value on error
            return self.dora_start_step

    def set_dora_enabled(self, enabled: bool) -> None:
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

    def _parse_manual_dora_schedule(
        self,
        dora_toggle_mode: Optional[str],
        dora_manual_schedule: Optional[str],
        steps: int
    ) -> Tuple[Optional[List[int]], Optional[str]]:
        """
        Parse manual DoRA schedule if manual toggle mode is selected.

        Args:
            dora_toggle_mode: Toggle mode ('manual', 'standard', 'smart', or None)
            dora_manual_schedule: CSV string of 0/1 values
            steps: Total number of generation steps

        Returns:
            Tuple of (parsed_schedule, warning_message)
        """
        if dora_toggle_mode != "manual":
            return None, None

        if not dora_manual_schedule:
            logger.warning("Manual toggle mode selected but no schedule provided - DoRA will be OFF for all steps")
            return None, None

        manual_schedule, manual_schedule_warning = parse_manual_dora_schedule(dora_manual_schedule, steps)

        if manual_schedule_warning:
            logger.warning(manual_schedule_warning)

        return manual_schedule, manual_schedule_warning

    def _enforce_toggle_mode_exclusivity(self, dora_toggle_mode: Optional[str]) -> bool:
        """
        Enforce mutual exclusivity constraints for toggle modes.
        Toggle modes always start from step 1 and override dora_start_step.

        DoRA toggle implementation uses efficient weight scaling (set_adapters with 0.0 or adapter_strength),
        not reload/unload. Adapter weights remain resident in VRAM throughout generation.
        Compatible with CPU offloading with minimal performance impact (<5%).

        Args:
            dora_toggle_mode: Toggle mode ('manual', 'standard', 'smart', or None)

        Returns:
            bool: Always True (toggle mode compatibility is universal)
        """
        if dora_toggle_mode and self.enable_dora and self.dora_loaded:
            if self.dora_start_step > 1:
                logger.warning(
                    f"Toggle mode '{dora_toggle_mode}' enabled with dora_start_step={self.dora_start_step}. "
                    f"Resetting start_step to 1."
                )
                self.dora_start_step = 1

        return True  # Toggle mode can always proceed

    def _setup_initial_dora_state(
        self,
        dora_toggle_mode: Optional[str],
        manual_schedule: Optional[List[int]]
    ) -> None:
        """
        Set up initial DoRA adapter state before generation starts.

        Args:
            dora_toggle_mode: Toggle mode ('manual', 'standard', 'smart', or None)
            manual_schedule: Parsed manual schedule (if manual mode)
        """
        if not self.enable_dora or not self.dora_loaded or not self.pipe:
            return

        # Handle delayed activation (normal mode)
        if self.dora_start_step > 1 and not dora_toggle_mode:
            self.pipe.set_adapters(["noobai_dora"], adapter_weights=[0.0])
            return

        # Handle toggle modes
        if dora_toggle_mode:
            if dora_toggle_mode == "manual":
                # Manual mode: Set based on schedule index 0 (default OFF)
                if manual_schedule and len(manual_schedule) > 0 and manual_schedule[0] == 1:
                    self.pipe.set_adapters(["noobai_dora"], adapter_weights=[self.adapter_strength])
                else:
                    self.pipe.set_adapters(["noobai_dora"], adapter_weights=[0.0])
            elif dora_toggle_mode == "optimized":
                # Optimized mode: Index 0 = OFF (structure formation phase)
                self.pipe.set_adapters(["noobai_dora"], adapter_weights=[0.0])
            else:
                # Standard and Smart modes: Index 0 = ON
                self.pipe.set_adapters(["noobai_dora"], adapter_weights=[self.adapter_strength])

    def _build_generation_info(
        self,
        steps: int,
        cfg_scale: float,
        width: int,
        height: int
    ) -> List[str]:
        """
        Build informational messages about generation parameters.

        Args:
            steps: Number of generation steps
            cfg_scale: CFG scale value
            width: Image width
            height: Image height

        Returns:
            List of info message strings
        """
        info_parts = []

        # Check optimal ranges
        if not (32 <= steps <= 40):
            info_parts.append(f"⚠️ Steps {steps} outside optimal range 32-40")
        if not (3.5 <= cfg_scale <= 5.5):
            info_parts.append(f"⚠️ CFG {cfg_scale} outside optimal range 3.5-5.5")

        # Check resolution
        current_res = (height, width)
        if current_res in RECOMMENDED_RESOLUTIONS:
            info_parts.append(f"✅ Optimal resolution: {width}x{height}")
        elif current_res in OFFICIAL_RESOLUTIONS:
            info_parts.append(f"✅ Official resolution: {width}x{height}")
        else:
            info_parts.append(f"⚠️ Non-official resolution: {width}x{height}")

        return info_parts

    def _create_progress_callback(
        self,
        steps: int,
        start_time: float,
        dora_toggle_mode: Optional[str],
        manual_schedule: Optional[List[int]],
        progress_callback: Optional[Callable[[float, str], None]]
    ) -> Callable:
        """
        Create the progress callback function for generation.

        Args:
            steps: Total number of generation steps
            start_time: Generation start time
            dora_toggle_mode: Toggle mode ('manual', 'standard', 'smart', 'optimized', or None)
            manual_schedule: Parsed manual schedule (if manual mode)
            progress_callback: User-provided progress callback

        Returns:
            Callback function for step_end events
        """
        # Pre-compute optimized schedule once (performance optimization)
        cached_optimized_schedule = None
        if dora_toggle_mode == "optimized":
            cached_optimized_schedule = generate_optimized_schedule(steps)
            # Validate schedule bounds
            if cached_optimized_schedule and len(cached_optimized_schedule) != steps:
                logger.warning(
                    f"Optimized schedule size mismatch: {len(cached_optimized_schedule)} vs {steps} steps. "
                    "Schedule will be bounds-checked during generation."
                )

        # Track callback failures for escalation
        callback_failure_count = [0]  # Use list to allow mutation in nested function

        def callback_on_step_end(pipe, step_index: int, timestep, callback_kwargs: Dict) -> Dict:
            """Diffusers callback invoked at the end of each denoising step.

            Handles interruption checking, progress tracking, DoRA toggling,
            and user progress callback invocation.

            Args:
                pipe: The StableDiffusionXLPipeline instance
                step_index: Current step index (0-based)
                timestep: Current diffusion timestep
                callback_kwargs: Pipeline callback state dictionary

            Returns:
                callback_kwargs: Unmodified callback state for pipeline

            Raises:
                GenerationInterruptedError: If user requested interruption
            """
            # Check for interruption
            if state_manager.is_interrupted():
                raise GenerationInterruptedError()

            current_step = step_index + 1
            progress = current_step / steps
            elapsed = time.time() - start_time
            eta = (elapsed / current_step) * (steps - current_step) if current_step > 0 else 0

            # Build progress description (use cached optimized schedule for performance)
            desc = self._build_progress_description(
                step_index, current_step, steps, eta,
                dora_toggle_mode, manual_schedule, cached_optimized_schedule
            )

            # Call user callback with error isolation and escalation tracking
            if progress_callback:
                try:
                    progress_callback(progress, desc)
                    # Reset failure count on success
                    callback_failure_count[0] = 0
                except Exception as e:
                    callback_failure_count[0] += 1
                    # Log with full traceback for debugging
                    logger.error(
                        f"Progress callback error at step {current_step}: {e}",
                        exc_info=True
                    )
                    # In CLI mode, re-raise to alert user immediately
                    if os.environ.get('NOOBAI_CLI_MODE') == '1':
                        raise
                    # In GUI mode, try one more time with minimal description
                    try:
                        progress_callback(progress, f"Step {current_step}/{steps}")
                        callback_failure_count[0] = 0  # Reset on retry success
                    except Exception as retry_error:
                        callback_failure_count[0] += 1
                        # Log retry failure even in GUI mode for debugging
                        logger.error(f"Progress callback retry also failed: {retry_error}")
                        # Escalate warning after repeated failures
                        if callback_failure_count[0] >= 3:
                            logger.warning(
                                f"Progress callback failed {callback_failure_count[0]} times. "
                                "Generation continuing but progress display may be unreliable."
                            )

            return callback_kwargs

        return callback_on_step_end

    def _build_progress_description(
        self,
        step_index: int,
        current_step: int,
        steps: int,
        eta: float,
        dora_toggle_mode: Optional[str],
        manual_schedule: Optional[List[int]],
        cached_optimized_schedule: Optional[List[int]] = None
    ) -> str:
        """
        Build progress description string and update DoRA state for next step.

        Args:
            step_index: Current step index (0-based)
            current_step: Current step number (1-based)
            steps: Total steps
            eta: Estimated time remaining
            dora_toggle_mode: Toggle mode
            manual_schedule: Manual schedule
            cached_optimized_schedule: Pre-computed optimized schedule (for performance)

        Returns:
            Progress description string
        """
        # Handle DoRA toggle modes
        if dora_toggle_mode and self.enable_dora and self.dora_loaded and self.pipe:
            next_step_index = step_index + 1

            if dora_toggle_mode == "standard":
                return self._handle_standard_toggle(step_index, current_step, steps, eta, next_step_index)
            elif dora_toggle_mode == "smart":
                return self._handle_smart_toggle(step_index, current_step, steps, eta, next_step_index)
            elif dora_toggle_mode == "optimized":
                # Use cached schedule for performance (avoid regenerating every step)
                optimized_schedule = cached_optimized_schedule or generate_optimized_schedule(steps)
                return self._handle_optimized_toggle(
                    step_index, current_step, steps, eta, next_step_index, optimized_schedule
                )
            elif dora_toggle_mode == "manual":
                return self._handle_manual_toggle(
                    step_index, current_step, steps, eta, next_step_index, manual_schedule
                )

        # Handle normal DoRA start step
        elif self.enable_dora and self.dora_loaded and self.pipe:
            if current_step == self.dora_start_step - 1 and self.dora_start_step > 1:
                self.pipe.set_adapters(["noobai_dora"], adapter_weights=[self.adapter_strength])
                return f"Step {current_step}/{steps} (DoRA will activate at step {self.dora_start_step}, ETA: {eta:.1f}s)"
            elif current_step >= self.dora_start_step:
                return f"Step {current_step}/{steps} (DoRA active, ETA: {eta:.1f}s)"
            else:
                return f"Step {current_step}/{steps} (DoRA starts at step {self.dora_start_step}, ETA: {eta:.1f}s)"

        # No DoRA
        return f"Step {current_step}/{steps} (ETA: {eta:.1f}s)"

    def _handle_standard_toggle(
        self, step_index: int, current_step: int, steps: int, eta: float, next_step_index: int
    ) -> str:
        """Handle standard toggle mode progress description with device synchronization."""
        if next_step_index < steps:
            current_state = "ON" if step_index % 2 == 0 else "OFF"
            # Use lock to prevent race conditions during adapter weight changes
            with self._toggle_lock:
                self._synchronize_device()
                if next_step_index % 2 == 0:  # Next is even - turn ON
                    self.pipe.set_adapters(["noobai_dora"], adapter_weights=[self.adapter_strength])
                    return f"Step {current_step}/{steps} (DoRA: {current_state}, next[{next_step_index}]: ON, ETA: {eta:.1f}s)"
                else:  # Next is odd - turn OFF
                    self.pipe.set_adapters(["noobai_dora"], adapter_weights=[0.0])
                    return f"Step {current_step}/{steps} (DoRA: {current_state}, next[{next_step_index}]: OFF, ETA: {eta:.1f}s)"
        else:
            current_state = "ON" if step_index % 2 == 0 else "OFF"
            return f"Step {current_step}/{steps} (DoRA: {current_state}, final, ETA: {eta:.1f}s)"

    def _handle_smart_toggle(
        self, step_index: int, current_step: int, steps: int, eta: float, next_step_index: int
    ) -> str:
        """Handle smart toggle mode progress description with device synchronization."""
        if next_step_index < steps:
            # Use lock to prevent race conditions during adapter weight changes
            with self._toggle_lock:
                self._synchronize_device()
                if next_step_index <= 19:
                    # Alternating phase (indices 0-19)
                    current_state = "ON" if step_index % 2 == 0 else "OFF"
                    if next_step_index % 2 == 0:  # Next is even - turn ON
                        self.pipe.set_adapters(["noobai_dora"], adapter_weights=[self.adapter_strength])
                        return f"Step {current_step}/{steps} (DoRA: {current_state}, next[{next_step_index}]: ON, ETA: {eta:.1f}s)"
                    else:  # Next is odd - turn OFF
                        self.pipe.set_adapters(["noobai_dora"], adapter_weights=[0.0])
                        return f"Step {current_step}/{steps} (DoRA: {current_state}, next[{next_step_index}]: OFF, ETA: {eta:.1f}s)"
                else:
                    # Always ON phase (index 20+)
                    self.pipe.set_adapters(["noobai_dora"], adapter_weights=[self.adapter_strength])
                    return f"Step {current_step}/{steps} (DoRA: ON [smart-locked], next[{next_step_index}]: ON, ETA: {eta:.1f}s)"
        else:
            # Final step
            if step_index <= 19:
                current_state = "ON" if step_index % 2 == 0 else "OFF"
            else:
                current_state = "ON"
            return f"Step {current_step}/{steps} (DoRA: {current_state}, final, ETA: {eta:.1f}s)"

    def _handle_optimized_toggle(
        self,
        step_index: int,
        current_step: int,
        steps: int,
        eta: float,
        next_step_index: int,
        optimized_schedule: List[int]
    ) -> str:
        """Handle optimized toggle mode progress description with device synchronization.

        This mode uses an empirically-tuned DoRA activation pattern optimized for
        NoobAI V-Pred with the stabilizer DoRA adapter. It achieves ~44% activation
        with strategic phase alignment for optimal quality.
        """
        if not optimized_schedule:
            return f"Step {current_step}/{steps} (DoRA: OFF [no schedule], ETA: {eta:.1f}s)"

        # Bounds check to prevent IndexError if schedule is malformed
        if step_index >= len(optimized_schedule):
            current_state = "OFF"
        else:
            current_state = "ON" if optimized_schedule[step_index] == 1 else "OFF"

        if next_step_index < steps:
            # Use lock to prevent race conditions during adapter weight changes
            with self._toggle_lock:
                # Bounds check for next step
                if next_step_index >= len(optimized_schedule):
                    next_state = "OFF"
                    self._synchronize_device()
                    self.pipe.set_adapters(["noobai_dora"], adapter_weights=[0.0])
                else:
                    next_state = "ON" if optimized_schedule[next_step_index] == 1 else "OFF"
                    self._synchronize_device()
                    # Set adapter for next step
                    if optimized_schedule[next_step_index] == 1:
                        self.pipe.set_adapters(["noobai_dora"], adapter_weights=[self.adapter_strength])
                    else:
                        self.pipe.set_adapters(["noobai_dora"], adapter_weights=[0.0])
            return f"Step {current_step}/{steps} (DoRA: {current_state} [optimized], next[{next_step_index}]: {next_state}, ETA: {eta:.1f}s)"
        else:
            return f"Step {current_step}/{steps} (DoRA: {current_state} [optimized], final, ETA: {eta:.1f}s)"

    def _handle_manual_toggle(
        self,
        step_index: int,
        current_step: int,
        steps: int,
        eta: float,
        next_step_index: int,
        manual_schedule: Optional[List[int]]
    ) -> str:
        """Handle manual toggle mode progress description with device synchronization."""
        if not manual_schedule:
            return f"Step {current_step}/{steps} (DoRA: OFF [no schedule], ETA: {eta:.1f}s)"

        # Bounds check to prevent IndexError if schedule is malformed
        if step_index >= len(manual_schedule):
            logger.warning(f"Manual schedule too short: index {step_index} >= length {len(manual_schedule)}")
            current_state = "OFF"
        else:
            current_state = "ON" if manual_schedule[step_index] == 1 else "OFF"

        if next_step_index < steps:
            # Use lock to prevent race conditions during adapter weight changes
            with self._toggle_lock:
                # Bounds check to prevent IndexError if schedule is malformed
                if next_step_index >= len(manual_schedule):
                    logger.warning(f"Manual schedule too short: index {next_step_index} >= length {len(manual_schedule)}, treating as OFF")
                    next_state = "OFF"
                    self._synchronize_device()
                    self.pipe.set_adapters(["noobai_dora"], adapter_weights=[0.0])
                else:
                    next_state = "ON" if manual_schedule[next_step_index] == 1 else "OFF"
                    self._synchronize_device()
                    # Set adapter for next step
                    if manual_schedule[next_step_index] == 1:
                        self.pipe.set_adapters(["noobai_dora"], adapter_weights=[self.adapter_strength])
                    else:
                        self.pipe.set_adapters(["noobai_dora"], adapter_weights=[0.0])
            return f"Step {current_step}/{steps} (DoRA: {current_state}, next[{next_step_index}]: {next_state}, ETA: {eta:.1f}s)"
        else:
            return f"Step {current_step}/{steps} (DoRA: {current_state}, final, ETA: {eta:.1f}s)"

    def _add_dora_info_to_result(
        self, info_parts: List[str], dora_toggle_mode: Optional[str], seed: int
    ) -> None:
        """
        Add DoRA information to generation info.

        Args:
            info_parts: List to append info messages to
            dora_toggle_mode: Toggle mode used
            seed: Generation seed
        """
        info_parts.append(f"🌱 Generated with seed: {seed}")

        if self.dora_loaded:
            dora_name = os.path.basename(self.dora_path) if self.dora_path else "DoRA"
            if self.enable_dora:
                dora_info = f"🎯 DoRA: {dora_name} (strength: {self.adapter_strength}"
                if dora_toggle_mode == "standard":
                    dora_info += ", toggle: ON,OFF throughout"
                elif dora_toggle_mode == "smart":
                    dora_info += ", smart toggle: ON,OFF to step 20, then ON"
                elif dora_toggle_mode == "optimized":
                    dora_info += ", optimized toggle: empirically-tuned phases (~44% active)"
                elif dora_toggle_mode == "manual":
                    dora_info += ", manual toggle schedule"
                elif self.dora_start_step > 1:
                    dora_info += f", starts at step {self.dora_start_step}"
                dora_info += ")"
                info_parts.append(dora_info)
            else:
                info_parts.append(f"⚪ DoRA: {dora_name} (disabled)")
        elif self.dora_path:
            info_parts.append("⚠️ DoRA: Available but not loaded")

    def _create_image_metadata(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        rescale_cfg: float,
        dora_toggle_mode: Optional[str]
    ) -> Dict[str, str]:
        """
        Create image metadata dictionary.

        Args:
            prompt: Positive prompt
            negative_prompt: Negative prompt
            seed: Generation seed
            width: Image width
            height: Image height
            steps: Number of steps
            cfg_scale: CFG scale
            rescale_cfg: Rescale CFG
            dora_toggle_mode: Toggle mode

        Returns:
            Metadata dictionary
        """
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

        # Add DoRA metadata
        if self.dora_loaded:
            metadata["dora_enabled"] = str(self.enable_dora).lower()
            metadata["dora_path"] = os.path.basename(self.dora_path) if self.dora_path else "unknown"
            if self.enable_dora:
                metadata["adapter_strength"] = str(self.adapter_strength)
                metadata["dora_start_step"] = str(self.dora_start_step)
                metadata["dora_toggle_mode"] = dora_toggle_mode if dora_toggle_mode else "none"
            else:
                metadata["adapter_strength"] = "0.0"
                metadata["dora_start_step"] = "1"
                metadata["dora_toggle_mode"] = "none"

        return metadata

    def save_image_standardized(self, image: Image.Image, output_path: str,
                               include_metadata: bool = True) -> str:
        """Save image with standardized settings for consistent hashing.

        Creates output directory if it doesn't exist to prevent race conditions.
        """
        # Ensure output directory exists (prevents race condition if deleted)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
            except (IOError, OSError) as e:
                logger.error(f"Failed to create output directory {output_dir}: {e}")
                raise

        # Prepare PNG metadata with validation
        pnginfo = None
        if include_metadata and hasattr(image, 'info') and image.info:
            pnginfo = PngImagePlugin.PngInfo()

            # Add metadata with validation (sorted keys for consistency)
            for key in sorted(image.info.keys()):
                try:
                    # Validate key is string and printable
                    if not isinstance(key, str):
                        logger.debug(f"Skipping non-string metadata key: {type(key)}")
                        continue
                    if not key.isprintable():
                        logger.debug(f"Skipping non-printable metadata key: {repr(key)}")
                        continue
                    if len(key) > 79:  # PNG key length limit
                        logger.debug(f"Truncating metadata key: {key[:76]}...")
                        key = key[:79]

                    # Validate and truncate value
                    value_str = str(image.info[key])
                    if len(value_str) > 2000:  # Reasonable limit
                        logger.debug(f"Truncating metadata value for key '{key}'")
                        value_str = value_str[:1997] + "..."

                    pnginfo.add_text(key, value_str)
                except Exception as e:
                    logger.warning(f"Skipping metadata key '{key}': {e}")

        # Save with consistent parameters and verification
        try:
            image.save(
                output_path,
                format='PNG',
                pnginfo=pnginfo,
                compress_level=6,  # Standard compression level
                optimize=False  # Disable optimization for consistency
            )

            # Verify save succeeded
            if not os.path.exists(output_path):
                raise IOError(f"File was not created: {output_path}")

            file_size = os.path.getsize(output_path)
            if file_size == 0:
                raise IOError(f"File is empty: {output_path}")
            if file_size < 1000:  # Minimum realistic PNG size
                raise IOError(
                    f"File suspiciously small ({file_size} bytes): {output_path}"
                )

            # Verify PNG is readable
            try:
                with Image.open(output_path) as test_img:
                    test_img.verify()
            except Exception as e:
                raise IOError(f"Saved PNG is corrupted: {e}")

            return output_path

        except Exception as e:
            # Clean up failed save
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except Exception:
                    pass
            raise IOError(f"Failed to save image: {e}")

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
        dora_toggle_mode: Optional[str] = None,
        dora_manual_schedule: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[Image.Image, int, str]:
        """
        Generate an image with the specified parameters.

        Args:
            prompt: Positive prompt text
            negative_prompt: Negative prompt text
            width: Image width in pixels
            height: Image height in pixels
            steps: Number of inference steps
            cfg_scale: Classifier-free guidance scale
            rescale_cfg: CFG rescale factor
            seed: Random seed (None for random)
            adapter_strength: DoRA adapter strength
            enable_dora: Enable/disable DoRA
            dora_start_step: Step at which DoRA activates
            dora_toggle_mode: Toggle mode ('standard', 'smart', 'manual', or None)
            dora_manual_schedule: CSV schedule for manual mode
            progress_callback: Callback for progress updates

        Returns:
            Tuple of (generated_image, used_seed, info_string)

        Raises:
            EngineNotInitializedError: If engine is not initialized
            GenerationInterruptedError: If generation is interrupted
        """
        if not self.is_initialized:
            raise EngineNotInitializedError("NoobAI engine is not initialized")

        # Apply dynamic DoRA settings
        if enable_dora is not None:
            self.set_dora_enabled(enable_dora)
        if adapter_strength is not None:
            # set_adapter_strength handles all cases (loaded, not loaded, disabled)
            self.set_adapter_strength(adapter_strength)
        if dora_start_step is not None:
            self.set_dora_start_step(dora_start_step)

        # Validate and coerce steps parameter
        if not isinstance(steps, int):
            try:
                steps = int(steps)
            except (TypeError, ValueError):
                raise InvalidParameterError(
                    f"Steps must be integer, got {type(steps).__name__}"
                )
        if steps <= 0:
            raise InvalidParameterError(
                f"Steps must be positive, got {steps}"
            )

        # Validate prompt parameters
        if not isinstance(prompt, str):
            raise InvalidParameterError(f"Prompt must be string, got {type(prompt).__name__}")
        if not prompt.strip():
            raise InvalidParameterError("Prompt cannot be empty")
        if len(prompt) > GEN_CONFIG.MAX_PROMPT_LENGTH:
            raise InvalidParameterError(
                f"Prompt too long ({len(prompt)} > {GEN_CONFIG.MAX_PROMPT_LENGTH})"
            )
        if not isinstance(negative_prompt, str):
            raise InvalidParameterError(f"Negative prompt must be string, got {type(negative_prompt).__name__}")
        if len(negative_prompt) > GEN_CONFIG.MAX_PROMPT_LENGTH:
            raise InvalidParameterError(
                f"Negative prompt too long ({len(negative_prompt)} > {GEN_CONFIG.MAX_PROMPT_LENGTH})"
            )

        # Validate dimensions (coerce once and reuse validated values)
        validated_dims: Dict[str, int] = {}
        for dim_name, dim_value in [("width", width), ("height", height)]:
            if not isinstance(dim_value, int):
                try:
                    dim_value = int(dim_value)
                except (TypeError, ValueError):
                    raise InvalidParameterError(
                        f"{dim_name} must be integer, got {type(dim_value).__name__}"
                    )
            if not (GEN_CONFIG.MIN_RESOLUTION <= dim_value <= GEN_CONFIG.MAX_RESOLUTION):
                raise InvalidParameterError(
                    f"{dim_name} must be in [{GEN_CONFIG.MIN_RESOLUTION}, {GEN_CONFIG.MAX_RESOLUTION}], got {dim_value}"
                )
            if dim_value % 8 != 0:
                raise InvalidParameterError(
                    f"{dim_name} must be divisible by 8, got {dim_value}"
                )
            validated_dims[dim_name] = dim_value

        width = validated_dims["width"]
        height = validated_dims["height"]

        # Validate CFG parameters
        if not isinstance(cfg_scale, (int, float)):
            raise InvalidParameterError(f"cfg_scale must be numeric, got {type(cfg_scale).__name__}")
        if not (GEN_CONFIG.MIN_CFG_SCALE <= cfg_scale <= GEN_CONFIG.MAX_CFG_SCALE):
            raise InvalidParameterError(
                f"cfg_scale must be in [{GEN_CONFIG.MIN_CFG_SCALE}, {GEN_CONFIG.MAX_CFG_SCALE}], got {cfg_scale}"
            )
        if not isinstance(rescale_cfg, (int, float)):
            raise InvalidParameterError(f"rescale_cfg must be numeric, got {type(rescale_cfg).__name__}")
        if not (GEN_CONFIG.MIN_RESCALE_CFG <= rescale_cfg <= GEN_CONFIG.MAX_RESCALE_CFG):
            raise InvalidParameterError(
                f"rescale_cfg must be in [{GEN_CONFIG.MIN_RESCALE_CFG}, {GEN_CONFIG.MAX_RESCALE_CFG}], got {rescale_cfg}"
            )

        # Validate toggle mode
        VALID_TOGGLE_MODES = {"standard", "smart", "optimized", "manual", None}
        if dora_toggle_mode not in VALID_TOGGLE_MODES:
            raise InvalidParameterError(
                f"Invalid dora_toggle_mode: '{dora_toggle_mode}'. "
                f"Must be one of {{'standard', 'smart', 'optimized', 'manual'}} or None"
            )

        # Validate dora_start_step doesn't exceed steps (when not using toggle mode)
        if dora_start_step is not None and dora_toggle_mode is None:
            if dora_start_step > steps:
                raise InvalidParameterError(
                    f"DoRA start step ({dora_start_step}) cannot exceed total steps ({steps})"
                )

        with perf_monitor.time_section("image_generation"):
            # Build generation info
            info_parts = self._build_generation_info(steps, cfg_scale, width, height)

            # Setup seed and generator
            if seed is None:
                seed = random.randint(0, 2**32 - 1)
            else:
                # Coerce seed to int if needed
                if not isinstance(seed, int):
                    try:
                        seed = int(seed)
                    except (TypeError, ValueError):
                        raise InvalidParameterError(
                            f"Seed must be integer, got {type(seed).__name__}"
                        )
                if seed < 0:
                    raise InvalidParameterError(
                        f"Seed must be non-negative, got {seed}"
                    )
                if seed >= 2**32:
                    raise InvalidParameterError(
                        f"Seed must be < 2^32 ({2**32}), got {seed}"
                    )

            # Synchronize RNG state across Python, CPU, and GPU so that all
            # stochastic operations (including any fallback paths) honor the
            # requested seed. The per-device generator remains on CPU for
            # deterministic noise sampling across platforms.
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            generator = torch.Generator(device="cpu").manual_seed(seed)

            # Parse manual DoRA schedule
            manual_schedule, _ = self._parse_manual_dora_schedule(
                dora_toggle_mode, dora_manual_schedule, steps
            )

            # Enforce toggle mode exclusivity (resets dora_start_step if needed)
            # Toggle mode is compatible with all configurations including CPU offloading
            self._enforce_toggle_mode_exclusivity(dora_toggle_mode)

            # Setup initial DoRA state
            self._setup_initial_dora_state(dora_toggle_mode, manual_schedule)

            # Create progress callback
            start_time = time.time()
            callback_on_step_end = self._create_progress_callback(
                steps, start_time, dora_toggle_mode, manual_schedule, progress_callback
            )

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

                # Add DoRA info to result
                self._add_dora_info_to_result(info_parts, dora_toggle_mode, seed)

                # Get generated image
                image = result.images[0]

                # Create and attach metadata
                metadata = self._create_image_metadata(
                    prompt, negative_prompt, seed, width, height,
                    steps, cfg_scale, rescale_cfg, dora_toggle_mode
                )
                image.info = metadata

                return image, seed, "\n".join(info_parts)

            finally:
                self.clear_memory()

    def teardown_engine(self) -> None:
        """Engine teardown with resource cleanup.
        
        Handles meta tensor scenarios that occur with CPU offloading (accelerate).
        Separates .to("cpu") from component deletion to ensure cleanup proceeds
        even when meta tensors prevent device transfer.
        """
        try:
            # 1. Unload any DoRA adapters completely
            if self.pipe and self.dora_loaded:
                try:
                    self.pipe.unload_lora_weights()
                    if hasattr(self.pipe, 'delete_adapters'):
                        self.pipe.delete_adapters(["noobai_dora"])
                except Exception as e:
                    logger.warning(f"Error unloading DoRA adapters: {e}")

            # 2. Clear pipeline components
            if self.pipe:
                # Step 2a: Try to move pipeline to CPU (optional, helps free GPU memory)
                # This may fail with meta tensors when CPU offloading is enabled
                try:
                    # Check if CPU offloading was enabled (use getattr for safety on partial init)
                    cpu_offload = getattr(self, '_cpu_offload_enabled', False)
                    
                    if not cpu_offload:
                        self.pipe = self.pipe.to("cpu")
                except RuntimeError as e:
                    # Handle meta tensor case gracefully - common with accelerate/CPU offloading
                    # Use robust keyword matching for various PyTorch/CUDA versions
                    error_str = str(e).lower()
                    meta_keywords = ["meta tensor", "cannot copy out of meta", "meta device", "not materialized"]
                    if any(keyword in error_str for keyword in meta_keywords):
                        logger.debug("Skipping .to(cpu) for meta tensor pipeline (expected with CPU offloading)")
                    else:
                        logger.warning(f"Error moving pipeline to CPU: {e}")
                except Exception as e:
                    logger.warning(f"Error moving pipeline to CPU: {e}")

                # Step 2b: For CPU-offloaded or accelerate-managed pipelines, release hooks
                try:
                    if hasattr(self.pipe, 'maybe_free_model_hooks'):
                        self.pipe.maybe_free_model_hooks()
                except Exception as e:
                    logger.debug(f"Could not free model hooks: {e}")

                # Step 2c: Always try to delete components regardless of .to() success
                try:
                    components_to_delete = ['unet', 'vae', 'text_encoder', 'text_encoder_2', 'scheduler']
                    for component_name in components_to_delete:
                        if hasattr(self.pipe, component_name):
                            component = getattr(self.pipe, component_name)
                            if component is not None:
                                del component
                                setattr(self.pipe, component_name, None)
                except Exception as e:
                    logger.warning(f"Error cleaning pipeline components: {e}")

                # 3. Delete entire pipeline
                try:
                    del self.pipe
                    self.pipe = None
                except Exception as e:
                    logger.warning(f"Error deleting pipeline: {e}")

            # 4. Clear device caches with proper synchronization
            try:
                if self._device == "mps":
                    try:
                        torch.mps.synchronize()
                    except (AttributeError, RuntimeError):
                        pass
                    torch.mps.empty_cache()
                elif self._device == "cuda":
                    try:
                        torch.cuda.synchronize()
                    except (AttributeError, RuntimeError):
                        pass
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, 'ipc_collect'):
                        torch.cuda.ipc_collect()
            except Exception as e:
                logger.warning(f"Error clearing device caches: {e}")

            # 5. Force garbage collection
            gc.collect()

            logger.info("Engine teardown completed")

        except Exception as e:
            logger.error(f"Error during engine teardown: {e}")
        finally:
            # Always reset state variables even if teardown partially fails
            try:
                self.pipe = None
            except Exception:
                self.__dict__['pipe'] = None

            try:
                self.dora_loaded = False
            except Exception:
                self.__dict__['dora_loaded'] = False

            try:
                self.dora_path = None
            except Exception:
                self.__dict__['dora_path'] = None

            try:
                self.is_initialized = False
            except Exception:
                self.__dict__['is_initialized'] = False

    def _synchronize_device(self) -> None:
        """Synchronize device operations."""
        try:
            if self._device == "mps":
                try:
                    torch.mps.synchronize()
                except (AttributeError, RuntimeError) as e:
                    logger.debug(f"MPS synchronize not available: {e}")
            elif self._device == "cuda":
                try:
                    torch.cuda.synchronize()
                except (AttributeError, RuntimeError) as e:
                    logger.debug(f"CUDA synchronize not available: {e}")
        except Exception as e:
            logger.debug(f"Device synchronization error (non-critical): {e}")

    def clear_memory(self) -> None:
        """Clear GPU/memory caches."""
        try:
            self._synchronize_device()
            if self._device == "mps":
                torch.mps.empty_cache()
            elif self._device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass
