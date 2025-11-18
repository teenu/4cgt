"""
NoobAI XL V-Pred 1.0 - Engine

This module contains the NoobAIEngine class which handles model loading,
image generation, and DoRA adapter management.
"""

import os
import time
import random
import gc
import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from PIL import Image
from PIL import PngImagePlugin
from typing import Optional, Tuple, Dict, Any, Callable, List
from config import (
    logger, MODEL_CONFIG, DEFAULT_NEGATIVE_PROMPT, OPTIMAL_SETTINGS,
    OFFICIAL_RESOLUTIONS, RECOMMENDED_RESOLUTIONS,
    EngineNotInitializedError, GenerationInterruptedError
)
from state import perf_monitor, state_manager, GenerationState
from utils import (
    detect_base_model_precision, detect_adapter_precision,
    validate_dora_path, find_dora_path, format_file_size,
    parse_manual_dora_schedule
)

# ============================================================================
# DETERMINISM ENFORCEMENT
# ============================================================================
# Enable deterministic algorithms for reproducible outputs across platforms
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Platform-specific determinism settings
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)  # Will be overridden by generator
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    # MPS determinism (PyTorch 2.0+)
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

                # Detect base model precision
                base_precision = detect_base_model_precision(self.model_path)

                # CRITICAL: For lossless quality and cross-platform parity, we enforce:
                # 1. BF16 model (NoobAI-XL-Vpred-v1.0.safetensors) is canonical/highest quality
                # 2. Platforms without BF16 → upcast to FP32 (lossless)
                # 3. Never downcast to FP16 (lossy compared to BF16)
                # 4. All platforms use identical precision pipeline

                # Check platform BF16 support
                bf16_supported = False
                if self._device == "cuda":
                    try:
                        compute_capability = torch.cuda.get_device_capability(0)
                        bf16_supported = compute_capability[0] >= 8
                        logger.debug(f"GPU compute capability: {compute_capability}, BF16 support: {bf16_supported}")
                    except (AttributeError, RuntimeError) as e:
                        logger.debug(f"Could not check CUDA compute capability: {e}")
                elif self._device == "mps":
                    # Apple Silicon supports BF16 natively via AMX
                    bf16_supported = True
                    logger.debug("Apple Silicon MPS: BF16 natively supported")
                elif self._device == "cpu":
                    # CPU supports BF16 but slow; use FP32
                    bf16_supported = False

                # Determine inference precision based on model and platform
                if base_precision == torch.bfloat16:
                    if bf16_supported:
                        # Use BF16 natively for optimal quality and performance
                        inference_dtype = torch.bfloat16
                        logger.info("Using native BF16 precision (optimal quality)")
                    else:
                        # Upcast BF16 → FP32 (lossless) for platforms without BF16 support
                        inference_dtype = torch.float32
                        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
                        logger.warning(
                            f"{gpu_name} does not support BF16 tensor cores. "
                            f"Upcasting to FP32 (lossless) for quality preservation. "
                            f"Note: Slower than BF16 but maintains full precision."
                        )
                elif base_precision == torch.float16:
                    # FP16 model detected - warn user about quality loss
                    logger.warning(
                        "FP16 model detected (NoobAI-XL-Vpred-v1.0-fp16*.safetensors). "
                        "For highest quality and cross-platform parity, use BF16 model "
                        "(NoobAI-XL-Vpred-v1.0.safetensors) which will be upcast to FP32 on "
                        "platforms without BF16 support."
                    )
                    # Upcast FP16 → FP32 for consistency
                    inference_dtype = torch.float32
                else:
                    # FP32 or other precision
                    inference_dtype = base_precision

                logger.info(f"Inference precision: {inference_dtype} (base model: {base_precision})")

                # Load pipeline with lossless VAE precision
                # VAE runs in FP32 for highest quality decode (prevents color banding/artifacts)
                logger.info("Loading VAE in FP32 for lossless image decode")

                from diffusers import AutoencoderKL

                # Load VAE separately in FP32
                vae = AutoencoderKL.from_single_file(
                    self.model_path,
                    torch_dtype=torch.float32,
                    use_safetensors=True,
                )

                # Load pipeline with inference precision, but override VAE
                self.pipe = StableDiffusionXLPipeline.from_single_file(
                    self.model_path,
                    torch_dtype=inference_dtype,
                    vae=vae,
                    use_safetensors=True,
                )

                logger.info(f"Pipeline loaded: UNet/TextEncoders={inference_dtype}, VAE=FP32")

                # Configure scheduler
                self.pipe.scheduler = EulerDiscreteScheduler.from_config(
                    self.pipe.scheduler.config,
                    prediction_type="v_prediction",
                    rescale_betas_zero_snr=True,
                    timestep_spacing="trailing"
                )

                # Move to device and enable optimizations
                # Use CPU offloading for GPUs with limited VRAM to reduce memory pressure
                use_cpu_offload = False
                if self._device == "cuda":
                    try:
                        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        gpu_name = torch.cuda.get_device_name(0)

                        # Enable CPU offloading for GPUs <8GB to reduce VRAM usage and avoid swapping
                        if vram_gb < 8.0:
                            use_cpu_offload = True
                            logger.info(
                                f"GPU ({gpu_name}) has {vram_gb:.1f}GB VRAM. "
                                f"Enabling sequential CPU offloading for optimal performance."
                            )
                            self.pipe.enable_sequential_cpu_offload()
                        else:
                            logger.info(f"GPU ({gpu_name}) has {vram_gb:.1f}GB VRAM. Using full GPU loading.")
                            self.pipe = self.pipe.to(self._device)
                    except Exception as e:
                        logger.debug(f"Could not detect VRAM capacity: {e}. Using full GPU loading.")
                        self.pipe = self.pipe.to(self._device)
                else:
                    self.pipe = self.pipe.to(self._device)

                # Enable memory optimizations uniformly across platforms for deterministic behavior
                if self._device != "cpu":
                    self.pipe.enable_vae_slicing()
                    # Note: Attention slicing disabled for cross-platform determinism
                    # Different platforms handle slicing differently, causing output divergence
                    # For maximum quality and consistency, we keep full attention computation

                # Validate precision consistency
                pipeline_dtype = next(self.pipe.unet.parameters()).dtype
                logger.info(f"Pipeline initialized with {pipeline_dtype} precision")

                self.is_initialized = True
                logger.info("NoobAI engine initialized successfully")

                # Load DoRA adapter if enabled
                if self.enable_dora:
                    self._load_dora_adapter()

        except (IOError, OSError, RuntimeError, ValueError) as e:
            self.is_initialized = False
            # Clean up partially initialized GPU resources
            if self.pipe is not None:
                try:
                    if self._device in ["cuda", "mps"]:
                        self.clear_memory()
                except Exception:
                    pass  # Best effort cleanup, don't mask original error
                finally:
                    self.pipe = None
            logger.error(f"Failed to initialize engine: {e}")
            raise
        except Exception as e:
            self.is_initialized = False
            # Clean up partially initialized GPU resources
            if self.pipe is not None:
                try:
                    if self._device in ["cuda", "mps"]:
                        self.clear_memory()
                except Exception:
                    pass  # Best effort cleanup, don't mask original error
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

            # Log precision information
            adapter_precision = detect_adapter_precision(validated_path)
            pipeline_dtype = next(self.pipe.unet.parameters()).dtype

            logger.info(f"Loading DoRA adapter: {validated_path}")
            logger.info(f"Adapter stored as: {adapter_precision}, Pipeline using: {pipeline_dtype}")

            if adapter_precision == "fp16" and pipeline_dtype == torch.bfloat16:
                logger.info("DoRA adapter will be automatically converted from FP16 to BF16")
            elif adapter_precision == "fp16" and pipeline_dtype == torch.float32:
                logger.info("DoRA adapter will be automatically converted from FP16 to FP32")

            # Load DoRA adapter using the LoRA loading mechanism
            # The diffusers library will handle precision conversion automatically
            self.pipe.load_lora_weights(
                os.path.dirname(validated_path),
                weight_name=os.path.basename(validated_path),
                adapter_name="noobai_dora"
            )

            # Set adapter scale
            self.pipe.set_adapters(["noobai_dora"], adapter_weights=[self.adapter_strength])

            # Set path only after successful load to maintain state consistency
            self.dora_path = validated_path
            self.dora_loaded = True
            logger.info(f"DoRA adapter loaded successfully with {pipeline_dtype} precision")

        except (IOError, OSError) as e:
            logger.error(f"Failed to load DoRA adapter (file error): {e}")
            self.dora_loaded = False
        except (RuntimeError, ValueError) as e:
            logger.error(f"Failed to load DoRA adapter (runtime/validation error): {e}")
            self.dora_loaded = False
        except Exception as e:
            logger.error(f"Unexpected error loading DoRA adapter: {e}")
            self.dora_loaded = False
        finally:
            # Ensure path is reset on failure (all error paths set dora_loaded=False)
            if not self.dora_loaded:
                self.dora_path = None

    def unload_dora_adapter(self) -> None:
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
                # GPU synchronization now handled by clear_memory()

            # 2. Update adapter path
            self.dora_path = validated_path

            # 3. Load new adapter with fresh state
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
            logger.info(f"DoRA start step set to {start_step}")

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

        # parse_manual_dora_schedule imported at module level
        manual_schedule, manual_schedule_warning = parse_manual_dora_schedule(dora_manual_schedule, steps)

        if manual_schedule_warning:
            logger.warning(manual_schedule_warning)

        logger.info(f"Manual DoRA schedule: {manual_schedule}")
        logger.info("Manual mode active - dora_start_step setting is ignored")

        return manual_schedule, manual_schedule_warning

    def _enforce_toggle_mode_exclusivity(self, dora_toggle_mode: Optional[str]) -> None:
        """
        Enforce mutual exclusivity between toggle mode and start_step.
        Toggle modes always start from step 1.

        Args:
            dora_toggle_mode: Toggle mode ('manual', 'standard', 'smart', or None)
        """
        if dora_toggle_mode and self.enable_dora and self.dora_loaded:
            if self.dora_start_step > 1:
                logger.warning(
                    f"Toggle mode '{dora_toggle_mode}' enabled with dora_start_step={self.dora_start_step}. "
                    f"Resetting start_step to 1."
                )
                self.dora_start_step = 1

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
                if manual_schedule and manual_schedule[0] == 1:
                    self.pipe.set_adapters(["noobai_dora"], adapter_weights=[self.adapter_strength])
                else:
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
            dora_toggle_mode: Toggle mode ('manual', 'standard', 'smart', or None)
            manual_schedule: Parsed manual schedule (if manual mode)
            progress_callback: User-provided progress callback

        Returns:
            Callback function for step_end events
        """
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

            # Build progress description
            desc = self._build_progress_description(
                step_index, current_step, steps, eta,
                dora_toggle_mode, manual_schedule
            )

            # Call user callback with error isolation
            if progress_callback:
                try:
                    progress_callback(progress, desc)
                except Exception as e:
                    # Log callback error but don't let it crash generation
                    logger.warning(f"Progress callback error at step {current_step}: {e}")
                    # Continue generation despite callback failure

            return callback_kwargs

        return callback_on_step_end

    def _build_progress_description(
        self,
        step_index: int,
        current_step: int,
        steps: int,
        eta: float,
        dora_toggle_mode: Optional[str],
        manual_schedule: Optional[List[int]]
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
            # Synchronize device before changing adapter weights to prevent race conditions
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
            # Synchronize device before changing adapter weights to prevent race conditions
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

        current_state = "ON" if manual_schedule[step_index] == 1 else "OFF"

        if next_step_index < steps:
            # Bounds check to prevent IndexError if schedule is malformed
            if next_step_index >= len(manual_schedule):
                logger.warning(f"Manual schedule too short: index {next_step_index} >= length {len(manual_schedule)}, treating as OFF")
                next_state = "OFF"
                # Synchronize before changing weights
                self._synchronize_device()
                self.pipe.set_adapters(["noobai_dora"], adapter_weights=[0.0])
            else:
                next_state = "ON" if manual_schedule[next_step_index] == 1 else "OFF"
                # Synchronize device before changing adapter weights
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

        # Prepare PNG metadata
        pnginfo = None
        if include_metadata and hasattr(image, 'info') and image.info:
            pnginfo = PngImagePlugin.PngInfo()

            # Add metadata in a consistent order (sorted keys)
            for key in sorted(image.info.keys()):
                pnginfo.add_text(key, str(image.info[key]))

        # Save with consistent parameters
        # Note: save() does not mutate the image object
        image.save(
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

        with perf_monitor.time_section("image_generation"):
            # Build generation info
            info_parts = self._build_generation_info(steps, cfg_scale, width, height)

            # Setup seed and generator
            if seed is None:
                seed = random.randint(0, 2**32 - 1)

            # Use CPU generator for maximum cross-platform determinism
            # Device-specific generators (CUDA/MPS) have different RNG implementations
            # CPU generator ensures identical noise patterns across all platforms
            generator = torch.Generator(device="cpu").manual_seed(seed)
            logger.debug(f"Using CPU generator (seed={seed}) for cross-platform reproducibility")

            # Parse manual DoRA schedule
            manual_schedule, _ = self._parse_manual_dora_schedule(
                dora_toggle_mode, dora_manual_schedule, steps
            )

            # Enforce toggle mode exclusivity
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

            # 4. Clear device caches with proper synchronization
            try:
                if self._device == "mps":
                    try:
                        torch.mps.synchronize()  # Wait for MPS operations to complete FIRST
                    except (AttributeError, RuntimeError):
                        pass
                    torch.mps.empty_cache()
                elif self._device == "cuda":
                    try:
                        torch.cuda.synchronize()  # Wait for CUDA operations to complete FIRST
                    except (AttributeError, RuntimeError):
                        pass
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, 'ipc_collect'):
                        torch.cuda.ipc_collect()  # Clear inter-process memory
                logger.info(f"Device caches cleared for {self._device}")
            except Exception as e:
                logger.warning(f"Error clearing device caches: {e}")

            # 5. Force garbage collection to free released resources
            collected = gc.collect()
            logger.info(f"Garbage collection freed {collected} objects")

            logger.info("Engine teardown completed successfully")

        except Exception as e:
            logger.error(f"Error during comprehensive engine teardown: {e}")
        finally:
            # CRITICAL: Always reset state variables even if teardown partially fails
            # This ensures the engine doesn't remain in an inconsistent state
            # Each attribute is reset individually with fallback to __dict__
            reset_success = []
            reset_failures = []

            # Reset pipe
            try:
                self.pipe = None
                reset_success.append('pipe')
            except Exception as e:
                try:
                    self.__dict__['pipe'] = None
                    reset_success.append('pipe (via __dict__)')
                except Exception as e2:
                    reset_failures.append(f'pipe: {e2}')

            # Reset dora_loaded
            try:
                self.dora_loaded = False
                reset_success.append('dora_loaded')
            except Exception as e:
                try:
                    self.__dict__['dora_loaded'] = False
                    reset_success.append('dora_loaded (via __dict__)')
                except Exception as e2:
                    reset_failures.append(f'dora_loaded: {e2}')

            # Reset dora_path
            try:
                self.dora_path = None
                reset_success.append('dora_path')
            except Exception as e:
                try:
                    self.__dict__['dora_path'] = None
                    reset_success.append('dora_path (via __dict__)')
                except Exception as e2:
                    reset_failures.append(f'dora_path: {e2}')

            # Reset is_initialized
            try:
                self.is_initialized = False
                reset_success.append('is_initialized')
            except Exception as e:
                try:
                    self.__dict__['is_initialized'] = False
                    reset_success.append('is_initialized (via __dict__)')
                except Exception as e2:
                    reset_failures.append(f'is_initialized: {e2}')

            # Log results
            if reset_success:
                logger.info(f"Engine state variables reset: {', '.join(reset_success)}")
            if reset_failures:
                logger.error(f"Failed to reset some state variables: {', '.join(reset_failures)}")

    def _synchronize_device(self) -> None:
        """Synchronize device operations to ensure all pending GPU operations complete.

        Critical for DoRA weight changes to prevent race conditions where adapter
        weights are modified before previous operations finish.
        """
        try:
            if self._device == "mps":
                try:
                    torch.mps.synchronize()
                except (AttributeError, RuntimeError):
                    pass  # Older PyTorch versions or MPS not available
            elif self._device == "cuda":
                try:
                    torch.cuda.synchronize()
                except (AttributeError, RuntimeError):
                    pass  # CUDA not available
        except Exception as e:
            logger.debug(f"Could not synchronize device: {e}")

    def clear_memory(self) -> None:
        """Clear GPU/memory caches with proper synchronization."""
        try:
            # Synchronize GPU operations before clearing cache to ensure
            # all pending operations complete (replaces unreliable time.sleep)
            self._synchronize_device()
            if self._device == "mps":
                torch.mps.empty_cache()
            elif self._device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            logger.warning(f"Could not clear memory cache: {e}")
