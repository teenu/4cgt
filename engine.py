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
from typing import Optional, Tuple, Dict, Any, Callable
from config import (
    logger, MODEL_CONFIG, DEFAULT_NEGATIVE_PROMPT, OPTIMAL_SETTINGS,
    OFFICIAL_RESOLUTIONS, RECOMMENDED_RESOLUTIONS,
    EngineNotInitializedError, GenerationInterruptedError
)
from state import perf_monitor, state_manager, GenerationState
from utils import (
    detect_base_model_precision, detect_adapter_precision,
    validate_dora_path, find_dora_path, format_file_size
)

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
        dora_toggle_mode: Optional[str] = None,
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

            # Validate toggle mode and start step don't conflict
            if dora_toggle_mode and self.enable_dora and self.dora_loaded and self.dora_start_step > 1:
                logger.warning(f"Toggle mode '{dora_toggle_mode}' enabled with dora_start_step={self.dora_start_step}. Toggle mode will override start_step setting.")

            # Pre-deactivate DoRA if start step is later than step 1
            # This ensures DoRA is inactive from the beginning when delayed activation is requested
            if self.enable_dora and self.dora_loaded and self.dora_start_step > 1 and not dora_toggle_mode:
                self.pipe.set_adapters(["noobai_dora"], adapter_weights=[0.0])

            # For toggle mode, set initial DoRA state to ON (both standard and smart start ON)
            if dora_toggle_mode and self.enable_dora and self.dora_loaded:
                # Both modes: Index 0 = ON
                self.pipe.set_adapters(["noobai_dora"], adapter_weights=[self.adapter_strength])

            # Generation callback
            start_time = time.time()
            def callback_on_step_end(pipe, step_index, timestep, callback_kwargs):
                if state_manager.is_interrupted():
                    raise GenerationInterruptedError()

                current_step = step_index + 1
                progress = current_step / steps
                elapsed = time.time() - start_time
                eta = (elapsed / current_step) * (steps - current_step) if current_step > 0 else 0

                # Handle DoRA toggle mode - alternate ON/OFF for improved anatomical accuracy
                if dora_toggle_mode and self.enable_dora and self.dora_loaded and self.pipe is not None:
                    # Callback runs AFTER step_index N completes, sets state for step_index N+1
                    # step_index is 0-indexed: 0, 1, 2, ..., 29 (for 30 steps)
                    next_step_index = step_index + 1

                    if dora_toggle_mode == "standard":
                        # Standard: ON,OFF,ON,OFF throughout all steps
                        # Even indices = ON, Odd indices = OFF
                        if next_step_index < steps:
                            if next_step_index % 2 == 0:  # Next is even - turn ON
                                self.pipe.set_adapters(["noobai_dora"], adapter_weights=[self.adapter_strength])
                                current_state = "ON" if step_index % 2 == 0 else "OFF"
                                desc = f"Step {current_step}/{steps} (DoRA: {current_state}, next[{next_step_index}]: ON, ETA: {eta:.1f}s)"
                            else:  # Next is odd - turn OFF
                                self.pipe.set_adapters(["noobai_dora"], adapter_weights=[0.0])
                                current_state = "ON" if step_index % 2 == 0 else "OFF"
                                desc = f"Step {current_step}/{steps} (DoRA: {current_state}, next[{next_step_index}]: OFF, ETA: {eta:.1f}s)"
                        else:
                            current_state = "ON" if step_index % 2 == 0 else "OFF"
                            desc = f"Step {current_step}/{steps} (DoRA: {current_state}, final, ETA: {eta:.1f}s)"

                    elif dora_toggle_mode == "smart":
                        # Smart: ON,OFF,ON,OFF through index 19, then ON from index 20 onwards
                        if next_step_index < steps:
                            if next_step_index <= 19:
                                # Alternating phase (indices 0-19)
                                if next_step_index % 2 == 0:  # Next is even - turn ON
                                    self.pipe.set_adapters(["noobai_dora"], adapter_weights=[self.adapter_strength])
                                    current_state = "ON" if step_index % 2 == 0 else "OFF"
                                    desc = f"Step {current_step}/{steps} (DoRA: {current_state}, next[{next_step_index}]: ON, ETA: {eta:.1f}s)"
                                else:  # Next is odd - turn OFF
                                    self.pipe.set_adapters(["noobai_dora"], adapter_weights=[0.0])
                                    current_state = "ON" if step_index % 2 == 0 else "OFF"
                                    desc = f"Step {current_step}/{steps} (DoRA: {current_state}, next[{next_step_index}]: OFF, ETA: {eta:.1f}s)"
                            else:
                                # Always ON phase (index 20+)
                                self.pipe.set_adapters(["noobai_dora"], adapter_weights=[self.adapter_strength])
                                current_state = "ON"
                                desc = f"Step {current_step}/{steps} (DoRA: {current_state} [smart-locked], next[{next_step_index}]: ON, ETA: {eta:.1f}s)"
                        else:
                            # Final step
                            if step_index <= 19:
                                current_state = "ON" if step_index % 2 == 0 else "OFF"
                            else:
                                current_state = "ON"  # Always ON from index 20+
                            desc = f"Step {current_step}/{steps} (DoRA: {current_state}, final, ETA: {eta:.1f}s)"
                # Handle DoRA start step control (normal mode)
                elif self.enable_dora and self.dora_loaded and self.pipe is not None:
                    if current_step == self.dora_start_step - 1 and self.dora_start_step > 1:
                        # Activate DoRA adapter before the target step
                        self.pipe.set_adapters(["noobai_dora"], adapter_weights=[self.adapter_strength])
                        desc = f"Step {current_step}/{steps} (DoRA will activate at step {self.dora_start_step}, ETA: {eta:.1f}s)"
                    elif current_step >= self.dora_start_step:
                        desc = f"Step {current_step}/{steps} (DoRA active, ETA: {eta:.1f}s)"
                    else:
                        desc = f"Step {current_step}/{steps} (DoRA starts at step {self.dora_start_step}, ETA: {eta:.1f}s)"
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
                        metadata["dora_start_step"] = str(self.dora_start_step)
                        metadata["dora_toggle_mode"] = dora_toggle_mode if dora_toggle_mode else "none"
                    else:
                        metadata["adapter_strength"] = "0.0"
                        metadata["dora_start_step"] = "1"
                        metadata["dora_toggle_mode"] = "none"

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
