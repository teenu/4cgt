"""NoobAI XL V-Pred 1.0 - Core Engine Class."""

import os
import time
import random
import torch
from PIL import Image, PngImagePlugin
from typing import Any, Optional, Tuple, Dict, Callable
from config import (
    logger, MODEL_CONFIG, DEFAULT_NEGATIVE_PROMPT, OPTIMAL_SETTINGS,
    OFFICIAL_RESOLUTIONS, RECOMMENDED_RESOLUTIONS,
    OPTIMAL_STEPS_RANGE, OPTIMAL_CFG_RANGE,
    EngineNotInitializedError, InvalidParameterError
)
from state import perf_monitor
from utils import parse_manual_dora_schedule
from engine.model_loader import detect_device, load_pipeline
from engine.dora_manager import DoRAManager
from engine.progress import ProgressManager
from engine.memory import clear_memory, teardown_pipeline

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    cublas_config = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if cublas_config not in [':4096:8', ':16:8']:
        logger.warning(
            f"CUBLAS_WORKSPACE_CONFIG is '{cublas_config}' "
            f"(expected ':4096:8' or ':16:8'). Determinism may be affected."
        )

if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

logger.info("Deterministic mode enabled for reproducible generation across platforms")


class NoobAIEngine:
    """Clean, modular NoobAI engine with optimal configuration."""

    def __init__(self, model_path: str, enable_dora: bool = False, dora_path: Optional[str] = None,
                 adapter_strength: float = MODEL_CONFIG.DEFAULT_ADAPTER_STRENGTH,
                 dora_start_step: int = MODEL_CONFIG.DEFAULT_DORA_START_STEP,
                 force_fp32: bool = False, optimize: bool = False):
        self.model_path = model_path
        self.enable_dora = enable_dora
        self.adapter_strength = adapter_strength
        self.dora_start_step = dora_start_step
        self.force_fp32 = force_fp32
        self.optimize = optimize
        self.pipe = None
        self.is_initialized = False
        self._device = None
        self._cpu_offload_enabled = False
        self._dora_manager = None
        self._progress_manager = None
        self._initialize(dora_path)

    def _initialize(self, dora_path: Optional[str]):
        """Initialize the diffusion pipeline."""
        try:
            with perf_monitor.time_section("engine_initialization"):
                logger.info(f"Initializing NoobAI engine with model: {self.model_path}")

                self._device = detect_device()
                logger.info(f"Using device: {self._device.upper()}")

                self.pipe, self._cpu_offload_enabled = load_pipeline(
                    self.model_path, self._device, self.force_fp32, self.optimize
                )
                self.is_initialized = True
                logger.info("Engine initialized")

                self._dora_manager = DoRAManager(self.pipe, self._device)
                self._progress_manager = ProgressManager(self.pipe, self._device, self._dora_manager)

                if self.enable_dora:
                    self._dora_manager.load_adapter(dora_path)
                    if self._dora_manager.dora_loaded:
                        self._dora_manager.set_strength(self.adapter_strength)

                # Apply torch.compile AFTER DoRA loading
                # Compiling before DoRA breaks adapter injection into compiled modules
                if self.optimize and self._device == "cuda":
                    logger.info("Compiling UNet with torch.compile (first inference will be slower)...")
                    self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead")
                    logger.info("UNet compiled successfully")

        except Exception as e:
            self.is_initialized = False
            if self.pipe is not None:
                try:
                    if self._device in ["cuda", "mps"]:
                        clear_memory(self._device)
                except Exception as cleanup_error:
                    logger.debug(f"Memory cleanup during init failure: {cleanup_error}")
                finally:
                    self.pipe = None
            logger.error(f"Failed to initialize engine: {e}")
            raise

    @property
    def dora_loaded(self):
        return self._dora_manager.dora_loaded if self._dora_manager else False

    @property
    def dora_path(self):
        return self._dora_manager.dora_path if self._dora_manager else None

    def unload_dora_adapter(self) -> None:
        if self._dora_manager:
            self._dora_manager.unload_adapter()

    def switch_dora_adapter(self, new_adapter_path: str) -> bool:
        if self._dora_manager:
            return self._dora_manager.switch_adapter(new_adapter_path)
        return False

    def set_adapter_strength(self, strength: float) -> float:
        self.adapter_strength = strength
        if self._dora_manager:
            return self._dora_manager.set_strength(strength)
        return strength

    def set_dora_start_step(self, start_step: int) -> int:
        try:
            if not isinstance(start_step, int):
                try:
                    start_step = int(start_step)
                except (TypeError, ValueError):
                    logger.warning(f"Invalid DoRA start step type: {type(start_step)}, using default")
                    start_step = MODEL_CONFIG.DEFAULT_DORA_START_STEP

            original_start_step = start_step
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
            return self.dora_start_step

    def set_dora_enabled(self, enabled: bool) -> None:
        self.enable_dora = enabled
        if self._dora_manager:
            self._dora_manager.set_enabled(enabled)

    def get_dora_info(self) -> Dict[str, Any]:
        return {
            'enabled': self.enable_dora,
            'loaded': self.dora_loaded,
            'path': self.dora_path,
            'strength': self.adapter_strength if self.dora_loaded else 0.0,
            'start_step': self.dora_start_step
        }

    def save_image_standardized(self, image: Image.Image, output_path: str, include_metadata: bool = True) -> str:
        """Save image with standardized settings for consistent hashing."""
        output_dir = os.path.dirname(output_path)
        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
            except (IOError, OSError) as e:
                logger.error(f"Failed to create output directory {output_dir}: {e}")
                raise

        pnginfo = None
        if include_metadata and hasattr(image, 'info') and image.info:
            pnginfo = PngImagePlugin.PngInfo()
            for key in sorted(image.info.keys()):
                try:
                    if not isinstance(key, str):
                        logger.debug(f"Skipping non-string metadata key: {type(key)}")
                        continue
                    if not key.isprintable():
                        logger.debug(f"Skipping non-printable metadata key: {repr(key)}")
                        continue
                    if len(key) > 79:
                        logger.debug(f"Truncating metadata key: {key[:76]}...")
                        key = key[:79]

                    value_str = str(image.info[key])
                    if len(value_str) > 2000:
                        logger.debug(f"Truncating metadata value for key '{key}'")
                        value_str = value_str[:1997] + "..."

                    pnginfo.add_text(key, value_str)
                except Exception as e:
                    logger.warning(f"Skipping metadata key '{key}': {e}")

        try:
            image.save(output_path, format='PNG', pnginfo=pnginfo, compress_level=6, optimize=False)

            if not os.path.exists(output_path):
                raise IOError(f"File was not created: {output_path}")

            file_size = os.path.getsize(output_path)
            if file_size == 0:
                raise IOError(f"File is empty: {output_path}")
            if file_size < 1000:
                raise IOError(f"File suspiciously small ({file_size} bytes): {output_path}")

            try:
                with Image.open(output_path) as test_img:
                    test_img.verify()
            except Exception as e:
                raise IOError(f"Saved PNG is corrupted: {e}")

            return output_path

        except Exception as e:
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
        if not self.is_initialized:
            raise EngineNotInitializedError("NoobAI engine is not initialized")

        if enable_dora is not None:
            self.set_dora_enabled(enable_dora)
        if adapter_strength is not None:
            self.set_adapter_strength(adapter_strength)
        if dora_start_step is not None:
            self.set_dora_start_step(dora_start_step)

        if not isinstance(steps, int):
            try:
                steps = int(steps)
            except (TypeError, ValueError):
                raise InvalidParameterError(f"Steps must be integer, got {type(steps).__name__}")
        if steps <= 0:
            raise InvalidParameterError(f"Steps must be positive, got {steps}")

        with perf_monitor.time_section("image_generation"):
            info_parts = self._build_generation_info(steps, cfg_scale, width, height)

            if seed is None:
                seed = random.randint(0, 2**32 - 1)
            else:
                if not isinstance(seed, int):
                    try:
                        seed = int(seed)
                    except (TypeError, ValueError):
                        raise InvalidParameterError(f"Seed must be integer, got {type(seed).__name__}")
                if seed < 0:
                    raise InvalidParameterError(f"Seed must be non-negative, got {seed}")
                if seed >= 2**32:
                    raise InvalidParameterError(f"Seed must be < 2^32 ({2**32}), got {seed}")

            generator = torch.Generator(device="cpu").manual_seed(seed)

            manual_schedule, _ = parse_manual_dora_schedule(dora_manual_schedule, steps) if dora_toggle_mode in ["manual", "optimized"] else (None, None)

            effective_dora_start_step = self.dora_start_step
            if dora_toggle_mode and self.enable_dora and self._dora_manager.dora_loaded:
                if effective_dora_start_step > 0:
                    logger.warning(f"Toggle mode '{dora_toggle_mode}' overrides dora_start_step={effective_dora_start_step}")
                    effective_dora_start_step = 0

            self._progress_manager.setup_initial_dora_state(dora_toggle_mode, effective_dora_start_step, manual_schedule, self.enable_dora)

            start_time = time.time()
            callback_on_step_end = self._progress_manager.create_callback(
                steps, start_time, dora_toggle_mode, effective_dora_start_step, manual_schedule, progress_callback, self.enable_dora
            )

            try:
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

                self._add_dora_info_to_result(info_parts, dora_toggle_mode, seed)
                image = result.images[0]
                metadata = self._create_image_metadata(prompt, negative_prompt, seed, width, height, steps, cfg_scale, rescale_cfg, dora_toggle_mode)
                image.info = metadata

                return image, seed, "\n".join(info_parts)

            finally:
                clear_memory(self._device)

    def _build_generation_info(self, steps: int, cfg_scale: float, width: int, height: int) -> list:
        """Build informational messages about generation parameters."""
        info_parts = []
        if not (OPTIMAL_STEPS_RANGE[0] <= steps <= OPTIMAL_STEPS_RANGE[1]):
            info_parts.append(f"⚠️ Steps {steps} outside optimal range {OPTIMAL_STEPS_RANGE[0]}-{OPTIMAL_STEPS_RANGE[1]}")
        if not (OPTIMAL_CFG_RANGE[0] <= cfg_scale <= OPTIMAL_CFG_RANGE[1]):
            info_parts.append(f"⚠️ CFG {cfg_scale} outside optimal range {OPTIMAL_CFG_RANGE[0]}-{OPTIMAL_CFG_RANGE[1]}")

        current_res = (height, width)
        if current_res in RECOMMENDED_RESOLUTIONS:
            info_parts.append(f"✅ Optimal resolution: {width}x{height}")
        elif current_res in OFFICIAL_RESOLUTIONS:
            info_parts.append(f"✅ Official resolution: {width}x{height}")
        else:
            info_parts.append(f"⚠️ Non-official resolution: {width}x{height}")

        return info_parts

    def _add_dora_info_to_result(self, info_parts: list, dora_toggle_mode: Optional[str], seed: int) -> None:
        info_parts.append(f"🌱 Generated with seed: {seed}")

        if self._dora_manager.dora_loaded:
            dora_name = os.path.basename(self._dora_manager.dora_path) if self._dora_manager.dora_path else "DoRA"
            if self.enable_dora:
                dora_info = f"🎯 DoRA: {dora_name} (strength: {self.adapter_strength}"
                if dora_toggle_mode == "manual":
                    dora_info += ", manual toggle schedule"
                elif dora_toggle_mode == "optimized":
                    dora_info += ", optimized toggle schedule"
                elif self.dora_start_step > 0:
                    dora_info += f", starts at step {self.dora_start_step}"
                dora_info += ")"
                info_parts.append(dora_info)
            else:
                info_parts.append(f"⚪ DoRA: {dora_name} (disabled)")
        elif self._dora_manager.dora_path:
            info_parts.append("⚠️ DoRA: Available but not loaded")

    def _create_image_metadata(self, prompt: str, negative_prompt: str, seed: int, width: int, height: int, steps: int, cfg_scale: float, rescale_cfg: float, dora_toggle_mode: Optional[str]) -> Dict[str, str]:
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

        if self._dora_manager.dora_loaded:
            metadata["dora_enabled"] = str(self.enable_dora).lower()
            metadata["dora_path"] = os.path.basename(self._dora_manager.dora_path) if self._dora_manager.dora_path else "unknown"
            if self.enable_dora:
                metadata["adapter_strength"] = str(self.adapter_strength)
                metadata["dora_start_step"] = str(self.dora_start_step)
                metadata["dora_toggle_mode"] = dora_toggle_mode if dora_toggle_mode else "none"
            else:
                metadata["adapter_strength"] = "0.0"
                metadata["dora_start_step"] = "1"
                metadata["dora_toggle_mode"] = "none"

        return metadata

    def teardown_engine(self) -> None:
        """Engine teardown with resource cleanup."""
        try:
            if self._dora_manager:
                self._dora_manager.unload_adapter()

            teardown_pipeline(self.pipe, self._device, self._cpu_offload_enabled, self._dora_manager.dora_loaded if self._dora_manager else False)

            try:
                del self.pipe
            except Exception as e:
                logger.warning(f"Error deleting pipeline: {e}")

        except Exception as e:
            logger.error(f"Error during engine teardown: {e}")
        finally:
            self.pipe = None
            self.is_initialized = False

    def clear_memory(self) -> None:
        """Clear GPU/memory caches."""
        clear_memory(self._device)
