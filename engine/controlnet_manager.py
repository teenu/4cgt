"""ControlNet model management for pose-controlled image generation."""

import os
import struct
import json
import torch
from typing import Optional
from PIL import Image
from diffusers import ControlNetModel
from config import logger, CONTROLNET_CONFIG, DTYPE_MAP
from utils.controlnet import (
    validate_controlnet_path,
    find_controlnet_path,
    preprocess_pose_image as utils_preprocess_pose_image,
    validate_pose_image
)
from engine.memory import clear_memory


def detect_controlnet_model_precision(model_path: str) -> torch.dtype:
    """Detect the actual precision of tensors in a ControlNet safetensors file.

    Returns:
        torch.dtype: The detected precision (float32, bfloat16, or float16)
    """
    try:
        with open(model_path, 'rb') as f:
            header_size = struct.unpack('<Q', f.read(8))[0]
            header_data = json.loads(f.read(header_size).decode('utf-8'))

        for key, value in header_data.items():
            if key != '__metadata__' and isinstance(value, dict) and 'dtype' in value:
                dtype_str = value['dtype']
                return DTYPE_MAP.get(dtype_str, torch.float32)

        return torch.float32

    except Exception as e:
        logger.warning(f"Could not detect ControlNet precision, assuming FP32: {e}")
        return torch.float32


class ControlNetManager:
    """Manages ControlNet model loading, unloading, and conditioning."""

    def __init__(self, device: str, force_fp32: bool = False):
        """Initialize ControlNet manager.

        Args:
            device: Target device (cuda/mps/cpu)
            force_fp32: If True, force FP32 loading even for BF16 models
        """
        self.device = device
        self.force_fp32 = force_fp32
        self.controlnet: Optional[ControlNetModel] = None
        self.controlnet_path: Optional[str] = None
        self.controlnet_loaded = False
        self.conditioning_scale = CONTROLNET_CONFIG.DEFAULT_CONDITIONING_SCALE
        self._last_error: Optional[str] = None

    def get_last_error(self) -> Optional[str]:
        """Get the last error message from failed operations.

        Returns:
            The last error message, or None if no error occurred.
        """
        return self._last_error

    def _check_bf16_support(self) -> bool:
        """Check if device supports bfloat16."""
        if self.device == "cuda":
            try:
                compute_capability = torch.cuda.get_device_capability(0)
                return compute_capability[0] >= 8
            except (AttributeError, RuntimeError):
                return False
        elif self.device == "mps":
            return True
        return False

    def load_controlnet(self, controlnet_path: Optional[str] = None) -> bool:
        """Load ControlNet model.

        Args:
            controlnet_path: Path to ControlNet safetensors file.
                            If None, attempts to auto-discover.

        Returns:
            True if loaded successfully, False otherwise.
        """
        try:
            if not controlnet_path:
                controlnet_path = find_controlnet_path("openpose")
                if not controlnet_path:
                    logger.warning("ControlNet enabled but no valid ControlNet file found")
                    return False

            is_valid, validated_path = validate_controlnet_path(controlnet_path)
            if not is_valid:
                logger.warning(f"ControlNet validation failed: {validated_path}")
                return False

            # Detect model precision
            model_precision = detect_controlnet_model_precision(validated_path)
            bf16_supported = self._check_bf16_support()

            # Determine loading precision
            if self.force_fp32 or model_precision == torch.float32:
                load_dtype = torch.float32
                logger.info("Loading ControlNet as FP32")
            elif model_precision == torch.bfloat16 and bf16_supported:
                load_dtype = torch.bfloat16
                logger.info("Loading ControlNet as BF16 (native support)")
            elif model_precision == torch.bfloat16 and not bf16_supported:
                load_dtype = torch.float32
                logger.info("Loading ControlNet as FP32 (BF16 not supported on this device)")
            else:
                load_dtype = torch.float32
                logger.info("Loading ControlNet as FP32 (default)")

            # Load ControlNet model with explicit SDXL config
            # All SDXL ControlNets (Canny, OpenPose, Depth, etc.) share the same architecture.
            # Using explicit config avoids auto-detection issues on some systems where
            # diffusers incorrectly detects SDXL models as SD1.5.
            self.controlnet = ControlNetModel.from_single_file(
                validated_path,
                torch_dtype=load_dtype,
                config="diffusers/controlnet-canny-sdxl-1.0"
            )
            self.controlnet = self.controlnet.to(self.device)

            self.controlnet_path = validated_path
            self.controlnet_loaded = True
            self._last_error = None  # Clear any previous error on success
            logger.info(f"ControlNet loaded: {os.path.basename(validated_path)}")
            return True

        except (IOError, OSError) as e:
            self._last_error = f"File error: {e}"
            logger.error(f"Failed to load ControlNet (file error): {e}")
            self.controlnet_loaded = False
            self.controlnet = None
            return False
        except (RuntimeError, ValueError) as e:
            self._last_error = f"Runtime/validation error: {e}"
            logger.error(f"Failed to load ControlNet (runtime/validation error): {e}")
            clear_memory(self.device)
            self.controlnet_loaded = False
            self.controlnet = None
            return False
        except Exception as e:
            self._last_error = f"Unexpected error: {e}"
            logger.error(f"Unexpected error loading ControlNet: {e}")
            clear_memory(self.device)
            self.controlnet_loaded = False
            self.controlnet = None
            return False
        finally:
            if not self.controlnet_loaded:
                self.controlnet_path = None

    def unload_controlnet(self) -> None:
        """Completely unload ControlNet model with full memory cleanup."""
        try:
            if self.controlnet is not None:
                del self.controlnet
                self.controlnet = None
                clear_memory(self.device)
                logger.info("ControlNet unloaded")

            self.controlnet_loaded = False
            self.controlnet_path = None

        except Exception as e:
            logger.error(f"Error unloading ControlNet: {e}")
            self.controlnet_loaded = False
            self.controlnet_path = None
            self.controlnet = None

    def switch_controlnet(self, new_controlnet_path: str) -> bool:
        """Switch ControlNet models with complete cleanup and fresh loading.

        Args:
            new_controlnet_path: Path to new ControlNet model

        Returns:
            True if switched successfully, False otherwise.
        """
        try:
            if not new_controlnet_path:
                logger.error("Cannot switch to empty ControlNet path")
                return False

            is_valid, validated_path = validate_controlnet_path(new_controlnet_path)
            if not is_valid:
                logger.error(f"Invalid new ControlNet path: {validated_path}")
                return False

            logger.info(f"Switching ControlNet to: {validated_path}")

            old_path = self.controlnet_path
            if self.controlnet_loaded:
                self.unload_controlnet()

            self.controlnet_path = validated_path
            success = self.load_controlnet(validated_path)

            if success:
                logger.info(f"Successfully switched to ControlNet: {os.path.basename(validated_path)}")
                return True
            else:
                self.controlnet_path = old_path
                logger.error("Failed to load new ControlNet")
                return False

        except Exception as e:
            logger.error(f"Error switching ControlNet: {e}")
            self.controlnet_loaded = False
            self.controlnet_path = None
            return False

    def set_conditioning_scale(self, scale: float) -> float:
        """Set ControlNet conditioning scale with clamping.

        Args:
            scale: Desired conditioning scale (0.0-2.0)

        Returns:
            The actual scale value after clamping.
        """
        original_scale = scale
        min_scale = CONTROLNET_CONFIG.MIN_CONDITIONING_SCALE
        max_scale = CONTROLNET_CONFIG.MAX_CONDITIONING_SCALE

        if not (min_scale <= scale <= max_scale):
            scale = max(min_scale, min(scale, max_scale))
            logger.warning(
                f"Conditioning scale {original_scale} out of bounds "
                f"[{min_scale}-{max_scale}], clamped to {scale}"
            )

        self.conditioning_scale = scale
        return scale

    def preprocess_pose_image(
        self,
        pose_image: Image.Image,
        target_width: int,
        target_height: int
    ) -> Optional[Image.Image]:
        """Preprocess a pose image for ControlNet conditioning.

        Args:
            pose_image: Input pose image (OpenPose skeleton)
            target_width: Target width for generation
            target_height: Target height for generation

        Returns:
            Preprocessed image in RGB format, or None if validation fails.
        """
        if pose_image is None:
            return None

        # Validate the pose image
        is_valid, message = validate_pose_image(pose_image)
        if not is_valid:
            logger.warning(f"Pose image validation failed: {message}")
            return None

        # Preprocess using utility function
        try:
            preprocessed = utils_preprocess_pose_image(
                pose_image, target_width, target_height
            )
            logger.info(f"Pose image preprocessed: {target_width}x{target_height}")
            return preprocessed
        except Exception as e:
            logger.error(f"Error preprocessing pose image: {e}")
            return None

    def get_info(self) -> dict:
        """Get ControlNet state information.

        Returns:
            Dictionary with current ControlNet state.
        """
        return {
            'loaded': self.controlnet_loaded,
            'path': self.controlnet_path,
            'conditioning_scale': self.conditioning_scale if self.controlnet_loaded else 0.0,
            'model_name': os.path.basename(self.controlnet_path) if self.controlnet_path else None
        }
