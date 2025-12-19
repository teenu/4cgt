"""Model loading and initialization."""

import os
import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler, AutoencoderKL
from config import logger
from utils import detect_base_model_precision


def detect_device() -> str:
    """Detect available device for inference."""
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if mps_available:
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def check_bf16_support(device: str) -> bool:
    """Check if device supports bfloat16."""
    if device == "cuda":
        try:
            compute_capability = torch.cuda.get_device_capability(0)
            return compute_capability[0] >= 8
        except (AttributeError, RuntimeError):
            return False
    elif device == "mps":
        return True
    elif device == "cpu":
        return False
    return False


def load_pipeline(model_path: str, device: str, force_fp32: bool = False) -> tuple:
    """Load diffusion pipeline with proper precision handling.

    Args:
        model_path: Path to model file or directory
        device: Target device (cuda/mps/cpu)
        force_fp32: If True, force FP32 inference even for BF16 models (parity mode)

    Returns:
        tuple: (pipeline, cpu_offload_enabled)
    """
    base_precision = detect_base_model_precision(model_path)
    is_directory = os.path.isdir(model_path)

    if base_precision not in [torch.bfloat16, torch.float32]:
        raise ValueError(
            f"Model precision validation failed: got {base_precision}. "
            f"Only BF16 model or FP32 pre-converted models are supported."
        )

    if base_precision == torch.float32 and is_directory:
        vae_path = os.path.join(model_path, "vae")
        if os.path.isdir(vae_path):
            vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float32)
            logger.info("VAE loaded as FP32 from directory for lossless decode")
        else:
            raise ValueError(
                f"VAE subdirectory not found at {vae_path}. "
                f"FP32 model directories must contain a 'vae' subdirectory."
            )

        pipe = StableDiffusionXLPipeline.from_pretrained(model_path, vae=vae)

        actual_unet_dtype = next(pipe.unet.parameters()).dtype
        actual_te_dtype = next(pipe.text_encoder.parameters()).dtype
        actual_vae_dtype = next(pipe.vae.parameters()).dtype

        if actual_unet_dtype != torch.float32:
            raise ValueError(
                f"Expected FP32 model but UNet loaded with {actual_unet_dtype}. "
                f"Directory may contain wrong precision weights."
            )
        if actual_te_dtype != torch.float32:
            raise ValueError(
                f"Expected FP32 model but TextEncoder loaded with {actual_te_dtype}. "
                f"Directory may contain wrong precision weights."
            )
        if actual_vae_dtype != torch.float32:
            raise ValueError(
                f"Expected FP32 VAE but loaded with {actual_vae_dtype}. "
                f"This could cause decode quality issues."
            )

        logger.info("FP32 directory model loaded with all components validated as FP32")
    else:
        # Single file loading (safetensors)
        if force_fp32:
            # Parity mode: Load all components in FP32 for bitwise-identical outputs
            load_dtype = torch.float32
            logger.info("Parity mode: Loading BF16 model with FP32 precision")
        else:
            bf16_supported = check_bf16_support(device)
            load_dtype = torch.bfloat16 if bf16_supported else torch.float32

        pipe = StableDiffusionXLPipeline.from_single_file(
            model_path,
            torch_dtype=load_dtype,
            use_safetensors=True,
        )

        if not force_fp32:
            # Only need explicit VAE upcast in non-parity mode (BF16 inference)
            try:
                pipe.vae.to(dtype=torch.float32)
                logger.info("Single file model loaded; VAE upcast to FP32 for lossless decode")
            except Exception as vae_error:
                try:
                    del pipe
                except Exception:
                    pass
                if device == "cuda":
                    torch.cuda.empty_cache()
                elif device == "mps" and hasattr(torch, 'mps'):
                    torch.mps.empty_cache()
                raise ValueError(f"Failed to upcast VAE to FP32: {vae_error}")
        else:
            logger.info("Parity mode: All components loaded as FP32 (VAE already FP32)")

    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        prediction_type="v_prediction",
        rescale_betas_zero_snr=True,
        timestep_spacing="trailing"
    )

    cpu_offload_enabled = False
    if device == "cuda":
        try:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb < 8.0:
                pipe.enable_sequential_cpu_offload()
                cpu_offload_enabled = True
                logger.info(f"CPU offloading enabled ({vram_gb:.1f}GB VRAM)")
            else:
                pipe = pipe.to(device)
        except Exception:
            pipe = pipe.to(device)
    else:
        pipe = pipe.to(device)

    pipe.enable_vae_slicing()

    return pipe, cpu_offload_enabled
