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


def load_pipeline(model_path: str, device: str, force_fp32: bool = False, optimize: bool = False) -> tuple:
    """Load diffusion pipeline with proper precision handling.

    Args:
        model_path: Path to model file or directory
        device: Target device (cuda/mps/cpu)
        force_fp32: If True, force FP32 inference even for BF16 models (parity mode)
        optimize: If True, enable TF32 + torch.compile for faster inference

    Returns:
        tuple: (pipeline, cpu_offload_enabled)
    """
    # Enable TF32 for faster matmuls when optimize is enabled
    if optimize and device == "cuda":
        torch.set_float32_matmul_precision('high')
        logger.info("TF32 enabled for faster matrix multiplications")

        # Enable persistent inductor cache for faster subsequent runs
        import tempfile
        cache_dir = os.path.join(tempfile.gettempdir(), "noobai_inductor_cache")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', cache_dir)
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
        bf16_supported = check_bf16_support(device)

        if force_fp32 or not bf16_supported:
            # Parity mode or unsupported device: Load all components in FP32
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float32,
                use_safetensors=True,
            )
            if force_fp32:
                logger.info("Parity mode: All components loaded as FP32")
            else:
                logger.info("Device does not support BF16; all components loaded as FP32")
        else:
            # BF16 mode with native VAE precision: Two-pass loading
            # First pass: Load at FP32 to extract VAE at native precision
            logger.info("Loading VAE at native FP32 precision from checkpoint...")
            fp32_pipe = StableDiffusionXLPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float32,
                use_safetensors=True,
            )
            native_vae = fp32_pipe.vae

            # Clean up FP32 pipeline components we don't need
            del fp32_pipe.unet
            del fp32_pipe.text_encoder
            del fp32_pipe.text_encoder_2
            del fp32_pipe

            # Clear memory before second pass
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps" and hasattr(torch, 'mps'):
                torch.mps.empty_cache()

            # Second pass: Load main pipeline at BF16
            logger.info("Loading UNet and text encoders at BF16 precision...")
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_path,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
            )

            # Replace BF16 VAE with native FP32 VAE
            pipe.vae = native_vae

            # Wrap VAE decode to automatically cast BF16 latents to FP32
            # This is needed because diffusers SDXL pipeline only auto-upcasts
            # latents when VAE is FP16, not when VAE is FP32 with BF16 latents
            original_decode = pipe.vae.decode

            def decode_with_upcast(latents, *args, **kwargs):
                if latents.dtype != torch.float32:
                    latents = latents.to(dtype=torch.float32)
                return original_decode(latents, *args, **kwargs)

            pipe.vae.decode = decode_with_upcast
            logger.info("BF16 pipeline loaded with native FP32 VAE (lossless)")

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

    # Apply torch.compile optimization when enabled
    if optimize and device == "cuda":
        logger.info("Compiling UNet with torch.compile (first inference will be slower)...")
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
        logger.info("UNet compiled successfully")

    return pipe, cpu_offload_enabled
