"""Engine memory management and cleanup."""

import gc
import torch
from config import logger


def synchronize_device(device: str) -> None:
    """Synchronize device operations."""
    try:
        if device == "mps":
            try:
                torch.mps.synchronize()
            except (AttributeError, RuntimeError) as e:
                logger.debug(f"MPS synchronization skipped: {e}")
        elif device == "cuda":
            try:
                torch.cuda.synchronize()
            except (AttributeError, RuntimeError) as e:
                logger.debug(f"CUDA synchronization skipped: {e}")
    except Exception as e:
        logger.debug(f"Device synchronization failed: {e}")


def clear_memory(device: str) -> None:
    """Clear GPU/memory caches."""
    if device not in ("cuda", "mps", "cpu"):
        return
    try:
        synchronize_device(device)
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        logger.debug(f"Error during memory cleanup: {e}")


def teardown_pipeline(pipe, device: str, cpu_offload_enabled: bool, dora_loaded: bool) -> None:
    """Teardown pipeline with resource cleanup."""
    try:
        if pipe and dora_loaded:
            try:
                pipe.unload_lora_weights()
                if hasattr(pipe, 'delete_adapters'):
                    pipe.delete_adapters(["noobai_dora"])
            except Exception as e:
                logger.warning(f"Error unloading DoRA adapters: {e}")

        if pipe:
            try:
                if not cpu_offload_enabled:
                    pipe = pipe.to("cpu")
            except RuntimeError as e:
                error_str = str(e)
                if "meta tensor" in error_str.lower() or "Cannot copy out of meta" in error_str:
                    logger.debug("Skipping .to(cpu) for meta tensor pipeline (expected with CPU offloading)")
                else:
                    logger.warning(f"Error moving pipeline to CPU: {e}")
            except Exception as e:
                logger.warning(f"Error moving pipeline to CPU: {e}")

            try:
                if hasattr(pipe, 'maybe_free_model_hooks'):
                    pipe.maybe_free_model_hooks()
            except Exception as e:
                logger.debug(f"Could not free model hooks: {e}")

            try:
                components_to_delete = [
                    'unet', 'vae', 'text_encoder', 'text_encoder_2',
                    'scheduler', 'tokenizer', 'tokenizer_2', 'image_processor'
                ]
                for component_name in components_to_delete:
                    if hasattr(pipe, component_name):
                        setattr(pipe, component_name, None)
            except Exception as e:
                logger.warning(f"Error cleaning pipeline components: {e}")

        try:
            if device == "mps":
                try:
                    torch.mps.synchronize()
                except (AttributeError, RuntimeError):
                    pass
                torch.mps.empty_cache()
            elif device == "cuda":
                try:
                    torch.cuda.synchronize()
                except (AttributeError, RuntimeError):
                    pass
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, 'ipc_collect'):
                    torch.cuda.ipc_collect()
        except Exception as e:
            logger.warning(f"Error clearing device caches: {e}")

        gc.collect()
        logger.info("Engine teardown completed")

    except Exception as e:
        logger.error(f"Error during engine teardown: {e}")
