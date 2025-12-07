"""Generation handlers for UI."""

import os
import gradio as gr
from typing import Tuple, Optional
from config import logger, OUTPUT_DIR, InvalidParameterError, GenerationInterruptedError
from state import state_manager, GenerationState
from ui.engine_manager import get_engine_safely
from ui.validation import parse_resolution_string, _coerce_int, _coerce_float, validate_parameters
from utils import calculate_image_hash, get_user_friendly_error


def start_generation() -> Tuple[gr.update, gr.update]:
    """Start generation UI update."""
    state_manager.set_state(GenerationState.GENERATING)
    return gr.update(visible=True, interactive=True), gr.update(value="🔄 Generating...", interactive=False)


def finish_generation() -> Tuple[gr.update, gr.update]:
    """Finish generation UI update."""
    state_manager.finish_generation()
    return gr.update(visible=False), gr.update(value="🎨 Generate Image", interactive=True)


def interrupt_generation() -> Tuple[gr.update, gr.update]:
    """Interrupt generation."""
    state_manager.request_interrupt()
    return gr.update(visible=False), gr.update(value="🔄 Interrupting...", interactive=False)


def generate_image_with_progress(
    prompt: str, negative_prompt: str, resolution: str, cfg_scale: float, steps: int,
    rescale_cfg: float, seed: str, use_custom_resolution: bool, custom_width: int,
    custom_height: int, auto_randomize_seed: bool, adapter_strength: float, enable_dora: bool,
    dora_start_step: int, dora_toggle_mode: Optional[str], dora_manual_schedule: str, progress=gr.Progress()
) -> Tuple[Optional[str], str, str]:
    """Generate image with progress tracking."""
    try:
        # Get engine reference - get_engine_safely() handles its own locking
        # DO NOT wrap this in _engine_lock as get_engine_safely() already acquires that lock,
        # and threading.Lock() is not reentrant (would cause deadlock)
        current_engine = get_engine_safely()
        engine_ready = current_engine is not None and current_engine.is_initialized

        if not engine_ready:
            state_manager.set_state(GenerationState.ERROR)
            return None, "❌ Engine not initialized", seed

        if not prompt.strip():
            state_manager.set_state(GenerationState.ERROR)
            return None, "❌ Please enter a prompt", seed

        if use_custom_resolution:
            width = _coerce_int(custom_width, "Custom width")
            height = _coerce_int(custom_height, "Custom height")
        else:
            width, height = parse_resolution_string(resolution)

        steps = _coerce_int(steps, "Steps")
        dora_start_step = _coerce_int(dora_start_step, "DoRA start step")
        cfg_scale = _coerce_float(cfg_scale, "CFG scale")
        rescale_cfg = _coerce_float(rescale_cfg, "Rescale CFG")
        adapter_strength = _coerce_float(adapter_strength, "Adapter strength")

        param_error = validate_parameters(width, height, steps, cfg_scale, rescale_cfg, adapter_strength, dora_start_step)
        if param_error:
            state_manager.set_state(GenerationState.ERROR)
            return None, param_error, seed

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

        progress(0, desc="Starting generation...")

        image, final_seed, info = current_engine.generate(
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
            dora_toggle_mode=dora_toggle_mode if enable_dora else None,
            dora_manual_schedule=dora_manual_schedule if enable_dora else None,
            progress_callback=lambda p, d: progress(p, desc=d)
        )

        output_path = os.path.join(OUTPUT_DIR, f"noobai_{final_seed}.png")
        saved_path = current_engine.save_image_standardized(image, output_path)

        if not saved_path or not os.path.exists(saved_path):
            state_manager.set_state(GenerationState.ERROR)
            return None, f"❌ Failed to save image to {output_path}", str(final_seed)

        try:
            image_hash = calculate_image_hash(saved_path)
            if image_hash == "ERROR":
                info += f"\n⚠️ Hash calculation failed (file still saved)"
            else:
                info += f"\n📄 MD5 Hash: {image_hash}"
        except Exception as hash_error:
            logger.warning(f"Hash calculation failed: {hash_error}")
            info += f"\n⚠️ Hash calculation failed: {hash_error}"

        progress(1.0, desc="Complete!")
        state_manager.set_state(GenerationState.COMPLETED)

        return saved_path, info, str(final_seed)

    except GenerationInterruptedError:
        state_manager.set_state(GenerationState.INTERRUPTED)
        return None, "⚠️ Generation interrupted", seed
    except InvalidParameterError as e:
        state_manager.set_state(GenerationState.ERROR)
        logger.error(f"Generation failed (invalid parameter): {e}")
        return None, f"❌ {str(e)}", seed
    except (IOError, OSError) as e:
        state_manager.set_state(GenerationState.ERROR)
        error_msg = get_user_friendly_error(e)
        logger.error(f"Generation failed (file error): {e}")
        return None, f"❌ Generation failed: {error_msg}", seed
    except (RuntimeError, ValueError) as e:
        state_manager.set_state(GenerationState.ERROR)
        error_msg = get_user_friendly_error(e)
        logger.error(f"Generation failed (runtime/validation error): {e}")
        return None, f"❌ Generation failed: {error_msg}", seed
    except Exception as e:
        state_manager.set_state(GenerationState.ERROR)
        error_msg = get_user_friendly_error(e)
        logger.error(f"Unexpected error during generation: {e}")
        return None, f"❌ Generation failed: {error_msg}", seed
