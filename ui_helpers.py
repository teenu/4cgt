"""
NoobAI XL V-Pred 1.0 - UI Helper Functions

This module contains helper functions for the Gradio UI including search,
validation, parameter handling, and generation management.
"""

import os
import re
import gc
import time
import gradio as gr
from threading import Lock
from typing import Tuple, Optional, List, Dict, Any
from config import (
    logger, MODEL_CONFIG, GEN_CONFIG, SEARCH_CONFIG, OPTIMAL_SETTINGS,
    OUTPUT_DIR, MODEL_SEARCH_PATHS, InvalidParameterError, GenerationInterruptedError
)
from state import state_manager, GenerationState, resource_pool
from utils import (
    normalize_text, validate_model_path, validate_dora_path,
    discover_dora_adapters, get_dora_adapter_by_name, find_dora_path,
    detect_adapter_precision, format_file_size, get_user_friendly_error,
    calculate_image_hash
)
from prompt_formatter import get_prompt_data
from engine import NoobAIEngine

# ============================================================================
# GLOBAL ENGINE INSTANCE
# ============================================================================

# Global engine instance with thread-safe access
engine: Optional[NoobAIEngine] = None
_engine_lock = Lock()

# Thread-safe engine access functions
def is_engine_ready() -> bool:
    """Thread-safe check if engine is initialized and ready."""
    with _engine_lock:
        return engine is not None and engine.is_initialized

def get_engine_safely() -> Optional[NoobAIEngine]:
    """Thread-safe getter for engine instance."""
    with _engine_lock:
        return engine

# ============================================================================
# UI HELPER FUNCTIONS
# ============================================================================

def search_for_autocomplete(query: str, data_type: str) -> Dict[str, Any]:
    """Handle autocomplete search."""
    try:
        if not query or len(query.strip()) < SEARCH_CONFIG.MIN_QUERY_LENGTH:
            return gr.update(choices=[], value=None)

        results = get_prompt_data().search(query, data_type, limit=SEARCH_CONFIG.MAX_RESULTS)
        choices = [f"{'🔴' if r['source'] == 'danbooru' else '🔵'} {r['display']}" for r in results]
        return gr.update(choices=choices, value=choices[0] if choices else None)

    except (AttributeError, KeyError, ValueError) as e:
        logger.error(f"Error in {data_type} search (data error): {e}")
        return gr.update(choices=[], value=None)
    except Exception as e:
        logger.error(f"Unexpected error in {data_type} search: {e}")
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

    except (AttributeError, KeyError, IndexError) as e:
        logger.error(f"Error in {data_type} selection (data error): {e}")
        return selected_choice or ""
    except Exception as e:
        logger.error(f"Unexpected error in {data_type} selection: {e}")
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

def _coerce_int(value, label: str) -> int:
    """Coerce value to integer with descriptive error message."""
    try:
        return int(value)
    except (TypeError, ValueError):
        raise InvalidParameterError(f"{label} must be an integer value")

def validate_parameters(w: int, h: int, s: int, c: float, r: float, a: Optional[float] = None, ds: Optional[int] = None) -> Optional[str]:
    """Validate generation parameters."""
    errors = []

    if not (GEN_CONFIG.MIN_RESOLUTION <= w <= GEN_CONFIG.MAX_RESOLUTION):
        errors.append(f"Width must be {GEN_CONFIG.MIN_RESOLUTION}-{GEN_CONFIG.MAX_RESOLUTION}")
    elif w % 8 != 0:
        errors.append("Width must be divisible by 8 (diffusion requirement)")

    if not (GEN_CONFIG.MIN_RESOLUTION <= h <= GEN_CONFIG.MAX_RESOLUTION):
        errors.append(f"Height must be {GEN_CONFIG.MIN_RESOLUTION}-{GEN_CONFIG.MAX_RESOLUTION}")
    elif h % 8 != 0:
        errors.append("Height must be divisible by 8 (diffusion requirement)")

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

# ============================================================================
# ENGINE MANAGEMENT
# ============================================================================

def initialize_engine(model_path: str, enable_dora: bool = False, dora_path: str = "", dora_selection: str = "") -> str:
    """Initialize the NoobAI engine with comprehensive teardown of previous instance.

    Thread-safe: Holds lock during entire check-teardown-initialization sequence
    to prevent race conditions with concurrent generation or reinitialization.
    """
    global engine

    # Hold lock for ENTIRE operation to prevent TOCTOU race conditions
    # This ensures no generation can start during teardown
    with _engine_lock:
        try:
            # Atomic check: prevent teardown during active generation
            if state_manager.is_generating():
                return "❌ Cannot reinitialize: Image generation in progress. Please wait for completion or interrupt first."

            # COMPREHENSIVE TEARDOWN of existing engine before fresh initialization
            if engine is not None:
                logger.info("Performing comprehensive teardown of previous engine instance")

                try:
                    # Full engine teardown with all cleanup
                    engine.teardown_engine()
                    # GPU synchronization handled by clear_memory() in teardown

                    # Explicit deletion and nullification
                    del engine
                    engine = None

                    # Additional global cleanup
                    resource_pool.clear()

                    # Force garbage collection after teardown
                    gc.collect()

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

            # Initialize engine (still under lock - prevents concurrent access)
            engine = NoobAIEngine(
                model_path=validated_model_path,
                enable_dora=enable_dora,
                dora_path=dora_path_to_use,
                dora_start_step=OPTIMAL_SETTINGS['dora_start_step']
            )

            # Get model size (handle both files and directories)
            if os.path.isdir(validated_model_path):
                # For directories, calculate total size of all files
                model_size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, _, filenames in os.walk(validated_model_path)
                    for filename in filenames
                )
            else:
                model_size = os.path.getsize(validated_model_path)

            status_msg = f"✅ Engine initialized!\n📊 Model: {format_file_size(model_size)}{dora_status}"

            return status_msg

        except (IOError, OSError) as e:
            engine = None
            error_msg = get_user_friendly_error(e)
            logger.error(f"Engine initialization failed (file error): {e}")
            return f"❌ Initialization failed: {error_msg}"
        except (RuntimeError, ValueError) as e:
            engine = None
            error_msg = get_user_friendly_error(e)
            logger.error(f"Engine initialization failed (runtime/validation error): {e}")
            return f"❌ Initialization failed: {error_msg}"
        except Exception as e:
            engine = None
            error_msg = get_user_friendly_error(e)
            logger.error(f"Unexpected error during engine initialization: {e}")
            return f"❌ Initialization failed: {error_msg}"

def find_model_path() -> Optional[str]:
    """Search for the model file or directory in common locations."""
    for path in MODEL_SEARCH_PATHS:
        if os.path.exists(path):
            try:
                if os.path.isdir(path):
                    # Directory (diffusers format) - check if it has required files
                    unet_path = os.path.join(path, "unet")
                    vae_path = os.path.join(path, "vae")
                    if os.path.isdir(unet_path) and os.path.isdir(vae_path):
                        return path
                else:
                    # File (safetensors format) - check size
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

def auto_initialize(preferred_model_path: str = None) -> Tuple[str, str, bool, str, str]:
    """Enhanced auto-initialize with smart DoRA defaults.

    Args:
        preferred_model_path: Optional model path to use. If provided, overrides auto-discovery.

    Returns:
        Tuple of (status, model_path, enable_dora, dora_path, default_adapter)
    """
    # Use provided path or auto-discover
    model_path = preferred_model_path if preferred_model_path else find_model_path()

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

# ============================================================================
# GENERATION HANDLERS
# ============================================================================

def start_generation() -> Tuple[gr.update, gr.update]:
    """Start generation UI update."""
    state_manager.set_state(GenerationState.GENERATING)
    return gr.update(visible=True, interactive=True), gr.update(value="🔄 Generating...", interactive=False)

def generate_image_with_progress(
    prompt: str, negative_prompt: str, resolution: str, cfg_scale: float, steps: int,
    rescale_cfg: float, seed: str, use_custom_resolution: bool, custom_width: int,
    custom_height: int, auto_randomize_seed: bool, adapter_strength: float, enable_dora: bool, dora_start_step: int, dora_toggle_mode: Optional[str], dora_manual_schedule: str, progress=gr.Progress()
) -> Tuple[Optional[str], str, str]:
    """Generate image with progress tracking and return file path for hash consistency."""
    try:
        # Check engine with thread-safe access and get local reference
        with _engine_lock:
            current_engine = engine
            engine_ready = current_engine is not None and current_engine.is_initialized

        if not engine_ready:
            state_manager.set_state(GenerationState.ERROR)
            return None, "❌ Engine not initialized", seed

        # Validate prompt
        if not prompt.strip():
            state_manager.set_state(GenerationState.ERROR)
            return None, "❌ Please enter a prompt", seed

        # Parse resolution
        if use_custom_resolution:
            width = _coerce_int(custom_width, "Custom width")
            height = _coerce_int(custom_height, "Custom height")
        else:
            width, height = parse_resolution_string(resolution)

        steps = _coerce_int(steps, "Steps")
        dora_start_step = _coerce_int(dora_start_step, "DoRA start step")

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

        # Debug: Log DoRA settings
        if enable_dora:
            if dora_toggle_mode == "manual":
                logger.info(f"DoRA enabled: toggle_mode=manual")
                logger.info(f"Manual DoRA schedule CSV: {dora_manual_schedule}")
            elif dora_toggle_mode:
                logger.info(f"DoRA enabled: toggle_mode={dora_toggle_mode}")
            else:
                logger.info(f"DoRA enabled: normal mode, start_step={dora_start_step}")

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

        # Save image with standardized settings
        output_path = os.path.join(OUTPUT_DIR, f"noobai_{final_seed}.png")
        saved_path = current_engine.save_image_standardized(image, output_path)

        # Validate save succeeded before calculating hash
        if not saved_path or not os.path.exists(saved_path):
            raise IOError(f"Failed to save image to {output_path}")

        # Add hash info to the generation info
        image_hash = calculate_image_hash(saved_path)
        info += f"\n📄 MD5 Hash: {image_hash}"

        progress(1.0, desc="Complete!")
        state_manager.set_state(GenerationState.COMPLETED)

        return output_path, info, str(final_seed)

    except GenerationInterruptedError:
        state_manager.set_state(GenerationState.INTERRUPTED)
        return None, "⚠️ Generation interrupted", seed
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

def finish_generation() -> Tuple[gr.update, gr.update]:
    """Finish generation UI update."""
    state_manager.set_state(GenerationState.IDLE)
    return gr.update(visible=False), gr.update(value="🎨 Generate Image", interactive=True)

def interrupt_generation() -> Tuple[gr.update, gr.update]:
    """Interrupt generation."""
    state_manager.request_interrupt()
    return gr.update(visible=False), gr.update(value="🔄 Interrupting...", interactive=False)
