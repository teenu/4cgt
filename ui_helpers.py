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
from typing import Tuple, Optional, List, Dict, Any, Union
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

# Global engine instance
engine: Optional[NoobAIEngine] = None
_engine_lock = Lock()

def is_engine_ready() -> bool:
    """Check if engine is initialized and ready."""
    with _engine_lock:
        return engine is not None and engine.is_initialized

def get_engine_safely() -> Optional[NoobAIEngine]:
    """Get engine instance safely."""
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
    """Parse resolution string to width and height.

    Validates parsed values are within acceptable bounds (512-2048).
    """
    try:
        matches = re.findall(r'\d+', res_str)
        if len(matches) < 2:
            logger.warning(f"Could not parse resolution from: '{res_str}', using defaults")
            return OPTIMAL_SETTINGS['width'], OPTIMAL_SETTINGS['height']

        w, h = int(matches[0]), int(matches[1])

        # Validate bounds
        if not (GEN_CONFIG.MIN_RESOLUTION <= w <= GEN_CONFIG.MAX_RESOLUTION):
            logger.warning(f"Width {w} out of bounds ({GEN_CONFIG.MIN_RESOLUTION}-{GEN_CONFIG.MAX_RESOLUTION}), using default")
            w = OPTIMAL_SETTINGS['width']
        if not (GEN_CONFIG.MIN_RESOLUTION <= h <= GEN_CONFIG.MAX_RESOLUTION):
            logger.warning(f"Height {h} out of bounds ({GEN_CONFIG.MIN_RESOLUTION}-{GEN_CONFIG.MAX_RESOLUTION}), using default")
            h = OPTIMAL_SETTINGS['height']

        return w, h
    except Exception as e:
        logger.error(f"Resolution parsing error: {e}")
        return OPTIMAL_SETTINGS['width'], OPTIMAL_SETTINGS['height']

def _coerce_int(value: Union[int, float, str, Any], label: str) -> int:
    """Coerce value to integer with descriptive error message.

    Handles floats from Gradio sliders (rounds instead of truncates),
    strings from textboxes, and numpy types.

    Note: OverflowError is caught to handle edge cases like int(float('inf')).
    """
    try:
        # Handle None
        if value is None:
            raise InvalidParameterError(f"{label} cannot be None")

        # Handle numpy types
        if hasattr(value, 'item'):
            value = value.item()

        # For floats, use round() to avoid silent truncation
        if isinstance(value, float):
            if not value.is_integer():
                logger.debug(f"{label}: rounding {value} to {round(value)}")
            return round(value)

        # Convert to int (handles str, int, etc.)
        return int(value)
    except InvalidParameterError:
        # Re-raise our own exceptions without wrapping
        raise
    except (TypeError, ValueError, OverflowError) as e:
        raise InvalidParameterError(f"{label} must be an integer value, got {type(value).__name__}: {e}")

def _coerce_float(value: Union[int, float, str, Any], label: str) -> float:
    """Coerce value to float with descriptive error message."""
    try:
        # Handle None
        if value is None:
            raise InvalidParameterError(f"{label} cannot be None")
        
        # Handle numpy types
        if hasattr(value, 'item'):
            value = value.item()
        
        return float(value)
    except InvalidParameterError:
        # Re-raise our own exceptions without wrapping
        raise
    except (TypeError, ValueError) as e:
        raise InvalidParameterError(f"{label} must be a numeric value, got {type(value).__name__}: {e}")

def validate_parameters(
    w: Union[int, float], 
    h: Union[int, float], 
    s: Union[int, float], 
    c: Union[int, float], 
    r: Union[int, float], 
    a: Optional[Union[int, float]] = None, 
    ds: Optional[Union[int, float]] = None
) -> Optional[str]:
    """Validate generation parameters with type coercion."""
    errors = []

    # Coerce types first
    try:
        w = _coerce_int(w, "Width")
    except InvalidParameterError as e:
        errors.append(str(e))
        w = OPTIMAL_SETTINGS['width']  # Use default for further validation
    
    try:
        h = _coerce_int(h, "Height")
    except InvalidParameterError as e:
        errors.append(str(e))
        h = OPTIMAL_SETTINGS['height']
    
    try:
        s = _coerce_int(s, "Steps")
    except InvalidParameterError as e:
        errors.append(str(e))
        s = OPTIMAL_SETTINGS['steps']
    
    try:
        c = _coerce_float(c, "CFG scale")
    except InvalidParameterError as e:
        errors.append(str(e))
        c = OPTIMAL_SETTINGS['cfg_scale']
    
    try:
        r = _coerce_float(r, "Rescale CFG")
    except InvalidParameterError as e:
        errors.append(str(e))
        r = OPTIMAL_SETTINGS['rescale_cfg']

    # Validate ranges
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

    # Validate optional adapter strength
    if a is not None:
        try:
            a = _coerce_float(a, "Adapter strength")
            if not (MODEL_CONFIG.MIN_ADAPTER_STRENGTH <= a <= MODEL_CONFIG.MAX_ADAPTER_STRENGTH):
                errors.append(f"Adapter strength must be {MODEL_CONFIG.MIN_ADAPTER_STRENGTH}-{MODEL_CONFIG.MAX_ADAPTER_STRENGTH}")
        except InvalidParameterError as e:
            errors.append(str(e))

    # Validate optional DoRA start step
    if ds is not None:
        try:
            ds = _coerce_int(ds, "DoRA start step")
            if not (MODEL_CONFIG.MIN_DORA_START_STEP <= ds <= MODEL_CONFIG.MAX_DORA_START_STEP):
                errors.append(f"DoRA start step must be {MODEL_CONFIG.MIN_DORA_START_STEP}-{MODEL_CONFIG.MAX_DORA_START_STEP}")
            elif ds > s:
                errors.append(f"DoRA start step ({ds}) cannot be greater than total steps ({s})")
        except InvalidParameterError as e:
            errors.append(str(e))

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
        try:
            value = float(value)
        except (TypeError, ValueError):
            return '<div style="color: red;">❌ Invalid value</div>'
        optimal_cfg = OPTIMAL_SETTINGS['cfg_scale']
        if abs(value - optimal_cfg) < 0.1:
            return f'<div style="color: green;">✅ Ideal ({optimal_cfg})</div>'
        elif 3.5 <= value <= 5.5:
            return '<div style="color: green;">✅ Optimal range (3.5-5.5)</div>'
        else:
            return '<div style="color: orange;">⚠️ Outside optimal range (3.5-5.5)</div>'

    def update_steps_status(value):
        try:
            value = int(value)
        except (TypeError, ValueError):
            return '<div style="color: red;">❌ Invalid value</div>'
        optimal_steps = OPTIMAL_SETTINGS['steps']
        if value == optimal_steps:
            return f'<div style="color: green;">✅ Ideal ({optimal_steps})</div>'
        elif 32 <= value <= 40:
            return '<div style="color: green;">✅ Optimal range (32-40)</div>'
        elif value >= 10:
            return '<div style="color: orange;">⚠️ Below optimal range (32-40)</div>'
        else:
            return '<div style="color: red;">❌ Too low for quality results</div>'

    def update_rescale_status(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return '<div style="color: red;">❌ Invalid value</div>'
        optimal_rescale = OPTIMAL_SETTINGS['rescale_cfg']
        if abs(value - optimal_rescale) < 0.05:
            return f'<div style="color: green;">✅ Ideal (around {optimal_rescale})</div>'
        elif 0.4 <= value <= 0.8:
            return '<div style="color: green;">✅ Optimal range (0.4-0.8)</div>'
        else:
            return '<div style="color: blue;">📊 Valid</div>'

    def update_adapter_status(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return '<div style="color: red;">❌ Invalid value</div>'
        if 0.8 <= value <= 1.2:
            return '<div style="color: green;">✅ Optimal range (0.8-1.2)</div>'
        elif value == 0.0:
            return '<div style="color: gray;">⚪ Disabled</div>'
        elif value > 1.2:
            return '<div style="color: orange;">⚠️ High strength (amplified)</div>'
        else:
            return '<div style="color: blue;">📊 Valid</div>'

    def update_dora_start_step_status(value):
        try:
            value = int(value)
        except (TypeError, ValueError):
            return '<div style="color: red;">❌ Invalid value</div>'
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
    """Initialize the engine."""
    global engine

    with _engine_lock:
        try:
            # CRITICAL FIX: Block initialization only during active pipeline usage
            # - GENERATING: Pipeline actively running, unsafe to teardown
            # - INTERRUPTED: Generation unwinding, callbacks may still execute
            # - COMPLETED/ERROR/IDLE: Pipeline not in use, safe to reinitialize
            current_state = state_manager.get_state()
            if current_state in (GenerationState.GENERATING, GenerationState.INTERRUPTED):
                state_name = current_state.value
                return f"❌ Cannot reinitialize: System is {state_name}. Please wait for current operation to complete."

            if engine is not None:
                logger.info("Performing teardown of previous engine instance")

                try:
                    engine.teardown_engine()
                    del engine
                    engine = None
                    resource_pool.clear()
                    gc.collect()
                    logger.info("Previous engine instance torn down")

                except Exception as e:
                    logger.error(f"Error during engine teardown: {e}")
                    engine = None
                    resource_pool.clear()
                    gc.collect()

            is_valid, validated_model_path = validate_model_path(model_path)
            if not is_valid:
                return f"❌ {validated_model_path}"

            dora_path_to_use = None
            dora_status = ""

            if enable_dora:
                if dora_selection and dora_selection != "None":
                    adapter_info = get_dora_adapter_by_name(dora_selection)
                    if adapter_info:
                        dora_path_to_use = adapter_info['path']
                        dora_status = f"\n🎯 DoRA: {adapter_info['display_name']}"
                    else:
                        dora_status = f"\n⚠️ DoRA: Selected adapter '{dora_selection}' not found"
                elif dora_path.strip():
                    dora_valid, dora_result = validate_dora_path(dora_path)
                    if dora_valid:
                        dora_path_to_use = dora_result
                        precision = detect_adapter_precision(dora_result)
                        dora_status = f"\n🎯 DoRA: {os.path.basename(dora_result)} ({precision})"
                    else:
                        dora_status = f"\n⚠️ DoRA Error: {dora_result}"
                else:
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
    """Auto-initialize with DoRA defaults."""
    model_path = preferred_model_path if preferred_model_path else find_model_path()

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
    """Generate image with progress tracking and return file path for hash consistency.
    
    Note: State transitions (GENERATING -> COMPLETED/ERROR/INTERRUPTED -> IDLE) are managed
    by the Gradio event chain (.then handlers), specifically finish_generation().
    This function sets intermediate states (ERROR, INTERRUPTED) for error cases,
    and COMPLETED for success, but IDLE transition happens in finish_generation().
    """
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

        # Coerce all numeric parameters
        steps = _coerce_int(steps, "Steps")
        dora_start_step = _coerce_int(dora_start_step, "DoRA start step")
        cfg_scale = _coerce_float(cfg_scale, "CFG scale")
        rescale_cfg = _coerce_float(rescale_cfg, "Rescale CFG")
        adapter_strength = _coerce_float(adapter_strength, "Adapter strength")

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
            state_manager.set_state(GenerationState.ERROR)
            return None, f"❌ Failed to save image to {output_path}", str(final_seed)

        # Calculate hash with error handling - don't fail generation if hash fails
        try:
            image_hash = calculate_image_hash(saved_path)
            if image_hash is None:
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

def finish_generation() -> Tuple[gr.update, gr.update]:
    """Finish generation UI update.
    
    This is called via Gradio's .then() handler after generate_image_with_progress completes.
    It transitions the state machine to IDLE regardless of success/failure/interruption.
    """
    state_manager.finish_generation()
    return gr.update(visible=False), gr.update(value="🎨 Generate Image", interactive=True)

def interrupt_generation() -> Tuple[gr.update, gr.update]:
    """Interrupt generation."""
    state_manager.request_interrupt()
    return gr.update(visible=False), gr.update(value="🔄 Interrupting...", interactive=False)

# ============================================================================
# QUEUE HANDLERS
# ============================================================================

def add_to_queue(
    prompt: str, negative_prompt: str, resolution: str, cfg_scale: float,
    steps: int, rescale_cfg: float, seed: str, use_custom_resolution: bool,
    custom_width: int, custom_height: int, auto_randomize_seed: bool,
    adapter_strength: float, enable_dora: bool, dora_start_step: int,
    dora_toggle_mode: Optional[str], dora_manual_schedule: str
) -> Tuple[str, str]:
    """Add current settings to generation queue.

    Returns:
        Tuple of (queue_html, status_html)
    """
    from state import queue_manager, QueueItem
    from config import QUEUE_CONFIG

    try:
        if not prompt.strip():
            return render_queue_html(), '<span style="color: red;">❌ Cannot queue: empty prompt</span>'

        item = QueueItem(
            prompt=prompt,
            negative_prompt=negative_prompt,
            resolution=resolution,
            cfg_scale=float(cfg_scale) if cfg_scale else 4.2,
            steps=int(steps) if steps else 34,
            rescale_cfg=float(rescale_cfg) if rescale_cfg else 0.55,
            seed=seed,
            use_custom_resolution=use_custom_resolution,
            custom_width=int(custom_width) if custom_width else 1216,
            custom_height=int(custom_height) if custom_height else 832,
            auto_randomize_seed=auto_randomize_seed,
            adapter_strength=float(adapter_strength) if adapter_strength else 1.0,
            enable_dora=enable_dora,
            dora_start_step=int(dora_start_step) if dora_start_step else 1,
            dora_toggle_mode=dora_toggle_mode,
            dora_manual_schedule=dora_manual_schedule or ""
        )

        success, message = queue_manager.add(item)
        if success:
            status = f'<span style="color: green;">✅ Queue: {queue_manager.size()}/{QUEUE_CONFIG.MAX_QUEUE_SIZE}</span>'
        else:
            status = f'<span style="color: red;">❌ {message}</span>'

        return render_queue_html(), status

    except Exception as e:
        logger.error(f"Queue add error: {e}")
        error_msg = str(e)[:50] if len(str(e)) > 50 else str(e)
        return render_queue_html(), f'<span style="color: red;">❌ Queue error: {error_msg}</span>'


def remove_from_queue(item_id: str) -> Tuple[str, str]:
    """Remove item from queue by ID.

    Returns:
        Tuple of (queue_html, status_html)
    """
    from state import queue_manager
    from config import QUEUE_CONFIG

    queue_manager.remove(item_id)
    status = f'<span style="color: gray;">Queue: {queue_manager.size()}/{QUEUE_CONFIG.MAX_QUEUE_SIZE}</span>'

    return render_queue_html(), status


def clear_queue() -> Tuple[str, str]:
    """Clear all items from queue.

    Returns:
        Tuple of (queue_html, status_html)
    """
    from state import queue_manager
    from config import QUEUE_CONFIG

    count = queue_manager.clear()
    status = f'<span style="color: gray;">Queue: 0/{QUEUE_CONFIG.MAX_QUEUE_SIZE} (cleared {count})</span>'

    return render_queue_html(), status


def set_auto_process(enabled: bool) -> None:
    """Enable/disable auto-processing of queue."""
    from state import queue_manager
    queue_manager.set_auto_process(enabled)


def render_queue_html() -> str:
    """Render queue items as HTML cards."""
    from state import queue_manager
    from config import QUEUE_CONFIG

    items = queue_manager.get_all()

    if not items:
        return '<div class="queue-empty">Queue is empty - add items with "Add to Queue"</div>'

    html_parts = []
    for i, item in enumerate(items):
        snippet = item.get_prompt_snippet(QUEUE_CONFIG.PROMPT_SNIPPET_LENGTH)
        position = i + 1
        dora_label = "DoRA" if item.enable_dora else "No DoRA"

        html_parts.append(f'''
        <div class="queue-card" data-id="{item.id}">
            <div class="queue-card-content">
                <div class="queue-card-position">#{position}</div>
                <div class="queue-card-snippet">{snippet}</div>
                <div class="queue-card-meta">
                    Steps: {item.steps} | CFG: {item.cfg_scale} | {dora_label}
                </div>
            </div>
            <button class="queue-card-remove" title="Remove from queue">✕</button>
        </div>
        ''')

    return f'<div class="queue-container">{"".join(html_parts)}</div>'


def get_queue_status_html() -> str:
    """Get HTML for queue status."""
    from state import queue_manager
    from config import QUEUE_CONFIG

    size = queue_manager.size()
    max_size = QUEUE_CONFIG.MAX_QUEUE_SIZE
    return f'<span style="color: gray;">Queue: {size}/{max_size}</span>'


def get_next_queue_item() -> Optional[Dict[str, Any]]:
    """Get next item from queue for processing.

    Returns:
        Dictionary of generation parameters or None if queue empty
    """
    from state import queue_manager

    item = queue_manager.pop_next()
    if item:
        return item.to_dict()
    return None


def should_process_queue() -> bool:
    """Check if queue processing should continue.

    Returns:
        True if auto-process enabled and queue has items
    """
    from state import queue_manager
    return queue_manager.is_auto_process_enabled() and not queue_manager.is_empty()


# ============================================================================
# GALLERY HANDLERS
# ============================================================================

def add_to_gallery(image_path: str, seed: int, prompt: str, generation_info: str) -> List[str]:
    """Add completed image to gallery.

    Args:
        image_path: Path to generated image
        seed: Seed used for generation
        prompt: Generation prompt
        generation_info: Full generation info string

    Returns:
        List of all gallery image paths for Gradio Gallery component
    """
    from state import gallery_manager, GalleryItem

    if not image_path or not os.path.exists(image_path):
        return gallery_manager.get_paths()

    item = GalleryItem(
        image_path=image_path,
        seed=seed,
        prompt=prompt,
        generation_info=generation_info
    )

    gallery_manager.add(item)
    return gallery_manager.get_paths()


def clear_gallery() -> Tuple[List[str], str]:
    """Clear all items from gallery.

    Returns:
        Tuple of (empty_paths_list, count_html)
    """
    from state import gallery_manager
    from config import QUEUE_CONFIG

    count = gallery_manager.clear()
    return [], f'<span style="color: gray;">0/{QUEUE_CONFIG.MAX_GALLERY_SIZE} images (cleared {count})</span>'


def select_gallery_image(evt: gr.SelectData) -> Tuple[Optional[str], str]:
    """Handle gallery image selection.

    Args:
        evt: Gradio select event with index

    Returns:
        Tuple of (image_path, generation_info)
    """
    from state import gallery_manager

    try:
        item = gallery_manager.get_by_index(evt.index)
        if item:
            return item.image_path, item.generation_info
        logger.warning(f"Gallery item at index {evt.index} not found")
        return None, "⚠️ Image not found in gallery"
    except Exception as e:
        logger.error(f"Gallery selection error: {e}")
        error_msg = str(e)[:50] if len(str(e)) > 50 else str(e)
        return None, f"⚠️ Selection error: {error_msg}"


def get_gallery_count_html() -> str:
    """Get HTML for gallery item count."""
    from state import gallery_manager
    from config import QUEUE_CONFIG

    count = gallery_manager.size()
    max_size = QUEUE_CONFIG.MAX_GALLERY_SIZE
    return f'<span style="color: gray;">{count}/{max_size} images</span>'


# ============================================================================
# MODIFIED GENERATION FLOW WITH QUEUE/GALLERY SUPPORT
# ============================================================================

def finish_generation_with_gallery(
    saved_image_path: Optional[str],
    generation_info: str,
    final_seed: str
) -> Tuple[gr.update, gr.update, List[str], str, str, str, bool]:
    """Finish generation and handle queue/gallery updates.

    Returns:
        Tuple of:
        - interrupt_btn update
        - generate_btn update
        - gallery images list
        - gallery count html
        - queue html
        - queue status html
        - should_continue (True if more queue items to process)
    """
    from state import gallery_manager, queue_manager

    # Reset generation state
    state_manager.finish_generation()

    # Add to gallery if successful
    if saved_image_path and os.path.exists(saved_image_path):
        try:
            seed_int = int(final_seed) if final_seed.isdigit() else 0
        except (ValueError, TypeError):
            seed_int = 0

        # Extract prompt from generation_info (look for actual content)
        prompt_line = ""
        for line in generation_info.split('\n'):
            line = line.strip()
            # Skip status/info lines
            if line and not line.startswith(('✅', '⚠️', '🌱', '🎯', '⚪', '📄', '❌', '📊')):
                prompt_line = line
                break

        add_to_gallery(saved_image_path, seed_int, prompt_line, generation_info)

    gallery_paths = gallery_manager.get_paths()
    gallery_count = get_gallery_count_html()
    queue_html = render_queue_html()
    queue_status = get_queue_status_html()

    # Check if we should continue processing queue
    should_continue = should_process_queue()

    return (
        gr.update(visible=False),
        gr.update(value="🎨 Generate Image", interactive=True),
        gallery_paths,
        gallery_count,
        queue_html,
        queue_status,
        should_continue
    )


def process_next_queue_item(should_continue: bool) -> Tuple:
    """Load next queue item into generation inputs if auto-process enabled.

    Returns:
        Tuple of all generation input values + trigger_generation flag
    """
    if not should_continue:
        # Return unchanged updates, no trigger
        return (
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(),
            False  # trigger_generation
        )

    next_item = get_next_queue_item()
    if not next_item:
        return (
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(),
            False
        )

    # Return values to populate inputs, with trigger
    return (
        next_item['prompt'],
        next_item['negative_prompt'],
        next_item['resolution'],
        next_item['cfg_scale'],
        next_item['steps'],
        next_item['rescale_cfg'],
        next_item['seed'],
        next_item['use_custom_resolution'],
        next_item['custom_width'],
        next_item['custom_height'],
        next_item['auto_randomize_seed'],
        next_item['adapter_strength'],
        next_item['enable_dora'],
        next_item['dora_start_step'],
        next_item['dora_toggle_mode'],
        next_item['dora_manual_schedule'],
        True  # trigger_generation
    )


def trigger_queue_generation(trigger: bool) -> Tuple[gr.update, gr.update, str]:
    """Trigger generation if queue item was loaded.

    Args:
        trigger: Whether to trigger generation

    Returns:
        Tuple of (interrupt_btn update, generate_btn update, trigger_input_value)
    """
    if trigger:
        # Return "trigger" value which JS will detect and click the generate button
        return gr.update(), gr.update(value="🎨 Generate Image", interactive=True), "trigger"
    return gr.update(), gr.update(), ""
