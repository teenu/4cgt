"""Gradio UI interface creation."""

import random
import gradio as gr
from config import (
    OFFICIAL_RESOLUTIONS, RECOMMENDED_RESOLUTIONS, OPTIMAL_SETTINGS,
    GEN_CONFIG, MODEL_CONFIG, DEFAULT_POSITIVE_PREFIX, DEFAULT_NEGATIVE_PROMPT,
    OPTIMIZED_DORA_SETTINGS, OPTIMIZED_DORA_SCHEDULE_CSV, CONTROLNET_CONFIG
)
from ui.engine_manager import (
    is_engine_ready, auto_initialize, get_dora_ui_state,
    initialize_engine, get_engine_safely, get_adapter_choices
)
from ui.widgets import (
    create_search_ui, connect_search_events, create_clear_handler,
    create_status_updater, format_token_count_html
)
from ui.search_helpers import compose_final_prompt, get_random_value
from ui.generation import (
    start_generation, generate_image_with_progress,
    finish_generation, interrupt_generation
)
from ui.styles import CSS_STYLES, JAVASCRIPT_HEAD
from utils import parse_manual_dora_schedule
from ui.controlnet_helpers import (
    get_controlnet_choices, get_default_controlnet,
    get_controlnet_path_from_display_name, refresh_controlnet_dropdown
)


def generate_dora_grid(num_steps: int, schedule_csv: str = "", show_locked_badge: bool = False) -> str:
    """Generate HTML for DoRA toggle grid visualization.

    Args:
        num_steps: Number of steps to display in the grid
        schedule_csv: CSV string of 0/1 values for each step
        show_locked_badge: If True, show "Preset Locked" badge above grid (for Optimized mode)
    """
    schedule, _ = parse_manual_dora_schedule(schedule_csv, num_steps) if schedule_csv else (None, None)

    if not schedule:
        schedule = [0] * num_steps

    cells = []
    for i, value in enumerate(schedule):
        cell_class = "dora-cell on" if value == 1 else "dora-cell"
        cells.append(f'<div class="{cell_class}" data-step="{i}" title="Step {i}: {"ON" if value == 1 else "OFF"}"></div>')

    badge_html = ''
    if show_locked_badge:
        badge_html = '<div class="dora-preset-badge">Preset Locked - Click any cell to customize</div>'

    grid_html = f'{badge_html}<div class="dora-grid-container" id="dora-grid-{random.randint(1000, 9999)}-container">{"".join(cells)}</div>'
    return grid_html


def schedules_match_optimized(schedule_csv: str, steps: int, cfg: float, rescale: float, adapter_strength: float) -> bool:
    """Check if current settings exactly match the Optimized preset.

    Args:
        schedule_csv: Current schedule as CSV string
        steps: Current step count
        cfg: Current CFG scale value
        rescale: Current rescale CFG value
        adapter_strength: Current adapter strength value

    Returns:
        True if all settings match the Optimized preset exactly
    """
    opt = OPTIMIZED_DORA_SETTINGS
    # Check numeric parameters with tolerance for floating point
    if int(steps) != opt['steps']:
        return False
    if abs(float(cfg) - opt['cfg_scale']) > 0.01:
        return False
    if abs(float(rescale) - opt['rescale_cfg']) > 0.01:
        return False
    if abs(float(adapter_strength) - opt['adapter_strength']) > 0.01:
        return False
    # Compare schedules
    schedule, _ = parse_manual_dora_schedule(schedule_csv, int(steps)) if schedule_csv else (None, None)
    if not schedule or schedule != opt['schedule']:
        return False
    return True


def create_interface(model_path: str = None, force_fp32: bool = False, optimize: bool = False) -> gr.Blocks:
    """Create the Gradio interface.

    Args:
        model_path: Optional path to model file or directory
        force_fp32: Force FP32 inference for parity mode
        optimize: Enable TF32 + torch.compile for faster inference
    """
    init_status, default_model_path, default_enable_dora, default_dora_path, default_adapter_selection = auto_initialize(model_path, force_fp32=force_fp32, optimize=optimize)
    is_ready = is_engine_ready()
    dora_ui_state = get_dora_ui_state()

    # Compute initial token counter values based on engine state
    # Final prompt starts empty, negative prompt has default value
    if is_ready:
        engine = get_engine_safely()
        # Empty prompt shows "Enter a prompt" message
        initial_token_html = '<div style="color: gray; font-size: 0.9em;">Enter a prompt to see token count</div>'
        # Negative prompt has default value, show its token count
        if engine and engine.is_initialized:
            neg_token_info = engine.count_prompt_tokens(DEFAULT_NEGATIVE_PROMPT)
            initial_neg_token_html = format_token_count_html(neg_token_info, is_negative=True)
        else:
            initial_neg_token_html = '<div style="color: gray; font-size: 0.9em;">Engine ready but token count unavailable</div>'
    else:
        initial_token_html = '<div style="color: gray; font-size: 0.9em;">Token count will appear when engine is ready</div>'
        initial_neg_token_html = '<div style="color: gray; font-size: 0.9em;">Negative token count will appear when engine is ready</div>'

    resolution_options = [
        f"{w}x{h}{' (Optimal)' if (h, w) in RECOMMENDED_RESOLUTIONS else ''}"
        for h, w in OFFICIAL_RESOLUTIONS
    ]

    with gr.Blocks(
        title="NoobAI XL V-Pred 1.0 (Hash Consistency Edition)",
        theme=gr.themes.Soft(),
        css=CSS_STYLES,
        head=JAVASCRIPT_HEAD
    ) as demo:

        gr.HTML('<div class="title-text">🎯 NoobAI XL V-Pred 1.0 - Hash Consistency Edition</div>')

        with gr.Row():
            with gr.Column(scale=2):
                # Engine initialization
                with gr.Group():
                    gr.HTML("<h3>🚀 Engine Initialization</h3>")
                    model_path_input = gr.Textbox(label="Model Path", value=default_model_path)

                    with gr.Row():
                        enable_dora = gr.Checkbox(
                            label="🎯 Enable DoRA Adapter",
                            value=dora_ui_state['enable_dora_value'],
                            interactive=dora_ui_state['enable_dora_interactive'],
                            info=dora_ui_state['checkbox_info']
                        )
                        dora_refresh_btn = gr.Button("🔄 Refresh Adapters", size="sm")

                    dora_selection = gr.Dropdown(
                        label="DoRA Adapter Selection",
                        choices=dora_ui_state['dropdown_choices'],
                        value=dora_ui_state['dropdown_value'],
                        interactive=dora_ui_state['dropdown_interactive'],
                        info=dora_ui_state['info_message']
                    )

                    dora_path = gr.Textbox(
                        label="Manual DoRA Path (Override)",
                        value=default_dora_path,
                        placeholder="Optional: Specify custom path to override adapter selection above"
                    )

                    with gr.Row():
                        init_btn = gr.Button("Initialize Engine", variant="primary")
                        status_indicator = gr.Button(
                            "✅ Ready" if is_ready else "❌ Not Ready",
                            variant="secondary" if is_ready else "stop",
                            interactive=False
                        )
                    init_status_display = gr.Textbox(
                        label="Status",
                        value=init_status,
                        interactive=False,
                        elem_classes=["status-success" if is_ready else "status-error"]
                    )

                # Positive prompt formatter
                with gr.Group():
                    gr.HTML('<h3>🎨 Positive Prompt Formatter</h3><div style="margin-bottom: 10px; color: #666;">🔴 Danbooru | 🔵 E621</div>')

                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Group(elem_classes=["segment-container"]):
                                gr.HTML('<div class="segment-header">1️⃣ Quality Tags</div>')
                                prefix_text = gr.Textbox(value=DEFAULT_POSITIVE_PREFIX, lines=2)
                                prefix_reset_btn = gr.Button("🔄 Reset", size="sm")

                        with gr.Column(scale=1):
                            character_search, character_dropdown, character_text, character_clear_btn, character_randomize_btn, character_source_filter = create_search_ui("Character", 2)

                    with gr.Row():
                        with gr.Column(scale=1):
                            artist_search, artist_dropdown, artist_text, artist_clear_btn, artist_randomize_btn, artist_source_filter = create_search_ui("Artist", 3)

                        with gr.Column(scale=1):
                            with gr.Group(elem_classes=["segment-container"]):
                                gr.HTML('<div class="segment-header">4️⃣ Custom Tags</div>')
                                custom_text = gr.Textbox(placeholder="Additional tags...", lines=4)
                                custom_clear_btn = gr.Button("🧹 Clear", size="sm")

                    with gr.Group(elem_classes=["final-prompt-container"]):
                        gr.HTML('<div class="segment-header">🎯 Final Prompt</div>')
                        final_prompt = gr.Textbox(label="Positive Prompt", lines=4)
                        token_counter = gr.HTML(
                            value=initial_token_html,
                            elem_classes=["token-counter"]
                        )
                        with gr.Row():
                            compose_btn = gr.Button("🔄 Compose", variant="primary")
                            randomize_all_btn = gr.Button("🎲 Randomize All", variant="secondary")
                            clear_all_btn = gr.Button("🧹 Clear All", variant="secondary")

                # Negative prompt
                with gr.Group():
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        value=DEFAULT_NEGATIVE_PROMPT,
                        lines=3
                    )
                    negative_token_counter = gr.HTML(
                        value=initial_neg_token_html,
                        elem_classes=["token-counter"]
                    )
                    negative_reset_btn = gr.Button("🔄 Reset Negative", size="sm")

                # Resolution settings
                with gr.Group():
                    gr.HTML("<h4>📐 Resolution</h4>")
                    use_custom_resolution = gr.Checkbox(label="Use Custom Resolution", value=False)
                    resolution = gr.Dropdown(
                        label="Preset",
                        choices=resolution_options,
                        value="1216x832 (Optimal)",
                        visible=True
                    )
                    with gr.Row(visible=False) as custom_res_row:
                        custom_width = gr.Number(
                            label="Width",
                            value=OPTIMAL_SETTINGS['width'],
                            minimum=GEN_CONFIG.MIN_RESOLUTION,
                            maximum=GEN_CONFIG.MAX_RESOLUTION,
                            step=64,
                            info="Must be divisible by 8"
                        )
                        custom_height = gr.Number(
                            label="Height",
                            value=OPTIMAL_SETTINGS['height'],
                            minimum=GEN_CONFIG.MIN_RESOLUTION,
                            maximum=GEN_CONFIG.MAX_RESOLUTION,
                            step=64,
                            info="Must be divisible by 8"
                        )

                # Generation parameters
                with gr.Group():
                    gr.HTML("<h4>⚙️ Parameters</h4>")
                    with gr.Row():
                        with gr.Column():
                            cfg_scale = gr.Slider(
                                label="CFG Scale",
                                minimum=GEN_CONFIG.MIN_CFG_SCALE,
                                maximum=GEN_CONFIG.MAX_CFG_SCALE,
                                step=0.1,
                                value=OPTIMAL_SETTINGS['cfg_scale']
                            )
                            cfg_status = gr.HTML('<div style="color: green;">✅ Optimal</div>')

                        with gr.Column():
                            rescale_cfg = gr.Slider(
                                label="Rescale CFG",
                                minimum=GEN_CONFIG.MIN_RESCALE_CFG,
                                maximum=GEN_CONFIG.MAX_RESCALE_CFG,
                                step=0.05,
                                value=OPTIMAL_SETTINGS['rescale_cfg']
                            )
                            rescale_status = gr.HTML('<div style="color: green;">✅ Optimal</div>')

                        with gr.Column():
                            adapter_strength = gr.Slider(
                                label="🎯 Adapter Strength",
                                minimum=MODEL_CONFIG.MIN_ADAPTER_STRENGTH,
                                maximum=MODEL_CONFIG.MAX_ADAPTER_STRENGTH,
                                step=0.1,
                                value=OPTIMAL_SETTINGS['adapter_strength'],
                                visible=default_enable_dora,
                                info="Control DoRA adapter influence"
                            )
                            adapter_status = gr.HTML(
                                '<div style="color: green;">✅ Optimal</div>',
                                visible=default_enable_dora
                            )

                            dora_start_step = gr.Slider(
                                label="🚀 DoRA Start Step",
                                minimum=MODEL_CONFIG.MIN_DORA_START_STEP,
                                maximum=min(OPTIMAL_SETTINGS['steps'], MODEL_CONFIG.MAX_DORA_START_STEP),
                                step=1,
                                value=OPTIMAL_SETTINGS['dora_start_step'],
                                visible=default_enable_dora,
                                info="Step at which DoRA adapter activates (0 = first step)"
                            )
                            dora_start_step_status = gr.HTML(
                                '<div style="color: green;">✅ Start at step 0 (first step)</div>',
                                visible=default_enable_dora
                            )

                            dora_toggle_mode = gr.Radio(
                                label="🔄 DoRA Toggle Mode",
                                choices=[("None", None), ("Manual", "manual"), ("Optimized", "optimized")],
                                value=None,
                                visible=default_enable_dora,
                                info="Manual: Custom grid. Optimized: Pre-tuned 34-step schedule with locked settings."
                            )

                            dora_manual_grid = gr.HTML(value="", visible=False, label="Manual DoRA Schedule")

                            dora_manual_schedule_state = gr.Textbox(
                                value="",
                                visible=False,
                                label="Manual Schedule (CSV)",
                                elem_id="dora_manual_schedule_hidden",
                                interactive=True,
                                placeholder="e.g., 1, 0, 0, 1, 1, 0",
                                info="Comma-separated 0/1 values. Click grid cells above or edit manually."
                            )

                            # Hidden state to track programmatic changes (prevents circular event triggers)
                            is_programmatic_change = gr.State(value=False)

                    steps = gr.Slider(
                        label="Steps",
                        minimum=GEN_CONFIG.MIN_STEPS,
                        maximum=GEN_CONFIG.MAX_STEPS,
                        step=1,
                        value=OPTIMAL_SETTINGS['steps']
                    )
                    steps_status = gr.HTML('<div style="color: green;">✅ Optimal</div>')

                # Seed settings
                with gr.Group():
                    gr.HTML("<h4>🌱 Seed</h4>")
                    with gr.Row():
                        with gr.Column(scale=3):
                            seed = gr.Textbox(value=str(random.randint(0, 2**32-1)), label="Seed")
                            auto_randomize_seed = gr.Checkbox(
                                label="🔄 Ignore seed box and use random for next run",
                                value=True
                            )
                        with gr.Column(scale=1):
                            random_seed_btn = gr.Button("🎲 New Random Seed", size="lg")

                # ControlNet settings
                with gr.Group():
                    gr.HTML("<h4>🎛️ ControlNet (Pose Control)</h4>")
                    with gr.Row():
                        enable_controlnet = gr.Checkbox(
                            label="Enable ControlNet",
                            value=False,
                            info="Use pose image for anatomical control"
                        )
                        controlnet_refresh_btn = gr.Button("🔄 Refresh Models", size="sm")

                    controlnet_choices = get_controlnet_choices()
                    controlnet_default = get_default_controlnet()
                    has_controlnet_models = controlnet_choices and controlnet_choices[0] != "None"

                    controlnet_selection = gr.Dropdown(
                        label="ControlNet Model",
                        choices=controlnet_choices,
                        value=controlnet_default,
                        interactive=has_controlnet_models,
                        visible=False,
                        info="Select ControlNet model from /controlnet directory"
                    )

                    pose_image = gr.Image(
                        label="Pose Image (OpenPose skeleton)",
                        type="pil",
                        visible=False
                    )

                    controlnet_scale = gr.Slider(
                        label="ControlNet Scale",
                        minimum=CONTROLNET_CONFIG.MIN_CONDITIONING_SCALE,
                        maximum=CONTROLNET_CONFIG.MAX_CONDITIONING_SCALE,
                        step=0.05,
                        value=CONTROLNET_CONFIG.DEFAULT_CONDITIONING_SCALE,
                        visible=False,
                        info="Strength of pose conditioning. V-pred models need ~2.0 for proper pose adherence"
                    )

                # Generate button
                generate_btn = gr.Button(
                    "🎨 Generate Image" if is_ready else "❌ Initialize Engine First",
                    variant="primary" if is_ready else "stop",
                    size="lg",
                    interactive=is_ready
                )
                interrupt_btn = gr.Button("⏹️ Interrupt", variant="stop", size="sm", visible=False)

            # Output column
            with gr.Column(scale=2):
                with gr.Group():
                    gr.HTML("<h3>🖼️ Result</h3>")
                    output_image = gr.Image(type="filepath", interactive=False, height=400, format="png")
                    generation_info = gr.Textbox(label="Generation Info", lines=9, interactive=False)

        with gr.Row():
            reset_btn = gr.Button("🔄 Reset to Optimal", variant="secondary", size="sm")

        # Event Handlers
        def refresh_dora_adapters():
            """Refresh DoRA adapters and update UI."""
            dora_ui_state = get_dora_ui_state()
            current_engine = get_engine_safely()
            current_settings = None

            if current_engine is not None and current_engine.is_initialized:
                current_settings = {
                    'enable_dora': current_engine.enable_dora,
                }

            suggestion_msg = ""
            if current_settings and current_settings['enable_dora']:
                suggestion_msg = " (Re-initialize engine if switching adapters)"

            return (
                gr.update(
                    interactive=dora_ui_state['enable_dora_interactive'],
                    value=dora_ui_state['enable_dora_value'] if not current_settings else current_settings['enable_dora'],
                    info=dora_ui_state['checkbox_info']
                ),
                gr.update(
                    choices=dora_ui_state['dropdown_choices'],
                    value=dora_ui_state['dropdown_value'],
                    interactive=dora_ui_state['dropdown_interactive'],
                    info=dora_ui_state['info_message'] + suggestion_msg
                )
            )

        # Component lists
        all_prompt_inputs = [prefix_text, character_text, artist_text, custom_text]
        gen_inputs = [
            final_prompt, negative_prompt, resolution, cfg_scale, steps,
            rescale_cfg, seed, use_custom_resolution, custom_width,
            custom_height, auto_randomize_seed, adapter_strength, enable_dora,
            dora_start_step, dora_toggle_mode, dora_manual_schedule_state,
            enable_controlnet, controlnet_selection, pose_image, controlnet_scale
        ]
        gen_outputs = [output_image, generation_info, seed]

        # ControlNet UI handlers
        def update_controlnet_ui(enable_val):
            """Show/hide ControlNet options based on enable state."""
            return (
                gr.update(visible=enable_val),  # controlnet_selection
                gr.update(visible=enable_val),  # pose_image
                gr.update(visible=enable_val)   # controlnet_scale
            )

        enable_controlnet.change(
            update_controlnet_ui,
            inputs=[enable_controlnet],
            outputs=[controlnet_selection, pose_image, controlnet_scale]
        )

        def refresh_controlnet_models():
            """Refresh ControlNet model list."""
            choices = refresh_controlnet_dropdown()
            default = choices[0] if choices and choices[0] != "None" else "None"
            has_models = choices and choices[0] != "None"
            return gr.update(choices=choices, value=default, interactive=has_models)

        controlnet_refresh_btn.click(
            refresh_controlnet_models,
            outputs=[controlnet_selection]
        )

        def init_and_update(path, enable_dora_val, dora_path_val, dora_selection_val):
            """Initialize engine with UI feedback."""
            if get_engine_safely() is not None:
                teardown_status = "🔄 Performing comprehensive engine teardown..."
                yield (
                    teardown_status,
                    gr.update(value="🔄 Tearing Down", variant="stop"),
                    gr.update(value="🔄 Cleaning up...", variant="stop", interactive=False),
                    gr.update(elem_classes=["status-warning"])
                )

            status = initialize_engine(path, enable_dora_val, dora_path_val, dora_selection_val, force_fp32=force_fp32, optimize=optimize)
            ready = is_engine_ready()

            yield (
                status,
                gr.update(value="✅ Ready" if ready else "❌ Not Ready", variant="secondary" if ready else "stop"),
                gr.update(value="🎨 Generate Image" if ready else "❌ Initialize Engine First", variant="primary" if ready else "stop", interactive=ready),
                gr.update(elem_classes=["status-success" if ready else "status-error"])
            )

        # Wire up event handlers
        # Store init event to chain token counter updates later (after functions are defined)
        init_event = init_btn.click(
            init_and_update,
            inputs=[model_path_input, enable_dora, dora_path, dora_selection],
            outputs=[init_status_display, status_indicator, generate_btn, init_status_display]
        )

        dora_refresh_btn.click(refresh_dora_adapters, outputs=[enable_dora, dora_selection])

        # DoRA UI updates - also reset toggle mode and hide grid when disabling
        def update_dora_ui(enable_val):
            if enable_val:
                return (
                    gr.update(visible=True),   # adapter_strength
                    gr.update(visible=True),   # adapter_status
                    gr.update(visible=True),   # dora_start_step
                    gr.update(visible=True),   # dora_start_step_status
                    gr.update(visible=True),   # dora_toggle_mode
                    gr.update(visible=False),  # dora_manual_grid (hidden until mode selected)
                    gr.update(visible=False)   # dora_manual_schedule_state
                )
            else:
                return (
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False, value=None),  # Reset toggle mode to None
                    gr.update(visible=False),
                    gr.update(visible=False)
                )

        enable_dora.change(
            update_dora_ui,
            inputs=[enable_dora],
            outputs=[adapter_strength, adapter_status, dora_start_step, dora_start_step_status,
                     dora_toggle_mode, dora_manual_grid, dora_manual_schedule_state]
        )

        # DoRA toggle mode changes - comprehensive handler with state management
        def handle_toggle_mode_change(toggle_mode, num_steps, is_programmatic):
            """Handle DoRA toggle mode changes with full state management."""
            if is_programmatic:
                # Reset flag, no other changes
                return (gr.update(), gr.update(), gr.update(), gr.update(),
                        gr.update(), gr.update(), gr.update(), False)

            try:
                steps_int = max(int(num_steps), 1)
            except (TypeError, ValueError):
                steps_int = 1

            if toggle_mode == "optimized":
                opt = OPTIMIZED_DORA_SETTINGS
                grid_html = generate_dora_grid(opt['steps'], OPTIMIZED_DORA_SCHEDULE_CSV, show_locked_badge=True)
                return (
                    gr.update(visible=True, value=grid_html),           # dora_manual_grid
                    gr.update(value=OPTIMIZED_DORA_SCHEDULE_CSV, visible=False),  # dora_manual_schedule_state
                    gr.update(interactive=False, value=0),              # dora_start_step (disabled)
                    gr.update(value=opt['cfg_scale']),                  # cfg_scale
                    gr.update(value=opt['rescale_cfg']),                # rescale_cfg
                    gr.update(value=opt['adapter_strength']),           # adapter_strength
                    gr.update(value=opt['steps']),                      # steps
                    True                                                 # is_programmatic_change
                )
            elif toggle_mode == "manual":
                grid_html = generate_dora_grid(steps_int)
                return (
                    gr.update(visible=True, value=grid_html),
                    gr.update(visible=True),
                    gr.update(interactive=False),                        # disabled for manual too
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    False
                )
            else:  # None
                return (
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(interactive=True),                         # enabled for None
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    False
                )

        dora_toggle_mode.change(
            handle_toggle_mode_change,
            inputs=[dora_toggle_mode, steps, is_programmatic_change],
            outputs=[dora_manual_grid, dora_manual_schedule_state, dora_start_step,
                     cfg_scale, rescale_cfg, adapter_strength, steps, is_programmatic_change]
        )

        # Prompt composition (compose_btn.click is handled below with token counter update)
        prefix_reset_btn.click(lambda: DEFAULT_POSITIVE_PREFIX, outputs=[prefix_text])
        negative_reset_btn.click(lambda: DEFAULT_NEGATIVE_PROMPT, outputs=[negative_prompt])
        custom_clear_btn.click(create_clear_handler('text'), outputs=[custom_text])

        def clear_all_prompts():
            return "", "", "", "", ""

        clear_all_btn.click(clear_all_prompts, outputs=[prefix_text, character_text, artist_text, custom_text, final_prompt]).then(
            lambda: '<div style="color: gray; font-size: 0.9em;">Enter a prompt to see token count</div>',
            outputs=[token_counter],
            show_progress=False
        )

        # Connect search events with auto-compose
        connect_search_events('character', character_search, character_dropdown, character_text, character_clear_btn, character_randomize_btn, character_source_filter,
                              compose_fn=compose_final_prompt, compose_inputs=all_prompt_inputs, final_prompt_output=final_prompt)
        connect_search_events('artist', artist_search, artist_dropdown, artist_text, artist_clear_btn, artist_randomize_btn, artist_source_filter,
                              compose_fn=compose_final_prompt, compose_inputs=all_prompt_inputs, final_prompt_output=final_prompt)

        # Token counter update functions
        def update_token_counter(prompt_text: str) -> str:
            """Update the token counter display based on current prompt."""
            engine = get_engine_safely()
            if engine is None or not engine.is_initialized:
                return '<div style="color: gray; font-size: 0.9em;">Initialize engine for token count</div>'
            token_info = engine.count_prompt_tokens(prompt_text)
            return format_token_count_html(token_info)

        def update_negative_token_counter(negative_text: str) -> str:
            """Update the negative prompt token counter display."""
            engine = get_engine_safely()
            if engine is None or not engine.is_initialized:
                return '<div style="color: gray; font-size: 0.9em;">Initialize engine for token count</div>'
            token_info = engine.count_prompt_tokens(negative_text)
            return format_token_count_html(token_info, is_negative=True)

        # Wire up token counter to final prompt changes
        final_prompt.change(
            update_token_counter,
            inputs=[final_prompt],
            outputs=[token_counter],
            show_progress=False
        )

        # Wire up negative prompt token counter
        negative_prompt.change(
            update_negative_token_counter,
            inputs=[negative_prompt],
            outputs=[negative_token_counter],
            show_progress=False
        )

        # Update both token counters after engine initialization
        # (now that update functions are defined, chain them to init_event)
        init_event.then(
            update_token_counter,
            inputs=[final_prompt],
            outputs=[token_counter],
            show_progress=False
        ).then(
            update_negative_token_counter,
            inputs=[negative_prompt],
            outputs=[negative_token_counter],
            show_progress=False
        )

        # Also update token counter after compose button
        compose_btn.click(
            compose_final_prompt,
            inputs=all_prompt_inputs,
            outputs=[final_prompt]
        ).then(
            update_token_counter,
            inputs=[final_prompt],
            outputs=[token_counter],
            show_progress=False
        )

        # Randomize All button handler
        def randomize_all(char_filter, artist_filter):
            """Randomize both character and artist simultaneously."""
            char_value = get_random_value('character', char_filter if char_filter != 'all' else None)
            artist_value = get_random_value('artist', artist_filter if artist_filter != 'all' else None)
            return char_value, artist_value

        randomize_all_btn.click(
            randomize_all,
            inputs=[character_source_filter, artist_source_filter],
            outputs=[character_text, artist_text]
        ).then(
            compose_final_prompt,
            inputs=all_prompt_inputs,
            outputs=[final_prompt]
        ).then(
            update_token_counter,
            inputs=[final_prompt],
            outputs=[token_counter],
            show_progress=False
        )

        # Generation
        generate_btn.click(start_generation, outputs=[interrupt_btn, generate_btn]).then(
            generate_image_with_progress,
            inputs=gen_inputs,
            outputs=gen_outputs
        ).then(finish_generation, outputs=[interrupt_btn, generate_btn])

        interrupt_btn.click(interrupt_generation, outputs=[interrupt_btn, generate_btn])

        # Utility handlers
        random_seed_btn.click(lambda: str(random.randint(0, 2**32-1)), outputs=[seed])
        use_custom_resolution.change(
            lambda x: (gr.update(visible=not x), gr.update(visible=x)),
            inputs=[use_custom_resolution],
            outputs=[resolution, custom_res_row]
        )

        # Parameter change handler for detecting optimized mode modifications
        def handle_param_change_for_optimized(current_toggle_mode, is_programmatic, param_value, param_name,
                                               current_schedule, steps_val, cfg_val, rescale_val, adapter_val):
            """Check if parameter change should trigger mode switch.

            - Optimized → Manual: When user changes any parameter
            - Manual → Optimized: When all settings match Optimized preset exactly
            """
            status_updater = create_status_updater(param_name)
            status_html = status_updater(param_value)

            if is_programmatic:
                return status_html, gr.update(), gr.update(), False

            if current_toggle_mode == "optimized":
                # User manually changed parameter → switch to Manual, remove badge
                # Set is_programmatic=True to prevent handle_toggle_mode_change from overwriting the grid
                grid_html = generate_dora_grid(int(steps_val), current_schedule, show_locked_badge=False)
                return status_html, gr.update(value="manual"), gr.update(value=grid_html), True

            if current_toggle_mode == "manual":
                # Check if settings now match Optimized preset exactly
                # Use the CURRENT values (param_value is already reflected in the specific param input)
                if schedules_match_optimized(current_schedule, steps_val, cfg_val, rescale_val, adapter_val):
                    # Auto-switch back to Optimized
                    opt = OPTIMIZED_DORA_SETTINGS
                    grid_html = generate_dora_grid(opt['steps'], OPTIMIZED_DORA_SCHEDULE_CSV, show_locked_badge=True)
                    return status_html, gr.update(value="optimized"), gr.update(value=grid_html), True

            return status_html, gr.update(), gr.update(), False

        # DoRA start step change handler - auto-switches to None mode
        def handle_dora_start_step_change(start_step_value, current_toggle_mode):
            """Handle DoRA start step changes - auto-switch to None mode."""
            status_html = create_status_updater('dora_start_step')(start_step_value)

            if current_toggle_mode is not None:
                return (
                    status_html,
                    gr.update(value=None),           # Switch toggle mode to None
                    gr.update(visible=False),        # Hide manual grid
                    gr.update(visible=False)         # Hide manual schedule state
                )
            return status_html, gr.update(), gr.update(), gr.update()

        # Status updaters with optimized mode detection and auto-switch back
        cfg_scale.change(
            lambda v, m, p, sched, st, cfg, resc, adp: handle_param_change_for_optimized(m, p, v, 'cfg', sched, st, v, resc, adp),
            inputs=[cfg_scale, dora_toggle_mode, is_programmatic_change, dora_manual_schedule_state, steps, cfg_scale, rescale_cfg, adapter_strength],
            outputs=[cfg_status, dora_toggle_mode, dora_manual_grid, is_programmatic_change]
        )

        rescale_cfg.change(
            lambda v, m, p, sched, st, cfg, resc, adp: handle_param_change_for_optimized(m, p, v, 'rescale', sched, st, cfg, v, adp),
            inputs=[rescale_cfg, dora_toggle_mode, is_programmatic_change, dora_manual_schedule_state, steps, cfg_scale, rescale_cfg, adapter_strength],
            outputs=[rescale_status, dora_toggle_mode, dora_manual_grid, is_programmatic_change]
        )

        adapter_strength.change(
            lambda v, m, p, sched, st, cfg, resc, adp: handle_param_change_for_optimized(m, p, v, 'adapter', sched, st, cfg, resc, v),
            inputs=[adapter_strength, dora_toggle_mode, is_programmatic_change, dora_manual_schedule_state, steps, cfg_scale, rescale_cfg, adapter_strength],
            outputs=[adapter_status, dora_toggle_mode, dora_manual_grid, is_programmatic_change]
        )

        # DoRA start step with auto-switch to None
        dora_start_step.change(
            handle_dora_start_step_change,
            inputs=[dora_start_step, dora_toggle_mode],
            outputs=[dora_start_step_status, dora_toggle_mode, dora_manual_grid, dora_manual_schedule_state]
        )

        # Reset to optimal
        def reset_to_optimal():
            return (
                OPTIMAL_SETTINGS['cfg_scale'],
                OPTIMAL_SETTINGS['rescale_cfg'],
                OPTIMAL_SETTINGS['steps'],
                OPTIMAL_SETTINGS['adapter_strength'],
                OPTIMAL_SETTINGS['dora_start_step']
            )

        reset_btn.click(reset_to_optimal, outputs=[cfg_scale, rescale_cfg, steps, adapter_strength, dora_start_step])

        # Update manual grid when steps change, with optimized mode detection
        def handle_steps_change(num_steps, toggle_mode, is_programmatic, current_schedule, cfg_val, rescale_val, adapter_val):
            """Handle steps change with bidirectional mode switching.

            - Optimized → Manual: When user changes steps
            - Manual → Optimized: When all settings match Optimized preset exactly
            """
            status_html = create_status_updater('steps')(num_steps)

            try:
                steps_int = max(int(num_steps), 1)
            except (TypeError, ValueError):
                steps_int = 1

            if is_programmatic:
                if toggle_mode == "optimized":
                    grid_html = generate_dora_grid(OPTIMIZED_DORA_SETTINGS['steps'], OPTIMIZED_DORA_SCHEDULE_CSV, show_locked_badge=True)
                    return (status_html, gr.update(), False, gr.update(value=grid_html),
                            gr.update(value=OPTIMIZED_DORA_SCHEDULE_CSV), gr.update(maximum=steps_int))
                return status_html, gr.update(), False, gr.update(), gr.update(), gr.update(maximum=steps_int)

            if toggle_mode == "optimized":
                # User manually changed steps while in optimized mode -> switch to manual
                # Set is_programmatic=True to prevent handle_toggle_mode_change from overwriting the grid
                grid_html = generate_dora_grid(steps_int, OPTIMIZED_DORA_SCHEDULE_CSV, show_locked_badge=False)
                return (status_html, gr.update(value="manual"), True,
                        gr.update(value=grid_html, visible=True),
                        gr.update(value=OPTIMIZED_DORA_SCHEDULE_CSV, visible=True),
                        gr.update(maximum=steps_int, interactive=False))

            if toggle_mode == "manual":
                schedule, _ = parse_manual_dora_schedule(current_schedule, steps_int) if current_schedule else (None, None)
                schedule_csv = ", ".join(str(x) for x in schedule) if schedule else ", ".join("0" for _ in range(steps_int))

                # Check if settings now match Optimized preset exactly
                if schedules_match_optimized(schedule_csv, steps_int, cfg_val, rescale_val, adapter_val):
                    # Auto-switch back to Optimized
                    opt = OPTIMIZED_DORA_SETTINGS
                    grid_html = generate_dora_grid(opt['steps'], OPTIMIZED_DORA_SCHEDULE_CSV, show_locked_badge=True)
                    return (status_html, gr.update(value="optimized"), True, gr.update(value=grid_html),
                            gr.update(value=OPTIMIZED_DORA_SCHEDULE_CSV), gr.update(maximum=opt['steps']))

                grid_html = generate_dora_grid(steps_int, schedule_csv)
                return (status_html, gr.update(), False, gr.update(value=grid_html),
                        gr.update(value=schedule_csv), gr.update(maximum=steps_int))

            return status_html, gr.update(), False, gr.update(), gr.update(), gr.update(maximum=steps_int, interactive=True)

        steps.change(
            handle_steps_change,
            inputs=[steps, dora_toggle_mode, is_programmatic_change, dora_manual_schedule_state, cfg_scale, rescale_cfg, adapter_strength],
            outputs=[steps_status, dora_toggle_mode, is_programmatic_change, dora_manual_grid,
                     dora_manual_schedule_state, dora_start_step]
        )

        # Update grid when textbox is manually edited
        def update_grid_from_textbox(toggle_mode, num_steps, schedule_csv, cfg_val, rescale_val, adapter_val, is_programmatic):
            """Handle schedule changes with mode auto-switching.

            - If in Optimized mode and schedule changed (user clicked grid): Switch to Manual
            - If in Manual mode and all settings match Optimized: Switch back to Optimized
            """
            try:
                steps_int = max(int(num_steps), 1)
            except (TypeError, ValueError):
                steps_int = 1

            # If this is a programmatic change (from mode switch), just regenerate grid without mode switch
            if is_programmatic:
                if toggle_mode == "optimized":
                    grid_html = generate_dora_grid(OPTIMIZED_DORA_SETTINGS['steps'], schedule_csv, show_locked_badge=True)
                elif toggle_mode == "manual":
                    grid_html = generate_dora_grid(steps_int, schedule_csv)
                else:
                    return gr.update(), gr.update(), False
                return gr.update(value=grid_html), gr.update(), False

            # User clicked a grid cell - handle mode switching
            if toggle_mode == "optimized":
                # User modified grid in Optimized mode → switch to Manual
                # Use optimized steps count since that's what the grid currently shows
                opt_steps = OPTIMIZED_DORA_SETTINGS['steps']
                grid_html = generate_dora_grid(opt_steps, schedule_csv, show_locked_badge=False)
                # Set is_programmatic=True to prevent handle_toggle_mode_change from overwriting the grid
                return gr.update(value=grid_html), gr.update(value="manual"), True

            if toggle_mode == "manual":
                # Check if settings now match Optimized preset exactly
                if schedules_match_optimized(schedule_csv, steps_int, cfg_val, rescale_val, adapter_val):
                    # Auto-switch back to Optimized
                    opt = OPTIMIZED_DORA_SETTINGS
                    grid_html = generate_dora_grid(opt['steps'], OPTIMIZED_DORA_SCHEDULE_CSV, show_locked_badge=True)
                    return gr.update(value=grid_html), gr.update(value="optimized"), True
                else:
                    grid_html = generate_dora_grid(steps_int, schedule_csv)
                    return gr.update(value=grid_html), gr.update(), False

            return gr.update(), gr.update(), False

        dora_manual_schedule_state.change(
            update_grid_from_textbox,
            inputs=[dora_toggle_mode, steps, dora_manual_schedule_state, cfg_scale, rescale_cfg, adapter_strength, is_programmatic_change],
            outputs=[dora_manual_grid, dora_toggle_mode, is_programmatic_change]
        )

    return demo
