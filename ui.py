"""
NoobAI XL V-Pred 1.0 - Gradio UI

This module contains the Gradio interface creation and all event handlers.
"""

import random
import gradio as gr
from config import (
    OFFICIAL_RESOLUTIONS, RECOMMENDED_RESOLUTIONS, OPTIMAL_SETTINGS,
    GEN_CONFIG, MODEL_CONFIG, DEFAULT_POSITIVE_PREFIX, DEFAULT_NEGATIVE_PROMPT
)
from ui_helpers import (
    engine, auto_initialize, get_dora_ui_state, initialize_engine,
    create_search_ui, connect_search_events, create_clear_handler,
    create_status_updater, compose_final_prompt, start_generation,
    generate_image_with_progress, finish_generation, interrupt_generation
)

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_interface() -> gr.Blocks:
    """Create the Gradio interface."""
    init_status, default_model_path, default_enable_dora, default_dora_path, default_adapter_selection = auto_initialize()
    is_ready = engine is not None and engine.is_initialized

    # Get smart DoRA UI state
    dora_ui_state = get_dora_ui_state()

    resolution_options = [
        f"{w}x{h}{' (Optimal)' if (h, w) in RECOMMENDED_RESOLUTIONS else ''}"
        for h, w in OFFICIAL_RESOLUTIONS
    ]

    with gr.Blocks(
        title="NoobAI XL V-Pred 1.0 (Hash Consistency Edition)",
        theme=gr.themes.Soft(),
        css="""
        .title-text {
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .status-success {
            background: rgba(34, 197, 94, 0.1) !important;
            color: rgb(34, 197, 94) !important;
            border: 1px solid rgba(34, 197, 94, 0.3) !important;
        }
        .status-error {
            background: rgba(239, 68, 68, 0.1) !important;
            color: rgb(239, 68, 68) !important;
            border: 1px solid rgba(239, 68, 68, 0.3) !important;
        }
        .segment-container {
            border: 1px solid var(--border-color-primary);
            border-radius: 6px;
            padding: 10px;
            margin-bottom: 10px;
        }
        .segment-header {
            font-weight: bold;
            margin-bottom: 8px;
        }
        .final-prompt-container {
            background: var(--background-fill-secondary);
            border: 2px solid var(--border-color-accent);
            border-radius: 8px;
            padding: 15px;
        }
        """
    ) as demo:

        gr.HTML('<div class="title-text">🎯 NoobAI XL V-Pred 1.0 - Hash Consistency Edition</div>')

        with gr.Row():
            with gr.Column(scale=2):
                # Engine initialization
                with gr.Group():
                    gr.HTML("<h3>🚀 Engine Initialization</h3>")
                    model_path = gr.Textbox(
                        label="Model Path",
                        value=default_model_path
                    )

                    # DoRA controls with smart UI state
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
                        interactive=True,
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
                    gr.HTML(
                        '<h3>🎨 Positive Prompt Formatter</h3>'
                        '<div style="margin-bottom: 10px; color: #666;">🔴 Danbooru | 🔵 E621</div>'
                    )

                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Group(elem_classes=["segment-container"]):
                                gr.HTML('<div class="segment-header">1️⃣ Quality Tags</div>')
                                prefix_text = gr.Textbox(
                                    value=DEFAULT_POSITIVE_PREFIX,
                                    lines=2
                                )
                                prefix_reset_btn = gr.Button("🔄 Reset", size="sm")

                        with gr.Column(scale=1):
                            character_search, character_dropdown, character_text, character_clear_btn = create_search_ui("Character", 2)

                    with gr.Row():
                        with gr.Column(scale=1):
                            artist_search, artist_dropdown, artist_text, artist_clear_btn = create_search_ui("Artist", 3)

                        with gr.Column(scale=1):
                            with gr.Group(elem_classes=["segment-container"]):
                                gr.HTML('<div class="segment-header">4️⃣ Custom Tags</div>')
                                custom_text = gr.Textbox(
                                    placeholder="Additional tags...",
                                    lines=4
                                )
                                custom_clear_btn = gr.Button("🧹 Clear", size="sm")

                    with gr.Group(elem_classes=["final-prompt-container"]):
                        gr.HTML('<div class="segment-header">🎯 Final Prompt</div>')
                        final_prompt = gr.Textbox(label="Positive Prompt", lines=4)
                        with gr.Row():
                            compose_btn = gr.Button("🔄 Compose", variant="primary")
                            clear_all_btn = gr.Button("🧹 Clear All", variant="secondary")

                # Negative prompt
                with gr.Group():
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        value=DEFAULT_NEGATIVE_PROMPT,
                        lines=3
                    )
                    negative_reset_btn = gr.Button("🔄 Reset Negative", size="sm")

                # Resolution settings
                with gr.Group():
                    gr.HTML("<h4>📐 Resolution</h4>")
                    use_custom_resolution = gr.Checkbox(
                        label="Use Custom Resolution",
                        value=False
                    )
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
                            step=64
                        )
                        custom_height = gr.Number(
                            label="Height",
                            value=OPTIMAL_SETTINGS['height'],
                            minimum=GEN_CONFIG.MIN_RESOLUTION,
                            maximum=GEN_CONFIG.MAX_RESOLUTION,
                            step=64
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
                                info="Step at which DoRA adapter activates"
                            )
                            dora_start_step_status = gr.HTML(
                                '<div style="color: green;">✅ Start at step 1</div>',
                                visible=default_enable_dora
                            )

                            dora_toggle_mode = gr.Checkbox(
                                label="🔄 Enable DoRA Toggle Mode",
                                value=False,
                                visible=default_enable_dora,
                                info="Alternate ON,OFF,ON,OFF for improved anatomical accuracy"
                            )

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
                            seed = gr.Textbox(
                                value=str(random.randint(0, 2**32-1)),
                                label="Seed"
                            )
                            auto_randomize_seed = gr.Checkbox(
                                label="🔄 Ignore seed box and use random for next run",
                                value=True
                            )
                        with gr.Column(scale=1):
                            random_seed_btn = gr.Button("🎲 New Random Seed", size="lg")

                # Generate button
                generate_btn = gr.Button(
                    "🎨 Generate Image" if is_ready else "❌ Initialize Engine First",
                    variant="primary" if is_ready else "stop",
                    size="lg",
                    interactive=is_ready
                )
                interrupt_btn = gr.Button(
                    "⏹️ Interrupt",
                    variant="stop",
                    size="sm",
                    visible=False
                )

            # Output column
            with gr.Column(scale=2):
                with gr.Group():
                    gr.HTML("<h3>🖼️ Result</h3>")
                    output_image = gr.Image(
                        type="filepath",  # Changed from "pil" for hash consistency
                        interactive=False,
                        height=400,
                        format="png"
                    )
                    generation_info = gr.Textbox(
                        label="Generation Info",
                        lines=9,  # Increased to show hash
                        interactive=False
                    )

        with gr.Row():
            reset_btn = gr.Button("🔄 Reset to Optimal", variant="secondary", size="sm")

        # === Event Handlers ===

        # DoRA refresh handler with comprehensive UI updates
        def refresh_dora_adapters():
            """Enhanced refresh with conditional UI updates for both checkbox and dropdown."""
            from ui_helpers import engine

            # Get fresh adapter state
            dora_ui_state = get_dora_ui_state()

            # Get current engine settings if engine exists
            current_settings = None
            if engine is not None and engine.is_initialized:
                current_settings = {
                    'model_path': engine.model_path,
                    'enable_dora': engine.enable_dora,
                    'adapter_strength': engine.adapter_strength
                }

            # Add re-initialization suggestion if engine was using DoRA
            suggestion_msg = ""
            if current_settings and current_settings['enable_dora']:
                suggestion_msg = " (Re-initialize engine if switching adapters)"

            # Update both checkbox and dropdown based on adapter availability
            return (
                # Update checkbox state
                gr.update(
                    interactive=dora_ui_state['enable_dora_interactive'],
                    value=dora_ui_state['enable_dora_value'] if not current_settings else current_settings['enable_dora'],
                    info=dora_ui_state['checkbox_info']
                ),
                # Update dropdown state
                gr.update(
                    choices=dora_ui_state['dropdown_choices'],
                    value=dora_ui_state['dropdown_value'],
                    interactive=dora_ui_state['dropdown_interactive'],
                    info=dora_ui_state['info_message'] + suggestion_msg
                )
            )

        # Component lists for event handling
        all_prompt_inputs = [prefix_text, character_text, artist_text, custom_text]
        all_prompt_components = [
            prefix_text, character_search, character_text, character_dropdown,
            artist_search, artist_text, artist_dropdown, custom_text, final_prompt
        ]
        gen_inputs = [
            final_prompt, negative_prompt, resolution, cfg_scale, steps,
            rescale_cfg, seed, use_custom_resolution, custom_width,
            custom_height, auto_randomize_seed, adapter_strength, enable_dora, dora_start_step, dora_toggle_mode
        ]
        gen_outputs = [output_image, generation_info, seed]

        # Engine initialization
        def init_and_update(path, enable_dora_val, dora_path_val, dora_selection_val):
            """Enhanced initialization with teardown feedback."""
            from ui_helpers import engine

            # Provide teardown feedback if engine exists
            if engine is not None:
                # Show teardown progress
                teardown_status = "🔄 Performing comprehensive engine teardown..."
                yield (
                    teardown_status,
                    gr.update(value="🔄 Tearing Down", variant="stop"),
                    gr.update(value="🔄 Cleaning up...", variant="stop", interactive=False),
                    gr.update(elem_classes=["status-warning"])
                )

            # Perform initialization with comprehensive teardown
            status = initialize_engine(path, enable_dora_val, dora_path_val, dora_selection_val)
            from ui_helpers import engine
            ready = engine is not None and engine.is_initialized

            # Final status update
            final_status = (
                status,
                gr.update(
                    value="✅ Ready" if ready else "❌ Not Ready",
                    variant="secondary" if ready else "stop"
                ),
                gr.update(
                    value="🎨 Generate Image" if ready else "❌ Initialize...",
                    variant="primary" if ready else "stop",
                    interactive=ready
                ),
                gr.update(elem_classes=["status-success" if ready else "status-error"])
            )

            if engine is not None:
                yield final_status
            else:
                return final_status

        init_btn.click(
            init_and_update,
            inputs=[model_path, enable_dora, dora_path, dora_selection],
            outputs=[init_status_display, status_indicator, generate_btn, init_status_display]
        )

        # DoRA refresh handler with multiple outputs
        dora_refresh_btn.click(
            refresh_dora_adapters,
            outputs=[enable_dora, dora_selection]
        )

        # DoRA visibility toggle with feedback
        def toggle_dora_visibility(enabled):
            """Handle DoRA toggle with immediate feedback."""
            # Update adapter strength slider visibility
            adapter_strength_update = gr.update(visible=enabled)

            # Update DoRA start step slider visibility
            dora_start_step_update = gr.update(visible=enabled)
            dora_start_step_status_update = gr.update(visible=enabled)

            # Update DoRA toggle mode checkbox visibility
            dora_toggle_mode_update = gr.update(visible=enabled)

            # Provide status feedback with visibility and message
            if enabled:
                status_msg = '<div style="color: green;">🎯 DoRA will be enabled for next generation</div>'
            else:
                status_msg = '<div style="color: gray;">⚪ DoRA will be disabled for next generation</div>'

            adapter_status_update = gr.update(visible=enabled, value=status_msg)

            return adapter_strength_update, adapter_status_update, dora_start_step_update, dora_start_step_status_update, dora_toggle_mode_update

        enable_dora.change(
            toggle_dora_visibility,
            inputs=[enable_dora],
            outputs=[adapter_strength, adapter_status, dora_start_step, dora_start_step_status, dora_toggle_mode]
        )

        # Toggle mode handler - disable start step when toggle mode is active
        def handle_toggle_mode_change(toggle_enabled):
            """Disable DoRA start step when toggle mode is enabled."""
            if toggle_enabled:
                return (
                    gr.update(interactive=False, value=1),  # Reset and disable start step
                    gr.update(value='<div style="color: gray;">⚪ Disabled (toggle mode active)</div>')
                )
            else:
                return (
                    gr.update(interactive=True),
                    gr.update(value='<div style="color: green;">✅ Start at step 1</div>')
                )

        dora_toggle_mode.change(
            handle_toggle_mode_change,
            inputs=[dora_toggle_mode],
            outputs=[dora_start_step, dora_start_step_status]
        )

        # Search handlers
        connect_search_events(
            "character",
            character_search,
            character_dropdown,
            character_text,
            character_clear_btn,
        )
        connect_search_events(
            "artist", artist_search, artist_dropdown, artist_text, artist_clear_btn
        )

        # Prompt composition
        compose_btn.click(
            compose_final_prompt,
            inputs=all_prompt_inputs,
            outputs=[final_prompt],
            show_progress=False
        )

        # Reset/clear handlers
        prefix_reset_btn.click(
            lambda: DEFAULT_POSITIVE_PREFIX,
            outputs=[prefix_text],
            show_progress=False
        )
        negative_reset_btn.click(
            lambda: DEFAULT_NEGATIVE_PROMPT,
            outputs=[negative_prompt],
            show_progress=False
        )
        custom_clear_btn.click(
            create_clear_handler('text'),
            outputs=[custom_text],
            show_progress=False
        )

        clear_all_btn.click(
            lambda: (
                DEFAULT_POSITIVE_PREFIX, "", "", gr.update(choices=[], value=None),
                "", "", gr.update(choices=[], value=None), "", ""
            ),
            outputs=all_prompt_components,
            show_progress=False
        )

        # Generation handlers
        generate_btn.click(
            start_generation,
            outputs=[interrupt_btn, generate_btn]
        ).then(
            generate_image_with_progress,
            inputs=gen_inputs,
            outputs=gen_outputs
        ).then(
            finish_generation,
            outputs=[interrupt_btn, generate_btn]
        )

        interrupt_btn.click(
            interrupt_generation,
            outputs=[interrupt_btn, generate_btn]
        )

        # Seed management
        random_seed_btn.click(
            lambda: str(random.randint(0, 2**32-1)),
            outputs=[seed]
        )

        # Resolution toggle
        use_custom_resolution.change(
            lambda x: [gr.update(visible=not x), gr.update(visible=x)],
            inputs=[use_custom_resolution],
            outputs=[resolution, custom_res_row]
        )

        # Parameter status updates
        cfg_scale.change(
            create_status_updater('cfg'),
            inputs=[cfg_scale],
            outputs=[cfg_status]
        )
        steps.change(
            create_status_updater('steps'),
            inputs=[steps],
            outputs=[steps_status]
        )

        # Update DoRA start step maximum when steps change
        def update_dora_start_step_max(steps_value):
            return gr.update(maximum=steps_value)

        steps.change(
            update_dora_start_step_max,
            inputs=[steps],
            outputs=[dora_start_step]
        )
        rescale_cfg.change(
            create_status_updater('rescale'),
            inputs=[rescale_cfg],
            outputs=[rescale_status]
        )
        adapter_strength.change(
            create_status_updater('adapter'),
            inputs=[adapter_strength],
            outputs=[adapter_status]
        )
        dora_start_step.change(
            create_status_updater('dora_start_step'),
            inputs=[dora_start_step],
            outputs=[dora_start_step_status]
        )

        # Reset to optimal
        def reset_to_optimal():
            cfg_updater = create_status_updater('cfg')
            steps_updater = create_status_updater('steps')
            rescale_updater = create_status_updater('rescale')
            adapter_updater = create_status_updater('adapter')
            dora_start_step_updater = create_status_updater('dora_start_step')

            return (
                OPTIMAL_SETTINGS['cfg_scale'],
                OPTIMAL_SETTINGS['steps'],
                OPTIMAL_SETTINGS['rescale_cfg'],
                OPTIMAL_SETTINGS['adapter_strength'],
                OPTIMAL_SETTINGS['dora_start_step'],
                "1216x832 (Optimal)",
                False,
                OPTIMAL_SETTINGS['width'],
                OPTIMAL_SETTINGS['height'],
                False,  # Reset toggle mode to disabled
                cfg_updater(OPTIMAL_SETTINGS['cfg_scale']),
                steps_updater(OPTIMAL_SETTINGS['steps']),
                rescale_updater(OPTIMAL_SETTINGS['rescale_cfg']),
                adapter_updater(OPTIMAL_SETTINGS['adapter_strength']),
                dora_start_step_updater(OPTIMAL_SETTINGS['dora_start_step'])
            )

        reset_btn.click(
            reset_to_optimal,
            outputs=[
                cfg_scale, steps, rescale_cfg, adapter_strength, dora_start_step, resolution,
                use_custom_resolution, custom_width, custom_height,
                dora_toggle_mode,  # Add toggle mode reset
                cfg_status, steps_status, rescale_status, adapter_status, dora_start_step_status
            ]
        )

        return demo
