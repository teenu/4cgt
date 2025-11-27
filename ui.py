"""
NoobAI XL V-Pred 1.0 - Gradio UI

This module contains the Gradio interface creation and all event handlers.
"""

import random
import uuid
import gradio as gr
from config import (
    OFFICIAL_RESOLUTIONS, RECOMMENDED_RESOLUTIONS, OPTIMAL_SETTINGS,
    GEN_CONFIG, MODEL_CONFIG, DEFAULT_POSITIVE_PREFIX, DEFAULT_NEGATIVE_PROMPT
)
from ui_helpers import (
    is_engine_ready, auto_initialize, get_dora_ui_state, initialize_engine,
    create_search_ui, connect_search_events, create_clear_handler,
    create_status_updater, compose_final_prompt, start_generation,
    generate_image_with_progress, finish_generation, interrupt_generation
)

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_interface(model_path: str = None) -> gr.Blocks:
    """Create the Gradio interface.

    Args:
        model_path: Optional path to model file or directory. If provided, overrides auto-discovery.
    """
    init_status, default_model_path, default_enable_dora, default_dora_path, default_adapter_selection = auto_initialize(model_path)
    is_ready = is_engine_ready()

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
        .dora-grid-container {
            display: flex;
            flex-wrap: wrap;
            gap: 3px;
            padding: 10px;
            background: var(--background-fill-secondary);
            border-radius: 6px;
            margin-top: 10px;
        }
        .dora-cell {
            width: 18px;
            height: 18px;
            background: #dc2626 !important;
            border-radius: 2px;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .dora-cell.on {
            background: #16a34a !important;
        }
        .dora-cell:hover {
            transform: scale(1.1);
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        """,
        head="""
        <script>
        function setupDoraGridHandlers() {
            setTimeout(function() {
                const containers = document.querySelectorAll('[id*="dora-grid"][id$="-container"]');

                containers.forEach(function(container) {
                    if (container.hasAttribute('data-dora-initialized')) return;
                    container.setAttribute('data-dora-initialized', 'true');

                    container.addEventListener('click', function(e) {
                        const cell = e.target;
                        if (!cell.classList.contains('dora-cell')) return;

                        // Toggle class and style
                        cell.classList.toggle('on');

                        // Update schedule
                        const cells = container.querySelectorAll('.dora-cell');
                        const schedule = Array.from(cells).map(c =>
                            c.classList.contains('on') ? '1' : '0'
                        );
                        const scheduleCSV = schedule.join(', ');

                        // Update hidden textbox
                        const hiddenBox = document.getElementById('dora_manual_schedule_hidden');
                        if (hiddenBox) {
                            const input = hiddenBox.querySelector('textarea, input');
                            if (input) {
                                input.value = scheduleCSV;
                                input.dispatchEvent(new Event('input', { bubbles: true }));
                            }
                        }
                    });
                });
            }, 500);
        }

        // Setup on load
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', setupDoraGridHandlers);
        } else {
            setupDoraGridHandlers();
        }

        // Re-setup on mutations
        const observer = new MutationObserver(setupDoraGridHandlers);
        setTimeout(function() {
            observer.observe(document.body, { childList: true, subtree: true });
        }, 1000);
        </script>
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
                                info="Step at which DoRA adapter activates"
                            )
                            dora_start_step_status = gr.HTML(
                                '<div style="color: green;">✅ Start at step 1</div>',
                                visible=default_enable_dora
                            )

                            dora_toggle_mode = gr.Radio(
                                label="🔄 DoRA Toggle Mode",
                                choices=[("None", None), ("Standard", "standard"), ("Smart", "smart"), ("Manual", "manual")],
                                value=None,
                                visible=default_enable_dora,
                                info="Standard: ON,OFF throughout. Smart: ON,OFF to step 20, then ON. Manual: Custom grid"
                            )

                            # Manual DoRA schedule grid (GitHub contributions style)
                            dora_manual_grid = gr.HTML(
                                value="",
                                visible=False,
                                label="Manual DoRA Schedule"
                            )

                            # Visible textbox to store manual schedule as CSV string
                            # Shows the actual schedule and allows manual editing
                            dora_manual_schedule_state = gr.Textbox(
                                value="",
                                visible=False,
                                label="Manual Schedule (CSV)",
                                elem_id="dora_manual_schedule_hidden",
                                interactive=True,
                                placeholder="e.g., 1, 0, 0, 1, 1, 0",
                                info="Comma-separated 0/1 values. Click grid cells above or edit manually."
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
            """Refresh DoRA adapters and update UI."""
            from ui_helpers import get_engine_safely

            # Get fresh adapter state
            dora_ui_state = get_dora_ui_state()

            # Get current engine settings if engine exists (thread-safe)
            current_settings = None
            current_engine = get_engine_safely()
            if current_engine is not None and current_engine.is_initialized:
                current_settings = {
                    'model_path': current_engine.model_path,
                    'enable_dora': current_engine.enable_dora,
                    'adapter_strength': current_engine.adapter_strength
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
            custom_height, auto_randomize_seed, adapter_strength, enable_dora, dora_start_step, dora_toggle_mode, dora_manual_schedule_state
        ]
        gen_outputs = [output_image, generation_info, seed]

        # Engine initialization
        def init_and_update(path, enable_dora_val, dora_path_val, dora_selection_val):
            """Initialize engine with UI feedback."""
            from ui_helpers import get_engine_safely, is_engine_ready

            # Provide teardown feedback if engine exists (thread-safe check)
            if get_engine_safely() is not None:
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
            ready = is_engine_ready()

            # Final status update - always yield for consistent generator behavior
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

            # Always yield to maintain generator semantics
            yield final_status

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

        # Generate DoRA grid HTML with proper event handling
        def generate_dora_grid(num_steps, schedule_csv=""):
            """Generate GitHub-style grid HTML for manual DoRA scheduling."""
            from utils import parse_manual_dora_schedule

            # Gradio sliders emit floats; normalize and guard against zero/negatives.
            try:
                steps_int = max(int(num_steps), 1)
            except (TypeError, ValueError):
                steps_int = 1

            # Parse existing schedule or create default (all OFF)
            if schedule_csv:
                schedule, _ = parse_manual_dora_schedule(schedule_csv, steps_int)
                if schedule is None:
                    schedule = [0] * steps_int
            else:
                schedule = [0] * steps_int

            # Build grid HTML (20 cells per row)
            cells_html = []
            for i, state in enumerate(schedule):
                cell_class = "dora-cell on" if state == 1 else "dora-cell"
                cells_html.append(f'<div class="{cell_class}" data-index="{i}"></div>')

            # Unique ID for this grid instance
            grid_id = f"dora-grid-{steps_int}-{uuid.uuid4().hex[:8]}"

            grid_html = f'''
            <div id="{grid_id}" style="margin-top: 10px;">
                <div style="font-weight: bold; margin-bottom: 5px;">Manual DoRA Schedule ({steps_int} steps, 20 per row)</div>
                <div style="font-size: 12px; color: gray; margin-bottom: 8px;">
                    Click cells to toggle: <span style="color: #16a34a;">■</span> ON (DoRA enabled) |
                    <span style="color: #dc2626;">■</span> OFF (DoRA disabled)
                </div>
                <div class="dora-grid-container" id="{grid_id}-container" style="max-width: 100%; display: grid; grid-template-columns: repeat(20, 18px); gap: 3px;">
                    {"".join(cells_html)}
                </div>
            </div>
            '''
            return grid_html

        # DoRA visibility toggle with feedback and grid synchronization
        def toggle_dora_visibility(enabled, toggle_mode, num_steps, current_schedule):
            """Handle DoRA toggle with immediate feedback and grid synchronization."""
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

            # Handle grid visibility based on enable_dora and toggle_mode
            # This ensures grid state is consistent when enable_dora is toggled
            if not enabled:
                # Hide grid and schedule when DoRA is disabled
                grid_update = gr.update(visible=False)
                schedule_update = gr.update(visible=False)
            elif toggle_mode == "manual":
                # Show grid when DoRA enabled and manual mode selected
                try:
                    steps_int = max(int(num_steps), 1)
                except (TypeError, ValueError):
                    steps_int = 1
                grid_html = generate_dora_grid(steps_int, current_schedule)
                # Initialize schedule if empty
                if not current_schedule or not current_schedule.strip():
                    current_schedule = ", ".join("0" for _ in range(steps_int))
                grid_update = gr.update(visible=True, value=grid_html)
                schedule_update = gr.update(visible=True, value=current_schedule)
            else:
                # Hide grid for non-manual modes (standard, smart, or None)
                grid_update = gr.update(visible=False)
                schedule_update = gr.update(visible=False)

            return (adapter_strength_update, adapter_status_update, dora_start_step_update,
                    dora_start_step_status_update, dora_toggle_mode_update, grid_update, schedule_update)

        enable_dora.change(
            toggle_dora_visibility,
            inputs=[enable_dora, dora_toggle_mode, steps, dora_manual_schedule_state],
            outputs=[adapter_strength, adapter_status, dora_start_step, dora_start_step_status,
                     dora_toggle_mode, dora_manual_grid, dora_manual_schedule_state]
        )

        # Toggle mode handler - disable start step when any toggle mode is active
        def handle_toggle_mode_change(toggle_mode, num_steps, current_schedule):
            """Handle DoRA toggle mode changes including manual grid visibility."""
            from utils import generate_standard_schedule, generate_smart_schedule

            try:
                steps_int = max(int(num_steps), 1)
            except (TypeError, ValueError):
                steps_int = 1

            if toggle_mode == "manual":
                # Show grid AND textbox, disable start step
                grid_html = generate_dora_grid(steps_int, current_schedule)
                # If no current schedule, initialize with all OFF
                if not current_schedule or not current_schedule.strip():
                    current_schedule = ", ".join("0" for _ in range(steps_int))
                return (
                    gr.update(interactive=False, value=1),  # Reset and disable start step
                    gr.update(value='<div style="color: gray;">⚪ Disabled (Manual toggle active)</div>'),
                    gr.update(visible=True, value=grid_html),  # Show grid
                    gr.update(visible=True, value=current_schedule)  # Show and update textbox
                )
            elif toggle_mode == "standard":
                # Hide grid and textbox, auto-populate with standard pattern
                schedule = generate_standard_schedule(steps_int)
                schedule_csv = ", ".join(str(x) for x in schedule)
                return (
                    gr.update(interactive=False, value=1),
                    gr.update(value='<div style="color: gray;">⚪ Disabled (Standard toggle active)</div>'),
                    gr.update(visible=False),  # Hide grid
                    gr.update(visible=False, value=schedule_csv)  # Hide textbox but set value
                )
            elif toggle_mode == "smart":
                # Hide grid and textbox, auto-populate with smart pattern
                schedule = generate_smart_schedule(steps_int)
                schedule_csv = ", ".join(str(x) for x in schedule)
                return (
                    gr.update(interactive=False, value=1),
                    gr.update(value='<div style="color: gray;">⚪ Disabled (Smart toggle active)</div>'),
                    gr.update(visible=False),  # Hide grid
                    gr.update(visible=False, value=schedule_csv)  # Hide textbox but set value
                )
            else:  # None selected
                return (
                    gr.update(interactive=True),
                    gr.update(value='<div style="color: green;">✅ Start at step 1</div>'),
                    gr.update(visible=False),  # Hide grid
                    gr.update(visible=False, value="")  # Hide textbox and clear
                )

        dora_toggle_mode.change(
            handle_toggle_mode_change,
            inputs=[dora_toggle_mode, steps, dora_manual_schedule_state],
            outputs=[dora_start_step, dora_start_step_status, dora_manual_grid, dora_manual_schedule_state]
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

        # Update manual grid when steps change (if in manual mode)
        def update_manual_grid_on_steps_change(toggle_mode, num_steps, current_schedule):
            """Update the manual DoRA grid when steps slider changes."""
            try:
                steps_int = max(int(num_steps), 1)
            except (TypeError, ValueError):
                steps_int = 1

            if toggle_mode == "manual":
                # Regenerate grid with new step count
                grid_html = generate_dora_grid(steps_int, current_schedule)
                # Parse and extend/truncate schedule
                from utils import parse_manual_dora_schedule
                if current_schedule:
                    schedule, _ = parse_manual_dora_schedule(current_schedule, steps_int)
                    if schedule:
                        schedule_csv = ", ".join(str(x) for x in schedule)
                    else:
                        schedule_csv = ", ".join("0" for _ in range(steps_int))
                else:
                    schedule_csv = ", ".join("0" for _ in range(steps_int))
                return gr.update(value=grid_html), gr.update(value=schedule_csv)
            elif toggle_mode == "standard":
                # Regenerate standard schedule for new step count
                from utils import generate_standard_schedule
                schedule = generate_standard_schedule(steps_int)
                schedule_csv = ", ".join(str(x) for x in schedule)
                return gr.update(), gr.update(value=schedule_csv)
            elif toggle_mode == "smart":
                # Regenerate smart schedule for new step count
                from utils import generate_smart_schedule
                schedule = generate_smart_schedule(steps_int)
                schedule_csv = ", ".join(str(x) for x in schedule)
                return gr.update(), gr.update(value=schedule_csv)
            else:
                # Not in any toggle mode - no update needed
                return gr.update(), gr.update()

        steps.change(
            update_manual_grid_on_steps_change,
            inputs=[dora_toggle_mode, steps, dora_manual_schedule_state],
            outputs=[dora_manual_grid, dora_manual_schedule_state]
        )

        # Update grid when textbox is manually edited (only in manual mode)
        def update_grid_from_textbox(toggle_mode, num_steps, schedule_csv):
            """Regenerate grid when user manually edits the CSV textbox."""
            if toggle_mode == "manual":
                try:
                    steps_int = max(int(num_steps), 1)
                except (TypeError, ValueError):
                    steps_int = 1
                grid_html = generate_dora_grid(steps_int, schedule_csv)
                return gr.update(value=grid_html)
            return gr.update()

        dora_manual_schedule_state.change(
            update_grid_from_textbox,
            inputs=[dora_toggle_mode, steps, dora_manual_schedule_state],
            outputs=[dora_manual_grid]
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
                None,  # Reset toggle mode to None
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
