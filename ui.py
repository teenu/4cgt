"""
NoobAI XL V-Pred 1.0 - Gradio UI

This module contains the Gradio interface creation and all event handlers.
"""

import random
import uuid
from dataclasses import dataclass
import gradio as gr
from config import (
    OFFICIAL_RESOLUTIONS, RECOMMENDED_RESOLUTIONS, OPTIMAL_SETTINGS,
    GEN_CONFIG, MODEL_CONFIG, DEFAULT_POSITIVE_PREFIX, DEFAULT_NEGATIVE_PROMPT
)
from ui_helpers import (
    is_engine_ready, auto_initialize, get_dora_ui_state, initialize_engine,
    create_search_ui, connect_search_events, create_clear_handler,
    create_status_updater, compose_final_prompt, start_generation,
    generate_image_with_progress, finish_generation, interrupt_generation,
    # Queue and gallery handlers
    add_to_queue, remove_from_queue, clear_queue, set_auto_process,
    render_queue_html, get_queue_status_html, clear_gallery, select_gallery_image,
    get_gallery_count_html, finish_generation_with_gallery, process_next_queue_item,
    trigger_queue_generation, conditional_queue_start
)
from config import QUEUE_CONFIG


@dataclass
class QueueUIComponents:
    """Structured references to queue-related UI elements."""

    add_button: gr.Button
    status: gr.HTML
    auto_process: gr.Checkbox
    clear_button: gr.Button
    display: gr.HTML
    remove_input: gr.Textbox
    trigger_input: gr.Textbox


@dataclass
class GalleryUIComponents:
    """Structured references to gallery-related UI elements."""

    carousel: gr.Gallery
    count: gr.HTML
    clear_button: gr.Button


def _build_queue_section() -> QueueUIComponents:
    """Create the queue management UI section."""

    with gr.Group():
        gr.HTML("<h4>📋 Generation Queue</h4>")
        with gr.Row():
            add_button = gr.Button(
                "➕ Add to Queue",
                variant="secondary",
                size="sm",
                scale=2
            )
            status = gr.HTML(
                f'<span style="color: gray;">Queue: 0/{QUEUE_CONFIG.MAX_QUEUE_SIZE}</span>'
            )
        with gr.Row():
            auto_process = gr.Checkbox(
                label="Auto-process queue",
                value=True,
                scale=2
            )
            clear_button = gr.Button("🗑️ Clear", size="sm", scale=1)
        display = gr.HTML(
            '<div class="queue-empty">Queue is empty - add items with "Add to Queue"</div>'
        )
        with gr.Column(elem_classes=["hidden-input-wrapper"]):
            remove_input = gr.Textbox(
                value="",
                show_label=False,
                container=False,
                elem_id="queue_command_input",
            )
        with gr.Column(elem_classes=["hidden-input-wrapper"]):
            trigger_input = gr.Textbox(
                value="",
                show_label=False,
                container=False,
                elem_id="queue_trigger_input",
            )

    return QueueUIComponents(
        add_button=add_button,
        status=status,
        auto_process=auto_process,
        clear_button=clear_button,
        display=display,
        remove_input=remove_input,
        trigger_input=trigger_input,
    )


def _build_gallery_section() -> GalleryUIComponents:
    """Create the gallery UI section."""

    with gr.Group():
        gr.HTML("<h3>🖼️ Session Gallery</h3>")
        carousel = gr.Gallery(
            value=[],
            columns=4,
            rows=None,
            height="auto",
            object_fit="contain",
            show_label=False,
            allow_preview=True,
            preview=True,
            elem_id="session-gallery",
            elem_classes=["session-gallery-container"],
        )
        with gr.Row():
            count = gr.HTML(
                f'<span style="color: gray;">0/{QUEUE_CONFIG.MAX_GALLERY_SIZE} images</span>'
            )
            clear_button = gr.Button("🗑️ Clear Gallery", size="sm", variant="secondary")

    return GalleryUIComponents(
        carousel=carousel,
        count=count,
        clear_button=clear_button,
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

        /* Queue System Styles */
        .queue-container {
            max-height: 250px;
            overflow-y: auto;
            padding: 5px;
        }
        .queue-card {
            background: var(--background-fill-secondary);
            border: 1px solid var(--border-color-primary);
            border-radius: 6px;
            padding: 8px 12px;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.15s ease;
        }
        .queue-card:hover {
            background: var(--background-fill-primary);
            border-color: var(--border-color-accent);
        }
        .queue-card-content {
            flex: 1;
            overflow: hidden;
            margin-right: 10px;
        }
        .queue-card-position {
            font-weight: bold;
            color: var(--color-accent);
            font-size: 12px;
            margin-bottom: 2px;
        }
        .queue-card-snippet {
            font-size: 13px;
            color: var(--body-text-color);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .queue-card-meta {
            font-size: 11px;
            color: var(--body-text-color-subdued);
            margin-top: 2px;
        }
        .queue-card-remove {
            padding: 4px 8px;
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 4px;
            color: rgb(239, 68, 68);
            cursor: pointer;
            font-size: 12px;
            flex-shrink: 0;
        }
        .queue-card-remove:hover {
            background: rgba(239, 68, 68, 0.2);
        }
        .queue-empty {
            text-align: center;
            color: var(--body-text-color-subdued);
            padding: 15px;
            font-style: italic;
        }

        /* Hidden inputs - keep in DOM but visually hidden for JS interaction */
        .hidden-input-wrapper {
            position: absolute !important;
            left: -10000px !important;
            width: 1px !important;
            height: 1px !important;
            overflow: hidden !important;
            opacity: 0 !important;
            pointer-events: none !important;
        }

        /* Session Gallery Styles - Optimized for 1024x1024 images scaled to ~256px thumbnails */
        .session-gallery-container {
            margin-top: 10px;
            max-height: 580px;  /* Fits ~2 rows of 256px thumbnails + padding */
            overflow-y: auto;
            overflow-x: hidden;
            border: 1px solid var(--border-color-primary);
            border-radius: 8px;
            padding: 10px;
            background: var(--background-fill-secondary);
        }
        .session-gallery-container .gallery {
            display: grid !important;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 12px;
        }
        .session-gallery-container .gallery > div {
            aspect-ratio: 1;
            border-radius: 6px;
            overflow: hidden;
            border: 2px solid transparent;
            transition: all 0.2s ease;
            cursor: pointer;
        }
        .session-gallery-container .gallery > div:hover {
            border-color: var(--color-accent);
            transform: scale(1.02);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .session-gallery-container .gallery > div.selected {
            border-color: var(--color-accent);
            box-shadow: 0 0 0 3px rgba(var(--color-accent-rgb), 0.3);
        }
        .session-gallery-container img {
            width: 100%;
            height: 100%;
            object-fit: contain !important;  /* Scale without cropping */
            background: var(--background-fill-primary);
        }
        /* Scrollbar styling */
        .session-gallery-container::-webkit-scrollbar {
            width: 8px;
        }
        .session-gallery-container::-webkit-scrollbar-track {
            background: var(--background-fill-secondary);
            border-radius: 4px;
        }
        .session-gallery-container::-webkit-scrollbar-thumb {
            background: var(--border-color-primary);
            border-radius: 4px;
        }
        .session-gallery-container::-webkit-scrollbar-thumb:hover {
            background: var(--border-color-accent);
        }
        """,
        head="""
        <script>
        // Store observers for cleanup
        window.noobaiObservers = window.noobaiObservers || [];
        
        // Queue continuation flag - set by Python via hidden element
        window.noobaiQueueContinue = false;

        function setupDoraGridHandlers() {
            // Use requestAnimationFrame with retry logic instead of fixed timeout
            function trySetup(attempts) {
                const containers = document.querySelectorAll('[id*="dora-grid"][id$="-container"]');

                if (containers.length === 0 && attempts < 10) {
                    requestAnimationFrame(function() { trySetup(attempts + 1); });
                    return;
                }

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
                        const schedule = Array.from(cells).map(function(c) {
                            return c.classList.contains('on') ? '1' : '0';
                        });
                        const scheduleCSV = schedule.join(', ');

                        // Update hidden textbox using Gradio-compatible method
                        const hiddenBox = document.getElementById('dora_manual_schedule_hidden');
                        if (hiddenBox) {
                            triggerGradioInput(hiddenBox, scheduleCSV);
                        }
                    });
                });
            }
            requestAnimationFrame(function() { trySetup(0); });
        }

        // Setup on load
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', setupDoraGridHandlers);
        } else {
            setupDoraGridHandlers();
        }

        // Re-setup on mutations with stored observer for cleanup
        var doraObserver = new MutationObserver(setupDoraGridHandlers);
        window.noobaiObservers.push(doraObserver);
        setTimeout(function() {
            doraObserver.observe(document.body, { childList: true, subtree: true });
        }, 1000);

        // Robust Gradio input trigger - works with Gradio 4.x
        function triggerGradioInput(container, value) {
            if (!container) return false;
            
            // Find input element (textarea or input)
            let input = container.querySelector('textarea') || 
                        container.querySelector('input[type="text"]') || 
                        container.querySelector('input');
            
            if (!input) {
                console.error('[NoobAI] No input found in container:', container.id);
                return false;
            }
            
            // Store original value to detect if change is needed
            const originalValue = input.value;
            
            // Set value using native setter to bypass any framework proxies
            const descriptor = Object.getOwnPropertyDescriptor(
                input.tagName.toLowerCase() === 'textarea' ? 
                    HTMLTextAreaElement.prototype : HTMLInputElement.prototype,
                'value'
            );
            
            if (descriptor && descriptor.set) {
                descriptor.set.call(input, value);
            } else {
                input.value = value;
            }
            
            // Dispatch events that Gradio listens to
            // Order matters: input first, then change
            input.dispatchEvent(new InputEvent('input', {
                bubbles: true,
                cancelable: true,
                inputType: 'insertText',
                data: value
            }));
            
            // Small delay before change event
            setTimeout(function() {
                input.dispatchEvent(new Event('change', { bubbles: true }));
                
                // Gradio 4.x sometimes needs blur to commit
                input.dispatchEvent(new Event('blur', { bubbles: true }));
            }, 10);
            
            return true;
        }

        // Queue item removal handler - robust implementation for Gradio 4.x
        function removeQueueItem(itemId) {
            if (!itemId) {
                console.error('[NoobAI Queue] No item ID provided');
                return;
            }

            console.log('[NoobAI Queue] Removing item:', itemId);

            // Find the hidden textbox for queue commands
            const cmdBox = document.getElementById('queue_command_input');
            if (!cmdBox) {
                console.error('[NoobAI Queue] Command input box not found');
                return;
            }

            // Use robust trigger function
            const success = triggerGradioInput(cmdBox, itemId);
            
            if (!success) {
                console.error('[NoobAI Queue] Failed to trigger input for removal');
            }
        }

        // Use event delegation for queue remove buttons (handles dynamically created elements)
        document.addEventListener('click', function(e) {
            // Check if clicked element is a queue remove button
            const btn = e.target.closest('.queue-card-remove');
            if (btn) {
                // Get the item ID from the parent queue card's data attribute
                const card = btn.closest('.queue-card');
                if (card && card.dataset.id) {
                    e.preventDefault();
                    e.stopPropagation();
                    removeQueueItem(card.dataset.id);
                }
            }
        });

        // Queue auto-processing: Click generate button when signaled
        // This is triggered by observing a hidden element that Python updates
        function setupQueueAutoProcess() {
            // Find the queue trigger element
            const triggerBox = document.getElementById('queue_trigger_input');
            if (!triggerBox) {
                // Retry if not found yet
                setTimeout(setupQueueAutoProcess, 500);
                return;
            }
            
            // Create a MutationObserver to watch for value changes
            const observer = new MutationObserver(function(mutations) {
                checkQueueTrigger();
            });
            
            // Also poll periodically as a fallback
            setInterval(checkQueueTrigger, 200);
            
            console.log('[NoobAI Queue] Auto-process observer initialized');
        }
        
        function checkQueueTrigger() {
            const triggerBox = document.getElementById('queue_trigger_input');
            if (!triggerBox) return;
            
            const input = triggerBox.querySelector('textarea') || 
                          triggerBox.querySelector('input');
            if (!input) return;
            
            const value = input.value.trim();
            
            if (value === 'trigger') {
                console.log('[NoobAI Queue] Trigger detected - clicking generate button');
                
                // Clear the trigger first to prevent re-triggering
                if (Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')) {
                    const setter = Object.getOwnPropertyDescriptor(
                        input.tagName.toLowerCase() === 'textarea' ? 
                            HTMLTextAreaElement.prototype : HTMLInputElement.prototype,
                        'value'
                    ).set;
                    setter.call(input, '');
                } else {
                    input.value = '';
                }
                
                // Find and click the generate button
                // Look for the button by its text content
                const buttons = document.querySelectorAll('button');
                for (const btn of buttons) {
                    const text = btn.textContent || btn.innerText || '';
                    if (text.includes('Generate Image') && !btn.disabled) {
                        console.log('[NoobAI Queue] Clicking generate button');
                        btn.click();
                        return;
                    }
                }
                
                // Fallback: look for primary variant button in generation section
                const primaryBtns = document.querySelectorAll('button.primary, button[class*="primary"]');
                for (const btn of primaryBtns) {
                    if (!btn.disabled && btn.offsetParent !== null) {
                        const text = btn.textContent || '';
                        if (text.includes('Generate') || text.includes('🎨')) {
                            console.log('[NoobAI Queue] Clicking primary button (fallback)');
                            btn.click();
                            return;
                        }
                    }
                }
                
                console.warn('[NoobAI Queue] Generate button not found or disabled');
            }
        }
        
        // Initialize queue auto-processing after DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', setupQueueAutoProcess);
        } else {
            setTimeout(setupQueueAutoProcess, 1000);
        }

        // Cleanup observers on page unload to prevent memory leaks
        window.addEventListener('beforeunload', function() {
            if (window.noobaiObservers) {
                window.noobaiObservers.forEach(function(obs) {
                    try { obs.disconnect(); } catch(e) {}
                });
                window.noobaiObservers = [];
            }
        });
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
                                interactive=False,  # Disabled by default when optimized mode is active
                                info="Step at which DoRA adapter activates"
                            )
                            dora_start_step_status = gr.HTML(
                                '<div style="color: green;">✅ Optimized: Using validated parameters (CFG=4.2, Steps=34)</div>',
                                visible=default_enable_dora
                            )

                            dora_toggle_mode = gr.Radio(
                                label="🔄 DoRA Toggle Mode",
                                choices=[("None", None), ("Standard", "standard"), ("Smart", "smart"), ("Optimized", "optimized"), ("Manual", "manual")],
                                value="optimized",
                                visible=default_enable_dora,
                                info="Optimized: Empirically-tuned phased activation (recommended). Standard: ON,OFF throughout. Smart: ON,OFF to step 20, then ON. Manual: Custom grid"
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
                                label="Seed (for next generation)",
                                info="Edit this while generation is in progress to queue a different seed"
                            )
                            auto_randomize_seed = gr.Checkbox(
                                label="🔄 Ignore seed box and use random for next run",
                                value=True
                            )
                        with gr.Column(scale=1):
                            random_seed_btn = gr.Button("🎲 New Random Seed", size="lg")
                    # Separate display for last used seed (updated after generation completes)
                    last_seed_display = gr.Textbox(
                        value="(none yet)",
                        label="Last Generated Seed",
                        interactive=False,
                        visible=True,
                        info="The seed used in the most recent generation"
                    )

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

                queue_ui = _build_queue_section()

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

                # Session Gallery - auto-scaling thumbnails with scroll support
                gallery_ui = _build_gallery_section()

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
        # Note: Output seed to last_seed_display (not the input seed textbox)
        # This allows users to edit the seed input while generation is in progress
        gen_outputs = [output_image, generation_info, last_seed_display]

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

        # Toggle mode handler - show status when any toggle mode is active (start step stays interactive for auto-switch)
        def handle_toggle_mode_change(toggle_mode, num_steps, current_schedule):
            """Handle DoRA toggle mode changes including manual grid visibility.

            Note: Start step slider remains interactive so users can change it,
            which will auto-switch toggle mode to None (see handle_dora_start_step_change).

            When "optimized" mode is selected, empirically validated parameters are auto-applied:
            - CFG scale: 4.2
            - Rescale CFG: 0.55
            - Adapter strength: 1.0
            - Steps: 34
            - DoRA start step: disabled (greyed out)
            """
            from utils import generate_standard_schedule, generate_smart_schedule, generate_optimized_schedule

            try:
                steps_int = max(int(num_steps), 1)
            except (TypeError, ValueError):
                steps_int = 1

            if toggle_mode == "manual":
                # Show grid AND textbox, show override hint
                grid_html = generate_dora_grid(steps_int, current_schedule)
                # If no current schedule, initialize with all OFF
                if not current_schedule or not current_schedule.strip():
                    current_schedule = ", ".join("0" for _ in range(steps_int))
                return (
                    gr.update(value=1, interactive=True),  # Reset to 1 (schedule controls activation), enable slider
                    gr.update(value='<div style="color: gray;">⚪ Manual toggle active (adjust to override)</div>'),
                    gr.update(visible=True, value=grid_html),  # Show grid
                    gr.update(visible=True, value=current_schedule),  # Show and update textbox
                    # No changes to other params in manual mode
                    gr.update(),  # steps
                    gr.update(),  # cfg_scale
                    gr.update(),  # rescale_cfg
                    gr.update(),  # adapter_strength
                )
            elif toggle_mode == "standard":
                # Hide grid and textbox, auto-populate with standard pattern
                schedule = generate_standard_schedule(steps_int)
                schedule_csv = ", ".join(str(x) for x in schedule)
                return (
                    gr.update(value=1, interactive=True),  # Reset to 1 (schedule controls activation), enable slider
                    gr.update(value='<div style="color: gray;">⚪ Standard toggle active (adjust to override)</div>'),
                    gr.update(visible=False),  # Hide grid
                    gr.update(visible=False, value=schedule_csv),  # Hide textbox but set value
                    gr.update(),  # steps
                    gr.update(),  # cfg_scale
                    gr.update(),  # rescale_cfg
                    gr.update(),  # adapter_strength
                )
            elif toggle_mode == "smart":
                # Hide grid and textbox, auto-populate with smart pattern
                schedule = generate_smart_schedule(steps_int)
                schedule_csv = ", ".join(str(x) for x in schedule)
                return (
                    gr.update(value=1, interactive=True),  # Reset to 1 (schedule controls activation), enable slider
                    gr.update(value='<div style="color: gray;">⚪ Smart toggle active (adjust to override)</div>'),
                    gr.update(visible=False),  # Hide grid
                    gr.update(visible=False, value=schedule_csv),  # Hide textbox but set value
                    gr.update(),  # steps
                    gr.update(),  # cfg_scale
                    gr.update(),  # rescale_cfg
                    gr.update(),  # adapter_strength
                )
            elif toggle_mode == "optimized":
                # OPTIMIZED MODE: Auto-apply empirically validated parameters
                # These settings were validated for optimal quality with this DoRA schedule
                optimized_steps = 34
                optimized_cfg = 4.2
                optimized_rescale = 0.55
                optimized_adapter = 1.0

                # Generate schedule for 34 steps (optimized setting)
                schedule = generate_optimized_schedule(optimized_steps)
                schedule_csv = ", ".join(str(x) for x in schedule)

                return (
                    gr.update(value=1, interactive=False),  # Disable DoRA start step slider in optimized mode
                    gr.update(value='<div style="color: green;">✅ Optimized: Using validated parameters (CFG=4.2, Steps=34)</div>'),
                    gr.update(visible=False),  # Hide grid
                    gr.update(visible=False, value=schedule_csv),  # Hide textbox but set value
                    gr.update(value=optimized_steps),  # Auto-set steps to 34
                    gr.update(value=optimized_cfg),  # Auto-set CFG to 4.2
                    gr.update(value=optimized_rescale),  # Auto-set rescale to 0.55
                    gr.update(value=optimized_adapter),  # Auto-set adapter strength to 1.0
                )
            else:  # None selected
                return (
                    gr.update(interactive=True),  # Re-enable slider
                    gr.update(value='<div style="color: green;">✅ Start at step 1</div>'),
                    gr.update(visible=False),  # Hide grid
                    gr.update(visible=False, value=""),  # Hide textbox and clear
                    gr.update(),  # steps
                    gr.update(),  # cfg_scale
                    gr.update(),  # rescale_cfg
                    gr.update(),  # adapter_strength
                )

        dora_toggle_mode.change(
            handle_toggle_mode_change,
            inputs=[dora_toggle_mode, steps, dora_manual_schedule_state],
            outputs=[dora_start_step, dora_start_step_status, dora_manual_grid, dora_manual_schedule_state,
                     steps, cfg_scale, rescale_cfg, adapter_strength]
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

        # Hidden state for queue processing continuation
        should_continue_state = gr.State(value=False)

        # Generation handlers with queue/gallery support
        generate_btn.click(
            start_generation,
            outputs=[interrupt_btn, generate_btn]
        ).then(
            generate_image_with_progress,
            inputs=gen_inputs,
            outputs=gen_outputs
        ).then(
            finish_generation_with_gallery,
            inputs=[output_image, generation_info, last_seed_display],  # Use last_seed_display, not seed input
            outputs=[
                interrupt_btn, generate_btn,
                gallery_ui.carousel, gallery_ui.count,
                queue_ui.display, queue_ui.status,
                should_continue_state
            ]
        ).then(
            process_next_queue_item,
            inputs=[should_continue_state],
            outputs=[
                final_prompt, negative_prompt, resolution, cfg_scale, steps,
                rescale_cfg, seed, use_custom_resolution, custom_width,
                custom_height, auto_randomize_seed, adapter_strength,
                enable_dora, dora_start_step, dora_toggle_mode, dora_manual_schedule_state,
                should_continue_state
            ]
        ).then(
            trigger_queue_generation,
            inputs=[should_continue_state],
            outputs=[interrupt_btn, generate_btn, queue_ui.trigger_input]
        )

        # Queue auto-processing via Gradio-native .change() event
        # This replaces the unreliable JavaScript polling approach
        queue_ui.trigger_input.change(
            conditional_queue_start,
            inputs=[queue_ui.trigger_input],
            outputs=[queue_ui.trigger_input, interrupt_btn, generate_btn]
        ).then(
            generate_image_with_progress,
            inputs=gen_inputs,
            outputs=gen_outputs
        ).then(
            finish_generation_with_gallery,
            inputs=[output_image, generation_info, last_seed_display],
            outputs=[
                interrupt_btn, generate_btn,
                gallery_ui.carousel, gallery_ui.count,
                queue_ui.display, queue_ui.status,
                should_continue_state
            ]
        ).then(
            process_next_queue_item,
            inputs=[should_continue_state],
            outputs=[
                final_prompt, negative_prompt, resolution, cfg_scale, steps,
                rescale_cfg, seed, use_custom_resolution, custom_width,
                custom_height, auto_randomize_seed, adapter_strength,
                enable_dora, dora_start_step, dora_toggle_mode, dora_manual_schedule_state,
                should_continue_state
            ]
        ).then(
            trigger_queue_generation,
            inputs=[should_continue_state],
            outputs=[interrupt_btn, generate_btn, queue_ui.trigger_input]
        )

        interrupt_btn.click(
            interrupt_generation,
            outputs=[interrupt_btn, generate_btn]
        )

        # Queue event handlers
        queue_ui.add_button.click(
            add_to_queue,
            inputs=gen_inputs,
            outputs=[queue_ui.display, queue_ui.status]
        )

        queue_ui.clear_button.click(
            clear_queue,
            outputs=[queue_ui.display, queue_ui.status]
        )

        queue_ui.auto_process.change(
            set_auto_process,
            inputs=[queue_ui.auto_process]
        )

        # Queue item removal via hidden input (triggered by JS)
        queue_ui.remove_input.change(
            remove_from_queue,
            inputs=[queue_ui.remove_input],
            outputs=[queue_ui.display, queue_ui.status]
        )

        # Gallery event handlers
        gallery_ui.carousel.select(
            select_gallery_image,
            outputs=[output_image, generation_info]
        )

        gallery_ui.clear_button.click(
            clear_gallery,
            outputs=[gallery_ui.carousel, gallery_ui.count]
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

        # Parameter status updates (CFG, Rescale, Adapter handled below with optimized mode check)

        # Update DoRA start step maximum when steps change
        def update_dora_start_step_max(steps_value):
            return gr.update(maximum=steps_value)

        steps.change(
            update_dora_start_step_max,
            inputs=[steps],
            outputs=[dora_start_step]
        )

        # Update grid/schedule when steps change, with optimized mode deviation check
        def handle_steps_change(toggle_mode, num_steps, current_schedule):
            """Handle steps slider changes with optimized mode deviation check.

            If in 'optimized' mode and steps deviates from 34, switch to 'manual' mode.
            Otherwise, regenerate schedule for the current mode.
            """
            from utils import (
                parse_manual_dora_schedule, generate_standard_schedule,
                generate_smart_schedule, generate_optimized_schedule
            )

            try:
                steps_int = max(int(num_steps), 1)
            except (TypeError, ValueError):
                steps_int = 1

            # Get steps status
            steps_status_updater = create_status_updater('steps')
            steps_status_html = steps_status_updater(num_steps)

            # Check for optimized mode deviation (optimal steps = 34)
            if toggle_mode == "optimized" and steps_int != 34:
                # Steps deviates from optimal - switch to manual mode
                # Use optimized schedule as starting point for manual customization
                schedule = generate_optimized_schedule(steps_int)
                schedule_csv = ", ".join(str(x) for x in schedule)
                grid_html = generate_dora_grid(steps_int, schedule_csv)

                return (
                    steps_status_html,
                    gr.update(value="manual"),  # Switch to manual mode
                    gr.update(value='<div style="color: gray;">⚪ Manual toggle active (adjust to override)</div>'),
                    gr.update(visible=True, value=grid_html),  # Show grid
                    gr.update(visible=True, value=schedule_csv)  # Show schedule
                )

            # Not switching modes - handle normally
            if toggle_mode == "manual":
                # Regenerate grid with new step count
                grid_html = generate_dora_grid(steps_int, current_schedule)
                # Parse and extend/truncate schedule
                if current_schedule:
                    schedule, _ = parse_manual_dora_schedule(current_schedule, steps_int)
                    if schedule:
                        schedule_csv = ", ".join(str(x) for x in schedule)
                    else:
                        schedule_csv = ", ".join("0" for _ in range(steps_int))
                else:
                    schedule_csv = ", ".join("0" for _ in range(steps_int))
                return (
                    steps_status_html,
                    gr.update(),  # No change to toggle mode
                    gr.update(),  # No change to start step status
                    gr.update(value=grid_html),
                    gr.update(value=schedule_csv)
                )
            elif toggle_mode == "standard":
                schedule = generate_standard_schedule(steps_int)
                schedule_csv = ", ".join(str(x) for x in schedule)
                return (
                    steps_status_html,
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(value=schedule_csv)
                )
            elif toggle_mode == "smart":
                schedule = generate_smart_schedule(steps_int)
                schedule_csv = ", ".join(str(x) for x in schedule)
                return (
                    steps_status_html,
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(value=schedule_csv)
                )
            elif toggle_mode == "optimized":
                # Steps is still 34, regenerate optimized schedule
                schedule = generate_optimized_schedule(steps_int)
                schedule_csv = ", ".join(str(x) for x in schedule)
                return (
                    steps_status_html,
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(value=schedule_csv)
                )
            else:
                # Not in any toggle mode - just update status
                return (
                    steps_status_html,
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update()
                )

        steps.change(
            handle_steps_change,
            inputs=[dora_toggle_mode, steps, dora_manual_schedule_state],
            outputs=[steps_status, dora_toggle_mode, dora_start_step_status, dora_manual_grid, dora_manual_schedule_state]
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

        # Auto-switch from "optimized" to "manual" when inference parameters deviate from validated values
        # The optimized DoRA schedule was empirically validated for: CFG=4.2, Rescale=0.55, Adapter=1.0, Steps=34
        def check_optimized_param_deviation(param_type, value, current_toggle_mode, num_steps, current_schedule):
            """Check if parameter deviates from optimized values and switch to manual mode if needed.

            The 'optimized' DoRA schedule was empirically validated only for specific settings.
            Changing these parameters means the user is choosing custom inference parameters,
            so we switch to 'manual' mode to give them full control over the DoRA schedule.
            """
            # Get the status update for this parameter
            status_updater = create_status_updater(param_type)
            status_html = status_updater(value)

            # Check if we're in optimized mode
            if current_toggle_mode != "optimized":
                # Not in optimized mode, just return status update
                return (
                    status_html,
                    gr.update(),  # No change to toggle mode
                    gr.update(),  # No change to start step status
                    gr.update(),  # No change to grid
                    gr.update()   # No change to schedule
                )

            # Check if value deviates from the empirically validated optimal
            optimal_values = {
                'cfg': 4.2,
                'rescale': 0.55,
                'adapter': 1.0,
                'steps': 34
            }

            optimal_value = optimal_values.get(param_type)
            if optimal_value is None:
                return (
                    status_html,
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update()
                )

            # Use tolerance for float comparison
            tolerance = 0.001
            if abs(float(value) - optimal_value) <= tolerance:
                # Value matches optimal, stay in optimized mode
                return (
                    status_html,
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update()
                )

            # Value deviates from optimal - switch to manual mode
            try:
                steps_int = max(int(num_steps), 1)
            except (TypeError, ValueError):
                steps_int = 1

            # Generate grid for manual mode
            grid_html = generate_dora_grid(steps_int, current_schedule)

            # Initialize schedule if empty (use optimized as starting point)
            if not current_schedule or not current_schedule.strip():
                from utils import generate_optimized_schedule
                schedule = generate_optimized_schedule(steps_int)
                schedule_csv = ", ".join(str(x) for x in schedule)
            else:
                schedule_csv = current_schedule

            return (
                status_html,
                gr.update(value="manual"),  # Switch to manual mode
                gr.update(value='<div style="color: gray;">⚪ Manual toggle active (adjust to override)</div>'),
                gr.update(visible=True, value=grid_html),  # Show grid
                gr.update(visible=True, value=schedule_csv)  # Show schedule
            )

        # CFG scale change - check for optimized mode deviation
        def handle_cfg_change(value, toggle_mode, num_steps, schedule):
            return check_optimized_param_deviation('cfg', value, toggle_mode, num_steps, schedule)

        cfg_scale.change(
            handle_cfg_change,
            inputs=[cfg_scale, dora_toggle_mode, steps, dora_manual_schedule_state],
            outputs=[cfg_status, dora_toggle_mode, dora_start_step_status, dora_manual_grid, dora_manual_schedule_state]
        )

        # Rescale CFG change - check for optimized mode deviation
        def handle_rescale_change(value, toggle_mode, num_steps, schedule):
            return check_optimized_param_deviation('rescale', value, toggle_mode, num_steps, schedule)

        rescale_cfg.change(
            handle_rescale_change,
            inputs=[rescale_cfg, dora_toggle_mode, steps, dora_manual_schedule_state],
            outputs=[rescale_status, dora_toggle_mode, dora_start_step_status, dora_manual_grid, dora_manual_schedule_state]
        )

        # Adapter strength change - check for optimized mode deviation
        def handle_adapter_change(value, toggle_mode, num_steps, schedule):
            return check_optimized_param_deviation('adapter', value, toggle_mode, num_steps, schedule)

        adapter_strength.change(
            handle_adapter_change,
            inputs=[adapter_strength, dora_toggle_mode, steps, dora_manual_schedule_state],
            outputs=[adapter_status, dora_toggle_mode, dora_start_step_status, dora_manual_grid, dora_manual_schedule_state]
        )
        # Auto-switch to None mode when user changes start step while a toggle mode is active
        def handle_dora_start_step_change(start_step, current_toggle_mode):
            """Auto-switch toggle mode to None if user changes start step while a toggle mode is active.

            This provides intuitive UX: adjusting the start step slider indicates the user
            wants direct control over when DoRA activates, so we switch to None mode automatically.

            Note: When the slider is disabled (in optimized mode), this handler won't be triggered
            because the user can't interact with it. Only after switching away from optimized mode
            (which re-enables the slider) will this handler be called.
            """
            dora_start_step_updater = create_status_updater('dora_start_step')
            status_html = dora_start_step_updater(start_step)

            if current_toggle_mode is not None:
                # User changed start step while a toggle mode was active
                # Auto-switch to None mode for intuitive behavior
                return (
                    status_html,  # Update status with current value info
                    gr.update(value=None),  # Switch toggle mode to None
                    gr.update(visible=False),  # Hide grid
                    gr.update(visible=False, value="")  # Hide and clear schedule
                )
            else:
                # Already in None mode, just update status
                return (
                    status_html,
                    gr.update(),  # No change to toggle mode
                    gr.update(),  # No change to grid
                    gr.update()   # No change to schedule
                )

        dora_start_step.change(
            handle_dora_start_step_change,
            inputs=[dora_start_step, dora_toggle_mode],
            outputs=[dora_start_step_status, dora_toggle_mode, dora_manual_grid, dora_manual_schedule_state]
        )

        # Reset to optimal (sets optimized DoRA mode with validated parameters)
        def reset_to_optimal():
            from utils import generate_optimized_schedule

            cfg_updater = create_status_updater('cfg')
            steps_updater = create_status_updater('steps')
            rescale_updater = create_status_updater('rescale')
            adapter_updater = create_status_updater('adapter')

            # Generate optimized schedule for 34 steps
            schedule = generate_optimized_schedule(34)
            schedule_csv = ", ".join(str(x) for x in schedule)

            return (
                OPTIMAL_SETTINGS['cfg_scale'],
                OPTIMAL_SETTINGS['steps'],
                OPTIMAL_SETTINGS['rescale_cfg'],
                OPTIMAL_SETTINGS['adapter_strength'],
                gr.update(value=OPTIMAL_SETTINGS['dora_start_step'], interactive=False),  # Disable slider in optimized mode
                "1216x832 (Optimal)",
                False,
                OPTIMAL_SETTINGS['width'],
                OPTIMAL_SETTINGS['height'],
                "optimized",  # Reset toggle mode to optimized (not None)
                gr.update(visible=False),  # Hide grid
                gr.update(visible=False, value=schedule_csv),  # Hide schedule but set value
                cfg_updater(OPTIMAL_SETTINGS['cfg_scale']),
                steps_updater(OPTIMAL_SETTINGS['steps']),
                rescale_updater(OPTIMAL_SETTINGS['rescale_cfg']),
                adapter_updater(OPTIMAL_SETTINGS['adapter_strength']),
                '<div style="color: green;">✅ Optimized: Using validated parameters (CFG=4.2, Steps=34)</div>'
            )

        reset_btn.click(
            reset_to_optimal,
            outputs=[
                cfg_scale, steps, rescale_cfg, adapter_strength, dora_start_step, resolution,
                use_custom_resolution, custom_width, custom_height,
                dora_toggle_mode, dora_manual_grid, dora_manual_schedule_state,
                cfg_status, steps_status, rescale_status, adapter_status, dora_start_step_status
            ]
        )

        return demo
