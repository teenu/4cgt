"""UI widget factories and status updaters."""

import gradio as gr
from typing import Tuple, Optional
from ui.search_helpers import search_for_autocomplete, select_from_dropdown, get_random_value, search_for_autocomplete_filtered


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
        if 3.5 <= value <= 5.5:
            return '<div style="color: green;">✅ Optimal range (3.5-5.5)</div>'
        else:
            return '<div style="color: orange;">⚠️ Outside optimal range (3.5-5.5)</div>'

    def update_steps_status(value):
        try:
            value = int(value)
        except (TypeError, ValueError):
            return '<div style="color: red;">❌ Invalid value</div>'
        if 32 <= value <= 40:
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
        if abs(value - 0.7) < 0.1:
            return '<div style="color: green;">✅ Optimal (around 0.7)</div>'
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
        if value == 0:
            return '<div style="color: green;">✅ Start at step 0 (first step)</div>'
        elif value <= 5:
            return '<div style="color: blue;">🚀 Early activation (step {})'.format(value) + '</div>'
        elif value <= 15:
            return '<div style="color: orange;">⏰ Mid activation (step {})'.format(value) + '</div>'
        else:
            return '<div style="color: purple;">🔄 Late activation (step {})'.format(value) + '</div>'

    updaters = {
        'cfg': update_cfg_status,
        'steps': update_steps_status,
        'rescale': update_rescale_status,
        'adapter': update_adapter_status,
        'dora_start_step': update_dora_start_step_status
    }

    return updaters.get(param_type, lambda x: "")


def create_search_ui(label: str, number: int) -> Tuple[gr.Textbox, gr.Dropdown, gr.Textbox, gr.Button, gr.Button, gr.Radio]:
    """
    Create the UI for a search segment with randomize functionality.

    Returns:
        Tuple of (search_box, dropdown, text_output, clear_btn, randomize_btn, source_filter)
    """
    with gr.Group(elem_classes=["segment-container"]):
        gr.HTML(f'<div class="segment-header">{number}️⃣ {label}</div>')

        # Source filter toggle
        source_filter = gr.Radio(
            choices=[("All", "all"), ("Danbooru", "danbooru"), ("E621", "e621")],
            value="all",
            label="Source Filter",
            info="Filter search and randomize by source"
        )

        search_box = gr.Textbox(placeholder=f"Search {label.lower()}s...", lines=1)
        dropdown = gr.Dropdown(choices=[], interactive=True, allow_custom_value=True)
        text_output = gr.Textbox(lines=2, interactive=False)

        with gr.Row():
            randomize_btn = gr.Button("🎲 Randomize", size="sm")
            clear_btn = gr.Button("🧹 Clear", size="sm")

    return search_box, dropdown, text_output, clear_btn, randomize_btn, source_filter


def connect_search_events(
    data_type: str,
    search_box: gr.Textbox,
    dropdown: gr.Dropdown,
    text_output: gr.Textbox,
    clear_btn: gr.Button,
    randomize_btn: Optional[gr.Button] = None,
    source_filter: Optional[gr.Radio] = None,
    compose_fn=None,
    compose_inputs: Optional[list] = None,
    final_prompt_output: Optional[gr.Textbox] = None,
):
    """Connect event handlers for a search segment.

    If compose_fn, compose_inputs, and final_prompt_output are provided,
    the randomize and dropdown selection will auto-compose the final prompt.
    """
    # Search with source filter support
    if source_filter is not None:
        search_box.change(
            lambda q, sf: search_for_autocomplete_filtered(q, data_type, sf if sf != 'all' else None),
            inputs=[search_box, source_filter],
            outputs=[dropdown],
            show_progress=False,
        )

        # Re-search when filter changes
        source_filter.change(
            lambda q, sf: search_for_autocomplete_filtered(q, data_type, sf if sf != 'all' else None),
            inputs=[search_box, source_filter],
            outputs=[dropdown],
            show_progress=False,
        )
    else:
        search_box.change(
            lambda q: search_for_autocomplete(q, data_type),
            inputs=[search_box],
            outputs=[dropdown],
            show_progress=False,
        )

    dropdown_event = dropdown.change(
        lambda q, c: select_from_dropdown(q, c, data_type),
        inputs=[search_box, dropdown],
        outputs=[text_output],
        show_progress=False,
    )
    # Auto-compose after dropdown selection if compose parameters are provided
    if compose_fn and compose_inputs is not None and final_prompt_output is not None:
        dropdown_event.then(
            compose_fn,
            inputs=compose_inputs,
            outputs=[final_prompt_output],
            show_progress=False,
        )

    clear_btn.click(
        create_clear_handler(data_type),
        outputs=[search_box, text_output, dropdown],
        show_progress=False,
    )

    # Randomize button handler
    if randomize_btn is not None and source_filter is not None:
        randomize_event = randomize_btn.click(
            lambda sf: get_random_value(data_type, sf if sf != 'all' else None),
            inputs=[source_filter],
            outputs=[text_output],
            show_progress=False,
        )
        # Auto-compose after randomize if compose parameters are provided
        if compose_fn and compose_inputs is not None and final_prompt_output is not None:
            randomize_event.then(
                compose_fn,
                inputs=compose_inputs,
                outputs=[final_prompt_output],
                show_progress=False,
            )
