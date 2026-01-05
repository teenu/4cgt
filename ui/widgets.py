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


def format_token_count_html(token_info: dict, is_negative: bool = False) -> str:
    """Format token count information as HTML for display.

    Args:
        token_info: Dictionary from engine.count_prompt_tokens() containing:
            - max_tokens: Maximum token count
            - chunks: Number of chunks needed
            - is_long: Whether prompt exceeds 77 tokens
            - warning: Warning message if applicable
            - long_prompt_supported: Whether sd_embed is available
        is_negative: If True, format for negative prompt display

    Returns:
        HTML string for display in the token counter element
    """
    prompt_type = "negative prompt" if is_negative else "prompt"
    if not token_info or token_info.get('max_tokens', 0) == 0:
        return f'<div style="color: gray; font-size: 0.9em;">Enter a {prompt_type} to see token count</div>'

    tokens = token_info.get('max_tokens', 0)
    chunks = token_info.get('chunks', 1)
    is_long = token_info.get('is_long', False)
    warning = token_info.get('warning')
    long_supported = token_info.get('long_prompt_supported', False)

    # Determine color and status based on token count
    if tokens <= 77:
        color = "#28a745"  # Green
        icon = "✅"
        status = f"{tokens}/77 tokens"
    elif chunks <= 2:
        color = "#17a2b8"  # Blue
        icon = "📝"
        status = f"{tokens} tokens ({chunks} chunks)"
    elif chunks <= 4:
        color = "#ffc107"  # Orange/yellow
        icon = "⚠️"
        status = f"{tokens} tokens ({chunks} chunks)"
    elif chunks <= 8:
        color = "#fd7e14"  # Darker orange
        icon = "⚠️"
        status = f"{tokens} tokens ({chunks} chunks - long prompt)"
    else:
        color = "#dc3545"  # Red
        icon = "❌"
        status = f"{tokens} tokens ({chunks} chunks - very long!)"

    # Build the HTML
    html = f'<div style="font-size: 0.9em;">'
    html += f'<span style="color: {color};">{icon} {status}</span>'

    # Add long prompt support indicator
    if is_long:
        if long_supported:
            html += ' <span style="color: #17a2b8; font-size: 0.85em;">(long prompt enabled)</span>'
        else:
            html += ' <span style="color: #dc3545; font-size: 0.85em;">(will be truncated!)</span>'

    # Add warning if present
    if warning:
        html += f'<br><span style="color: #fd7e14; font-size: 0.85em;">{warning}</span>'

    html += '</div>'
    return html


def create_status_updater(param_type: str):
    """Create a status update function for parameters."""
    def update_cfg_status(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return '<div style="color: red;">❌ Invalid value</div>'
        # Optimal range extended to 6.0 to include DoRA None mode setting (5.5261)
        if 3.5 <= value <= 6.0:
            return '<div style="color: green;">✅ Optimal range (3.5-6.0)</div>'
        else:
            return '<div style="color: orange;">⚠️ Outside optimal range (3.5-6.0)</div>'

    def update_steps_status(value):
        try:
            value = int(value)
        except (TypeError, ValueError):
            return '<div style="color: red;">❌ Invalid value</div>'
        # Optimal range extended to 45 to include DoRA None mode setting (42)
        if 32 <= value <= 45:
            return '<div style="color: green;">✅ Optimal range (32-45)</div>'
        elif value >= 10:
            return '<div style="color: orange;">⚠️ Below optimal range (32-45)</div>'
        else:
            return '<div style="color: red;">❌ Too low for quality results</div>'

    def update_rescale_status(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return '<div style="color: red;">❌ Invalid value</div>'
        # Optimal range includes both ~0.7 (standard) and ~0.6 (DoRA None mode: 0.6092)
        if 0.55 <= value <= 0.75:
            return '<div style="color: green;">✅ Optimal range (0.55-0.75)</div>'
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
        if value == 3:
            # Step 3 is optimal: skips semantic formation phase (steps 0-2)
            return '<div style="color: green;">✅ Optimal (step 3) - preserves structural formation</div>'
        elif value == 0:
            return '<div style="color: blue;">🚀 Start at step 0 (first step)</div>'
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
