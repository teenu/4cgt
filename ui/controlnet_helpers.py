"""ControlNet UI helper functions."""

from typing import List, Optional, Dict, Any
from utils.controlnet import discover_controlnet_models, get_controlnet_by_name


def get_controlnet_choices(force_refresh: bool = False) -> List[str]:
    """Get list of available ControlNet models for dropdown.

    Args:
        force_refresh: If True, bypass cache and rescan directories.

    Returns:
        List of model display names.
    """
    models = discover_controlnet_models(force_refresh=force_refresh)
    if models:
        return [model['display_name'] for model in models]
    return ["None"]


def get_default_controlnet() -> str:
    """Get default ControlNet model selection.

    Returns:
        Display name of first available model, or "None".
    """
    choices = get_controlnet_choices()
    return choices[0] if choices and choices[0] != "None" else "None"


def get_controlnet_path_from_display_name(display_name: str) -> Optional[str]:
    """Get ControlNet model path from display name.

    Args:
        display_name: The display name shown in the dropdown.

    Returns:
        Full path to the model, or None if not found.
    """
    if display_name == "None" or not display_name:
        return None

    models = discover_controlnet_models()
    for model in models:
        if model['display_name'] == display_name:
            return model['path']

    return None


def get_controlnet_info_from_display_name(display_name: str) -> Optional[Dict[str, Any]]:
    """Get full ControlNet model info from display name.

    Args:
        display_name: The display name shown in the dropdown.

    Returns:
        Model info dictionary, or None if not found.
    """
    if display_name == "None" or not display_name:
        return None

    models = discover_controlnet_models()
    for model in models:
        if model['display_name'] == display_name:
            return model

    return None


def refresh_controlnet_dropdown():
    """Force refresh of ControlNet dropdown choices.

    Returns:
        Updated list of choices.
    """
    return get_controlnet_choices(force_refresh=True)
