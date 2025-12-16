"""Parameter validation for UI."""

import re
from typing import Union, Any, Optional, Tuple
from config import logger, GEN_CONFIG, MODEL_CONFIG, OPTIMAL_SETTINGS, InvalidParameterError


def _coerce_int(value: Union[int, float, str, Any], label: str) -> int:
    """Coerce value to integer with descriptive error message."""
    try:
        if value is None:
            raise InvalidParameterError(f"{label} cannot be None")

        if hasattr(value, 'item'):
            value = value.item()

        return int(value)
    except (TypeError, ValueError) as e:
        raise InvalidParameterError(f"{label} must be an integer value, got {type(value).__name__}: {e}")


def _coerce_float(value: Union[int, float, str, Any], label: str) -> float:
    """Coerce value to float with descriptive error message."""
    try:
        if value is None:
            raise InvalidParameterError(f"{label} cannot be None")

        if hasattr(value, 'item'):
            value = value.item()

        return float(value)
    except (TypeError, ValueError) as e:
        raise InvalidParameterError(f"{label} must be a numeric value, got {type(value).__name__}: {e}")


def parse_resolution_string(res_str: str) -> Tuple[int, int]:
    """Parse resolution string to width and height.

    Expects format like "1216x832" or "1216x832 (Optimal)".
    """
    try:
        # Try explicit WIDTHxHEIGHT pattern first
        match = re.search(r'(\d+)\s*x\s*(\d+)', res_str, re.IGNORECASE)
        if match:
            return int(match.group(1)), int(match.group(2))
        # Fallback to extracting first two numbers
        numbers = re.findall(r'\d+', res_str)
        if len(numbers) >= 2:
            return int(numbers[0]), int(numbers[1])
        return OPTIMAL_SETTINGS['width'], OPTIMAL_SETTINGS['height']
    except (ValueError, TypeError, IndexError):
        return OPTIMAL_SETTINGS['width'], OPTIMAL_SETTINGS['height']


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

    try:
        w = _coerce_int(w, "Width")
    except InvalidParameterError as e:
        errors.append(str(e))
        w = OPTIMAL_SETTINGS['width']

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

    if a is not None:
        try:
            a = _coerce_float(a, "Adapter strength")
            if not (MODEL_CONFIG.MIN_ADAPTER_STRENGTH <= a <= MODEL_CONFIG.MAX_ADAPTER_STRENGTH):
                errors.append(f"Adapter strength must be {MODEL_CONFIG.MIN_ADAPTER_STRENGTH}-{MODEL_CONFIG.MAX_ADAPTER_STRENGTH}")
        except InvalidParameterError as e:
            errors.append(str(e))

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
