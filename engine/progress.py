"""Progress tracking and DoRA toggle management."""

import os
import time
import threading
from typing import Optional, Callable, List, Dict
from config import logger, GenerationInterruptedError
from state import state_manager, GenerationState
from engine.memory import synchronize_device


class ProgressManager:
    """Manages progress callbacks and DoRA toggle modes."""

    def __init__(self, pipe, device: str, dora_manager):
        self.pipe = pipe
        self.device = device
        self.dora_manager = dora_manager
        self._toggle_lock = threading.Lock()

    def create_callback(
        self,
        steps: int,
        start_time: float,
        dora_toggle_mode: Optional[str],
        dora_start_step: int,
        manual_schedule: Optional[List[int]],
        progress_callback: Optional[Callable[[float, str], None]],
        enable_dora: bool
    ) -> Callable:
        """Create progress callback function for generation."""

        def callback_on_step_end(pipe, step_index: int, timestep, callback_kwargs: Dict) -> Dict:
            if state_manager.is_interrupted():
                raise GenerationInterruptedError()

            current_step = step_index + 1
            progress = current_step / steps
            elapsed = time.time() - start_time
            eta = (elapsed / current_step) * (steps - current_step) if current_step > 0 else 0

            desc = self._build_description(
                step_index, current_step, steps, eta,
                dora_toggle_mode, dora_start_step, manual_schedule, enable_dora
            )

            if progress_callback:
                try:
                    progress_callback(progress, desc)
                except Exception as e:
                    logger.error(f"Progress callback error at step {current_step}: {e}", exc_info=True)
                    if os.environ.get('NOOBAI_CLI_MODE') == '1':
                        raise
                    try:
                        progress_callback(progress, f"Step {current_step}/{steps}")
                    except Exception as retry_error:
                        logger.error(f"Progress callback retry also failed: {retry_error}")

            return callback_kwargs

        return callback_on_step_end

    def _build_description(
        self,
        step_index: int,
        current_step: int,
        steps: int,
        eta: float,
        dora_toggle_mode: Optional[str],
        dora_start_step: int,
        manual_schedule: Optional[List[int]],
        enable_dora: bool
    ) -> str:
        """Build progress description string and update DoRA state."""
        if dora_toggle_mode and enable_dora and self.dora_manager.dora_loaded and self.pipe:
            next_step_index = step_index + 1

            if dora_toggle_mode in ["manual", "optimized"]:
                # Both manual and optimized modes use schedule-based toggling
                return self._handle_manual_toggle(step_index, current_step, steps, eta, next_step_index, manual_schedule)

        elif enable_dora and self.dora_manager.dora_loaded and self.pipe:
            # Using 0-based indexing: step_index matches dora_start_step directly
            if step_index == dora_start_step - 1 and dora_start_step > 0:
                # Activate DoRA on the step before dora_start_step
                self.pipe.set_adapters(["noobai_dora"], adapter_weights=[self.dora_manager.adapter_strength])
                return f"Step {current_step}/{steps} (DoRA will activate at step {dora_start_step}, ETA: {eta:.1f}s)"
            elif step_index >= dora_start_step:
                return f"Step {current_step}/{steps} (DoRA active, ETA: {eta:.1f}s)"
            else:
                return f"Step {current_step}/{steps} (DoRA starts at step {dora_start_step}, ETA: {eta:.1f}s)"

        return f"Step {current_step}/{steps} (ETA: {eta:.1f}s)"

    def _handle_manual_toggle(
        self,
        step_index: int,
        current_step: int,
        steps: int,
        eta: float,
        next_step_index: int,
        manual_schedule: Optional[List[int]]
    ) -> str:
        """Handle manual toggle mode progress description."""
        if not manual_schedule:
            return f"Step {current_step}/{steps} (DoRA: OFF [no schedule], ETA: {eta:.1f}s)"

        if step_index >= len(manual_schedule):
            logger.warning(f"Manual schedule too short: index {step_index} >= length {len(manual_schedule)}")
            current_state = "OFF"
        else:
            current_state = "ON" if manual_schedule[step_index] == 1 else "OFF"

        if next_step_index < steps:
            with self._toggle_lock:
                if next_step_index >= len(manual_schedule):
                    logger.warning(f"Manual schedule too short: index {next_step_index} >= length {len(manual_schedule)}, treating as OFF")
                    next_state = "OFF"
                    synchronize_device(self.device)
                    self.pipe.set_adapters(["noobai_dora"], adapter_weights=[0.0])
                else:
                    next_state = "ON" if manual_schedule[next_step_index] == 1 else "OFF"
                    synchronize_device(self.device)
                    if manual_schedule[next_step_index] == 1:
                        self.pipe.set_adapters(["noobai_dora"], adapter_weights=[self.dora_manager.adapter_strength])
                    else:
                        self.pipe.set_adapters(["noobai_dora"], adapter_weights=[0.0])
            return f"Step {current_step}/{steps} (DoRA: {current_state}, next[{next_step_index}]: {next_state}, ETA: {eta:.1f}s)"
        else:
            return f"Step {current_step}/{steps} (DoRA: {current_state}, final, ETA: {eta:.1f}s)"

    def setup_initial_dora_state(
        self,
        dora_toggle_mode: Optional[str],
        dora_start_step: int,
        manual_schedule: Optional[List[int]],
        enable_dora: bool
    ) -> None:
        """Set up initial DoRA adapter state before generation starts."""
        if not enable_dora or not self.dora_manager.dora_loaded or not self.pipe:
            return

        if dora_start_step > 0 and not dora_toggle_mode:
            # Start step > 0 means DoRA should be OFF initially (0-based indexing)
            self.pipe.set_adapters(["noobai_dora"], adapter_weights=[0.0])
            return

        if dora_toggle_mode:
            if dora_toggle_mode in ["manual", "optimized"]:
                # For manual and optimized modes, use the schedule to determine initial state
                if manual_schedule and len(manual_schedule) > 0 and manual_schedule[0] == 1:
                    self.pipe.set_adapters(["noobai_dora"], adapter_weights=[self.dora_manager.adapter_strength])
                else:
                    self.pipe.set_adapters(["noobai_dora"], adapter_weights=[0.0])
            else:
                self.pipe.set_adapters(["noobai_dora"], adapter_weights=[self.dora_manager.adapter_strength])
