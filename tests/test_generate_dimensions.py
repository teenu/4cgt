import types
import unittest

from PIL import Image

from config import GEN_CONFIG, OPTIMAL_SETTINGS
from engine import NoobAIEngine


class DummyPipe:
    def __init__(self):
        self.called_kwargs = None

    def __call__(self, **kwargs):
        self.called_kwargs = kwargs
        dummy_image = Image.new("RGB", (kwargs["width"], kwargs["height"]))
        return types.SimpleNamespace(images=[dummy_image])


def build_minimal_engine():
    """Create a NoobAIEngine instance with minimal stubs for testing."""
    engine = NoobAIEngine.__new__(NoobAIEngine)
    engine.is_initialized = True
    engine.pipe = DummyPipe()
    engine._device = "cpu"

    # Stub methods that are irrelevant for validation
    engine.set_dora_enabled = lambda *_, **__: None
    engine.set_adapter_strength = lambda *_, **__: None
    engine.set_dora_start_step = lambda *_, **__: None
    engine._parse_manual_dora_schedule = lambda *_, **__: (None, None)
    engine._enforce_toggle_mode_exclusivity = lambda *_, **__: None
    engine._setup_initial_dora_state = lambda *_, **__: None
    engine._create_progress_callback = lambda *_, **__: (lambda *a, **k: None)
    engine._add_dora_info_to_result = lambda *_, **__: None
    engine._create_image_metadata = lambda *_, **__: {}
    engine.clear_memory = lambda *_, **__: None

    return engine


class GenerateDimensionCoercionTests(unittest.TestCase):
    def test_width_height_coerced_to_ints(self):
        engine = build_minimal_engine()

        width_input = "1024"
        height_input = 768.0

        image, _, _ = engine.generate(
            prompt="test",
            negative_prompt="",
            width=width_input,
            height=height_input,
            steps=OPTIMAL_SETTINGS["steps"],
            cfg_scale=GEN_CONFIG.MIN_CFG_SCALE,
            rescale_cfg=GEN_CONFIG.MIN_RESCALE_CFG,
            seed=1,
        )

        self.assertIsInstance(engine.pipe.called_kwargs["width"], int)
        self.assertIsInstance(engine.pipe.called_kwargs["height"], int)
        self.assertEqual(engine.pipe.called_kwargs["width"], 1024)
        self.assertEqual(engine.pipe.called_kwargs["height"], 768)
        self.assertEqual(image.size, (1024, 768))


if __name__ == "__main__":
    unittest.main()
