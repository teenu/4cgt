"""Apple Sharp Image-to-3D conversion utilities.

Sharp converts a single 2D image to a 3D Gaussian Splat (.ply) in under one second.
It runs as a subprocess so its GPU memory and model weights are fully isolated from
the SDXL pipeline — loading Sharp after generation does not evict model weights.

Checkpoint (~1GB): auto-downloaded to ~/.cache/torch/hub/checkpoints/ on first run,
or specify --sharp-checkpoint for a custom path.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

from config import logger, OUTPUT_DIR

# Where 3D outputs are written (mirrors the outputs/ structure)
SHARP_3D_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "3d")

# Path to the Sharp CLI entry point — same bin dir as the running Python
_SHARP_BIN = os.path.join(os.path.dirname(sys.executable), "sharp")

# Default cache location for the auto-downloaded checkpoint
_CHECKPOINT_CACHE = os.path.expanduser(
    os.path.join("~", ".cache", "torch", "hub", "checkpoints", "sharp_2572gikvuh.pt")
)


def is_sharp_installed() -> bool:
    """Return True if Sharp is importable and its CLI is present."""
    try:
        result = subprocess.run(
            [_SHARP_BIN, "--help"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def find_sharp_checkpoint(checkpoint_path: Optional[str] = None) -> Optional[str]:
    """Resolve a Sharp checkpoint .pt file path.

    Search order:
    1. Caller-supplied path (``checkpoint_path``)
    2. Default download cache: ~/.cache/torch/hub/checkpoints/
    3. Script directory (repo root)

    Returns ``None`` if not found; Sharp will auto-download on first run.
    """
    if checkpoint_path and os.path.isfile(checkpoint_path):
        return checkpoint_path

    if os.path.isfile(_CHECKPOINT_CACHE):
        return _CHECKPOINT_CACHE

    # Repo root fallback
    repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    local = os.path.join(repo_root, "sharp_2572gikvuh.pt")
    if os.path.isfile(local):
        return local

    return None


def convert_to_3d(
    image_path: str,
    output_name: str,
    device: str = "cuda",
    render: bool = False,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a saved PNG/JPG to a 3D Gaussian Splat using Apple Sharp.

    Args:
        image_path:      Path to the input image file.
        output_name:     Stem name for the output subdirectory (e.g. "noobai_42").
        device:          Inference device: "cuda", "mps", or "cpu".
        render:          If True, also render a camera-trajectory .mp4 (CUDA only).
        checkpoint_path: Optional path to the Sharp .pt checkpoint.

    Returns:
        dict with keys:
            ply_path   (str)            – path to the output .ply file.
            video_path (str | None)     – path to rendered .mp4, or None.
            elapsed    (float)          – wall-clock seconds for the whole call.
            error      (str | None)     – error message on failure, else None.
    """
    if not os.path.isfile(image_path):
        return {"error": f"Input image not found: {image_path}",
                "ply_path": None, "video_path": None, "elapsed": 0.0}

    if not is_sharp_installed():
        return {"error": "Sharp is not installed. Run: pip install git+https://github.com/apple/ml-sharp.git --no-deps",
                "ply_path": None, "video_path": None, "elapsed": 0.0}

    if render and device != "cuda":
        logger.warning("Sharp --render requires CUDA; disabling render for device=%s", device)
        render = False

    os.makedirs(SHARP_3D_OUTPUT_DIR, exist_ok=True)
    gaussians_dir = os.path.join(SHARP_3D_OUTPUT_DIR, output_name)
    os.makedirs(gaussians_dir, exist_ok=True)

    # Sharp expects a directory of images as input.
    # Copy the single image into a temporary directory.
    with tempfile.TemporaryDirectory() as input_dir:
        ext = Path(image_path).suffix or ".png"
        input_copy = os.path.join(input_dir, f"input{ext}")
        shutil.copy2(image_path, input_copy)

        cmd = [_SHARP_BIN, "predict",
               "--input-path", input_dir,
               "--output-path", gaussians_dir,
               "--device", device]

        checkpoint = find_sharp_checkpoint(checkpoint_path)
        if checkpoint:
            cmd += ["--checkpoint-path", checkpoint]

        if render:
            cmd.append("--render")

        logger.info("Sharp predict: %s", " ".join(cmd))
        t0 = time.perf_counter()

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,
            )
        except subprocess.TimeoutExpired:
            return {"error": "Sharp timed out after 180 seconds.",
                    "ply_path": None, "video_path": None, "elapsed": 180.0}
        except Exception as exc:
            return {"error": f"Sharp subprocess error: {exc}",
                    "ply_path": None, "video_path": None, "elapsed": 0.0}

        elapsed = time.perf_counter() - t0

        if proc.returncode != 0:
            stderr_tail = proc.stderr[-800:] if proc.stderr else "(no stderr)"
            logger.error("Sharp exited %d:\n%s", proc.returncode, stderr_tail)
            return {"error": f"Sharp failed (exit {proc.returncode}): {stderr_tail}",
                    "ply_path": None, "video_path": None, "elapsed": elapsed}

    # Locate the output .ply
    ply_files = sorted(Path(gaussians_dir).glob("*.ply"))
    if not ply_files:
        return {"error": "Sharp completed but produced no .ply file.",
                "ply_path": None, "video_path": None, "elapsed": elapsed}

    ply_path = str(ply_files[-1])

    # Locate rendered video (if requested)
    video_path = None
    if render:
        mp4_files = sorted(Path(gaussians_dir).glob("**/*.mp4"))
        if mp4_files:
            video_path = str(mp4_files[-1])
        else:
            logger.warning("Sharp render requested but no .mp4 found in %s", gaussians_dir)

    logger.info("Sharp done in %.2fs — %s", elapsed, ply_path)
    return {
        "ply_path": ply_path,
        "video_path": video_path,
        "elapsed": elapsed,
        "error": None,
    }
