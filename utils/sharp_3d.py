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


def _build_render_env() -> dict:
    """Build subprocess env with CUDA_HOME and PATH set for gsplat JIT compilation.

    gsplat compiles CUDA kernels on the first --render call.  This requires:
      1. CUDA_HOME whose include/ has cuda_runtime.h + crt/host_config.h + thrust/
      2. cicc (nvcc internal tool) on PATH

    In conda environments CUDA components are spread across bin/, nvvm/bin/,
    targets/<arch>/include/, and pip nvidia packages — the defaults torch infers
    are often incomplete.  We locate each piece and fix the env here.
    """
    env = os.environ.copy()

    nvcc = shutil.which("nvcc") or ""
    if not nvcc:
        return env

    nvcc_dir = os.path.dirname(nvcc)          # e.g. ~/miniconda3/bin
    conda_root = os.path.normpath(os.path.join(nvcc_dir, ".."))

    # --- 1. Put cicc on PATH so nvcc can call it ---
    cicc_path = os.path.normpath(os.path.join(nvcc_dir, "..", "nvvm", "bin", "cicc"))
    if os.path.isfile(cicc_path):
        cicc_dir = os.path.dirname(cicc_path)
        path = env.get("PATH", "")
        if cicc_dir not in path.split(os.pathsep):
            env["PATH"] = cicc_dir + os.pathsep + path

    # --- 2. Find CUDA include with cuda_runtime.h + crt/ + thrust/ ---
    cuda_home = env.get("CUDA_HOME", "")
    if cuda_home:
        inc = os.path.join(cuda_home, "include")
        if (os.path.isfile(os.path.join(inc, "cuda_runtime.h")) and
                os.path.isfile(os.path.join(inc, "crt", "host_config.h"))):
            return env  # caller-supplied CUDA_HOME is already valid

    # Candidate include directories to search, in preference order:
    #  1. conda's targets/<arch>/include/ — full header tree including crt/
    #  2. common system CUDA toolkit paths
    candidates = []
    for arch in ("x86_64-linux", "sbsa-linux", "aarch64-linux"):
        candidates.append(os.path.join(conda_root, "targets", arch, "include"))
    for sys_path in ("/usr/local/cuda/include", "/usr/cuda/include"):
        candidates.append(sys_path)

    for inc in candidates:
        if not os.path.isfile(os.path.join(inc, "cuda_runtime.h")):
            continue
        if not os.path.isfile(os.path.join(inc, "crt", "host_config.h")):
            continue
        # Ensure thrust/ is accessible (newer CUDA moves it under cccl/).
        thrust_link = os.path.join(inc, "thrust")
        if not os.path.exists(thrust_link):
            cccl_thrust = os.path.join(inc, "cccl", "thrust")
            if os.path.isdir(cccl_thrust):
                try:
                    os.symlink(cccl_thrust, thrust_link)
                except OSError:
                    pass
        # Only proceed if thrust is now accessible (symlink may have failed).
        if not os.path.exists(os.path.join(inc, "thrust")):
            logger.debug("Sharp render: thrust not found in %s, skipping", inc)
            continue
        cuda_home_candidate = os.path.normpath(os.path.join(inc, ".."))
        env["CUDA_HOME"] = cuda_home_candidate
        logger.debug("Sharp render: CUDA_HOME=%s", cuda_home_candidate)
        return env

    logger.warning("Sharp render: could not locate a complete CUDA include tree; "
                   "render may fail to compile gsplat kernels")
    return env


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


def _reencode_video_hq(video_path: str) -> None:
    """Re-encode an mp4 in-place with CRF 18 for higher visual quality.

    Sharp's VideoWriter uses imageio's default (CRF 23), which introduces
    visible softness.  CRF 18 is visually lossless for typical rendered content.
    Uses the ffmpeg bundled with imageio-ffmpeg so no system install is needed.
    Silently skips if re-encoding fails.
    """
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return

    tmp = video_path + ".tmp.mp4"
    try:
        result = subprocess.run(
            [ffmpeg, "-i", video_path,
             "-vcodec", "libx264", "-crf", "18", "-preset", "slow",
             "-y", tmp],
            capture_output=True,
            timeout=120,
        )
        if result.returncode == 0 and os.path.isfile(tmp):
            os.replace(tmp, video_path)
            logger.debug("Re-encoded video at CRF 18: %s", video_path)
        else:
            logger.warning("Video re-encode failed (exit %d); keeping original", result.returncode)
    except Exception as exc:
        logger.warning("Video re-encode skipped: %s", exc)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


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

        run_env = _build_render_env() if render else None

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,
                env=run_env,
            )
        except subprocess.TimeoutExpired:
            return {"error": "Sharp timed out after 180 seconds.",
                    "ply_path": None, "video_path": None, "elapsed": 180.0}
        except Exception as exc:
            return {"error": f"Sharp subprocess error: {exc}",
                    "ply_path": None, "video_path": None, "elapsed": 0.0}

        elapsed = time.perf_counter() - t0

        if proc.returncode != 0:
            if proc.stderr:
                head = proc.stderr[:3000]
                tail = proc.stderr[-500:] if len(proc.stderr) > 3500 else ""
                stderr_tail = head + ("\n...\n" + tail if tail else "")
            else:
                stderr_tail = "(no stderr)"
            logger.error("Sharp exited %d:\n%s", proc.returncode, stderr_tail)
            return {"error": f"Sharp failed (exit {proc.returncode}): {stderr_tail}",
                    "ply_path": None, "video_path": None, "elapsed": elapsed}

    # Locate the output .ply
    ply_files = sorted(Path(gaussians_dir).glob("*.ply"))
    if not ply_files:
        return {"error": "Sharp completed but produced no .ply file.",
                "ply_path": None, "video_path": None, "elapsed": elapsed}

    ply_path = str(ply_files[-1])

    # Locate rendered video (if requested).
    # Sharp also writes a .depth.mp4 alongside the colour video — exclude it.
    video_path = None
    if render:
        mp4_files = [
            p for p in sorted(Path(gaussians_dir).glob("**/*.mp4"))
            if ".depth." not in p.name
        ]
        if mp4_files:
            video_path = str(mp4_files[-1])
            _reencode_video_hq(video_path)
        else:
            logger.warning("Sharp render requested but no .mp4 found in %s", gaussians_dir)

    logger.info("Sharp done in %.2fs — %s", elapsed, ply_path)
    return {
        "ply_path": ply_path,
        "video_path": video_path,
        "elapsed": elapsed,
        "error": None,
    }
