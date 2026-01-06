"""SHARP 3D model generation integration.

This module provides utilities for running Apple's SHARP model to convert
2D images into 3D Gaussian Splatting representations.
"""

import os
import struct
import subprocess
import shutil
from typing import Optional, Tuple
import numpy as np
from plyfile import PlyData
from config import logger, SHARP_CONFIG

# Spherical harmonics constant for DC component
SH_C0 = 0.28209479177387814


def convert_ply_to_splat(ply_path: str, splat_path: str) -> bool:
    """Convert 3DGS PLY file to .splat format for Gradio Model3D display.

    The .splat format is a compact 32-byte-per-vertex binary format:
    - Position: 12 bytes (3 x float32)
    - Scales: 12 bytes (3 x float32, exp-decoded)
    - Color: 4 bytes (RGBA as uint8)
    - Rotation: 4 bytes (quaternion as uint8)

    Args:
        ply_path: Path to input PLY file from SHARP
        splat_path: Path to output .splat file

    Returns:
        True if conversion successful, False otherwise
    """
    try:
        plydata = PlyData.read(ply_path)
        vertex = plydata['vertex']
        n_vertices = len(vertex)

        logger.info(f"SHARP: Converting {n_vertices} vertices from PLY to .splat format")

        # Extract data arrays
        x = np.array(vertex['x'], dtype=np.float32)
        y = np.array(vertex['y'], dtype=np.float32)
        z = np.array(vertex['z'], dtype=np.float32)

        # Center the model at origin and scale up for visibility
        # SHARP outputs very small models (~2-3 units) that need scaling for viewers
        x = x - np.mean(x)
        y = y - np.mean(y)
        z = z - np.mean(z)

        # Scale up for visibility in 3D viewer
        # SHARP outputs ~2-3 unit models, scale to ~100 units for proper viewing
        scale_factor = 42.0
        x = x * scale_factor
        y = y * scale_factor
        z = z * scale_factor

        # Spherical harmonics DC components for color
        f_dc_0 = np.array(vertex['f_dc_0'], dtype=np.float32)
        f_dc_1 = np.array(vertex['f_dc_1'], dtype=np.float32)
        f_dc_2 = np.array(vertex['f_dc_2'], dtype=np.float32)

        opacity = np.array(vertex['opacity'], dtype=np.float32)

        # Scales (stored as log-space values)
        scale_0 = np.array(vertex['scale_0'], dtype=np.float32)
        scale_1 = np.array(vertex['scale_1'], dtype=np.float32)
        scale_2 = np.array(vertex['scale_2'], dtype=np.float32)

        # Rotation quaternion
        rot_0 = np.array(vertex['rot_0'], dtype=np.float32)
        rot_1 = np.array(vertex['rot_1'], dtype=np.float32)
        rot_2 = np.array(vertex['rot_2'], dtype=np.float32)
        rot_3 = np.array(vertex['rot_3'], dtype=np.float32)

        # Convert scales from log-space and apply same scale factor as positions
        scales_exp = np.column_stack([
            np.exp(scale_0) * scale_factor,
            np.exp(scale_1) * scale_factor,
            np.exp(scale_2) * scale_factor
        ]).astype(np.float32)

        # Convert SH to RGB (0.5 + SH_C0 * f_dc)
        r = np.clip((0.5 + SH_C0 * f_dc_0) * 255, 0, 255).astype(np.uint8)
        g = np.clip((0.5 + SH_C0 * f_dc_1) * 255, 0, 255).astype(np.uint8)
        b = np.clip((0.5 + SH_C0 * f_dc_2) * 255, 0, 255).astype(np.uint8)

        # Convert opacity using sigmoid
        alpha = np.clip((1.0 / (1.0 + np.exp(-opacity))) * 255, 0, 255).astype(np.uint8)

        # Normalize and quantize rotation quaternion
        rot_norm = np.sqrt(rot_0**2 + rot_1**2 + rot_2**2 + rot_3**2)
        rot_norm = np.where(rot_norm == 0, 1, rot_norm)  # Avoid division by zero

        rot_0_q = np.clip((rot_0 / rot_norm) * 128 + 128, 0, 255).astype(np.uint8)
        rot_1_q = np.clip((rot_1 / rot_norm) * 128 + 128, 0, 255).astype(np.uint8)
        rot_2_q = np.clip((rot_2 / rot_norm) * 128 + 128, 0, 255).astype(np.uint8)
        rot_3_q = np.clip((rot_3 / rot_norm) * 128 + 128, 0, 255).astype(np.uint8)

        # Sort by importance (opacity-weighted scale) in descending order
        # Use exp(scale) since scales are stored in log-space
        # Higher opacity (more visible) and larger scale = higher importance
        exp_scale_sum = np.exp(scale_0) + np.exp(scale_1) + np.exp(scale_2)
        sigmoid_opacity = 1.0 / (1.0 + np.exp(-opacity))
        importance = exp_scale_sum * sigmoid_opacity
        sort_indices = np.argsort(-importance)

        # Write binary .splat file
        with open(splat_path, 'wb') as f:
            for i in sort_indices:
                # Position (12 bytes)
                f.write(struct.pack('<fff', x[i], y[i], z[i]))
                # Scales (12 bytes)
                f.write(struct.pack('<fff', scales_exp[i, 0], scales_exp[i, 1], scales_exp[i, 2]))
                # Color RGBA (4 bytes)
                f.write(struct.pack('<BBBB', r[i], g[i], b[i], alpha[i]))
                # Rotation (4 bytes)
                f.write(struct.pack('<BBBB', rot_0_q[i], rot_1_q[i], rot_2_q[i], rot_3_q[i]))

        logger.info(f"SHARP: Converted to .splat format: {splat_path}")
        return True

    except Exception as e:
        logger.error(f"SHARP: Failed to convert PLY to .splat: {e}")
        return False


def check_sharp_available() -> Tuple[bool, str]:
    """Check if SHARP is available in the system.

    Returns:
        Tuple of (is_available, message)
    """
    sharp_path = shutil.which("sharp")
    if sharp_path is None:
        return False, "SHARP command not found. Install with: pip install ml-sharp"

    try:
        result = subprocess.run(
            ["sharp", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return True, f"SHARP available at: {sharp_path}"
        else:
            return False, f"SHARP found but not working: {result.stderr}"
    except subprocess.TimeoutExpired:
        return False, "SHARP command timed out"
    except Exception as e:
        return False, f"Error checking SHARP: {e}"


def get_sharp_checkpoint_path() -> Optional[str]:
    """Locate the SHARP checkpoint file.

    Checks default cache location and returns path if found.

    Returns:
        Path to checkpoint file or None if not found
    """
    checkpoint_path = os.path.join(
        SHARP_CONFIG.DEFAULT_CACHE_DIR,
        SHARP_CONFIG.CHECKPOINT_NAME
    )

    if os.path.exists(checkpoint_path):
        return checkpoint_path

    # Check if it might be in current directory
    if os.path.exists(SHARP_CONFIG.CHECKPOINT_NAME):
        return os.path.abspath(SHARP_CONFIG.CHECKPOINT_NAME)

    return None


def run_sharp_inference(
    input_image_path: str,
    output_dir: str,
    checkpoint_path: Optional[str] = None
) -> Optional[str]:
    """Run SHARP inference to generate 3D Gaussian representation.

    Args:
        input_image_path: Path to input PNG image
        output_dir: Directory to save output .ply file
        checkpoint_path: Optional path to SHARP checkpoint (auto-detected if None)

    Returns:
        Path to generated .ply file or None on failure
    """
    # Validate input
    if not os.path.exists(input_image_path):
        logger.error(f"SHARP: Input image not found: {input_image_path}")
        return None

    # Get checkpoint
    if checkpoint_path is None:
        checkpoint_path = get_sharp_checkpoint_path()

    # Build output path with same base name
    base_name = os.path.splitext(os.path.basename(input_image_path))[0]

    # Create a temp directory for SHARP output (it outputs to a folder)
    sharp_output_dir = os.path.join(output_dir, f".sharp_temp_{base_name}")
    os.makedirs(sharp_output_dir, exist_ok=True)

    try:
        # Build command
        cmd = [
            "sharp", "predict",
            "-i", input_image_path,
            "-o", sharp_output_dir
        ]

        # Add checkpoint if available
        if checkpoint_path and os.path.exists(checkpoint_path):
            cmd.extend(["-c", checkpoint_path])

        logger.info(f"SHARP: Running inference on {input_image_path}")
        logger.debug(f"SHARP command: {' '.join(cmd)}")

        # Run SHARP
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SHARP_CONFIG.TIMEOUT_SECONDS
        )

        if result.returncode != 0:
            logger.error(f"SHARP failed with code {result.returncode}: {result.stderr}")
            return None

        # Find the generated .ply file
        ply_files = [f for f in os.listdir(sharp_output_dir) if f.endswith('.ply')]

        if not ply_files:
            logger.error(f"SHARP: No .ply file generated in {sharp_output_dir}")
            return None

        # Move the .ply file to the output directory with proper name
        source_ply = os.path.join(sharp_output_dir, ply_files[0])
        dest_ply = os.path.join(output_dir, f"{base_name}.ply")

        shutil.move(source_ply, dest_ply)
        logger.info(f"SHARP: 3D model saved to {dest_ply}")

        # Convert PLY to .splat format for Gradio Model3D display
        dest_splat = os.path.join(output_dir, f"{base_name}.splat")
        if convert_ply_to_splat(dest_ply, dest_splat):
            # Return .splat path for Gradio display (PLY is also kept for other uses)
            return dest_splat
        else:
            # Fall back to PLY if conversion fails
            logger.warning("SHARP: .splat conversion failed, returning .ply path")
            return dest_ply

    except subprocess.TimeoutExpired:
        logger.error(f"SHARP: Inference timed out after {SHARP_CONFIG.TIMEOUT_SECONDS}s")
        return None
    except Exception as e:
        logger.error(f"SHARP: Unexpected error: {e}")
        return None
    finally:
        # Cleanup temp directory
        try:
            if os.path.exists(sharp_output_dir):
                shutil.rmtree(sharp_output_dir)
        except Exception as cleanup_error:
            logger.warning(f"SHARP: Failed to cleanup temp dir: {cleanup_error}")
