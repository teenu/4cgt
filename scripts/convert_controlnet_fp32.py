#!/usr/bin/env python3
"""Convert ControlNet models from BF16 to FP32 for cross-platform determinism.

This script converts BF16 ControlNet safetensors files to FP32 format,
ensuring consistent behavior across all platforms (matching the approach
used for NoobAI-XL-Vpred-v1.0-FP32).

Usage:
    python scripts/convert_controlnet_fp32.py input.safetensors output.safetensors
    python scripts/convert_controlnet_fp32.py --auto  # Auto-convert from ref/
"""

import argparse
import os
import sys
import struct
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def get_safetensors_metadata(filepath: str) -> dict:
    """Read safetensors header to get tensor metadata."""
    with open(filepath, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header_bytes = f.read(header_size)
        header = json.loads(header_bytes.decode('utf-8'))
    return header


def analyze_precision(filepath: str) -> dict:
    """Analyze the precision of tensors in a safetensors file."""
    header = get_safetensors_metadata(filepath)

    dtype_counts = {}
    for key, value in header.items():
        if key == '__metadata__':
            continue
        dtype = value.get('dtype', 'unknown')
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1

    total = sum(dtype_counts.values())
    return {
        'total_tensors': total,
        'dtype_counts': dtype_counts,
        'file_size_gb': os.path.getsize(filepath) / (1024**3)
    }


def convert_to_fp32(input_path: str, output_path: str, verbose: bool = True) -> dict:
    """Convert a safetensors file from BF16 to FP32.

    Args:
        input_path: Path to input BF16 safetensors file
        output_path: Path to output FP32 safetensors file
        verbose: Print progress information

    Returns:
        Dictionary with conversion statistics
    """
    if verbose:
        print(f"Loading: {input_path}")

    # Analyze input precision
    input_analysis = analyze_precision(input_path)
    if verbose:
        print(f"  Input: {input_analysis['total_tensors']} tensors, {input_analysis['file_size_gb']:.2f} GB")
        print(f"  Dtypes: {input_analysis['dtype_counts']}")

    # Load all tensors
    state_dict = load_file(input_path)

    # Convert each tensor to FP32
    fp32_state_dict = {}
    converted_count = 0
    already_fp32_count = 0

    for key, tensor in state_dict.items():
        if tensor.dtype == torch.float32:
            fp32_state_dict[key] = tensor
            already_fp32_count += 1
        else:
            fp32_state_dict[key] = tensor.to(torch.float32)
            converted_count += 1

    if verbose:
        print(f"  Converted: {converted_count} tensors to FP32")
        print(f"  Already FP32: {already_fp32_count} tensors")

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save with metadata
    metadata = {
        "format": "pt",
        "precision": "FP32",
        "source_file": os.path.basename(input_path),
        "converted_by": "convert_controlnet_fp32.py",
        "source_precision": str(input_analysis['dtype_counts'])
    }

    if verbose:
        print(f"Saving: {output_path}")

    save_file(fp32_state_dict, output_path, metadata)

    # Verify output
    output_analysis = analyze_precision(output_path)
    if verbose:
        print(f"  Output: {output_analysis['total_tensors']} tensors, {output_analysis['file_size_gb']:.2f} GB")
        print(f"  Dtypes: {output_analysis['dtype_counts']}")

    # Validate all tensors are FP32
    non_fp32 = {k: v for k, v in output_analysis['dtype_counts'].items() if k != 'F32'}
    if non_fp32:
        raise ValueError(f"Conversion failed: non-FP32 tensors remain: {non_fp32}")

    return {
        'input_path': input_path,
        'output_path': output_path,
        'tensors_converted': converted_count,
        'tensors_unchanged': already_fp32_count,
        'input_size_gb': input_analysis['file_size_gb'],
        'output_size_gb': output_analysis['file_size_gb'],
        'size_increase_gb': output_analysis['file_size_gb'] - input_analysis['file_size_gb']
    }


def find_ref_controlnet() -> str:
    """Find openpose_pre.safetensors in the ref directory."""
    script_dir = Path(__file__).parent.parent
    ref_path = script_dir.parent / "ref" / "openpose_pre.safetensors"

    if ref_path.exists():
        return str(ref_path)

    # Also check relative to current working directory
    cwd_ref = Path.cwd().parent / "ref" / "openpose_pre.safetensors"
    if cwd_ref.exists():
        return str(cwd_ref)

    raise FileNotFoundError(
        f"Could not find openpose_pre.safetensors in ref directory.\n"
        f"Checked: {ref_path}\n"
        f"Please provide explicit input path."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert ControlNet models from BF16 to FP32"
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Input safetensors file path"
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="Output safetensors file path"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-convert ref/openpose_pre.safetensors to controlnet/openpose_fp32.safetensors"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze precision, don't convert"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    if args.auto:
        # Auto mode: find ref file and convert to controlnet directory
        input_path = find_ref_controlnet()
        script_dir = Path(__file__).parent.parent
        output_path = str(script_dir / "controlnet" / "openpose_fp32.safetensors")
    elif args.input:
        input_path = args.input
        output_path = args.output
        if not output_path and not args.analyze_only:
            # Default output name
            base = os.path.splitext(args.input)[0]
            output_path = f"{base}_fp32.safetensors"
    else:
        parser.print_help()
        sys.exit(1)

    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    if args.analyze_only:
        analysis = analyze_precision(input_path)
        print(f"File: {input_path}")
        print(f"Size: {analysis['file_size_gb']:.2f} GB")
        print(f"Tensors: {analysis['total_tensors']}")
        print(f"Dtypes: {analysis['dtype_counts']}")
        return

    verbose = not args.quiet

    if verbose:
        print("=" * 60)
        print("ControlNet BF16 to FP32 Conversion")
        print("=" * 60)

    try:
        stats = convert_to_fp32(input_path, output_path, verbose=verbose)

        if verbose:
            print("=" * 60)
            print("Conversion complete!")
            print(f"  Input:  {stats['input_size_gb']:.2f} GB")
            print(f"  Output: {stats['output_size_gb']:.2f} GB")
            print(f"  Size increase: {stats['size_increase_gb']:.2f} GB")
            print("=" * 60)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
