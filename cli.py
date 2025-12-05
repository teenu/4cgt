"""
NoobAI XL V-Pred 1.0 - CLI Functions

This module contains CLI-specific functions for listing adapters,
generating images, and parsing command-line arguments.
"""

import os
import sys
import argparse
from config import (
    logger, OPTIMAL_SETTINGS, MODEL_CONFIG, DEFAULT_NEGATIVE_PROMPT,
    DORA_SEARCH_DIRECTORIES
)
from utils import (
    discover_dora_adapters, get_dora_adapter_by_name, validate_model_path,
    validate_dora_path, detect_adapter_precision, get_user_friendly_error,
    calculate_image_hash, find_dora_path, parse_manual_dora_schedule
)
from ui_helpers import find_model_path, validate_parameters
from engine import NoobAIEngine

# ============================================================================
# CLI FUNCTIONS
# ============================================================================

def cli_list_adapters():
    """List all discovered DoRA adapters."""
    print("🎯 Discovered DoRA Adapters:")
    adapters = discover_dora_adapters()

    if not adapters:
        print("   No DoRA adapters found in search directories.")
        print("   Search directories:")
        for directory in DORA_SEARCH_DIRECTORIES:
            if os.path.exists(directory):
                print(f"     ✓ {directory}")
            else:
                print(f"     ✗ {directory} (not found)")
        return

    for i, adapter in enumerate(adapters):
        print(f"   [{i}] {adapter['display_name']}")
        print(f"       Path: {adapter['path']}")
    print()

def cli_generate(args):
    """Generate image in CLI mode."""
    # Set CLI mode flag for error handling behavior
    os.environ['NOOBAI_CLI_MODE'] = '1'

    # Track engine for cleanup in finally block
    engine = None

    try:
        # Handle list-adapters option
        if hasattr(args, 'list_dora_adapters') and args.list_dora_adapters:
            cli_list_adapters()
            return 0

        # Initialize engine
        if not args.model_path:
            args.model_path = find_model_path()
            if not args.model_path:
                print("❌ No model found. Please specify --model-path")
                return 1

        # Validate model path
        is_valid, path_or_error = validate_model_path(args.model_path)
        if not is_valid:
            print(f"❌ {path_or_error}")
            return 1

        print(f"🚀 Initializing engine with model: {path_or_error}")

        # Handle DoRA if enabled
        dora_path_to_use = None
        if args.enable_dora:
            if hasattr(args, 'dora_adapter') and args.dora_adapter is not None:
                # Select by index
                adapters = discover_dora_adapters()
                if 0 <= args.dora_adapter < len(adapters):
                    adapter_info = adapters[args.dora_adapter]
                    dora_path_to_use = adapter_info['path']
                    print(f"🎯 DoRA adapter: {adapter_info['display_name']}")
                else:
                    print(f"❌ Invalid adapter index {args.dora_adapter}. Available: 0-{len(adapters)-1}")
                    return 1
            elif hasattr(args, 'dora_name') and args.dora_name:
                # Select by name
                adapter_info = get_dora_adapter_by_name(args.dora_name)
                if adapter_info:
                    dora_path_to_use = adapter_info['path']
                    print(f"🎯 DoRA adapter: {adapter_info['display_name']}")
                else:
                    print(f"❌ DoRA adapter '{args.dora_name}' not found")
                    print("Available adapters:")
                    cli_list_adapters()
                    return 1
            elif args.dora_path:
                # Manual path specification
                dora_valid, dora_result = validate_dora_path(args.dora_path)
                if dora_valid:
                    dora_path_to_use = dora_result
                    precision = detect_adapter_precision(dora_result)
                    print(f"🎯 DoRA adapter: {os.path.basename(dora_result)} ({precision})")
                else:
                    print(f"⚠️ DoRA validation failed: {dora_result}")
                    return 1
            else:
                # Auto-detect DoRA
                auto_dora_path = find_dora_path()
                if auto_dora_path:
                    dora_path_to_use = auto_dora_path
                    precision = detect_adapter_precision(auto_dora_path)
                    print(f"🎯 DoRA adapter: {os.path.basename(auto_dora_path)} ({precision}, auto-detected)")
                else:
                    print("⚠️ DoRA enabled but no valid DoRA file found")
                    print("Use --list-dora-adapters to see available adapters")

        engine = NoobAIEngine(
            model_path=path_or_error,
            enable_dora=args.enable_dora,
            dora_path=dora_path_to_use,
            adapter_strength=args.adapter_strength,
            dora_start_step=args.dora_start_step
        )

        # Parse resolution
        width = args.width or OPTIMAL_SETTINGS['width']
        height = args.height or OPTIMAL_SETTINGS['height']

        # Validate parameters
        param_error = validate_parameters(
            width, height, args.steps, args.cfg_scale, args.rescale_cfg, args.adapter_strength, args.dora_start_step
        )
        if param_error:
            print(param_error)
            return 1

        # Prepare prompt
        prompt = (args.prompt or "").strip()
        if not prompt:
            print("❌ Please provide a prompt")
            return 1

        # Handle manual DoRA schedule if provided
        manual_schedule_csv = None
        if args.enable_dora and args.dora_toggle_mode == "manual":
            if args.dora_manual_schedule:
                manual_schedule, warning = parse_manual_dora_schedule(args.dora_manual_schedule, args.steps)
                if warning:
                    print(f"⚠️ {warning}")
                if manual_schedule:
                    manual_schedule_csv = args.dora_manual_schedule
                    print(f"🎯 Using manual DoRA schedule ({len(manual_schedule)} steps)")
                else:
                    print("⚠️ Manual DoRA schedule is invalid - DoRA will be OFF for all steps")
            else:
                print("⚠️ Manual toggle mode selected but no schedule provided (--dora-manual-schedule)")
                print("   DoRA will be OFF for all steps")

        # Generate image
        print(f"🎨 Generating image...")
        print(f"   Prompt: {prompt}")
        print(f"   Resolution: {width}x{height}")
        print(f"   Steps: {args.steps}")
        print(f"   CFG Scale: {args.cfg_scale}")

        def progress_callback(progress, desc):
            print(f"   {desc}")

        image, seed, info = engine.generate(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            width=width,
            height=height,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            rescale_cfg=args.rescale_cfg,
            seed=args.seed,
            adapter_strength=args.adapter_strength if args.enable_dora else None,
            enable_dora=args.enable_dora,
            dora_start_step=args.dora_start_step if args.enable_dora else None,
            dora_toggle_mode=(None if args.dora_toggle_mode == "none" else args.dora_toggle_mode) if args.enable_dora else None,
            dora_manual_schedule=manual_schedule_csv if args.enable_dora else None,
            progress_callback=progress_callback
        )

        # Save image with standardized settings
        output_path = args.output or f"noobai_{seed}.png"
        saved_path = engine.save_image_standardized(image, output_path)

        # Validate save succeeded before calculating hash
        if not saved_path or not os.path.exists(saved_path):
            print(f"❌ Failed to save image to {output_path}")
            return 1

        # Calculate and display hash
        image_hash = calculate_image_hash(saved_path)

        print(f"✅ Image saved to: {output_path}")
        print(f"🌱 Seed: {seed}")
        print(f"📄 MD5 Hash: {image_hash}")

        # Show DoRA info if enabled
        if engine.dora_loaded:
            print(f"🎯 DoRA: {os.path.basename(engine.dora_path)} (strength: {engine.adapter_strength})")

        if args.verbose:
            print("\nGeneration Info:")
            print(info)

        return 0

    except Exception as e:
        error_msg = get_user_friendly_error(e)
        print(f"❌ Generation failed: {error_msg}")
        logger.error(f"CLI generation error: {e}")
        return 1

    finally:
        # Always teardown engine if it was created, regardless of success or failure
        if engine is not None:
            try:
                engine.teardown_engine()
            except Exception as teardown_error:
                logger.warning(f"Error during engine teardown: {teardown_error}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="NoobAI XL V-Pred 1.0 - Professional AI Image Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                          # Launch GUI locally (default)
  %(prog)s --lan                                    # Launch GUI accessible on LAN (no auto-browser)
  %(prog)s --lan --port 8080                        # LAN mode on custom port
  %(prog)s --cli --prompt "cat girl, anime"        # CLI generation
  %(prog)s --cli --prompt "dragon" --steps 40      # CLI with custom steps
  %(prog)s --cli --prompt "landscape" --width 1024 --height 768  # Custom resolution
  %(prog)s --cli --prompt "portrait" --enable-dora --dora-toggle-mode standard  # DoRA standard toggle
  %(prog)s --cli --prompt "portrait" --enable-dora --dora-toggle-mode smart     # DoRA smart toggle
  %(prog)s --cli --prompt "portrait" --enable-dora --dora-toggle-mode manual --dora-manual-schedule "1, 0, 0, 1"  # DoRA manual toggle
  %(prog)s --list-dora-adapters                     # List available DoRA adapters
  %(prog)s --cli --prompt "anime girl" --enable-dora  # CLI with DoRA adapter (auto-detect)
  %(prog)s --cli --prompt "fantasy" --enable-dora --dora-adapter 0  # Select by index
  %(prog)s --cli --prompt "portrait" --enable-dora --dora-name "noobai_vp10_stabilizer_v0.280a.safetensors"  # Select by name
  %(prog)s --cli --prompt "portrait" --enable-dora --dora-path /path/to/dora.safetensors --adapter-strength 0.8 --dora-start-step 10  # DoRA activates at step 10/35
  %(prog)s --cli --prompt "landscape" --enable-dora --dora-start-step 1   # DoRA active from first step (default)
  %(prog)s --cli --prompt "abstract art" --enable-dora --dora-start-step 25 --steps 40  # Late DoRA activation for subtle effects
  %(prog)s --cli --prompt "portrait" --enable-dora --dora-toggle-mode standard  # Toggle: ON,OFF,ON,OFF throughout
  %(prog)s --cli --prompt "landscape" --enable-dora --dora-toggle-mode smart  # Toggle: ON,OFF through step 20, then ON
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--gui",
        action="store_true",
        default=False,
        help="Launch GUI mode (default)"
    )
    mode_group.add_argument(
        "--cli",
        action="store_true",
        default=False,
        help="Use CLI mode for batch generation"
    )

    # Model configuration
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to NoobAI model file (.safetensors)"
    )

    # DoRA adapter configuration
    parser.add_argument(
        "--enable-dora",
        action="store_true",
        help="Enable DoRA (Weight-Decomposed Low-Rank Adaptation) adapter"
    )
    parser.add_argument(
        "--list-dora-adapters",
        action="store_true",
        help="List all discovered DoRA adapters and exit"
    )
    parser.add_argument(
        "--dora-adapter",
        type=int,
        help="Select DoRA adapter by index (use --list-dora-adapters to see options)"
    )
    parser.add_argument(
        "--dora-name",
        type=str,
        help="Select DoRA adapter by filename (e.g., 'noobai_vp10_stabilizer_v0.280a.safetensors')"
    )
    parser.add_argument(
        "--dora-path",
        type=str,
        help="Manual path to DoRA adapter file (.safetensors). Overrides --dora-adapter and --dora-name."
    )

    # CLI-specific options
    cli_group = parser.add_argument_group("CLI Generation Options")
    cli_group.add_argument(
        "--prompt",
        type=str,
        help="Positive prompt for image generation"
    )
    cli_group.add_argument(
        "--negative-prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt (default: built-in negative prompt)"
    )
    cli_group.add_argument(
        "--width",
        type=int,
        default=OPTIMAL_SETTINGS['width'],
        help=f"Image width (default: {OPTIMAL_SETTINGS['width']})"
    )
    cli_group.add_argument(
        "--height",
        type=int,
        default=OPTIMAL_SETTINGS['height'],
        help=f"Image height (default: {OPTIMAL_SETTINGS['height']})"
    )
    cli_group.add_argument(
        "--steps",
        type=int,
        default=OPTIMAL_SETTINGS['steps'],
        help=f"Number of inference steps (default: {OPTIMAL_SETTINGS['steps']})"
    )
    cli_group.add_argument(
        "--cfg-scale",
        type=float,
        default=OPTIMAL_SETTINGS['cfg_scale'],
        help=f"CFG scale (default: {OPTIMAL_SETTINGS['cfg_scale']})"
    )
    cli_group.add_argument(
        "--rescale-cfg",
        type=float,
        default=OPTIMAL_SETTINGS['rescale_cfg'],
        help=f"Rescale CFG (default: {OPTIMAL_SETTINGS['rescale_cfg']})"
    )
    cli_group.add_argument(
        "--seed",
        type=int,
        help="Seed for generation (random if not specified)"
    )
    cli_group.add_argument(
        "--output",
        type=str,
        help="Output file path (default: noobai_<seed>.png in current directory)"
    )
    cli_group.add_argument(
        "--adapter-strength",
        type=float,
        default=OPTIMAL_SETTINGS['adapter_strength'],
        help=f"DoRA adapter strength when enabled (default: {OPTIMAL_SETTINGS['adapter_strength']}, range: {MODEL_CONFIG.MIN_ADAPTER_STRENGTH}-{MODEL_CONFIG.MAX_ADAPTER_STRENGTH})"
    )
    cli_group.add_argument(
        "--dora-start-step",
        type=int,
        default=OPTIMAL_SETTINGS['dora_start_step'],
        help=f"Step at which DoRA adapter activates (default: {OPTIMAL_SETTINGS['dora_start_step']}, range: {MODEL_CONFIG.MIN_DORA_START_STEP}-{MODEL_CONFIG.MAX_DORA_START_STEP})"
    )
    cli_group.add_argument(
        "--dora-toggle-mode",
        type=str,
        choices=["none", "standard", "smart", "optimized", "manual"],
        default="optimized",
        help="DoRA toggle mode. 'none': disable toggle mode (use dora-start-step instead). 'standard': ON,OFF,ON,OFF throughout. 'smart': ON,OFF to step 20, then ON. 'optimized': empirically-tuned phased activation (recommended). 'manual': use custom CSV schedule"
    )
    cli_group.add_argument(
        "--dora-manual-schedule",
        type=str,
        default=None,
        help="Manual DoRA schedule as CSV (e.g., '1, 0, 0, 1'). Only used with --dora-toggle-mode manual. Format: comma+space-separated 0/1 values. If entries < steps: missing = 0 (OFF). If entries > steps: extras ignored. Non-0/1 treated as 0."
    )
    cli_group.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed generation information"
    )

    # GUI options
    gui_group = parser.add_argument_group("GUI Options")
    gui_group.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for GUI server (default: 127.0.0.1)"
    )
    gui_group.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for GUI server (default: 7860)"
    )
    gui_group.add_argument(
        "--share",
        action="store_true",
        help="Create public Gradio link"
    )
    gui_group.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically"
    )
    gui_group.add_argument(
        "--lan",
        action="store_true",
        help="Enable LAN access (binds to 0.0.0.0, disables auto-browser). Access from any device on your network."
    )

    args = parser.parse_args()

    # Normalize mode flags to a single source of truth (GUI default)
    if args.gui:
        args.cli = False
    elif args.cli:
        args.gui = False
    else:
        # Neither flag provided: default to GUI
        args.gui = True
        args.cli = False

    # Process --lan flag (convenient shortcut for LAN access)
    if args.lan:
        args.host = "0.0.0.0"
        args.no_browser = True

    return args
