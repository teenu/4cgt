"""CLI functions for NoobAI XL V-Pred 1.0."""

import os
import argparse
from PIL import Image
from config import (
    logger, OPTIMAL_SETTINGS, MODEL_CONFIG, CONTROLNET_CONFIG, DEFAULT_NEGATIVE_PROMPT,
    DORA_SEARCH_DIRECTORIES, CONTROLNET_SEARCH_DIRECTORIES,
    OPTIMIZED_DORA_SETTINGS, OPTIMIZED_DORA_SCHEDULE_CSV
)
from utils import (
    discover_dora_adapters, get_dora_adapter_by_name, validate_model_path,
    validate_dora_path, detect_adapter_precision, get_user_friendly_error,
    calculate_image_hash, find_dora_path, parse_manual_dora_schedule,
    discover_controlnet_models, get_controlnet_by_name, find_controlnet_path,
    validate_controlnet_path, detect_controlnet_precision
)
from ui.engine_manager import find_model_path
from ui.validation import validate_parameters
from engine import NoobAIEngine


def cli_list_adapters():
    """List all discovered DoRA adapters."""
    print("🎯 Discovered DoRA Adapters:")
    adapters = discover_dora_adapters()

    if not adapters:
        print("   No DoRA adapters found in search directories.")
        print("   Search directories:")
        for directory in DORA_SEARCH_DIRECTORIES:
            status = "✓" if os.path.exists(directory) else "✗ (not found)"
            print(f"     {status} {directory}")
        return

    for i, adapter in enumerate(adapters):
        print(f"   [{i}] {adapter['display_name']}")
        print(f"       Path: {adapter['path']}")
    print()


def cli_list_controlnets():
    """List all discovered ControlNet models."""
    print("🎭 Discovered ControlNet Models:")
    models = discover_controlnet_models()

    if not models:
        print("   No ControlNet models found in search directories.")
        print("   Search directories:")
        for directory in CONTROLNET_SEARCH_DIRECTORIES:
            status = "✓" if os.path.exists(directory) else "✗ (not found)"
            print(f"     {status} {directory}")
        return

    for i, model in enumerate(models):
        print(f"   [{i}] {model['display_name']}")
        print(f"       Path: {model['path']}")
        print(f"       Type: {model['type']}")
    print()


def cli_generate(args):
    """Generate image in CLI mode."""
    os.environ['NOOBAI_CLI_MODE'] = '1'
    engine = None

    try:
        if hasattr(args, 'list_dora_adapters') and args.list_dora_adapters:
            cli_list_adapters()
            return 0

        if hasattr(args, 'list_controlnets') and args.list_controlnets:
            cli_list_controlnets()
            return 0

        if not args.model_path:
            args.model_path = find_model_path()
            if not args.model_path:
                print("❌ No model found. Please specify --model-path")
                return 1

        is_valid, path_or_error = validate_model_path(args.model_path)
        if not is_valid:
            print(f"❌ {path_or_error}")
            return 1

        print(f"🚀 Initializing engine with model: {path_or_error}")

        dora_path_to_use = None
        if args.enable_dora:
            if hasattr(args, 'dora_adapter') and args.dora_adapter is not None:
                adapters = discover_dora_adapters()
                if 0 <= args.dora_adapter < len(adapters):
                    adapter_info = adapters[args.dora_adapter]
                    dora_path_to_use = adapter_info['path']
                    print(f"🎯 DoRA adapter: {adapter_info['display_name']}")
                else:
                    print(f"❌ Invalid adapter index {args.dora_adapter}. Available: 0-{len(adapters)-1}")
                    return 1
            elif hasattr(args, 'dora_name') and args.dora_name:
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
                dora_valid, dora_result = validate_dora_path(args.dora_path)
                if dora_valid:
                    dora_path_to_use = dora_result
                    precision = detect_adapter_precision(dora_result)
                    print(f"🎯 DoRA adapter: {os.path.basename(dora_result)} ({precision})")
                else:
                    print(f"⚠️ DoRA validation failed: {dora_result}")
                    return 1
            else:
                auto_dora_path = find_dora_path()
                if auto_dora_path:
                    dora_path_to_use = auto_dora_path
                    precision = detect_adapter_precision(auto_dora_path)
                    print(f"🎯 DoRA adapter: {os.path.basename(auto_dora_path)} ({precision}, auto-detected)")
                else:
                    print("⚠️ DoRA enabled but no valid DoRA file found")
                    print("Use --list-dora-adapters to see available adapters")

        if args.force_fp32:
            print("🔒 Parity mode: FP32 inference enabled for reproducibility")

        if args.optimize:
            print("⚡ Performance mode: TF32 enabled" + (" (torch.compile skipped for DoRA compatibility)" if args.enable_dora else " + torch.compile (first run slower)"))

        # Handle ControlNet
        controlnet_path_to_use = None
        if args.pose_image:
            # If pose image is provided, we need a ControlNet
            if hasattr(args, 'controlnet_index') and args.controlnet_index is not None:
                models = discover_controlnet_models()
                if 0 <= args.controlnet_index < len(models):
                    model_info = models[args.controlnet_index]
                    controlnet_path_to_use = model_info['path']
                    print(f"🎭 ControlNet: {model_info['display_name']}")
                else:
                    print(f"❌ Invalid ControlNet index {args.controlnet_index}. Available: 0-{len(models)-1}")
                    return 1
            elif hasattr(args, 'controlnet_name') and args.controlnet_name:
                model_info = get_controlnet_by_name(args.controlnet_name)
                if model_info:
                    controlnet_path_to_use = model_info['path']
                    print(f"🎭 ControlNet: {model_info['display_name']}")
                else:
                    print(f"❌ ControlNet '{args.controlnet_name}' not found")
                    print("Available ControlNets:")
                    cli_list_controlnets()
                    return 1
            elif args.controlnet_path:
                cn_valid, cn_result = validate_controlnet_path(args.controlnet_path)
                if cn_valid:
                    controlnet_path_to_use = cn_result
                    precision = detect_controlnet_precision(cn_result)
                    print(f"🎭 ControlNet: {os.path.basename(cn_result)} ({precision})")
                else:
                    print(f"⚠️ ControlNet validation failed: {cn_result}")
                    return 1
            else:
                # Auto-detect ControlNet
                auto_cn_path = find_controlnet_path("openpose")
                if auto_cn_path:
                    controlnet_path_to_use = auto_cn_path
                    precision = detect_controlnet_precision(auto_cn_path)
                    print(f"🎭 ControlNet: {os.path.basename(auto_cn_path)} ({precision}, auto-detected)")
                else:
                    print("⚠️ Pose image provided but no ControlNet found")
                    print("Use --list-controlnets to see available models")
                    return 1

        engine = NoobAIEngine(
            model_path=path_or_error,
            enable_dora=args.enable_dora,
            dora_path=dora_path_to_use,
            adapter_strength=args.adapter_strength,
            dora_start_step=args.dora_start_step,
            force_fp32=args.force_fp32,
            optimize=args.optimize,
            controlnet_path=controlnet_path_to_use,
            controlnet_scale=args.controlnet_scale
        )

        width = args.width or OPTIMAL_SETTINGS['width']
        height = args.height or OPTIMAL_SETTINGS['height']

        param_error = validate_parameters(
            width, height, args.steps, args.cfg_scale, args.rescale_cfg,
            args.adapter_strength, args.dora_start_step
        )
        if param_error:
            print(param_error)
            return 1

        prompt = (args.prompt or "").strip()
        if not prompt:
            print("❌ Please provide a prompt")
            return 1

        manual_schedule_csv = None
        if args.enable_dora and args.dora_toggle_mode == "optimized":
            # Optimized mode: use predefined settings and schedule
            opt = OPTIMIZED_DORA_SETTINGS
            args.steps = opt['steps']
            args.cfg_scale = opt['cfg_scale']
            args.rescale_cfg = opt['rescale_cfg']
            args.adapter_strength = opt['adapter_strength']
            manual_schedule_csv = OPTIMIZED_DORA_SCHEDULE_CSV
            print(f"🎯 Using Optimized DoRA mode:")
            print(f"   Steps: {opt['steps']}, CFG: {opt['cfg_scale']}, Rescale: {opt['rescale_cfg']}, Strength: {opt['adapter_strength']}")
            print(f"   Schedule: {len(opt['schedule'])} steps predefined")
        elif args.enable_dora and args.dora_toggle_mode == "manual":
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

        # Load pose image if provided
        pose_image = None
        if args.pose_image:
            try:
                pose_image = Image.open(args.pose_image).convert("RGB")
                print(f"🎭 Pose image: {args.pose_image} ({pose_image.size[0]}x{pose_image.size[1]})")
            except Exception as e:
                print(f"❌ Failed to load pose image: {e}")
                return 1

        print(f"🎨 Generating image...")
        print(f"   Prompt: {prompt}")
        print(f"   Resolution: {width}x{height}")
        print(f"   Steps: {args.steps}")
        print(f"   CFG Scale: {args.cfg_scale}")
        if pose_image:
            print(f"   ControlNet Scale: {args.controlnet_scale}")

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
            dora_toggle_mode=args.dora_toggle_mode if args.enable_dora else None,
            dora_manual_schedule=manual_schedule_csv if args.enable_dora else None,
            progress_callback=progress_callback,
            pose_image=pose_image,
            controlnet_scale=args.controlnet_scale if pose_image else None
        )

        output_path = args.output or f"noobai_{seed}.png"
        saved_path = engine.save_image_standardized(image, output_path)

        if not saved_path or not os.path.exists(saved_path):
            print(f"❌ Failed to save image to {output_path}")
            return 1

        image_hash = calculate_image_hash(saved_path)

        print(f"✅ Image saved to: {output_path}")
        print(f"🌱 Seed: {seed}")
        print(f"📄 MD5 Hash: {image_hash}")

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
        epilog="""Examples:
  GUI Mode:
    %(prog)s                           # Launch GUI locally (default)
    %(prog)s --lan                     # Launch GUI on LAN (0.0.0.0:7860)
    %(prog)s --lan --port 8080         # Custom port
    %(prog)s --force-fp32              # Launch GUI with FP32 parity mode

  CLI Mode:
    %(prog)s --cli --prompt "cat girl, anime"
    %(prog)s --cli --prompt "dragon" --steps 40 --width 1024 --height 768

  Parity Mode (FP32):
    %(prog)s --force-fp32              # GUI with FP32 parity mode
    %(prog)s --lan --force-fp32        # LAN mode with FP32 parity
    %(prog)s --cli --prompt "test" --force-fp32  # CLI with FP32 parity

  Performance Mode:
    %(prog)s --optimize                          # GUI with TF32 + torch.compile
    %(prog)s --lan --optimize                    # LAN mode with performance optimizations
    %(prog)s --cli --prompt "test" --optimize    # CLI with ~2x faster inference

  DoRA Adapters:
    %(prog)s --list-dora-adapters                        # List available adapters
    %(prog)s --cli --prompt "portrait" --enable-dora     # Auto-detect adapter
    %(prog)s --cli --prompt "fantasy" --enable-dora --dora-adapter 0
    %(prog)s --cli --prompt "landscape" --enable-dora --dora-toggle-mode optimized
    %(prog)s --cli --prompt "portrait" --enable-dora --dora-toggle-mode manual --dora-manual-schedule "1,0,0,1"

  ControlNet (Pose):
    %(prog)s --list-controlnets                                          # List available ControlNets
    %(prog)s --cli --prompt "1girl" --pose-image pose.png                # Use pose for anatomy control
    %(prog)s --cli --prompt "1girl" --pose-image pose.png --controlnet-scale 0.8
"""
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--gui", action="store_true", default=True, help="Launch GUI mode (default)")
    mode_group.add_argument("--cli", action="store_true", help="Use CLI mode for batch generation")

    parser.add_argument("--model-path", type=str, help="Path to NoobAI model file (.safetensors)")
    parser.add_argument("--enable-dora", action="store_true", help="Enable DoRA adapter")
    parser.add_argument("--list-dora-adapters", action="store_true", help="List DoRA adapters and exit")
    parser.add_argument("--dora-adapter", type=int, help="Select DoRA adapter by index")
    parser.add_argument("--dora-name", type=str, help="Select DoRA adapter by filename")
    parser.add_argument("--dora-path", type=str, help="Manual path to DoRA adapter (.safetensors)")

    # ControlNet arguments
    parser.add_argument("--list-controlnets", action="store_true", help="List ControlNet models and exit")
    parser.add_argument("--pose-image", type=str, help="Path to OpenPose skeleton image for pose control")
    parser.add_argument("--controlnet-path", type=str, help="Manual path to ControlNet model (.safetensors)")
    parser.add_argument("--controlnet-index", type=int, help="Select ControlNet model by index")
    parser.add_argument("--controlnet-name", type=str, help="Select ControlNet model by filename")
    parser.add_argument("--controlnet-scale", type=float, default=CONTROLNET_CONFIG.DEFAULT_CONDITIONING_SCALE,
                       help=f"ControlNet conditioning scale (default: {CONTROLNET_CONFIG.DEFAULT_CONDITIONING_SCALE}, range: {CONTROLNET_CONFIG.MIN_CONDITIONING_SCALE}-{CONTROLNET_CONFIG.MAX_CONDITIONING_SCALE})")

    cli_group = parser.add_argument_group("CLI Generation Options")
    cli_group.add_argument("--prompt", type=str, help="Positive prompt")
    cli_group.add_argument("--negative-prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt")
    cli_group.add_argument("--width", type=int, help=f"Image width (default: {OPTIMAL_SETTINGS['width']})")
    cli_group.add_argument("--height", type=int, help=f"Image height (default: {OPTIMAL_SETTINGS['height']})")
    cli_group.add_argument("--steps", type=int, default=OPTIMAL_SETTINGS['steps'],
                          help=f"Inference steps (default: {OPTIMAL_SETTINGS['steps']})")
    cli_group.add_argument("--cfg-scale", type=float, default=OPTIMAL_SETTINGS['cfg_scale'],
                          help=f"CFG scale (default: {OPTIMAL_SETTINGS['cfg_scale']})")
    cli_group.add_argument("--rescale-cfg", type=float, default=OPTIMAL_SETTINGS['rescale_cfg'],
                          help=f"Rescale CFG (default: {OPTIMAL_SETTINGS['rescale_cfg']})")
    cli_group.add_argument("--seed", type=int, help="Seed for generation (random if not specified)")
    cli_group.add_argument("--output", type=str, help="Output file path (default: noobai_<seed>.png)")
    cli_group.add_argument("--adapter-strength", type=float, default=OPTIMAL_SETTINGS['adapter_strength'],
                          help=f"DoRA strength (default: {OPTIMAL_SETTINGS['adapter_strength']}, range: {MODEL_CONFIG.MIN_ADAPTER_STRENGTH}-{MODEL_CONFIG.MAX_ADAPTER_STRENGTH})")
    cli_group.add_argument("--dora-start-step", type=int, default=OPTIMAL_SETTINGS['dora_start_step'],
                          help=f"DoRA activation step (default: {OPTIMAL_SETTINGS['dora_start_step']})")
    cli_group.add_argument("--dora-toggle-mode", type=str, choices=["manual", "optimized"],
                          help="DoRA toggle mode: manual (custom schedule) or optimized (pre-tuned 34-step schedule)")
    cli_group.add_argument("--dora-manual-schedule", type=str,
                          help="Manual DoRA schedule CSV (e.g., '1,0,0,1')")
    cli_group.add_argument("--verbose", action="store_true", help="Show detailed generation info")

    precision_group = parser.add_argument_group("Precision and Performance Options")
    precision_group.add_argument("--force-fp32", action="store_true",
                          help="Force FP32 inference for parity with FP32 directory models (uses more VRAM)")
    precision_group.add_argument("--parity-mode", action="store_true", dest="force_fp32",
                          help="Alias for --force-fp32 (parity testing mode)")
    precision_group.add_argument("--optimize", action="store_true",
                          help="Enable TF32 matmuls + torch.compile (compile skipped when DoRA loaded). ~1.5-2x faster")

    gui_group = parser.add_argument_group("GUI Options")
    gui_group.add_argument("--host", type=str, default="127.0.0.1", help="GUI server host")
    gui_group.add_argument("--port", type=int, default=7860, help="GUI server port")
    gui_group.add_argument("--share", action="store_true", help="Create public Gradio link")
    gui_group.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    gui_group.add_argument("--lan", action="store_true", help="Enable LAN access (binds to 0.0.0.0)")

    args = parser.parse_args()

    if args.cli:
        args.gui = False
    else:
        args.gui = True

    if args.lan:
        args.host = "0.0.0.0"
        args.no_browser = True

    return args
