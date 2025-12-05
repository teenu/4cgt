#!/usr/bin/env python3
"""
NoobAI XL V-Pred 1.0 - Main Entry Point

This is the main entry point that orchestrates all modules and provides
both GUI and CLI interfaces.
"""

import os
import sys
import signal

# ============================================================================
# DETERMINISM SETUP - CRITICAL: Must be set before ANY PyTorch imports
# ============================================================================
# Configure CUBLAS workspace for deterministic operations
# This MUST be set before torch is imported to take effect
if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
else:
    existing_value = os.environ['CUBLAS_WORKSPACE_CONFIG']
    if existing_value not in [':4096:8', ':16:8']:
        corrected_value = ':4096:8'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = corrected_value
        print(
            "WARNING: Invalid CUBLAS_WORKSPACE_CONFIG detected. "
            f"Overriding '{existing_value}' with '{corrected_value}' to preserve deterministic CUDA execution.",
            file=sys.stderr
        )

import atexit
from config import logger, OUTPUT_DIR
from state import perf_monitor, resource_pool
from ui_helpers import get_engine_safely
from prompt_formatter import get_prompt_data
from ui import create_interface
from cli import cli_list_adapters, cli_generate, parse_args

# ============================================================================
# CLEANUP
# ============================================================================

def cleanup_resources():
    """Clean up resources on application exit."""
    errors = []

    # Attempt engine cleanup
    try:
        current_engine = get_engine_safely()
        if current_engine:
            current_engine.clear_memory()
    except Exception as e:
        errors.append(f"Engine cleanup: {e}")

    # Attempt resource pool cleanup
    try:
        resource_pool.clear()
    except Exception as e:
        errors.append(f"Resource pool cleanup: {e}")

    # Log results
    if errors:
        logger.error(f"Cleanup errors: {'; '.join(errors)}")
    else:
        logger.info("Resources cleaned up successfully")

# Register cleanup
atexit.register(cleanup_resources)

# ============================================================================
# SIGNAL HANDLERS
# ============================================================================

def signal_handler(signum, frame):
    """Handle interrupt signals (SIGINT, SIGTERM) for clean shutdown."""
    sig_name = signal.Signals(signum).name
    logger.info(f"Received {sig_name}, cleaning up...")
    cleanup_resources()
    sys.exit(0)

# Register signal handlers for clean shutdown
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main application entry point with CLI support."""
    args = None  # Initialize for exception handler access
    try:
        args = parse_args()

        # Handle list-dora-adapters (can be called without CLI flag)
        if hasattr(args, 'list_dora_adapters') and args.list_dora_adapters:
            cli_list_adapters()
            return 0

        # Handle CLI mode
        if args.cli:
            return cli_generate(args)

        # GUI mode (default)
        logger.info("Starting NoobAI XL V-Pred 1.0")
        logger.info(f"Output directory: {OUTPUT_DIR}")

        if args.host == "0.0.0.0":
            logger.info(f"LAN mode enabled on port {args.port}")
        else:
            logger.info(f"Server: {args.host}:{args.port}")

        # Pre-load CSV data
        get_prompt_data()

        # Create and launch interface (pass model_path if specified)
        demo = create_interface(model_path=args.model_path if hasattr(args, 'model_path') else None)

        # Enable queue for progress tracking (required by Gradio 3.50.x)
        demo.queue()

        demo.launch(
            share=args.share,
            inbrowser=not args.no_browser,
            show_error=True,
            server_name=args.host,
            server_port=args.port
        )

        return 0

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Application error: {e}")
        # Print error for CLI mode, re-raise for GUI mode
        if args is not None and (args.cli or (hasattr(args, 'list_dora_adapters') and args.list_dora_adapters)):
            print(f"❌ Error: {e}")
            return 1
        raise
    finally:
        if 'perf_monitor' in globals() and perf_monitor.enabled:
            logger.info("Performance summary:")
            for section, stats in perf_monitor.get_summary().items():
                logger.info(f"  {section}: {stats}")

if __name__ == "__main__":
    exit(main())
