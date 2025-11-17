#!/usr/bin/env python3
"""
NoobAI XL V-Pred 1.0 - Main Entry Point

This is the main entry point that orchestrates all modules and provides
both GUI and CLI interfaces.
"""

import os
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
    try:
        # Get current engine instance safely (thread-safe and up-to-date)
        current_engine = get_engine_safely()
        if current_engine:
            current_engine.clear_memory()
        resource_pool.clear()
        logger.info("Resources cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

# Register cleanup
atexit.register(cleanup_resources)

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
        logger.info("Starting NoobAI XL V-Pred 1.0 - Hash Consistency Edition")
        logger.info(f"Performance monitoring: {'Enabled' if perf_monitor.enabled else 'Disabled'}")
        logger.info(f"Output directory: {OUTPUT_DIR}")

        # Display network mode information
        if args.host == "0.0.0.0":
            logger.info("🌐 LAN Access Mode: Enabled")
            logger.info("   Interface will be accessible from any device on your local network")
            logger.info(f"   Server will bind to: 0.0.0.0:{args.port}")
        else:
            logger.info(f"🖥️  Local Access Mode: Running on {args.host}:{args.port}")

        # Pre-load CSV data
        logger.info("Loading CSV data for prompt formatter...")
        get_prompt_data()

        # Create and launch interface
        demo = create_interface()
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
