#!/usr/bin/env python3
"""
JARVIS Agentic Task Runner - CLI Interface
===========================================

Thin CLI wrapper for the unified AgenticTaskRunner.
The core functionality is in backend/core/agentic_task_runner.py.

This CLI can be used for:
- Standalone agentic task execution
- Testing Computer Use capabilities
- Interactive task experimentation

For production use, tasks should flow through the JARVISSupervisor
via the TieredCommandRouter (Tier 2 commands).

Usage:
    # Interactive mode
    python run_agentic_task.py

    # Execute single goal
    python run_agentic_task.py --goal "Open Safari and find the weather"

    # Execute with specific mode
    python run_agentic_task.py --goal "Organize my desktop" --mode autonomous

    # With voice narration
    python run_agentic_task.py --goal "Connect to my TV" --narrate

    # Debug mode
    python run_agentic_task.py --goal "Open System Preferences" --debug

Author: JARVIS AI System
Version: 2.0.0 (CLI Wrapper)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Load environment variables from .env file BEFORE any other imports
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        backend_env = Path(__file__).parent / "backend" / ".env"
        if backend_env.exists():
            load_dotenv(backend_env)
except ImportError:
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, _, value = line.partition('=')
                    value = value.strip().strip('"').strip("'")
                    os.environ.setdefault(key.strip(), value)

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(debug: bool = False, log_file: Path = None) -> logging.Logger:
    """Configure logging for the CLI."""
    level = logging.DEBUG if debug else logging.INFO

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return logging.getLogger("agentic_cli")


# ============================================================================
# CLI Functions
# ============================================================================

async def run_single_task(
    goal: str,
    mode: str,
    narrate: bool,
    logger: logging.Logger,
) -> int:
    """Execute a single agentic task."""
    from core.agentic_task_runner import (
        AgenticTaskRunner,
        AgenticRunnerConfig,
        RunnerMode,
    )

    logger.info(f"Goal: {goal}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Narrate: {narrate}")
    logger.info("-" * 40)

    # Create TTS callback if narration enabled
    tts_callback = None
    if narrate:
        try:
            from voice.text_to_speech_service import TextToSpeechService
            tts = TextToSpeechService()

            async def speak(text: str):
                await tts.speak(text)

            tts_callback = speak
        except ImportError:
            logger.warning("TTS not available, narration disabled")

    # Create and initialize runner
    config = AgenticRunnerConfig()
    runner = AgenticTaskRunner(
        config=config,
        tts_callback=tts_callback,
        logger=logger,
    )

    try:
        await runner.initialize()

        # Execute the task
        result = await runner.run(
            goal=goal,
            mode=RunnerMode(mode),
            narrate=narrate,
        )

        # Print results
        print("\n" + "=" * 40)
        print("RESULT")
        print("=" * 40)
        print(f"Success: {result.success}")
        print(f"Message: {result.final_message}")
        print(f"Time: {result.execution_time_ms:.0f}ms")
        print(f"Actions: {result.actions_count}")
        print(f"Mode: {result.mode}")

        if result.learning_insights:
            print("\nInsights:")
            for insight in result.learning_insights:
                print(f"  - {insight}")

        if result.error:
            print(f"\nError: {result.error}")

        return 0 if result.success else 1

    finally:
        await runner.shutdown()


async def interactive_mode(logger: logging.Logger):
    """Run in interactive REPL mode."""
    from core.agentic_task_runner import (
        AgenticTaskRunner,
        AgenticRunnerConfig,
        RunnerMode,
    )

    print("\n" + "=" * 60)
    print("JARVIS Agentic Task Runner - Interactive Mode")
    print("=" * 60)
    print("\nCommands:")
    print("  Type a goal to execute it")
    print("  /mode <autonomous|supervised|direct> - Change mode")
    print("  /status - Show runner status")
    print("  /quit - Exit")
    print()

    # Create runner
    config = AgenticRunnerConfig()
    runner = AgenticTaskRunner(config=config, logger=logger)
    await runner.initialize()

    mode = RunnerMode.SUPERVISED

    try:
        while True:
            try:
                user_input = input(f"[{mode.value}] Goal> ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ('/quit', '/exit', 'quit', 'exit'):
                    break

                if user_input.startswith('/mode '):
                    mode_str = user_input[6:].strip().lower()
                    try:
                        mode = RunnerMode(mode_str)
                        print(f"Mode changed to: {mode.value}")
                    except ValueError:
                        print("Invalid mode. Options: autonomous, supervised, direct")
                    continue

                if user_input == '/status':
                    stats = runner.get_stats()
                    print(json.dumps(stats, indent=2))
                    continue

                # Execute goal
                print(f"\nExecuting: {user_input}")
                print("-" * 40)

                result = await runner.run(
                    goal=user_input,
                    mode=mode,
                    narrate=True,
                )

                print("-" * 40)
                print(f"Success: {result.success}")
                print(f"Message: {result.final_message}")
                print(f"Time: {result.execution_time_ms:.0f}ms")
                print(f"Actions: {result.actions_count}")

                if result.learning_insights:
                    print("Insights:")
                    for insight in result.learning_insights:
                        print(f"  - {insight}")
                print()

            except KeyboardInterrupt:
                print("\nInterrupted")
                break
            except EOFError:
                break

    finally:
        await runner.shutdown()

    print("Goodbye!")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="JARVIS Agentic Task Runner CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_agentic_task.py
  python run_agentic_task.py --goal "Open Safari and find the weather"
  python run_agentic_task.py --goal "Organize my desktop" --mode autonomous
  python run_agentic_task.py --goal "Open Mail" --narrate --debug
"""
    )

    parser.add_argument(
        "--goal", "-g",
        help="Goal to execute (if not provided, enters interactive mode)"
    )

    parser.add_argument(
        "--mode", "-m",
        choices=["direct", "supervised", "autonomous"],
        default="supervised",
        help="Execution mode (default: supervised)"
    )

    parser.add_argument(
        "--narrate", "-n",
        action="store_true",
        help="Enable voice narration"
    )

    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug logging"
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        help="Log to file"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(debug=args.debug, log_file=args.log_file)

    # Print banner
    print("\n" + "=" * 60)
    print("JARVIS Agentic Task Runner v2.0")
    print("=" * 60)

    # Run
    if args.goal:
        return await run_single_task(
            goal=args.goal,
            mode=args.mode,
            narrate=args.narrate,
            logger=logger,
        )
    else:
        await interactive_mode(logger)
        return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
