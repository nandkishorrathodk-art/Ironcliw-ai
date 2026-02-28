#!/usr/bin/env python3
"""
Ironcliw Event-Driven Startup Script
Initializes and runs Ironcliw with the new event-driven architecture
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add backend to Python path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

# Import the event-driven coordinator
from jarvis_event_coordinator import main

# Setup environment variables if not set
def setup_environment():
    """Setup default environment variables"""
    defaults = {
        "Ironcliw_USER": "Sir",
        "Ironcliw_DEBUG": "false",
        "Ironcliw_WEB_UI": "true",
        "Ironcliw_LOG_LEVEL": "INFO"
    }
    
    for key, value in defaults.items():
        if key not in os.environ:
            os.environ[key] = value
            
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  WARNING: ANTHROPIC_API_KEY not set")
        print("   Some features will be limited without the API key")
        print("   Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        print()

def print_banner():
    """Print Ironcliw startup banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗            ║
    ║     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝            ║
    ║     ██║███████║██████╔╝██║   ██║██║███████╗            ║
    ║██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║            ║
    ║╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║            ║
    ║ ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝            ║
    ║                                                          ║
    ║          Event-Driven AI Assistant v2.0                  ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    
    🚀 Starting Ironcliw with Event-Driven Architecture...
    """
    print(banner)

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        "anthropic",
        "psutil",
        "aiohttp",
        "pyyaml",
        "numpy",
        "librosa",
        "scikit-learn"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
            
    if missing:
        print("❌ Missing required packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\nInstall with:")
        print(f"   pip install {' '.join(missing)}")
        sys.exit(1)

async def run_jarvis():
    """Run Ironcliw with proper initialization"""
    # Print startup banner
    print_banner()
    
    # Setup environment
    setup_environment()
    
    # Check requirements
    check_requirements()
    
    # Configure logging
    log_level = getattr(logging, os.getenv("Ironcliw_LOG_LEVEL", "INFO"))
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('jarvis_event.log')
        ]
    )
    
    # Suppress some noisy loggers
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    
    print("✅ Environment configured")
    print(f"👤 User: {os.getenv('Ironcliw_USER')}")
    print(f"🔍 Debug: {os.getenv('Ironcliw_DEBUG')}")
    print(f"🌐 Web UI: {os.getenv('Ironcliw_WEB_UI')}")
    print()
    
    # Run the main coordinator
    try:
        await main()
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    try:
        # Run Ironcliw
        asyncio.run(run_jarvis())
    except KeyboardInterrupt:
        print("\n\n👋 Ironcliw shutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)