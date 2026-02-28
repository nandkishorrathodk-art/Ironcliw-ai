#!/usr/bin/env python3
"""
Ironcliw Optimized Startup Script
Launches Ironcliw with full system integration and resource management
Optimized for 16GB RAM MacBook Pro with dynamic resource allocation
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add backend to Python path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

# Import the system integration coordinator
from core.system_integration_coordinator import SystemIntegrationCoordinator

# Import the event-driven coordinator
from jarvis_event_coordinator import IroncliwEventCoordinator

# Setup environment variables if not set
def setup_environment():
    """Setup default environment variables"""
    defaults = {
        "Ironcliw_USER": "Sir",
        "Ironcliw_DEBUG": "false",
        "Ironcliw_WEB_UI": "true",
        "Ironcliw_LOG_LEVEL": "INFO",
        "Ironcliw_PERFORMANCE_DASHBOARD": "true"
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

def print_optimized_banner():
    """Print Ironcliw optimized startup banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗                    ║
    ║     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝                    ║
    ║     ██║███████║██████╔╝██║   ██║██║███████╗                    ║
    ║██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║                    ║
    ║╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║                    ║
    ║ ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝                    ║
    ║                                                                  ║
    ║          🚀 OPTIMIZED AI ASSISTANT v2.0 🚀                      ║
    ║                                                                  ║
    ║  Features:                                                       ║
    ║  ✓ Dynamic Resource Management (16GB RAM Optimized)              ║
    ║  ✓ Cross-Component Memory Sharing                               ║
    ║  ✓ Intelligent Health Monitoring                                ║
    ║  ✓ Graceful Degradation Strategies                             ║
    ║  ✓ Comprehensive Error Recovery                                 ║
    ║  ✓ Real-time Performance Dashboard                              ║
    ║  ✓ Screen Monitoring (30 FPS with macOS purple indicator)       ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    
    🎯 Starting Ironcliw with System Integration & Performance Optimization...
    """
    print(banner)

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        "anthropic",
        "psutil",
        "aiohttp",
        "aiohttp_cors",
        "yaml",  # PyYAML imports as 'yaml'
        "numpy",
        "librosa",
        "sklearn",  # scikit-learn imports as 'sklearn'
        "watchdog",
        "jsonschema"
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

async def run_jarvis_optimized():
    """Run Ironcliw with full optimization"""
    # Print startup banner
    print_optimized_banner()
    
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
            logging.FileHandler('jarvis_optimized.log')
        ]
    )
    
    # Suppress some noisy loggers
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    
    print("✅ Environment configured")
    print(f"👤 User: {os.getenv('Ironcliw_USER')}")
    print(f"🔍 Debug: {os.getenv('Ironcliw_DEBUG')}")
    print(f"🌐 Web UI: {os.getenv('Ironcliw_WEB_UI')}")
    print(f"📊 Performance Dashboard: {os.getenv('Ironcliw_PERFORMANCE_DASHBOARD')}")
    print()
    
    # Create system integration coordinator
    print("🔧 Initializing System Integration...")
    system_coordinator = SystemIntegrationCoordinator()
    
    # Create Ironcliw event coordinator
    print("🧠 Initializing Ironcliw Core Systems...")
    jarvis_coordinator = IroncliwEventCoordinator(
        user_name=os.getenv("Ironcliw_USER", "Sir"),
        enable_web_ui=os.getenv("Ironcliw_WEB_UI", "true").lower() == "true"
    )
    
    try:
        # Start system integration first
        print("⚡ Starting Resource Management & Optimization...")
        await system_coordinator.start()
        
        # Start Ironcliw coordinator
        print("🎯 Starting Ironcliw Event System...")
        await jarvis_coordinator.start()
        
        print("\n" + "="*60)
        print("✨ Ironcliw is fully operational!")
        print("="*60)
        
        print(f"\n📍 Available interfaces:")
        print(f"   Event Web UI: http://localhost:8888")
        print(f"   Performance Dashboard: http://localhost:8889")
        print(f"\n💡 Say 'Hey Ironcliw' to interact")
        print(f"\n🎥 Screen Monitoring Commands:")
        print(f"   'start monitoring my screen' - Begin 30 FPS capture")
        print(f"   'stop monitoring' - End video streaming")
        print(f"   (Purple macOS indicator appears when active)")
        print("   Press Ctrl+C to stop\n")
        
        # Keep running
        while jarvis_coordinator.state.is_running:
            await asyncio.sleep(1)
            
            # Periodic status check
            if int(asyncio.get_event_loop().time()) % 60 == 0:  # Every minute
                # Get system report
                report = system_coordinator.get_system_report()
                
                # Log key metrics
                logging.debug(f"Memory Usage: {report['resource_management']['usage_percent']:.1f}%")
                logging.debug(f"Overall Health: {report['health_monitoring']['overall_health']:.1f}")
                
    except KeyboardInterrupt:
        print("\n\n👋 Shutdown requested...")
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        
        # Handle emergency
        print("\n⚠️  Emergency shutdown initiated...")
        await system_coordinator.handle_emergency(str(e))
        
    finally:
        # Clean shutdown
        print("\n🛑 Shutting down Ironcliw...")
        
        # Stop Ironcliw coordinator
        await jarvis_coordinator.stop()
        
        # Stop system coordinator
        await system_coordinator.stop()
        
        print("✅ Ironcliw shutdown complete")
        print("👋 Goodbye!")

if __name__ == "__main__":
    try:
        # Run Ironcliw with optimization
        asyncio.run(run_jarvis_optimized())
    except KeyboardInterrupt:
        print("\n👋 Ironcliw shutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)