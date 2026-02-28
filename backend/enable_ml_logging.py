#!/usr/bin/env python3
"""
Enable ML Model Real-Time Logging
=================================

Run this to enable detailed console logging of ML model loading/unloading.
This shows exactly when models are loaded, why, and current memory usage.
"""

import logging
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from ml_logging_config import setup_ml_logging, ml_logger


def configure_ml_logging():
    """Configure ML logging for Ironcliw startup"""
    
    # Setup enhanced logging
    ml_logger_instance, visualizer = setup_ml_logging(
        level=logging.INFO,
        enable_visualization=True
    )
    
    # Also configure root logger to show our custom logs
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove default handlers that might interfere
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    print("\n" + "=" * 70)
    print("🚀 Ironcliw ML MODEL REAL-TIME LOGGING ENABLED")
    print("=" * 70)
    print("\nYou will see detailed information about:")
    print("  • 🔄 Model loading start (with reason)")
    print("  • ✅ Successful loads (with timing)")
    print("  • ❌ Failed loads (with error)")
    print("  • 📤 Model unloading (with freed memory)")
    print("  • 🧠 Memory checks and decisions")
    print("  • 🔀 Context changes")
    print("  • 📍 Proximity-based loading")
    print("  • ⚡ Cache hits (instant loads)")
    print("  • 🗜️ Quantized model usage")
    print("  • 🔮 Predictive preloading")
    print("  • 🚨 Critical memory warnings")
    print("\nMemory bar legend:")
    print("  █ = System memory")
    print("  ▓ = ML models memory")  
    print("  | = 35% target")
    print("  ░ = Free memory")
    print("\n" + "=" * 70 + "\n")
    
    return ml_logger_instance, visualizer


# Auto-configure when imported
if __name__ != "__main__":
    configure_ml_logging()
    
    
def patch_jarvis_startup():
    """Patch Ironcliw startup to include enhanced logging"""
    try:
        # Import main components that need logging
        import ml_memory_manager
        import context_aware_loader
        
        print("✅ Enhanced ML logging integrated with Ironcliw")
        print("   Models will be logged as they load/unload\n")
        
    except ImportError as e:
        print(f"⚠️  Warning: Could not patch all components: {e}")
        

if __name__ == "__main__":
    # Test the logging
    ml_logger_instance, visualizer = configure_ml_logging()
    
    print("\n📋 Testing logging output:\n")
    
    # Test various log types
    ml_logger.load_start("test_model", 150.5, "Testing logging system")
    
    import time
    time.sleep(0.5)
    
    ml_logger.load_success("test_model", 1.23, {
        'system_percent': 22.5,
        'ml_models_mb': 150.5,
        'available_mb': 8192
    })
    
    ml_logger.context_change("IDLE", "VOICE_COMMAND", 2)
    
    ml_logger.proximity_change("NEAR", 0.8, "Preloading voice models")
    
    ml_logger.cache_hit("whisper_base")
    
    ml_logger.quantized_load("voice_biometric", 50.0, 6.2)
    
    ml_logger.memory_check({'system_percent': 28.0, 'available_mb': 6000}, "OK to load")
    
    ml_logger.unload("test_model", 150.5, "Context change")
    
    ml_logger.cleanup_triggered("Memory pressure", 3)
    
    ml_logger.critical_memory(72.5, "Emergency unloading all models")
    
    # Test memory visualization
    print("\n📊 Memory Visualization:\n")
    
    visualizer.visualize_memory(
        {
            'system_percent': 28.5,
            'ml_percent': 12.5,
            'ml_models_mb': 2000,
            'available_mb': 11468
        },
        {
            'whisper_base': {'size_mb': 150, 'last_used_s': 5, 'quantized': False},
            'embeddings': {'size_mb': 100, 'last_used_s': 30, 'quantized': False},
            'voice_biometric': {'size_mb': 6, 'last_used_s': 2, 'quantized': True}
        }
    )
    
    print("\n✅ Logging test complete!")