#!/usr/bin/env python3
"""
Simple backend startup script that handles import issues
"""

import os
import sys
import subprocess

# Set environment variables before imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['USE_TORCH'] = '1'
os.environ['USE_TF'] = '0'
os.environ['BACKEND_PORT'] = '8010'  # Set default port to 8010

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

print("🚀 Starting Ironcliw Backend...")
print("=" * 50)

# Check for API key
if not os.getenv("ANTHROPIC_API_KEY"):
    print("⚠️  Warning: ANTHROPIC_API_KEY not set")
    print("   Ironcliw will have limited functionality")
else:
    print("✅ Anthropic API key found")

# Check optional dependencies
try:
    import librosa
    print("✅ Audio processing (librosa) available")
except ImportError:
    print("⚠️  Audio ML features limited (librosa not installed)")

try:
    import torch
    print("✅ PyTorch available for deep learning")
except ImportError:
    print("⚠️  Deep learning features limited (PyTorch not installed)")

# Check video streaming capability
try:
    from vision.video_stream_capture import MACOS_CAPTURE_AVAILABLE
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
    if MACOS_CAPTURE_AVAILABLE:
        print("✅ Screen monitoring with native macOS capture (purple indicator)")
    else:
        print("⚠️  Screen monitoring available (fallback mode)")
    print("   Commands: 'start/stop monitoring my screen'")
except ImportError:
    print("⚠️  Screen monitoring not available")

# Run the backend
print("\n🎯 Starting FastAPI server on port 8010...")
print("   WebSocket will be available at ws://localhost:8010/ws")
print("=" * 50 + "\n")

try:
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    main_path = os.path.join(backend_dir, "main.py")
    # Use port 8010 to match frontend expectations
    subprocess.run([sys.executable, main_path, "--port", "8010"])
except KeyboardInterrupt:
    print("\n\n✋ Backend stopped by user")
except Exception as e:
    print(f"\n❌ Error: {e}")
    sys.exit(1)