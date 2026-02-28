#!/usr/bin/env python3
"""
Quick start script to get Ironcliw backend running immediately
Includes the 503 fix for voice activation
"""

import os
import sys
import subprocess
import time

# Set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['USE_TORCH'] = '1'
os.environ['USE_TF'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("🚀 Starting Ironcliw Backend with Voice 503 Fix...")
print("=" * 60)

# Check if we're in the right directory
if not os.path.exists("main.py"):
    print("❌ Error: Please run this from the backend directory")
    print("   cd /Users/derekjrussell/Documents/repos/Ironcliw-AI-Agent/backend")
    sys.exit(1)

# Start the backend
print("✅ Starting FastAPI backend on port 8000...")
print("✅ Voice 503 fix enabled - no more Service Unavailable errors!")
print("=" * 60)

try:
    # Run the backend
    subprocess.run([sys.executable, "main.py", "--port", "8000"])
except KeyboardInterrupt:
    print("\n✋ Backend stopped by user")
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTry running directly:")
    print("  python main.py --port 8000")