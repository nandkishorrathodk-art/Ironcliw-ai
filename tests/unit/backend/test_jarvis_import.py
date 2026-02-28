#!/usr/bin/env python3
"""Test Ironcliw import chain"""

import os
import sys

# Set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['USE_TORCH'] = '1'
os.environ['USE_TF'] = '0'

print("Testing Ironcliw import chain...")
print("=" * 50)

# Test 1: Basic imports
try:
    import torch
    import torchaudio
    print("✅ PyTorch and torchaudio imported successfully")
    print(f"   - torch version: {torch.__version__}")
    print(f"   - torchaudio version: {torchaudio.__version__}")
except Exception as e:
    print(f"❌ Failed to import PyTorch/torchaudio: {e}")
    sys.exit(1)

# Test 2: ML Enhanced Voice System
try:
    from backend.voice.ml_enhanced_voice_system import MLEnhancedVoiceSystem
    print("✅ ML Enhanced Voice System imported successfully")
except Exception as e:
    print(f"❌ Failed to import ML Enhanced Voice System: {e}")

# Test 3: Ironcliw Voice
try:
    from backend.voice.jarvis_voice import EnhancedIroncliwVoiceAssistant
    print("✅ Ironcliw Voice imported successfully")
except Exception as e:
    print(f"❌ Failed to import Ironcliw Voice: {e}")

# Test 4: Ironcliw API
try:
    from backend.api.jarvis_voice_api import IroncliwVoiceAPI
    print("✅ Ironcliw Voice API imported successfully")
except Exception as e:
    print(f"❌ Failed to import Ironcliw Voice API: {e}")

# Test 5: Initialize Ironcliw
try:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        jarvis_api = IroncliwVoiceAPI()
        print("✅ Ironcliw Voice API initialized successfully")
        print(f"   - Ironcliw available: {jarvis_api.jarvis_available}")
    else:
        print("⚠️  ANTHROPIC_API_KEY not set, skipping initialization test")
except Exception as e:
    print(f"❌ Failed to initialize Ironcliw Voice API: {e}")

print("\nImport chain test complete!")