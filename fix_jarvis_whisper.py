#!/usr/bin/env python3
"""
Fix Ironcliw STT by injecting Whisper override
"""

import sys
import os
import time

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), "backend")
sys.path.insert(0, backend_path)

def fix_jarvis_stt():
    """Apply Whisper override to Ironcliw"""

    print("🔧 FIXING Ironcliw STT WITH WHISPER")
    print("="*50)

    # Import the override
    from voice.whisper_override import patch_jarvis_stt, get_whisper_stt

    # Initialize Whisper
    print("📦 Loading Whisper model...")
    whisper_stt = get_whisper_stt()
    whisper_stt.initialize()

    # Apply patch
    print("🔨 Patching Ironcliw STT system...")
    patch_jarvis_stt()

    print("\n✅ Ironcliw STT FIXED!")
    print("-"*50)
    print("Whisper is now the default STT engine")
    print("Transcription should work correctly")
    print("\n🎤 Test with: 'Hey Ironcliw, unlock my screen'")

if __name__ == "__main__":
    fix_jarvis_stt()

    # Keep running to maintain patch
    print("\nPatch applied. Keep this running while testing Ironcliw.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nPatch removed")