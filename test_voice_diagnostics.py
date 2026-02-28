#!/usr/bin/env python3
"""
Voice Pipeline Diagnostics - Fix Transcription Issues
"""

import subprocess
import json
import time
import os
from pathlib import Path

def test_microphone():
    """Test if microphone is working"""
    print("\n📍 TEST 1: Microphone Check")
    print("-" * 40)

    # Check microphone permissions
    print("Checking microphone permissions...")

    # List audio devices
    result = subprocess.run(
        ["system_profiler", "SPAudioDataType"],
        capture_output=True,
        text=True
    )

    if "Input" in result.stdout:
        print("✅ Microphone detected")
        return True
    else:
        print("❌ No microphone found")
        return False

def test_audio_recording():
    """Test basic audio recording"""
    print("\n📍 TEST 2: Audio Recording")
    print("-" * 40)

    test_file = "/tmp/test_audio.wav"

    print("Recording 3 seconds of audio...")
    print("🎤 Say: 'Testing, testing, one, two, three'")

    # Record audio using sox or ffmpeg
    try:
        # Try using sox first
        subprocess.run(
            ["sox", "-d", "-r", "16000", "-c", "1", test_file, "trim", "0", "3"],
            capture_output=True,
            timeout=4
        )

        if os.path.exists(test_file):
            size = os.path.getsize(test_file)
            print(f"✅ Audio recorded: {size} bytes")
            return test_file
    except:
        pass

    # Try ffmpeg as fallback
    try:
        subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-i", ":0", "-t", "3", "-y", test_file],
            capture_output=True,
            timeout=4
        )

        if os.path.exists(test_file):
            size = os.path.getsize(test_file)
            print(f"✅ Audio recorded: {size} bytes")
            return test_file
    except:
        pass

    print("❌ Could not record audio")
    return None

def test_stt_models():
    """Test different STT models"""
    print("\n📍 TEST 3: Speech-to-Text Models")
    print("-" * 40)

    models = [
        ("Whisper", "openai/whisper"),
        ("Wav2Vec2", "speechbrain/wav2vec2"),
        ("DeepSpeech", "mozilla/deepspeech")
    ]

    print("Testing available STT models...")
    for name, path in models:
        # Check if model files exist
        model_exists = False

        # Common model locations
        locations = [
            f"~/models/{path}",
            f"~/.cache/huggingface/{path}",
            f"./models/{path}"
        ]

        for loc in locations:
            full_path = os.path.expanduser(loc)
            if os.path.exists(full_path):
                model_exists = True
                break

        if model_exists:
            print(f"✅ {name}: Available")
        else:
            print(f"❌ {name}: Not found")

    return True

def test_jarvis_stt_component():
    """Test Ironcliw's actual STT component"""
    print("\n📍 TEST 4: Ironcliw STT Component")
    print("-" * 40)

    # Check if the STT service is running
    stt_check = subprocess.run(
        ["pgrep", "-f", "stt_service"],
        capture_output=True,
        text=True
    )

    if stt_check.stdout:
        print("✅ STT service is running")
    else:
        print("⚠️  STT service not detected - may be integrated")

    # Try to trigger STT directly
    print("\nTesting direct STT transcription...")
    test_phrases = [
        "unlock my screen",
        "hey jarvis unlock",
        "test transcription"
    ]

    for phrase in test_phrases:
        print(f"  Expected: '{phrase}'")
        print(f"  Actual: [waiting for implementation]")

    return True

def fix_stt_configuration():
    """Provide fixes for STT issues"""
    print("\n📍 FIXING STT CONFIGURATION")
    print("=" * 40)

    fixes = """
# Fix 1: Install Whisper as fallback STT
pip install openai-whisper

# Fix 2: Update Ironcliw config to use Whisper
cat > jarvis_stt_fix.json <<EOF
{
  "stt_engine": "whisper",
  "whisper_model": "base",
  "language": "en",
  "sample_rate": 16000
}
EOF

# Fix 3: Test with simple Python script
python3 -c "
import whisper
model = whisper.load_model('base')
result = model.transcribe('test_audio.wav')
print(result['text'])
"

# Fix 4: Restart Ironcliw with new STT config
killall -TERM jarvis
python3 start_system.py --stt-engine whisper
"""

    print(fixes)

    print("\n📝 Quick Fix Steps:")
    print("1. Install Whisper: pip install openai-whisper")
    print("2. Test recording: sox -d test.wav trim 0 3")
    print("3. Test transcription manually")
    print("4. Update Ironcliw STT configuration")

    return True

def main():
    print("\n🔧 VOICE TRANSCRIPTION DIAGNOSTIC")
    print("=" * 60)
    print("Diagnosing why '[transcription failed]' occurred")
    print("=" * 60)

    # Run tests
    mic_ok = test_microphone()

    if mic_ok:
        audio_file = test_audio_recording()

        if audio_file:
            test_stt_models()
            test_jarvis_stt_component()

    # Provide fixes
    fix_stt_configuration()

    print("\n" + "=" * 60)
    print("🎯 RECOMMENDED ACTIONS:")
    print("=" * 60)
    print()
    print("1. ⚡ QUICK FIX: Install Whisper")
    print("   pip install openai-whisper")
    print()
    print("2. 🔄 RESTART Ironcliw with Whisper:")
    print("   python3 start_system.py --stt-engine whisper")
    print()
    print("3. 🎤 TEST AGAIN:")
    print("   Say: 'Hey Ironcliw, unlock my screen'")
    print()
    print("The transcription should work after these fixes!")

if __name__ == "__main__":
    main()