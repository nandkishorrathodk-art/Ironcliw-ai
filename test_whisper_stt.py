#!/usr/bin/env python3
"""
Test Whisper STT to fix Ironcliw transcription
"""

import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import os

def record_audio(duration=3, sample_rate=16000):
    """Record audio from microphone"""
    print(f"🎤 Recording for {duration} seconds...")
    print("   Say: 'unlock my screen'")

    # Record audio
    audio = sd.rec(int(duration * sample_rate),
                   samplerate=sample_rate,
                   channels=1,
                   dtype='int16')
    sd.wait()  # Wait for recording to finish

    # Save to temporary file
    temp_file = tempfile.mktemp(suffix='.wav')
    wav.write(temp_file, sample_rate, audio)

    print("✅ Recording complete")
    return temp_file

def transcribe_with_whisper(audio_file):
    """Transcribe audio using Whisper"""
    print("\n🔍 Transcribing with Whisper...")

    # Load Whisper model
    model = whisper.load_model("base")

    # Transcribe
    result = model.transcribe(audio_file)

    return result["text"]

def test_stt_pipeline():
    """Test the complete STT pipeline"""
    print("\n" + "="*60)
    print("🎯 TESTING WHISPER STT FOR Ironcliw")
    print("="*60)

    try:
        # Record audio
        audio_file = record_audio(duration=3)

        # Transcribe
        text = transcribe_with_whisper(audio_file)

        print(f"\n📝 Transcription: '{text.strip()}'")

        # Check if it matches expected commands
        expected_commands = ["unlock my screen", "unlock screen", "screen unlock"]

        text_lower = text.strip().lower()
        matched = False

        for cmd in expected_commands:
            if cmd in text_lower:
                matched = True
                print(f"✅ Matched command: '{cmd}'")
                break

        if not matched:
            print(f"⚠️  No exact match, but transcribed: '{text.strip()}'")

        # Clean up
        os.remove(audio_file)

        return text.strip()

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\n📝 Fallback: Install sounddevice")
        print("   pip install sounddevice")
        return None

def create_jarvis_stt_fix():
    """Create a fix for Ironcliw STT configuration"""
    print("\n" + "="*60)
    print("🔧 Ironcliw STT FIX")
    print("="*60)

    config_fix = {
        "stt_engine": "whisper",
        "whisper_model": "base",
        "language": "en",
        "sample_rate": 16000,
        "audio_duration": 3
    }

    print("\n📝 Add this to Ironcliw configuration:")
    print("-" * 40)
    import json
    print(json.dumps(config_fix, indent=2))
    print("-" * 40)

    print("\n✨ Then restart Ironcliw:")
    print("   python3 start_system.py --stt-engine whisper")

if __name__ == "__main__":
    print("🎤 WHISPER STT TEST FOR Ironcliw")
    print("This will fix the '[transcription failed]' issue")
    print("-" * 60)

    # First check if sounddevice is installed
    try:
        import sounddevice

        # Test the pipeline
        result = test_stt_pipeline()

        if result:
            print(f"\n✅ STT is working! Transcribed: '{result}'")
            create_jarvis_stt_fix()
        else:
            print("\n❌ STT test failed")

    except ImportError:
        print("📦 Installing required package...")
        os.system("pip install sounddevice")
        print("\n✅ Package installed. Run this script again.")