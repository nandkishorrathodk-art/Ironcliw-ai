#!/usr/bin/env python3
"""
Direct Whisper Test - Test if Whisper can transcribe "unlock my screen"
"""

import whisper
import sounddevice as sd
import numpy as np
import tempfile
import scipy.io.wavfile as wav

def test_whisper_transcription():
    print("🎤 WHISPER DIRECT TEST")
    print("="*50)

    # Load Whisper
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    print("✅ Whisper loaded")

    print("\n📢 INSTRUCTIONS:")
    print("When you see 'Recording...', say:")
    print("   'unlock my screen'")
    print("\nPress Enter to start recording...")
    input()

    # Record audio
    duration = 3
    sample_rate = 16000

    print("\n🔴 Recording for 3 seconds...")
    print("Say: 'unlock my screen' NOW!")

    audio = sd.rec(int(duration * sample_rate),
                   samplerate=sample_rate,
                   channels=1,
                   dtype='float32')
    sd.wait()

    print("✅ Recording complete")

    # Save and transcribe
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        wav.write(tmp.name, sample_rate, audio)

        print("🔍 Transcribing...")
        result = model.transcribe(tmp.name)

    text = result["text"].strip()

    print("\n" + "="*50)
    print("📝 RESULT:")
    print(f"   Transcribed: '{text}'")

    # Check if it matches
    if "unlock" in text.lower() and "screen" in text.lower():
        print("   ✅ SUCCESS! Whisper correctly heard 'unlock my screen'")
        print("\n🎉 Whisper is working!")
        print("The issue is that Ironcliw isn't using Whisper properly.")
    else:
        print(f"   ❌ Whisper heard: '{text}'")
        print("   Try speaking more clearly or adjusting microphone")

    print("="*50)

if __name__ == "__main__":
    test_whisper_transcription()