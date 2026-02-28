#!/usr/bin/env python3
"""
Install voice dependencies for Ironcliw
Handles platform-specific requirements
"""

import subprocess
import sys
import platform

def install_voice_dependencies():
    """Install voice-related dependencies"""
    print("🎤 Installing Ironcliw Voice Dependencies...")
    
    # Basic voice packages
    packages = [
        "SpeechRecognition",
        "pyttsx3",
        "pygame",
        "pyaudio",
        "numpy"
    ]
    
    # Platform-specific handling
    system = platform.system()
    
    if system == "Darwin":  # macOS
        print("\n📱 Detected macOS (M1 compatible)")
        print("Installing portaudio (required for pyaudio)...")
        try:
            subprocess.run(["brew", "install", "portaudio"], check=True)
        except subprocess.CalledProcessError:
            print("⚠️  Please install Homebrew first: https://brew.sh")
            print("Then run: brew install portaudio")
    elif system == "Linux":
        print("\n🐧 Detected Linux")
        print("You may need to install: sudo apt-get install portaudio19-dev")
    elif system == "Windows":
        print("\n🪟 Detected Windows")
        print("PyAudio should install automatically")
    
    # Install Python packages
    print("\n📦 Installing Python packages...")
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Failed to install {package}: {e}")
            if package == "pyaudio" and system == "Darwin":
                print("Try: pip install --global-option='build_ext' --global-option='-I/opt/homebrew/include' --global-option='-L/opt/homebrew/lib' pyaudio")
    
    print("\n✅ Voice dependencies installation complete!")
    print("\n🎯 Test Ironcliw voice system with:")
    print("   python test_jarvis_voice.py")
    print("\n🚀 Or start the full system with:")
    print("   python start_system.py")

if __name__ == "__main__":
    install_voice_dependencies()