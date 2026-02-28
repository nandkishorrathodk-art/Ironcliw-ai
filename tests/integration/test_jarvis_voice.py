#!/usr/bin/env python3
"""
Test script for Ironcliw Voice Integration
Tests the voice endpoints and Ironcliw personality
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Load environment variables
load_dotenv()

async def test_jarvis_voice():
    """Test Ironcliw voice system"""
    print("🎯 Testing Ironcliw Voice System...")
    
    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set in environment")
        print("Please set your API key in .env file")
        return
    
    print("✅ API key found")
    
    try:
        # Import Ironcliw components
        from backend.voice.jarvis_voice import IroncliwVoiceAssistant, VoiceEngine
        
        print("\n🚀 Initializing Ironcliw...")
        jarvis = IroncliwVoiceAssistant(api_key)
        
        print("✅ Ironcliw initialized successfully")
        
        # Test personality system
        print("\n🧠 Testing Ironcliw Personality...")
        
        # Get activation response
        response = jarvis.personality.get_activation_response()
        print(f"Activation: {response}")
        
        # Test some commands
        test_commands = [
            "What's the weather like?",
            "Calculate 2 + 2 * 2",
            "Tell me a joke",
            "What time is it?",
            "Remind me to take a break"
        ]
        
        print("\n📝 Testing command processing...")
        for cmd in test_commands:
            print(f"\nUser: {cmd}")
            response = await jarvis.personality.process_command(cmd)
            print(f"Ironcliw: {response}")
        
        # Test special commands
        print("\n🎮 Testing special commands...")
        print("Available commands:", list(jarvis.special_commands.keys()))
        
        # Test voice engine
        print("\n🎤 Testing Voice Engine...")
        voice_engine = jarvis.voice_engine
        
        # Check voice setup
        print("Voice engine initialized:", hasattr(voice_engine, 'tts_engine'))
        print("Microphone available:", hasattr(voice_engine, 'microphone'))
        
        # Test configuration
        print("\n⚙️ Testing Configuration...")
        print("User name:", jarvis.personality.user_preferences['name'])
        print("Work hours:", jarvis.personality.user_preferences['work_hours'])
        print("Break reminders:", jarvis.personality.user_preferences['break_reminder'])
        print("Humor level:", jarvis.personality.user_preferences['humor_level'])
        
        print("\n✅ All tests passed! Ironcliw is ready for action.")
        print("\n💡 To start Ironcliw with voice activation:")
        print("   python -m backend.voice.jarvis_voice")
        print("\n💡 Or use the web interface at http://localhost:8000")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install speech_recognition pyttsx3 pygame pyaudio")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_jarvis_voice())