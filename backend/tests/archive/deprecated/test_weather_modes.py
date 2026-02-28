#!/usr/bin/env python3
"""Test weather functionality in different modes"""

import asyncio
import os
import sys
import json

# Add backend to path
sys.path.insert(0, '.')

async def test_weather_modes():
    """Test weather in different modes to see responses"""
    print("🌤️ Testing Weather System Modes\n")
    print("=" * 60)
    
    # Test 1: Check if vision is available
    print("\n1️⃣ Checking system components...")
    
    has_api_key = bool(os.getenv('ANTHROPIC_API_KEY'))
    print(f"   ANTHROPIC_API_KEY: {'✅ Set' if has_api_key else '❌ Not set'}")
    
    try:
        from api.jarvis_factory import get_app_state, get_vision_analyzer
        app_state = get_app_state()
        
        if app_state:
            print("   App State: ✅ Available")
            
            if hasattr(app_state, 'vision_analyzer'):
                print("   Vision Analyzer: ✅ Available")
            else:
                print("   Vision Analyzer: ❌ Not in app state")
                
            if hasattr(app_state, 'weather_system'):
                print("   Weather System: ✅ Available") 
            else:
                print("   Weather System: ❌ Not in app state")
        else:
            print("   App State: ❌ Not available (server may not be running)")
            
    except Exception as e:
        print(f"   Error checking components: {e}")
    
    # Test 2: Direct weather system test
    print("\n2️⃣ Testing weather system directly...")
    
    try:
        from system_control.weather_system_config import get_weather_system
        weather_system = get_weather_system()
        
        if weather_system:
            print("   Weather System Retrieved: ✅")
            
            # Check if it has vision
            has_vision = hasattr(weather_system, 'weather_system') and \
                        hasattr(weather_system.weather_system, 'vision_handler') and \
                        weather_system.weather_system.vision_handler is not None
                        
            print(f"   Has Vision Handler: {'✅' if has_vision else '❌'}")
            
            # Test weather retrieval
            print("\n   Testing weather retrieval...")
            result = await weather_system.get_weather("What's the weather today?")
            
            if result.get('success'):
                print("   ✅ Weather retrieved successfully")
                print(f"   Source: {result.get('source', 'unknown')}")
                print(f"   Response preview: {result.get('formatted_response', '')[:100]}...")
            else:
                print(f"   ❌ Weather retrieval failed: {result.get('error', 'Unknown error')}")
                
        else:
            print("   Weather System: ❌ Not initialized")
            
    except Exception as e:
        print(f"   Error testing weather system: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Test through API
    print("\n3️⃣ Testing through Ironcliw API...")
    
    try:
        from api.jarvis_voice_api import jarvis_api, IroncliwCommand
        
        # Test weather command
        command = IroncliwCommand(text="What's the weather today?")
        
        # Get raw response (not JSONResponse)
        from api.jarvis_voice_api import IroncliwVoiceAPI
        api = IroncliwVoiceAPI()
        result = await api.process_command(command)
        
        print(f"   Response: {result.get('response', '')[:150]}...")
        print(f"   Status: {result.get('status', 'unknown')}")
        print(f"   Mode: {result.get('mode', 'not specified')}")
        print(f"   Command Type: {result.get('command_type', 'unknown')}")
        
    except Exception as e:
        print(f"   Error testing API: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("\n📊 Summary:")
    print("\nFor FULL weather mode with vision analysis, you need:")
    print("1. ANTHROPIC_API_KEY environment variable set")
    print("2. Backend server running (python main.py)")
    print("3. Vision analyzer initialized")
    print("4. Weather system initialized with vision")
    
    print("\nCurrent capabilities:")
    if has_api_key:
        print("✅ Can use vision analysis for weather")
    else:
        print("❌ Limited mode - will open Weather app and navigate")


if __name__ == "__main__":
    # Check if backend is running
    import requests
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            print("✅ Backend server is running\n")
        else:
            print("⚠️ Backend server returned unexpected status\n")
    except:
        print("❌ Backend server is not running!")
        print("   Please start it with: cd backend && python main.py\n")
    
    asyncio.run(test_weather_modes())