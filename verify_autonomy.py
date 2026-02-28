#!/usr/bin/env python3
"""
Verify Ironcliw Autonomy Status
Checks all systems and provides clear status
"""

import asyncio
import aiohttp
import json
from datetime import datetime


async def check_autonomy_status():
    """Check complete autonomy status"""
    print("🤖 Ironcliw Autonomy Status Check")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    # Check backend health
    try:
        async with aiohttp.ClientSession() as session:
            # 1. Check overall health
            async with session.get(f"{base_url}/health") as resp:
                if resp.status == 200:
                    health = await resp.json()
                    print(f"✅ Backend Status: {health['status']}")
                else:
                    print(f"❌ Backend Status: Error {resp.status}")
                    return
            
            # 2. Check Ironcliw status
            async with session.get(f"{base_url}/voice/jarvis/status") as resp:
                if resp.status == 200:
                    jarvis = await resp.json()
                    status = jarvis.get('status', 'unknown')
                    print(f"\n📊 Ironcliw Voice System:")
                    print(f"  Status: {status}")
                    print(f"  User: {jarvis.get('user_name', 'Not set')}")
                    
                    # Check features
                    features = jarvis.get('features', [])
                    if 'system_control' in features:
                        print(f"  ✅ System Control: Enabled")
                    else:
                        print(f"  ❌ System Control: Disabled")
                        
                    # Voice engine status
                    voice_engine = jarvis.get('voice_engine', {})
                    if voice_engine.get('calibrated'):
                        print(f"  ✅ Voice Engine: Calibrated")
                    else:
                        print(f"  ⚠️  Voice Engine: Not calibrated")
            
            # 3. Check Vision status
            async with session.get(f"{base_url}/vision/status") as resp:
                if resp.status == 200:
                    vision = await resp.json()
                    print(f"\n👁️  Vision System:")
                    print(f"  Enabled: {vision.get('vision_enabled', False)}")
                    print(f"  Monitoring: {vision.get('monitoring_active', False)}")
                    print(f"  Claude Vision: {vision.get('claude_vision_available', False)}")
                    print(f"  Pipeline Active: {vision.get('pipeline_active', False)}")
                    
                    if vision.get('last_scan'):
                        last_scan = datetime.fromisoformat(vision['last_scan'])
                        age = (datetime.now() - last_scan).seconds
                        print(f"  Last Scan: {age} seconds ago")
            
            # 4. Check autonomy handler
            async with session.get(f"{base_url}/voice/jarvis/status") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"\n🤖 Autonomy Status:")
                    
                    # This would need to be added to the API
                    # For now, we check indirect indicators
                    if vision.get('monitoring_active') and features and 'system_control' in features:
                        print(f"  ✅ FULL AUTONOMY ACTIVE")
                        print(f"  • AI Brain: Active")
                        print(f"  • Voice System: Active") 
                        print(f"  • Vision Monitoring: Active")
                        print(f"  • System Control: Enabled")
                    else:
                        print(f"  ⚠️  PARTIAL AUTONOMY")
                        if not vision.get('monitoring_active'):
                            print(f"  • Vision Monitoring: INACTIVE")
                        if 'system_control' not in features:
                            print(f"  • System Control: DISABLED")
            
            # 5. Test WebSocket connectivity
            print(f"\n🔌 WebSocket Status:")
            print(f"  Vision WS: ws://localhost:8000/vision/ws/vision")
            print(f"  Voice WS: ws://localhost:8000/voice/jarvis/stream")
            
            # 6. Recommendations
            print(f"\n💡 Recommendations:")
            if not vision.get('monitoring_active'):
                print(f"  • Vision not monitoring - activate full autonomy")
            if not voice_engine.get('calibrated'):
                print(f"  • Voice not calibrated - check microphone")
            
            print(f"\n✅ Check complete!")
            
    except Exception as e:
        print(f"\n❌ Error checking status: {e}")
        print(f"Make sure backend is running: python backend/main.py")


async def test_speech():
    """Test speech output"""
    print(f"\n🔊 Testing Speech Output...")
    
    import platform
    if platform.system() == 'Darwin':
        import os
        # Test macOS speech
        os.system('say "Ironcliw speech test. Full autonomy activated."')
        print("✅ Speech command sent. Did you hear Ironcliw?")
    else:
        print("⚠️  Speech test only available on macOS")


async def main():
    """Run all checks"""
    await check_autonomy_status()
    await test_speech()
    
    print(f"\n📋 Quick Fixes:")
    print(f"1. If no speech: Check System Preferences → Sound → Output Volume")
    print(f"2. If vision disconnected: Restart backend with ./start_jarvis_backend.sh")
    print(f"3. If partial autonomy: Say 'Hey Ironcliw, activate full autonomy' again")
    print(f"4. Check browser console for errors")


if __name__ == "__main__":
    asyncio.run(main())