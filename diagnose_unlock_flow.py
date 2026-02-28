#!/usr/bin/env python3
"""
Comprehensive Voice Unlock Diagnostic
Traces the entire flow from frontend → backend → verification
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend"))


async def diagnose_unlock_flow():
    print("=" * 70)
    print("  VOICE UNLOCK FLOW DIAGNOSTIC")
    print("=" * 70)
    print()

    # 1. Check backend process
    print("1️⃣  Checking backend process...")
    import subprocess
    result = subprocess.run(["pgrep", "-fl", "main.py"], capture_output=True, text=True)
    if result.stdout:
        print(f"   ✅ Backend running: {result.stdout.strip()}")
    else:
        print("   ❌ Backend NOT running!")
        return

    # 2. Check WebSocket endpoint
    print("\n2️⃣  Checking WebSocket endpoint...")
    try:
        import websockets
        async with websockets.connect("ws://localhost:8010/ws") as ws:
            print("   ✅ WebSocket connection successful")

            # Send test unlock command
            print("\n3️⃣  Sending test unlock command...")
            test_message = {
                "type": "command",
                "text": "unlock my screen",
                "audio_data": "test_audio_data_placeholder"  # Fake audio for testing
            }

            import json
            await ws.send(json.dumps(test_message))
            print("   ✅ Command sent")

            # Wait for response
            print("\n4️⃣  Waiting for response...")
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=10.0)
                response_data = json.loads(response)
                print(f"   ✅ Response received:")
                print(f"      Type: {response_data.get('type')}")
                print(f"      Text: {response_data.get('text', '')[:100]}")
                if 'confidence' in str(response_data):
                    print(f"      Contains confidence data: Yes")
            except asyncio.TimeoutError:
                print("   ⏱️  Response timeout (10s)")

    except Exception as e:
        print(f"   ❌ WebSocket error: {e}")

    # 5. Check unified_command_processor
    print("\n5️⃣  Checking unified_command_processor...")
    try:
        from api.unified_command_processor import UnifiedCommandProcessor
        processor = UnifiedCommandProcessor()

        # Check if our fix is present
        import inspect
        source = inspect.getsource(processor.process_command)
        if "AudioContainer" in source:
            print("   ✅ Audio passthrough fix IS present")
        else:
            print("   ❌ Audio passthrough fix NOT found (old code)")

    except Exception as e:
        print(f"   ❌ Error checking processor: {e}")

    # 6. Check speaker verification service
    print("\n6️⃣  Checking speaker verification service...")
    try:
        from voice.speaker_verification_service import get_speaker_verification_service
        service = await get_speaker_verification_service()

        print(f"   ✅ Service initialized")
        print(f"   Profiles loaded: {len(service.speaker_profiles)}")
        for name, profile in service.speaker_profiles.items():
            print(f"      - {name}: {profile.total_samples} samples")

        encoder_ready = getattr(service, '_encoder_preloaded', False)
        print(f"   Encoder preloaded: {encoder_ready}")

    except Exception as e:
        print(f"   ❌ Speaker verification error: {e}")

    # 7. Check log files
    print("\n7️⃣  Checking recent logs for unlock commands...")
    log_files = [
        "/Users/derekjrussell/Documents/repos/Ironcliw-AI-Agent/backend/logs/jarvis_optimized_*.log",
        "/Users/derekjrussell/Documents/repos/Ironcliw-AI-Agent/jarvis_startup.log"
    ]

    import glob
    for pattern in log_files:
        files = glob.glob(pattern)
        if files:
            latest = max(files, key=lambda x: Path(x).stat().st_mtime)
            print(f"   Checking: {Path(latest).name}")

            # Check last 100 lines for unlock-related content
            with open(latest, 'r') as f:
                lines = f.readlines()[-100:]
                unlock_lines = [l for l in lines if 'unlock' in l.lower() or 'VOICE-UNLOCK' in l]
                if unlock_lines:
                    print(f"   Found {len(unlock_lines)} unlock-related log entries")
                    print("   Last entry:", unlock_lines[-1][:100])
                else:
                    print(f"   ⚠️  No unlock commands in last 100 lines")

    print("\n" + "=" * 70)
    print("  DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print()
    print("Next Steps:")
    print("1. Try saying 'unlock my screen' in the browser")
    print("2. Check Chrome DevTools Console for errors")
    print("3. Check Chrome DevTools Network tab → WS → Frames")
    print("4. Verify audio is being captured (look for '🎤 Audio captured')")
    print()


if __name__ == "__main__":
    asyncio.run(diagnose_unlock_flow())
