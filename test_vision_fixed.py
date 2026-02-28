#!/usr/bin/env python
"""Test script to verify vision command routing is working"""

import asyncio
import aiohttp
import json
import time
import subprocess
import sys
import os

async def test_vision_command():
    """Test if vision commands are properly routed"""
    
    # Start the backend
    print("🚀 Starting Ironcliw backend...")
    backend_process = subprocess.Popen(
        [sys.executable, "-B", "main.py", "--port", "8010"],
        cwd="/Users/derekjrussell/Documents/repos/Ironcliw-AI-Agent/backend",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    )
    
    # Wait for backend to start
    print("⏳ Waiting for backend to initialize...")
    await asyncio.sleep(10)
    
    # Test the vision command
    print("\n🧪 Testing vision command routing...")
    
    test_queries = [
        "What's happening across my desktop spaces?",
        "Show me all my desktop spaces",
        "Analyze my screen",
        "What is on my screen?"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    'http://localhost:8010/api/command',
                    json={'command': query},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Check command type
                        command_type = result.get('command_type', 'unknown')
                        if command_type == 'vision':
                            print(f"✅ SUCCESS: Routed to VISION handler")
                            if 'error' in str(result.get('response', '')).lower():
                                print(f"⚠️  Vision processing had an error: {result.get('response', '')[:100]}")
                            else:
                                print(f"📸 Response: {result.get('response', '')[:200]}...")
                        else:
                            print(f"❌ FAILED: Routed to {command_type.upper()} instead of VISION")
                            print(f"   Response: {result.get('response', '')[:100]}...")
                    else:
                        print(f"❌ HTTP Error {response.status}")
                        text = await response.text()
                        print(f"   Response: {text[:200]}")
                        
        except asyncio.TimeoutError:
            print("❌ Request timed out")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    backend_process.terminate()
    await asyncio.sleep(2)
    backend_process.kill()
    
    print("\n✅ Test complete!")

if __name__ == "__main__":
    asyncio.run(test_vision_command())