#!/usr/bin/env python3
"""
Verify Claude API Key Configuration for Ironcliw Vision
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

print("🔍 Ironcliw API Key Verification")
print("=" * 60)

# Check various possible .env locations
env_locations = [Path("backend") / ".env", Path(".env"), Path.home() / ".env"]

print("1️⃣ Checking .env file locations:")
env_found = False
for env_path in env_locations:
    if env_path.exists():
        print(f"   ✅ Found: {env_path.absolute()}")
        # Try to load it
        load_dotenv(env_path)
        env_found = True

        # Check if it contains ANTHROPIC_API_KEY
        with open(env_path, "r") as f:
            content = f.read()
            if "ANTHROPIC_API_KEY" in content:
                print(f"      → Contains ANTHROPIC_API_KEY")
            else:
                print(f"      → Does NOT contain ANTHROPIC_API_KEY")
    else:
        print(f"   ❌ Not found: {env_path.absolute()}")

if not env_found:
    print("\n   ⚠️  No .env file found!")

# Check environment variable
print("\n2️⃣ Checking environment variable:")
api_key = os.getenv("ANTHROPIC_API_KEY")

if api_key:
    print(f"   ✅ ANTHROPIC_API_KEY is set")
    print(f"   → Starts with: {api_key[:15]}...")
    print(f"   → Length: {len(api_key)} characters")
else:
    print("   ❌ ANTHROPIC_API_KEY is NOT set in environment")

# Check if it's in shell environment
print("\n3️⃣ Checking shell environment:")
shell_check = os.popen("echo $ANTHROPIC_API_KEY").read().strip()
if shell_check:
    print(f"   ✅ Found in shell: {shell_check[:15]}...")
else:
    print("   ❌ Not found in shell environment")

# Test if API key works
print("\n4️⃣ Testing API key validity:")
if api_key:
    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)

        # Try a simple test
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": "Say 'Hello'"}],
        )
        print("   ✅ API key is VALID - Claude responded!")
    except Exception as e:
        print(f"   ❌ API key test failed: {e}")
else:
    print("   ⚠️  Cannot test - no API key found")

# Provide solutions
print("\n📋 SOLUTIONS:")
print("=" * 60)

if not api_key:
    print("\n🔧 To fix this, do ONE of the following:\n")

    print("Option 1: Add to backend/.env file (RECOMMENDED)")
    print("   echo 'ANTHROPIC_API_KEY=your-actual-api-key-here' >> backend/.env")

    print("\nOption 2: Export in current shell")
    print("   export ANTHROPIC_API_KEY='your-actual-api-key-here'")

    print("\nOption 3: Add to shell profile (permanent)")
    print("   echo 'export ANTHROPIC_API_KEY=\"your-actual-api-key-here\"' >> ~/.zshrc")
    print("   source ~/.zshrc")

    print("\n⚠️  After adding the key, restart Ironcliw!")
    print("\n🔑 Get your API key from: https://console.anthropic.com/")
else:
    print("\n✅ API key is configured correctly!")
    print("\n🚀 Next steps:")
    print("   1. Restart Ironcliw: python start_system.py")
    print("   2. Say: 'Hey Ironcliw, what's on my screen?'")
    print("   3. Enjoy intelligent vision responses!")

# Check if vision files are using the key
print("\n5️⃣ Checking vision system integration:")
vision_files = [
    "vision/screen_capture_fallback.py",
    "vision/enhanced_vision_system.py",
    "vision/jarvis_vision_enhanced.py",
]

for file in vision_files:
    if Path(file).exists():
        with open(file, "r") as f:
            content = f.read()
            if "ANTHROPIC_API_KEY" in content:
                print(f"   ✅ {file} checks for API key")
            else:
                print(f"   ⚠️  {file} might not be checking for API key")
