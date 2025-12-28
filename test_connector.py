#!/usr/bin/env python3
"""Quick test to verify Computer Use connector can initialize."""

import os
import sys

# Load .env
from dotenv import load_dotenv
load_dotenv()

print("=" * 70)
print("Computer Use Connector Diagnostic")
print("=" * 70)
print()

# Check API key
api_key = os.environ.get("ANTHROPIC_API_KEY")
if api_key:
    print(f"✅ ANTHROPIC_API_KEY found (length: {len(api_key)})")
    print(f"   First 10 chars: {api_key[:10]}...")
else:
    print("❌ ANTHROPIC_API_KEY not found in environment")
    print()
    print("Checking .env file...")
    if os.path.exists(".env"):
        print("✅ .env file exists")
        with open(".env") as f:
            for line in f:
                if "ANTHROPIC_API_KEY" in line and not line.strip().startswith("#"):
                    print(f"✅ Found in .env: {line[:30]}...")
                    break
    else:
        print("❌ .env file not found")
    sys.exit(1)

print()

# Try to import connector
print("Testing Computer Use Connector import...")
sys.path.insert(0, "backend")

try:
    from backend.display.computer_use_connector import ClaudeComputerUseConnector
    print("✅ ClaudeComputerUseConnector imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

print()
print("Testing connector initialization...")

try:
    connector = ClaudeComputerUseConnector()
    print("✅ ClaudeComputerUseConnector initialized successfully!")
    print(f"   Model: {connector.COMPUTER_USE_MODEL}")
    print(f"   Max actions: {connector.max_actions_per_task}")
    print()
    print("🎉 Computer Use is ready for autonomous action execution!")
except Exception as e:
    print(f"❌ Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 70)
print("✅ ALL CHECKS PASSED - Computer Use is operational!")
print("=" * 70)
