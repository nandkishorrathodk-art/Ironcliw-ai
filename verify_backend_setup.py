#!/usr/bin/env python3
"""
Verify backend setup for Ironcliw
"""
import os
import sys
from pathlib import Path

def verify_setup():
    print("🔍 Verifying Ironcliw Backend Setup")
    print("=" * 50)
    
    backend_dir = Path(__file__).parent / "backend"
    
    # Check main.py exists
    main_py = backend_dir / "main.py"
    if main_py.exists():
        print("✅ main.py exists")
        
        # Check if it has parallel imports
        with open(main_py, 'r') as f:
            content = f.read()
            if "parallel_import_components" in content:
                print("✅ main.py has parallel imports integrated")
            else:
                print("❌ main.py missing parallel imports")
                
            if "OPTIMIZE_STARTUP" in content:
                print("✅ main.py supports optimized startup")
            else:
                print("❌ main.py missing optimization support")
    else:
        print("❌ main.py not found!")
        
    # Check main_minimal.py
    minimal_py = backend_dir / "main_minimal.py"
    if minimal_py.exists():
        print("✅ main_minimal.py exists (fallback)")
    else:
        print("⚠️  main_minimal.py not found (fallback unavailable)")
        
    # Check start_system.py
    start_system = Path(__file__).parent / "start_system.py"
    if start_system.exists():
        print("✅ start_system.py exists")
        
        with open(start_system, 'r') as f:
            content = f.read()
            if "OPTIMIZE_STARTUP" in content:
                print("✅ start_system.py enables optimized startup")
            else:
                print("❌ start_system.py doesn't enable optimized startup")
                
            if "main.py (parallel startup integrated)" in content:
                print("✅ start_system.py uses integrated main.py")
            else:
                print("⚠️  start_system.py may not use the right main.py")
    else:
        print("❌ start_system.py not found!")
        
    # Check environment
    print("\n🌍 Environment:")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        print(f"✅ ANTHROPIC_API_KEY set ({len(api_key)} chars)")
    else:
        print("❌ ANTHROPIC_API_KEY not set - vision features won't work!")
        
    print("\n📋 Summary:")
    print("When you run 'python start_system.py':")
    print("1. It will try to start main.py with parallel imports")
    print("2. Backend should start in ~30s (not 107s)")
    print("3. If main.py fails, it falls back to main_minimal.py")
    print("4. Vision/monitoring commands need the full main.py")
    
    print("\n🚀 To start Ironcliw:")
    print("python start_system.py")
    
if __name__ == "__main__":
    verify_setup()