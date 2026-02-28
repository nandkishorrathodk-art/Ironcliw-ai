"""
Test Authentication Bypass System
==================================

Tests the authentication bypass handler in the Ironcliw root context.
"""
import sys
import asyncio
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.config.auth_config import AuthConfig
from backend.voice_unlock.bypass_handler import AuthBypassHandler


async def main():
    print("=" * 70)
    print("Ironcliw Authentication Bypass System - Test")
    print("=" * 70)
    
    # Test 1: Configuration
    print("\n[Test 1] Authentication Configuration")
    print("-" * 70)
    config = AuthConfig.get_config_summary()
    for key, value in config.items():
        print(f"  {key:25s}: {value}")
    
    # Test 2: Bypass Handler
    print("\n[Test 2] Bypass Handler Initialization")
    print("-" * 70)
    handler = AuthBypassHandler()
    status = handler.get_status()
    
    print(f"  Bypass Enabled: {status['bypass_enabled']}")
    print(f"  Bypass Reason:  {status['bypass_reason']}")
    print(f"  Password Req:   {status['password_required']}")
    
    # Test 3: Authentication (no password)
    print("\n[Test 3] Authentication Test (No Password)")
    print("-" * 70)
    result = await handler.authenticate()
    
    print("  Result:")
    for key, value in result.items():
        print(f"    {key:20s}: {value}")
    
    if result["authenticated"]:
        print("\n  [SUCCESS] Authentication bypassed successfully!")
        print(f"  User ID: {result['speaker_id']}")
        print(f"  Method:  {result['method']}")
    else:
        print("\n  [FAILED] Authentication required")
        print(f"  Reason: {result.get('error', 'Unknown')}")
    
    # Test 4: Password bypass (if configured)
    print("\n[Test 4] Password Bypass Test")
    print("-" * 70)
    if AuthConfig.BYPASS_PASSWORD:
        # Test with correct password
        result_correct = await handler.authenticate(password=AuthConfig.BYPASS_PASSWORD)
        print(f"  Correct password: {'PASS' if result_correct['authenticated'] else 'FAIL'}")
        
        # Test with incorrect password
        result_wrong = await handler.authenticate(password="wrong_password")
        print(f"  Wrong password:   {'FAIL' if not result_wrong['authenticated'] else 'UNEXPECTED PASS'}")
    else:
        print("  No password configured - skipping password test")
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    if AuthConfig.should_bypass_auth():
        print("  [ENABLED] Authentication bypass is ACTIVE")
        print(f"  Reason: {AuthConfig.get_bypass_reason()}")
        print("\n  Platform: Windows")
        print("  Voice biometric: Bypassed (unavailable on Windows)")
        print("  Security level:  Reduced (bypass mode)")
    else:
        print("  [DISABLED] Authentication bypass is INACTIVE")
        print("  Voice biometric authentication required")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
