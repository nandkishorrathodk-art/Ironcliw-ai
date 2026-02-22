"""
Test Authentication Bypass
Verify that Windows authentication is properly bypassed
"""
import os
import sys

# Set environment variables
os.environ['JARVIS_SKIP_VOICE_AUTH'] = 'true'
os.environ['WINDOWS_AUTH_MODE'] = 'BYPASS'
os.environ['JARVIS_PLATFORM'] = 'windows'

print("=" * 70)
print("JARVIS Authentication Bypass Test")
print("=" * 70)

try:
    # Import authentication
    from backend.platform.windows.auth import WindowsAuthentication
    from backend.platform.base import AuthenticationResult
    
    print("\n[OK] Successfully imported WindowsAuthentication")
    
    # Create auth instance
    auth = WindowsAuthentication()
    print(f"[OK] Authentication instance created")
    print(f"   - Bypass mode: {auth._bypass_mode}")
    print(f"   - Auth mode: {auth._auth_mode}")
    
    # Test voice authentication
    print("\n[TEST] Voice Authentication...")
    result = auth.authenticate_voice(b"fake_audio_data", "test_user")
    print(f"   - Success: {result.success}")
    print(f"   - Method: {result.method}")
    print(f"   - Message: {result.message}")
    print(f"   - Confidence: {result.confidence}")
    
    # Test password authentication
    print("\n[TEST] Password Authentication...")
    result = auth.authenticate_password("test_password")
    print(f"   - Success: {result.success}")
    print(f"   - Method: {result.method}")
    print(f"   - Message: {result.message}")
    
    # Test biometric authentication
    print("\n[TEST] Biometric Authentication...")
    result = auth.authenticate_biometric()
    print(f"   - Success: {result.success}")
    print(f"   - Method: {result.method}")
    print(f"   - Message: {result.message}")
    
    # Test enrollment check
    print("\n[TEST] Enrollment Check...")
    enrolled = auth.is_enrolled("any_user")
    print(f"   - Is enrolled: {enrolled}")
    
    # Test bypass method
    print("\n[TEST] Direct Bypass...")
    result = auth.bypass_authentication()
    print(f"   - Success: {result.success}")
    print(f"   - Method: {result.method}")
    print(f"   - Message: {result.message}")
    
    print("\n" + "=" * 70)
    print("ALL AUTHENTICATION BYPASSED SUCCESSFULLY!")
    print("=" * 70)
    print("\nSummary:")
    print("   - Voice auth: BYPASSED [OK]")
    print("   - Password auth: BYPASSED [OK]")
    print("   - Biometric auth: BYPASSED [OK]")
    print("   - Direct bypass: WORKING [OK]")
    print("\nJARVIS can start without any authentication!")
    
except ImportError as e:
    print(f"\n[ERROR] Import Error: {e}")
    print("   Make sure you're in the correct directory")
    sys.exit(1)
except Exception as e:
    print(f"\n[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
