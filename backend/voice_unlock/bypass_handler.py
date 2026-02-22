"""
Voice Unlock Bypass Handler
============================

Provides authentication bypass logic for platforms where voice biometric 
authentication is unavailable (Windows, Linux) or in development mode.

Integrates with existing voice unlock system while providing graceful fallback.
"""

import logging
from typing import Dict, Optional
from datetime import datetime

# Import authentication config
try:
    from backend.config.auth_config import AuthConfig
except ImportError:
    # Fallback if running standalone
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.auth_config import AuthConfig


logger = logging.getLogger(__name__)


class AuthBypassHandler:
    """
    Handle authentication bypass logic for JARVIS voice unlock system.
    
    This handler provides a fallback authentication mechanism for platforms
    where voice biometrics are unavailable or when running in development mode.
    """
    
    def __init__(self):
        self.bypass_enabled = AuthConfig.should_bypass_auth()
        self.bypass_reason = AuthConfig.get_bypass_reason()
        
        if self.bypass_enabled:
            logger.warning(
                f"Authentication bypass ENABLED: {self.bypass_reason}"
            )
            logger.warning(
                "Security reduced - voice biometric authentication disabled"
            )
    
    async def authenticate(
        self,
        audio_data: Optional[bytes] = None,
        password: Optional[str] = None,
    ) -> Dict:
        """
        Authenticate user with bypass support.
        
        Authentication flow:
        1. Check if bypass is enabled
        2. If bypass + password configured, verify password
        3. If bypass + no password, auto-authenticate
        4. If bypass disabled, return error (voice biometric required)
        
        Args:
            audio_data: Voice audio data (ignored in bypass mode)
            password: Optional password for bypass mode
        
        Returns:
            dict: Authentication result with keys:
                - authenticated: bool
                - speaker_id: str (user identifier)
                - confidence: float (1.0 for bypass)
                - method: str (authentication method used)
                - message: str (status message)
                - timestamp: str (ISO format)
        """
        timestamp = datetime.utcnow().isoformat()
        
        # Check if bypass is enabled
        if not self.bypass_enabled:
            return {
                "authenticated": False,
                "speaker_id": None,
                "confidence": 0.0,
                "method": "voice_biometric",
                "message": "Voice biometric authentication required (bypass disabled)",
                "error": "Authentication bypass not enabled. Set JARVIS_BYPASS_AUTH=true or JARVIS_DEV_MODE=true",
                "timestamp": timestamp,
            }
        
        # Bypass is enabled - check password if configured
        if AuthConfig.BYPASS_PASSWORD:
            if not password:
                return {
                    "authenticated": False,
                    "speaker_id": None,
                    "confidence": 0.0,
                    "method": "bypass_password",
                    "message": "Password required for bypass mode",
                    "error": "Bypass password not provided",
                    "timestamp": timestamp,
                }
            
            if not AuthConfig.verify_bypass_password(password):
                logger.warning(f"Failed bypass password attempt at {timestamp}")
                return {
                    "authenticated": False,
                    "speaker_id": None,
                    "confidence": 0.0,
                    "method": "bypass_password",
                    "message": "Invalid bypass password",
                    "error": "Password verification failed",
                    "timestamp": timestamp,
                }
            
            # Password verified
            logger.info(f"Authentication bypassed with password at {timestamp}")
            return {
                "authenticated": True,
                "speaker_id": "bypass_user_password",
                "confidence": 1.0,
                "method": "bypass_password",
                "message": "Authentication bypassed with password",
                "bypass_reason": self.bypass_reason,
                "timestamp": timestamp,
            }
        
        # No password required, auto-authenticate
        logger.info(
            f"Authentication auto-bypassed at {timestamp}: {self.bypass_reason}"
        )
        
        # Determine user ID based on platform
        import platform
        os_type = platform.system()
        username = "unknown"
        try:
            import getpass
            username = getpass.getuser()
        except:
            pass
        
        speaker_id = f"bypass_user_{os_type}_{username}".lower()
        
        return {
            "authenticated": True,
            "speaker_id": speaker_id,
            "confidence": 1.0,
            "method": "bypass_auto",
            "message": f"Authentication bypassed: {self.bypass_reason}",
            "bypass_reason": self.bypass_reason,
            "platform": os_type,
            "username": username,
            "timestamp": timestamp,
        }
    
    def get_status(self) -> Dict:
        """
        Get current bypass status.
        
        Returns:
            dict: Status information
        """
        return {
            "bypass_enabled": self.bypass_enabled,
            "bypass_reason": self.bypass_reason if self.bypass_enabled else "Not enabled",
            "password_required": AuthConfig.BYPASS_PASSWORD is not None,
            "config": AuthConfig.get_config_summary(),
        }


# Convenience function for quick authentication
async def quick_auth(
    audio_data: Optional[bytes] = None,
    password: Optional[str] = None,
) -> Dict:
    """
    Quick authentication with bypass support.
    
    Args:
        audio_data: Voice audio data (optional)
        password: Bypass password (optional)
    
    Returns:
        dict: Authentication result
    """
    handler = AuthBypassHandler()
    return await handler.authenticate(audio_data=audio_data, password=password)


if __name__ == "__main__":
    # Self-test
    import asyncio
    
    print("=" * 60)
    print("JARVIS Voice Unlock Bypass Handler - Test")
    print("=" * 60)
    
    handler = AuthBypassHandler()
    
    # Show status
    status = handler.get_status()
    print("\nBypass Status:")
    for key, value in status.items():
        if key != "config":
            print(f"  {key}: {value}")
    
    # Test authentication
    async def test():
        print("\nTesting Authentication:")
        result = await handler.authenticate()
        print("\nResult:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        
        if result["authenticated"]:
            print("\n[SUCCESS] Authentication bypassed successfully")
        else:
            print("\n[FAILED] Authentication required")
    
    asyncio.run(test())
    
    print("\n" + "=" * 60)
