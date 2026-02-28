"""
Authentication Configuration for Ironcliw
========================================

Provides configuration for authentication bypass mechanism.
Designed for Windows/Linux where voice biometric authentication may not be available.

Environment Variables:
- Ironcliw_BYPASS_AUTH: Set to "true" to bypass authentication completely
- Ironcliw_DEV_MODE: Set to "true" to enable development mode (auto-bypass)
- Ironcliw_BYPASS_PASSWORD: Optional password for bypass mode

Security Warning:
Bypassing authentication reduces system security. Only use in trusted environments.
"""

import os
from typing import Optional, Dict


class AuthConfig:
    """Authentication configuration and bypass management."""
    
    # Environment variable checks
    BYPASS_AUTH: bool = os.getenv("Ironcliw_BYPASS_AUTH", "false").lower() in ("true", "1", "yes")
    DEV_MODE: bool = os.getenv("Ironcliw_DEV_MODE", "false").lower() in ("true", "1", "yes")
    BYPASS_PASSWORD: Optional[str] = os.getenv("Ironcliw_BYPASS_PASSWORD")
    
    # Platform-based auto-bypass (for Windows/Linux by default)
    import platform
    AUTO_BYPASS_ON_WINDOWS: bool = os.getenv("Ironcliw_AUTO_BYPASS_WINDOWS", "true").lower() in ("true", "1", "yes")
    AUTO_BYPASS_ON_LINUX: bool = os.getenv("Ironcliw_AUTO_BYPASS_LINUX", "true").lower() in ("true", "1", "yes")
    IS_WINDOWS: bool = platform.system() == "Windows"
    IS_LINUX: bool = platform.system() == "Linux"
    IS_MACOS: bool = platform.system() == "Darwin"
    
    @classmethod
    def should_bypass_auth(cls) -> bool:
        """
        Check if authentication should be bypassed.
        
        Returns True if:
        - Ironcliw_BYPASS_AUTH is set to true
        - Ironcliw_DEV_MODE is set to true
        - Running on Windows and AUTO_BYPASS_ON_WINDOWS is true
        - Running on Linux and AUTO_BYPASS_ON_LINUX is true
        
        Returns:
            bool: Whether to bypass authentication
        """
        # Explicit bypass flags
        if cls.BYPASS_AUTH or cls.DEV_MODE:
            return True
        
        # Platform-based auto-bypass
        if cls.IS_WINDOWS and cls.AUTO_BYPASS_ON_WINDOWS:
            return True
        
        if cls.IS_LINUX and cls.AUTO_BYPASS_ON_LINUX:
            return True
        
        return False
    
    @classmethod
    def get_bypass_reason(cls) -> str:
        """
        Get human-readable reason for bypass.
        
        Returns:
            str: Reason for authentication bypass
        """
        if cls.BYPASS_AUTH:
            return "Authentication bypass enabled (Ironcliw_BYPASS_AUTH=true)"
        
        if cls.DEV_MODE:
            return "Development mode enabled (Ironcliw_DEV_MODE=true)"
        
        if cls.IS_WINDOWS and cls.AUTO_BYPASS_ON_WINDOWS:
            return "Windows platform auto-bypass (voice biometric unavailable)"
        
        if cls.IS_LINUX and cls.AUTO_BYPASS_ON_LINUX:
            return "Linux platform auto-bypass (voice biometric unavailable)"
        
        return "Authentication required"
    
    @classmethod
    def verify_bypass_password(cls, password: str) -> bool:
        """
        Verify bypass password if configured.
        
        Args:
            password: Password to verify
        
        Returns:
            bool: True if password matches or no password required
        """
        if not cls.BYPASS_PASSWORD:
            # No password configured, allow bypass
            return True
        
        return password == cls.BYPASS_PASSWORD
    
    @classmethod
    def get_config_summary(cls) -> Dict[str, any]:
        """
        Get summary of authentication configuration.
        
        Returns:
            dict: Configuration summary
        """
        return {
            "platform": cls.platform.system(),
            "bypass_enabled": cls.should_bypass_auth(),
            "bypass_reason": cls.get_bypass_reason(),
            "password_required": cls.BYPASS_PASSWORD is not None,
            "dev_mode": cls.DEV_MODE,
            "explicit_bypass": cls.BYPASS_AUTH,
            "platform_auto_bypass": (
                (cls.IS_WINDOWS and cls.AUTO_BYPASS_ON_WINDOWS) or
                (cls.IS_LINUX and cls.AUTO_BYPASS_ON_LINUX)
            ),
        }


# Convenience function for quick checks
def is_auth_bypassed() -> bool:
    """Quick check if authentication is bypassed."""
    return AuthConfig.should_bypass_auth()


def get_auth_status() -> Dict[str, any]:
    """Get authentication status."""
    return AuthConfig.get_config_summary()


if __name__ == "__main__":
    # Self-test
    print("=" * 60)
    print("Ironcliw Authentication Configuration")
    print("=" * 60)
    
    config = AuthConfig.get_config_summary()
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    if AuthConfig.should_bypass_auth():
        print("[ENABLED] Authentication Bypass Active")
        print(f"Reason: {AuthConfig.get_bypass_reason()}")
    else:
        print("[DISABLED] Authentication Required")
    print("=" * 60)
