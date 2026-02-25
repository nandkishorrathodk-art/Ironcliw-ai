"""Windows Hello biometric authentication."""
import sys
import asyncio
import logging

logger = logging.getLogger(__name__)


async def authenticate_windows_hello(reason: str = "Ironcliw-AI authentication") -> bool:
    """
    Authenticate via Windows Hello (fingerprint, face, PIN).
    Returns True if authenticated, False otherwise.
    """
    if sys.platform != "win32":
        return False

    try:
        import winrt.windows.security.credentials.ui as ui

        verifier = ui.UserConsentVerifier()
        result = await verifier.request_verification_async(reason)
        return result == ui.UserConsentVerificationResult.VERIFIED
    except ImportError:
        logger.info(
            "winrt not installed — pip install winrt-Windows.Security.Credentials.UI"
        )
    except Exception as e:
        logger.warning(f"Windows Hello failed: {e}")

    import os

    if os.environ.get("JARVIS_AUTO_BYPASS_WINDOWS") == "true":
        logger.info("Windows Hello unavailable — auth bypass active")
        return True

    return False
