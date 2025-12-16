#!/usr/bin/env python3
"""
Screen Lock Detection Module - Enhanced
========================================

Provides robust, multi-method screen lock detection for macOS.
Uses multiple detection strategies with fallbacks for reliability.
"""

import subprocess
import logging
import os
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def is_screen_locked() -> bool:
    """
    Check if the macOS screen is currently locked.

    Uses MULTIPLE detection methods in order of reliability:
    1. Direct Quartz CGSessionCopyCurrentDictionary (most reliable)
    2. IORegistry display power state check
    3. Security session idle check
    4. Login window frontmost check
    5. Screen capture test (definitive but slower)

    Returns:
        bool: True if screen is locked, False otherwise
    """
    detection_results = []

    try:
        # =====================================================================
        # Method 1: Direct Quartz API check (MOST RELIABLE)
        # =====================================================================
        try:
            import Quartz
            session_dict = Quartz.CGSessionCopyCurrentDictionary()
            if session_dict:
                # Check the definitive lock indicator
                screen_locked = session_dict.get('CGSSessionScreenIsLocked', False)
                if screen_locked:
                    logger.info("🔒 [SCREEN-DETECT] LOCKED via Quartz CGSSessionScreenIsLocked")
                    return True

                # Also check if screen saver with lock is active
                screensaver_time = session_dict.get('CGSSessionScreenLockedTime', 0)
                if screensaver_time and screensaver_time > 0:
                    logger.info("🔒 [SCREEN-DETECT] LOCKED via Quartz CGSSessionScreenLockedTime")
                    return True

                # Check on-console status
                on_console = session_dict.get('kCGSSessionOnConsoleKey', True)
                if not on_console:
                    logger.info("🔒 [SCREEN-DETECT] LOCKED via kCGSSessionOnConsoleKey=False")
                    return True

                detection_results.append(("Quartz", False))
        except ImportError:
            logger.debug("[SCREEN-DETECT] Quartz not available, trying fallbacks")
        except Exception as e:
            logger.debug(f"[SCREEN-DETECT] Quartz check failed: {e}")

        # =====================================================================
        # Method 2: IORegistry Display State Check
        # =====================================================================
        try:
            ioreg_cmd = ["ioreg", "-r", "-c", "IODisplayWrangler", "-d", "1"]
            result = subprocess.run(ioreg_cmd, capture_output=True, text=True, timeout=3)

            if result.returncode == 0:
                output = result.stdout.lower()
                # DevicePowerState: 0 = display off/locked, 4 = display on
                if '"devicepowerstate" = 0' in output or "'devicepowerstate' = 0" in output:
                    logger.info("🔒 [SCREEN-DETECT] Display appears OFF via IORegistry")
                    # Display off could mean locked - check further
                    detection_results.append(("IORegistry-DisplayOff", True))
        except Exception as e:
            logger.debug(f"[SCREEN-DETECT] IORegistry check failed: {e}")

        # =====================================================================
        # Method 3: Security Session Idle Check
        # =====================================================================
        try:
            idle_cmd = ["ioreg", "-c", "IOHIDSystem"]
            result = subprocess.run(idle_cmd, capture_output=True, text=True, timeout=3)

            if result.returncode == 0 and "HIDIdleTime" in result.stdout:
                # Parse idle time - very long idle often means locked
                import re
                match = re.search(r'"HIDIdleTime"\s*=\s*(\d+)', result.stdout)
                if match:
                    idle_ns = int(match.group(1))
                    idle_seconds = idle_ns / 1000000000
                    # If idle for more than 5 minutes, likely locked
                    if idle_seconds > 300:
                        detection_results.append(("IdleTime", True))
        except Exception as e:
            logger.debug(f"[SCREEN-DETECT] Idle check failed: {e}")

        # =====================================================================
        # Method 4: Login Window Frontmost Check (AppleScript)
        # =====================================================================
        try:
            script = '''
            tell application "System Events"
                set frontApp to name of first process whose frontmost is true
                return frontApp
            end tell
            '''
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0:
                front_app = result.stdout.strip().lower()
                logger.debug(f"[SCREEN-DETECT] Frontmost app: {front_app}")

                if "loginwindow" in front_app:
                    logger.info("🔒 [SCREEN-DETECT] LOCKED via loginwindow frontmost")
                    return True

                detection_results.append(("FrontApp", False))
            else:
                # If we can't get frontmost app, it might be locked
                logger.debug(f"[SCREEN-DETECT] Could not get frontmost app: {result.stderr}")
                detection_results.append(("FrontApp-Error", True))
        except subprocess.TimeoutExpired:
            # Timeout often indicates locked screen
            logger.info("🔒 [SCREEN-DETECT] AppleScript timeout - likely LOCKED")
            return True
        except Exception as e:
            logger.debug(f"[SCREEN-DETECT] Frontmost app check failed: {e}")

        # =====================================================================
        # Method 5: Screen Capture Test (Definitive but slower)
        # =====================================================================
        try:
            # Try to capture a tiny screenshot - fails if screen is locked
            capture_cmd = ["screencapture", "-x", "-c", "-T", "0"]
            result = subprocess.run(capture_cmd, capture_output=True, timeout=3)

            if result.returncode != 0:
                # screencapture fails when screen is locked
                logger.info("🔒 [SCREEN-DETECT] LOCKED via screencapture failure")
                return True

            detection_results.append(("ScreenCapture", False))
        except subprocess.TimeoutExpired:
            logger.info("🔒 [SCREEN-DETECT] screencapture timeout - likely LOCKED")
            return True
        except Exception as e:
            logger.debug(f"[SCREEN-DETECT] Screen capture test failed: {e}")

        # =====================================================================
        # Method 6: Check Console User (final fallback)
        # =====================================================================
        try:
            stat_cmd = ["stat", "-f", "%Su", "/dev/console"]
            result = subprocess.run(stat_cmd, capture_output=True, text=True, timeout=2)

            if result.returncode == 0:
                console_user = result.stdout.strip()
                current_user = os.environ.get("USER", "")

                if console_user != current_user and console_user not in ["root", ""]:
                    logger.info(f"🔒 [SCREEN-DETECT] Console user mismatch: {console_user} vs {current_user}")
                    return True
        except Exception as e:
            logger.debug(f"[SCREEN-DETECT] Console user check failed: {e}")

        # =====================================================================
        # Aggregate Results
        # =====================================================================
        locked_votes = sum(1 for _, is_locked in detection_results if is_locked)
        unlocked_votes = sum(1 for _, is_locked in detection_results if not is_locked)

        logger.debug(f"[SCREEN-DETECT] Results: {detection_results}")
        logger.debug(f"[SCREEN-DETECT] Votes: locked={locked_votes}, unlocked={unlocked_votes}")

        # If any strong indicator says locked, return True
        if locked_votes > 0 and unlocked_votes == 0:
            logger.info("🔒 [SCREEN-DETECT] LOCKED (aggregated)")
            return True

        # Default: screen is unlocked
        logger.info("🔓 [SCREEN-DETECT] UNLOCKED (all checks passed)")
        return False

    except Exception as e:
        logger.error(f"[SCREEN-DETECT] Critical error: {e}")
        # On error, assume unlocked to avoid blocking user
        return False


def get_screen_state_details() -> Dict[str, Any]:
    """
    Get detailed screen state information using multiple detection methods.

    Returns:
        dict: Detailed state including lock status, detection method, and diagnostics
    """
    details = {
        "isLocked": False,
        "detectionMethod": None,
        "methods": {},
        "diagnostics": {}
    }

    try:
        # Method 1: Quartz CGSession
        try:
            import Quartz
            session_dict = Quartz.CGSessionCopyCurrentDictionary()
            if session_dict:
                details["methods"]["quartz"] = {
                    "CGSSessionScreenIsLocked": session_dict.get('CGSSessionScreenIsLocked', False),
                    "CGSSessionScreenLockedTime": session_dict.get('CGSSessionScreenLockedTime', 0),
                    "kCGSSessionOnConsoleKey": session_dict.get('kCGSSessionOnConsoleKey', True),
                }
                if details["methods"]["quartz"]["CGSSessionScreenIsLocked"]:
                    details["isLocked"] = True
                    details["detectionMethod"] = "Quartz-CGSSessionScreenIsLocked"
        except Exception as e:
            details["diagnostics"]["quartz_error"] = str(e)

        # Method 2: Login window frontmost
        try:
            script = 'tell application "System Events" to get name of first process whose frontmost is true'
            result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                front_app = result.stdout.strip()
                details["methods"]["frontApp"] = front_app
                if "loginwindow" in front_app.lower():
                    details["isLocked"] = True
                    details["detectionMethod"] = details["detectionMethod"] or "LoginWindow-Frontmost"
        except Exception as e:
            details["diagnostics"]["frontApp_error"] = str(e)

        # Method 3: Screen capture test
        try:
            result = subprocess.run(["screencapture", "-x", "-c", "-T", "0"], capture_output=True, timeout=3)
            details["methods"]["screenCapture"] = result.returncode == 0
            if result.returncode != 0:
                details["isLocked"] = True
                details["detectionMethod"] = details["detectionMethod"] or "ScreenCapture-Failed"
        except Exception as e:
            details["diagnostics"]["screenCapture_error"] = str(e)

        # Method 4: IORegistry display state
        try:
            result = subprocess.run(["ioreg", "-r", "-c", "IODisplayWrangler", "-d", "1"],
                                    capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                output = result.stdout.lower()
                display_off = '"devicepowerstate" = 0' in output
                details["methods"]["ioregDisplayOff"] = display_off
        except Exception as e:
            details["diagnostics"]["ioreg_error"] = str(e)

        if not details["detectionMethod"]:
            details["detectionMethod"] = "AllMethodsPassed-Unlocked"

    except Exception as e:
        logger.error(f"Error getting screen state details: {e}")
        details["diagnostics"]["critical_error"] = str(e)

    return details


async def async_is_screen_locked() -> bool:
    """
    Async version of is_screen_locked() for use in async contexts.

    Returns:
        bool: True if screen is locked, False otherwise
    """
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, is_screen_locked)


if __name__ == "__main__":
    # Test the detection
    import json

    print("=" * 60)
    print("🔍 SCREEN LOCK DETECTION TEST")
    print("=" * 60)

    is_locked = is_screen_locked()
    print(f"\n📺 Screen is: {'🔒 LOCKED' if is_locked else '🔓 UNLOCKED'}")

    print("\n📊 Detailed State:")
    details = get_screen_state_details()
    print(json.dumps(details, indent=2, default=str))

    print("\n" + "=" * 60)