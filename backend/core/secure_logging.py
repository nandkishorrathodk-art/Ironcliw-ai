"""
Centralized log sanitization — CWE-117 (Log Injection) and CWE-532 (Clear-Text Logging).

Consolidates existing patterns from:
- backend/main.py:_sanitize_log() — control character stripping
- backend/neural_mesh/agents/google_workspace_agent.py:_sanitize_for_logging() — PII redaction

Usage:
    from backend.core.secure_logging import sanitize_for_log, mask_sensitive

    logger.info(f"User action: {sanitize_for_log(user_input)}")
    logger.debug(f"Using key: {mask_sensitive(api_key)}")
"""

import re

_CONTROL_CHAR_RE = re.compile(r'[\x00-\x1f\x7f]')


def sanitize_for_log(val, max_len: int = 200) -> str:
    """Strip control characters and limit length to prevent log injection (CWE-117).

    Removes: null bytes, newlines, carriage returns, tabs, ANSI escapes,
    and all other control characters (0x00-0x1f, 0x7f).

    Args:
        val: Value to sanitize (coerced to str).
        max_len: Maximum output length. Defaults to 200.

    Returns:
        Sanitized string safe for log output.
    """
    return _CONTROL_CHAR_RE.sub('', str(val))[:max_len]


def mask_sensitive(val, visible_prefix: int = 4) -> str:
    """Mask sensitive values, showing only first N characters (CWE-532 prevention).

    Args:
        val: Sensitive value to mask (coerced to str).
        visible_prefix: Number of leading characters to show. Defaults to 4.

    Returns:
        Masked string (e.g., "sk-a****").
    """
    s = str(val)
    if len(s) <= visible_prefix:
        return '****'
    return s[:visible_prefix] + '****'
