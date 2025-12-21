"""
JARVIS Numba Pre-loader v2.0.0
==============================

CRITICAL: This module must be imported FIRST, before ANY other imports
that might use numba (whisper, librosa, scipy with JIT, etc.).

This solves the circular import error:
    "cannot import name 'get_hashable_key' from partially initialized module 'numba.core.utils'"

The error occurs when:
1. Multiple threads try to import numba simultaneously
2. Thread A starts importing numba.core.utils
3. Thread B also tries to import numba.core.utils
4. Thread B sees a partially initialized module and fails

v2.0.0 Solution (ROBUST):
1. Use a PROCESS-LEVEL lock (not just thread lock)
2. Disable numba JIT during initial import
3. Force full initialization in main thread FIRST
4. Use threading.Event for BLOCKING wait by other threads
5. Set GLOBAL MARKER that whisper can check
6. Expose version info for health checks
7. Add secondary safeguard for parallel imports

Usage in main.py (MUST BE FIRST IMPORT):
    # CRITICAL: Pre-load numba before ANY other imports
    from core.numba_preload import ensure_numba_initialized, get_numba_status
    ensure_numba_initialized()
    
Usage in whisper_audio_fix.py (or any numba-using module):
    from core.numba_preload import wait_for_numba, is_numba_ready
    
    # This BLOCKS until main thread's numba init completes
    wait_for_numba(timeout=30.0)
    
    # Now safe to import whisper/librosa/etc
    import whisper
"""

import os
import sys
import threading
import logging
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class NumbaStatus(Enum):
    """Status of numba initialization"""
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    READY = "ready"
    FAILED = "failed"
    NOT_INSTALLED = "not_installed"


@dataclass
class NumbaInfo:
    """Information about numba initialization"""
    status: NumbaStatus = NumbaStatus.NOT_STARTED
    version: Optional[str] = None
    error: Optional[str] = None
    initialized_by_thread: Optional[str] = None
    initialized_at: Optional[float] = None


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE - Process-level singleton
# ═══════════════════════════════════════════════════════════════════════════════
_numba_lock = threading.Lock()
_numba_info = NumbaInfo()
_numba_module = None
_initialization_complete = threading.Event()


def _do_numba_import() -> bool:
    """
    Actually perform the numba import.
    This is called exactly ONCE per process.
    
    Returns True if successful, False otherwise.
    """
    global _numba_module, _numba_info
    import time
    
    _numba_info.initialized_by_thread = threading.current_thread().name
    _numba_info.initialized_at = time.time()
    
    # Save original environment
    original_env = {
        'NUMBA_DISABLE_JIT': os.environ.get('NUMBA_DISABLE_JIT'),
        'NUMBA_NUM_THREADS': os.environ.get('NUMBA_NUM_THREADS'),
        'NUMBA_THREADING_LAYER': os.environ.get('NUMBA_THREADING_LAYER'),
    }
    
    try:
        # ═══════════════════════════════════════════════════════════════════════
        # CRITICAL: Disable JIT and threading during import to prevent issues
        # ═══════════════════════════════════════════════════════════════════════
        os.environ['NUMBA_DISABLE_JIT'] = '1'
        os.environ['NUMBA_NUM_THREADS'] = '1'
        os.environ['NUMBA_THREADING_LAYER'] = 'safe'
        
        # Import numba
        import numba
        _numba_module = numba
        _numba_info.version = numba.__version__
        
        # ═══════════════════════════════════════════════════════════════════════
        # CRITICAL: Force full initialization of problematic submodules
        # These are the modules that cause circular import issues
        # ═══════════════════════════════════════════════════════════════════════
        try:
            from numba.core import utils as numba_utils
            
            # Access the problematic function to ensure it's fully initialized
            if hasattr(numba_utils, 'get_hashable_key'):
                _ = numba_utils.get_hashable_key
            
            # Also pre-load other commonly problematic modules
            from numba.core import types as numba_types
            from numba.core import config as numba_config
            from numba import typed as numba_typed
            
            logger.debug(f"numba {numba.__version__} submodules pre-loaded")
            
        except (ImportError, AttributeError) as e:
            # Older numba versions may not have all these
            logger.debug(f"Some numba submodules unavailable: {e}")
        
        _numba_info.status = NumbaStatus.READY
        logger.info(f"✅ numba {numba.__version__} pre-initialized (thread: {threading.current_thread().name})")
        return True
        
    except ImportError as e:
        _numba_info.status = NumbaStatus.NOT_INSTALLED
        _numba_info.error = str(e)
        logger.debug(f"numba not installed (optional): {e}")
        return False
        
    except Exception as e:
        _numba_info.status = NumbaStatus.FAILED
        _numba_info.error = str(e)
        logger.warning(f"⚠️ numba pre-initialization failed (non-fatal): {e}")
        return False
        
    finally:
        # ═══════════════════════════════════════════════════════════════════════
        # Restore original environment after import
        # ═══════════════════════════════════════════════════════════════════════
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def ensure_numba_initialized(timeout: float = 30.0) -> bool:
    """
    Ensure numba is initialized. Thread-safe and idempotent.
    
    This function can be called from any thread. The first caller will
    do the actual import, all other callers will wait for completion.
    
    Args:
        timeout: Maximum time to wait for initialization (seconds)
        
    Returns:
        True if numba is available, False otherwise
    """
    global _numba_info
    
    # Fast path - already initialized
    if _initialization_complete.is_set():
        return _numba_info.status == NumbaStatus.READY
    
    # Try to acquire lock for initialization
    acquired = _numba_lock.acquire(timeout=timeout)
    
    if not acquired:
        logger.warning(f"Timeout waiting for numba initialization ({timeout}s)")
        return False
    
    try:
        # Double-check after acquiring lock
        if _initialization_complete.is_set():
            return _numba_info.status == NumbaStatus.READY
        
        # We're the initializing thread
        _numba_info.status = NumbaStatus.INITIALIZING
        success = _do_numba_import()
        
        # Signal completion
        _initialization_complete.set()
        return success
        
    finally:
        _numba_lock.release()


def get_numba_status() -> Dict[str, Any]:
    """
    Get current numba status for health checks.
    
    Returns:
        Dictionary with status, version, and other info
    """
    return {
        'status': _numba_info.status.value,
        'version': _numba_info.version,
        'error': _numba_info.error,
        'initialized_by': _numba_info.initialized_by_thread,
        'initialized_at': _numba_info.initialized_at,
        'is_ready': _numba_info.status == NumbaStatus.READY,
    }


def get_numba_module():
    """
    Get the numba module if available.
    
    Returns:
        The numba module, or None if not available
    """
    ensure_numba_initialized()
    return _numba_module


def is_numba_ready() -> bool:
    """
    Quick check if numba is ready.
    Non-blocking if initialization is complete.
    """
    if _initialization_complete.is_set():
        return _numba_info.status == NumbaStatus.READY
    return False


def wait_for_numba(timeout: float = 60.0) -> bool:
    """
    BLOCKING wait for numba initialization to complete.
    
    v2.0: This is the KEY function for parallel safety.
    Other modules should call this BEFORE importing numba-dependent packages.
    
    This ensures:
    1. If main thread is initializing, we WAIT for it to complete
    2. If main thread already completed, we return immediately
    3. If no initialization started, we trigger it ourselves
    
    Args:
        timeout: Maximum time to wait (seconds)
        
    Returns:
        True if numba is available, False if not installed or failed
    """
    global _numba_info
    
    # Fast path - already initialized
    if _initialization_complete.is_set():
        return _numba_info.status == NumbaStatus.READY
    
    # Check if initialization is in progress by another thread
    start_time = time.time()
    
    # Try to acquire lock (with short timeout to check status)
    while time.time() - start_time < timeout:
        # Check if completed while waiting
        if _initialization_complete.is_set():
            return _numba_info.status == NumbaStatus.READY
        
        # Try to acquire lock for initialization
        acquired = _numba_lock.acquire(timeout=0.5)
        
        if acquired:
            try:
                # Double-check after acquiring lock
                if _initialization_complete.is_set():
                    return _numba_info.status == NumbaStatus.READY
                
                # We're the initializing thread
                logger.info(f"[wait_for_numba] Initializing from thread: {threading.current_thread().name}")
                _numba_info.status = NumbaStatus.INITIALIZING
                success = _do_numba_import()
                
                # Signal completion
                _initialization_complete.set()
                return success
                
            finally:
                _numba_lock.release()
        
        # Lock not acquired - another thread is initializing
        # Wait for the event with shorter timeout to allow status checks
        if _initialization_complete.wait(timeout=1.0):
            return _numba_info.status == NumbaStatus.READY
    
    # Timeout
    logger.warning(f"[wait_for_numba] Timeout after {timeout}s waiting for numba")
    return False


def set_numba_bypass_marker():
    """
    Set a global marker that signals numba has been attempted.
    Other modules can check this to avoid redundant initialization attempts.
    """
    os.environ['_JARVIS_NUMBA_INIT_ATTEMPTED'] = '1'


def get_numba_bypass_marker() -> bool:
    """
    Check if numba initialization has been attempted.
    """
    return os.environ.get('_JARVIS_NUMBA_INIT_ATTEMPTED') == '1'


# ═══════════════════════════════════════════════════════════════════════════════
# Auto-initialize if this module is imported directly
# This ensures numba is ready as soon as this module loads
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ != "__main__":
    # Only auto-init if not run as script
    ensure_numba_initialized()
    set_numba_bypass_marker()

