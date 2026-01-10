"""
JARVIS System Resource Optimizer v1.0
=====================================

Provides dynamic, intelligent optimization of system resources (ulimit, etc.)
automatically at startup. This eliminates the need for manual shell commands
or workarounds for "Time Limit Exceeded" or "Too many open files" errors.

Features:
- Smart File Descriptor Maximization (Safe macOS limits)
- Resource Limit logging
- Retry logic with fallback strategies
- No hard dependencies on external tools

Usage:
    from backend.core.system_optimization import get_system_optimizer
    optimizer = get_system_optimizer()
    optimizer.optimize()
"""

import resource
import platform
import logging
import sys
import os
from typing import Tuple, Dict, Any, Optional

# Setup dedicated logger
logger = logging.getLogger("jarvis.system_optimizer")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | [SystemOptimizer] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class SystemResourceOptimizer:
    """
    Intelligent system resource optimizer.
    """
    
    def __init__(self):
        self._is_macos = platform.system() == "Darwin"
        self._optimized = False
        
    def optimize(self) -> Dict[str, Any]:
        """
        Run all system optimizations.
        
        Returns:
            Dict containing optimization results and stats.
        """
        results = {
            "platform": platform.system(),
            "file_descriptors": self._optimize_file_descriptors(),
            # Future: add network stack optimization here
        }
        
        self._optimized = True
        return results

    def _optimize_file_descriptors(self) -> Dict[str, Any]:
        """
        Maximize available file descriptors (ulimit -n).
        
        Tries to raise the soft limit to the hard limit, or a safe maximum.
        """
        status = "unchanged"
        try:
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
            original_soft = soft_limit
            
            # Target higher limits
            # macOS often defaults to 256/unlimited, but has practical limits around 10240 or 65536 depending on version
            # We target a safe high number if hard limit is 'infinity' (often represented as very large int)
            
            target_limit = hard_limit
            
            # Cap extreme values to avoid "ValueError: not allowed to raise maximum limit" if hard is crazy high
            # or if we are just trying to set a sane default.
            # On macOS, typical "maxfiles" might be 10240 or similar for safe user usage without sudo
            SAFE_MAX = 65536 
            
            if hard_limit == resource.RLIM_INFINITY:
                 # If unlimited, pick a robust high number that is safe for select() / poll()
                 # Python usually handles large FDs well, but let's be safe.
                 target_limit = SAFE_MAX 
            elif hard_limit > SAFE_MAX:
                target_limit = SAFE_MAX
            
            # Don't lower it if it's already high
            if soft_limit >= target_limit:
                 logger.debug(f"File descriptors already optimized: {soft_limit}")
                 return {
                     "status": "already_optimized", 
                     "soft": soft_limit, 
                     "hard": hard_limit
                 }

            # Try to raise soft limit
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (target_limit, hard_limit))
                new_soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
                logger.info(f"✅ Optimized file descriptors: {original_soft} -> {new_soft} (Hard: {hard_limit})")
                status = "optimized"
                return {"status": status, "soft": new_soft, "hard": hard_limit}
            
            except ValueError as e:
                # If target failed, try intermediate fallback (e.g. 10240)
                fallback = 10240
                if soft_limit < fallback and hard_limit >= fallback:
                    try:
                        resource.setrlimit(resource.RLIMIT_NOFILE, (fallback, hard_limit))
                        new_soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
                        logger.warning(f"⚠️  Partial optimization (fallback): {original_soft} -> {new_soft}")
                        status = "partial_optimized"
                        return {"status": status, "soft": new_soft, "hard": hard_limit}
                    except Exception as e2:
                        logger.error(f"❌ Failed fallback optimization: {e2}")
                else:
                    logger.error(f"❌ Failed to set resource limit: {e}")

        except Exception as e:
            logger.error(f"❌ Error optimizing file descriptors: {e}")
            status = "error"
            
        return {"status": status, "error": str(e) if 'e' in locals() else None}

# Global Instance
_optimizer_instance: Optional[SystemResourceOptimizer] = None

def get_system_optimizer() -> SystemResourceOptimizer:
    """Get global optimizer instance."""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = SystemResourceOptimizer()
    return _optimizer_instance
