"""
JARVIS System Resource Optimizer v1.0
=====================================

Provides dynamic, intelligent optimization of system resources (ulimit, etc.)
automatically at startup. This eliminates the need for manual shell commands
or workarounds for "Time Limit Exceeded" or "Too many open files" errors.

Features:
- Smart File Descriptor Maximization (Safe macOS/Linux limits)
- Resource Limit logging
- Retry logic with fallback strategies
- No hard dependencies on external tools
- Cross-platform: no-op on Windows (resource module unavailable)

Usage:
    from backend.core.system_optimization import get_system_optimizer
    optimizer = get_system_optimizer()
    optimizer.optimize()
"""

import platform
import logging
import sys
import os
from typing import Tuple, Dict, Any, Optional

# resource module is Unix-only
try:
    import resource
    _HAS_RESOURCE = True
except ImportError:
    _HAS_RESOURCE = False

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
        self._is_windows = platform.system() == "Windows"
        self._optimized = False
        
    def optimize(self) -> Dict[str, Any]:
        """
        Run all system optimizations.
        
        Returns:
            Dict containing optimization results and stats.
        """
        if self._is_windows or not _HAS_RESOURCE:
            logger.debug("System optimization skipped (Windows/resource module unavailable)")
            self._optimized = True
            return {
                "platform": platform.system(),
                "file_descriptors": {"status": "skipped", "reason": "not applicable on Windows"},
            }

        results = {
            "platform": platform.system(),
            "file_descriptors": self._optimize_file_descriptors(),
        }
        
        self._optimized = True
        return results

    def _optimize_file_descriptors(self) -> Dict[str, Any]:
        """
        Maximize available file descriptors (ulimit -n).
        
        Tries to raise the soft limit to the hard limit, or a safe maximum.
        """
        if not _HAS_RESOURCE:
            return {"status": "skipped", "reason": "resource module not available"}

        status = "unchanged"
        try:
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
            original_soft = soft_limit
            
            target_limit = hard_limit
            
            SAFE_MAX = 65536 
            
            if hard_limit == resource.RLIM_INFINITY:
                 target_limit = SAFE_MAX 
            elif hard_limit > SAFE_MAX:
                target_limit = SAFE_MAX
            
            if soft_limit >= target_limit:
                 logger.debug(f"File descriptors already optimized: {soft_limit}")
                 return {
                     "status": "already_optimized", 
                     "soft": soft_limit, 
                     "hard": hard_limit
                 }

            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (target_limit, hard_limit))
                new_soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
                logger.info(f"✅ Optimized file descriptors: {original_soft} -> {new_soft} (Hard: {hard_limit})")
                status = "optimized"
                return {"status": status, "soft": new_soft, "hard": hard_limit}
            
            except ValueError as e:
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
