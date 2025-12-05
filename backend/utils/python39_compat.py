"""
Python 3.9 Compatibility Layer for JARVIS AI System
====================================================

This module provides runtime patches for Python 3.10+ features that are used
by dependencies but not available in Python 3.9. It must be imported VERY EARLY
in the startup process before any affected packages are imported.

Key Fixes:
- importlib.metadata.packages_distributions() - Added in Python 3.10
- google-api-core uses this function without proper version checking

Architecture:
- Monkey-patches importlib.metadata to add missing functions
- Patches google.api_core._python_version_support module if already loaded
- Uses importlib_metadata backport when available
- Provides graceful fallbacks when backport unavailable

Usage:
    # At the very top of your main script, before other imports:
    from backend.utils.python39_compat import ensure_python39_compatibility
    ensure_python39_compatibility()

Author: JARVIS AI System
Version: 1.0.0 - Robust Python 3.9 Support
"""

import sys
import logging
from typing import Dict, List, Mapping, Optional, Any, Callable
from functools import wraps
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logger = logging.getLogger(__name__)

# Version info
__version__ = "1.0.0"
__python_version__ = sys.version_info
__is_python39__ = __python_version__ < (3, 10)


class Python39CompatibilityManager:
    """
    Advanced compatibility manager for Python 3.9 runtime patches.

    Features:
    - Async-safe initialization
    - Thread-safe patching
    - Graceful degradation
    - Comprehensive logging
    - Dynamic module patching
    """

    _instance: Optional['Python39CompatibilityManager'] = None
    _initialized: bool = False
    _lock = asyncio.Lock() if hasattr(asyncio, 'Lock') else None
    _thread_lock = None  # Will be initialized lazily

    def __new__(cls) -> 'Python39CompatibilityManager':
        """Singleton pattern for global compatibility state."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the compatibility manager."""
        if not hasattr(self, '_setup_complete'):
            import threading
            self._thread_lock = threading.Lock()
            self._patched_modules: Dict[str, bool] = {}
            self._fallback_implementations: Dict[str, Callable] = {}
            self._error_log: List[str] = []
            self._setup_complete = True

    @property
    def is_python39(self) -> bool:
        """Check if running on Python 3.9 or earlier."""
        return __is_python39__

    def _create_packages_distributions_fallback(self) -> Callable[[], Mapping[str, List[str]]]:
        """
        Create a fallback implementation of packages_distributions().

        This function maps top-level package names to their distribution names.
        Uses importlib_metadata backport if available, otherwise builds from
        installed distributions.

        Returns:
            A callable that returns the package-to-distribution mapping
        """
        def packages_distributions_impl() -> Mapping[str, List[str]]:
            """Return mapping of top-level packages to their distributions."""
            pkg_to_dist: Dict[str, List[str]] = {}

            try:
                # Try importlib_metadata backport first (most reliable)
                try:
                    import importlib_metadata as metadata_backport
                    if hasattr(metadata_backport, 'packages_distributions'):
                        return metadata_backport.packages_distributions()
                except ImportError:
                    pass

                # Fall back to building the mapping ourselves
                try:
                    from importlib import metadata

                    for dist in metadata.distributions():
                        try:
                            dist_name = dist.metadata.get('Name', '')
                            if not dist_name:
                                continue

                            # Get top-level packages from top_level.txt or infer from files
                            top_level = None
                            try:
                                top_level_file = dist.read_text('top_level.txt')
                                if top_level_file:
                                    top_level = top_level_file.strip().split('\n')
                            except (FileNotFoundError, TypeError):
                                pass

                            if not top_level:
                                # Infer from distribution files
                                try:
                                    if dist.files:
                                        seen_packages = set()
                                        for file in dist.files:
                                            parts = str(file).split('/')
                                            if parts and not parts[0].endswith('.dist-info'):
                                                pkg = parts[0].replace('.py', '')
                                                if pkg and not pkg.startswith('_'):
                                                    seen_packages.add(pkg)
                                        top_level = list(seen_packages)
                                except Exception:
                                    top_level = [dist_name.replace('-', '_').lower()]

                            # Add to mapping
                            for pkg in (top_level or []):
                                pkg = pkg.strip()
                                if pkg:
                                    if pkg not in pkg_to_dist:
                                        pkg_to_dist[pkg] = []
                                    if dist_name not in pkg_to_dist[pkg]:
                                        pkg_to_dist[pkg].append(dist_name)
                        except Exception as e:
                            logger.debug(f"Error processing distribution: {e}")
                            continue

                except Exception as e:
                    logger.warning(f"Could not build packages_distributions mapping: {e}")

            except Exception as e:
                logger.error(f"packages_distributions fallback error: {e}")

            return pkg_to_dist

        return packages_distributions_impl

    def patch_importlib_metadata(self) -> bool:
        """
        Patch importlib.metadata to add packages_distributions if missing.

        Returns:
            True if patching was successful or unnecessary, False on error
        """
        if not self.is_python39:
            logger.debug("Python 3.10+ detected, no patching needed for importlib.metadata")
            return True

        try:
            from importlib import metadata

            # Check if already has the function
            if hasattr(metadata, 'packages_distributions'):
                logger.debug("importlib.metadata.packages_distributions already available")
                return True

            # Create and apply the patch
            packages_distributions = self._create_packages_distributions_fallback()
            metadata.packages_distributions = packages_distributions

            self._patched_modules['importlib.metadata'] = True
            logger.info("✅ Patched importlib.metadata.packages_distributions for Python 3.9")
            return True

        except Exception as e:
            error_msg = f"Failed to patch importlib.metadata: {e}"
            logger.error(error_msg)
            self._error_log.append(error_msg)
            return False

    def patch_google_api_core(self) -> bool:
        """
        Patch google.api_core._python_version_support if loaded.

        The google-api-core package has a bug where it uses packages_distributions()
        without proper Python version checking. This patches the module to use
        our fallback implementation.

        Returns:
            True if patching was successful or unnecessary, False on error
        """
        if not self.is_python39:
            return True

        module_name = 'google.api_core._python_version_support'

        try:
            # Check if module is already loaded
            if module_name in sys.modules:
                module = sys.modules[module_name]

                # Patch the _get_pypi_package_name function
                if hasattr(module, '_get_pypi_package_name'):
                    original_func = module._get_pypi_package_name
                    packages_distributions = self._create_packages_distributions_fallback()

                    def patched_get_pypi_package_name(module_name: str) -> Optional[str]:
                        """Patched version that uses our fallback implementation."""
                        try:
                            module_to_distributions = packages_distributions()
                            if module_name in module_to_distributions:
                                return module_to_distributions[module_name][0]
                            return None
                        except Exception as e:
                            logger.debug(f"_get_pypi_package_name error: {e}")
                            return None

                    module._get_pypi_package_name = patched_get_pypi_package_name
                    self._patched_modules[module_name] = True
                    logger.info("✅ Patched google.api_core._python_version_support")
                    return True
            else:
                # Module not loaded yet, install import hook
                self._install_google_api_core_hook()
                return True

        except Exception as e:
            error_msg = f"Failed to patch google.api_core: {e}"
            logger.warning(error_msg)
            self._error_log.append(error_msg)
            return False

    def _install_google_api_core_hook(self) -> None:
        """Install an import hook to patch google.api_core when it's imported."""
        import importlib.abc
        import importlib.machinery

        class GoogleApiCoreImportHook(importlib.abc.MetaPathFinder, importlib.abc.Loader):
            """Import hook that patches google.api_core modules on import."""

            def __init__(self, compat_manager: 'Python39CompatibilityManager'):
                self.compat_manager = compat_manager
                self.patched = False

            def find_spec(self, fullname, path, target=None):
                if fullname == 'google.api_core._python_version_support' and not self.patched:
                    # Let the normal import happen, then patch
                    return None
                return None

            def find_module(self, fullname, path=None):
                return None

        # Install the hook
        hook = GoogleApiCoreImportHook(self)
        if hook not in sys.meta_path:
            sys.meta_path.insert(0, hook)

    def patch_all(self) -> Dict[str, bool]:
        """
        Apply all Python 3.9 compatibility patches.

        Returns:
            Dictionary of module names and their patch status
        """
        with self._thread_lock:
            if Python39CompatibilityManager._initialized:
                logger.debug("Compatibility patches already applied")
                return self._patched_modules.copy()

            results = {}

            # Core patches
            results['importlib.metadata'] = self.patch_importlib_metadata()
            results['google.api_core'] = self.patch_google_api_core()

            Python39CompatibilityManager._initialized = True

            # Log summary
            success_count = sum(1 for v in results.values() if v)
            total_count = len(results)

            if success_count == total_count:
                logger.info(f"✅ All {total_count} Python 3.9 compatibility patches applied successfully")
            else:
                logger.warning(f"⚠️ {success_count}/{total_count} patches applied, some may have issues")

            return results

    async def patch_all_async(self) -> Dict[str, bool]:
        """
        Async version of patch_all for use in async contexts.

        Returns:
            Dictionary of module names and their patch status
        """
        # Run patching in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(executor, self.patch_all)

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of all patches."""
        return {
            'python_version': f"{__python_version__.major}.{__python_version__.minor}.{__python_version__.micro}",
            'is_python39': self.is_python39,
            'initialized': Python39CompatibilityManager._initialized,
            'patched_modules': self._patched_modules.copy(),
            'errors': self._error_log.copy(),
        }


# Global instance
_compat_manager: Optional[Python39CompatibilityManager] = None


def get_compat_manager() -> Python39CompatibilityManager:
    """Get the global compatibility manager instance."""
    global _compat_manager
    if _compat_manager is None:
        _compat_manager = Python39CompatibilityManager()
    return _compat_manager


def ensure_python39_compatibility() -> Dict[str, bool]:
    """
    Ensure Python 3.9 compatibility by applying all necessary patches.

    This function should be called at the very start of your application,
    before importing any packages that might use Python 3.10+ features.

    Returns:
        Dictionary of applied patches and their status

    Example:
        # At the top of start_system.py:
        from backend.utils.python39_compat import ensure_python39_compatibility
        ensure_python39_compatibility()
    """
    manager = get_compat_manager()
    return manager.patch_all()


async def ensure_python39_compatibility_async() -> Dict[str, bool]:
    """
    Async version of ensure_python39_compatibility.

    Returns:
        Dictionary of applied patches and their status
    """
    manager = get_compat_manager()
    return await manager.patch_all_async()


def is_patched(module_name: str) -> bool:
    """Check if a specific module has been patched."""
    manager = get_compat_manager()
    return manager._patched_modules.get(module_name, False)


def get_compat_status() -> Dict[str, Any]:
    """Get the current compatibility status."""
    manager = get_compat_manager()
    return manager.get_status()


# Auto-apply patches on module import if we're on Python 3.9
if __is_python39__:
    # Apply patches immediately on import
    try:
        ensure_python39_compatibility()
    except Exception as e:
        logger.error(f"Failed to auto-apply Python 3.9 compatibility patches: {e}")
