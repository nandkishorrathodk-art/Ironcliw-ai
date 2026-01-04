"""
v67.0 CEREBRO PROTOCOL - System-Wide App Awareness

This module provides JARVIS with dynamic, real-time awareness of all installed
applications on macOS using Spotlight integration (mdfind).

FEATURES:
- O(1) app lookup via macOS Spotlight index (mdfind)
- No hardcoding - discovers apps dynamically
- Async subprocess execution for non-blocking queries
- Intelligent caching with TTL for performance
- Running vs Installed app detection
- Fuzzy matching fallback when Spotlight returns multiple results
- Integration with Yabai for window-based app verification

ROOT CAUSE FIX:
Instead of maintaining a static alias list that needs manual updates,
we query the macOS Metadata Server directly. This means:
- Install new app → JARVIS knows it instantly
- Rename app → JARVIS adapts automatically
- Zero maintenance required

USAGE:
    from backend.system.app_library import get_app_library

    library = get_app_library()

    # Find installed app
    result = await library.resolve_app_name_async("Chrome")
    # Returns: {"found": True, "app_name": "Google Chrome", "path": "/Applications/Google Chrome.app"}

    # Check if running
    is_running = await library.is_app_running_async("Google Chrome")
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# v67.0: APP RESOLUTION RESULT
# =============================================================================

@dataclass
class AppResolutionResult:
    """Result of an app name resolution query."""
    found: bool
    app_name: Optional[str] = None  # Display name (e.g., "Google Chrome")
    bundle_name: Optional[str] = None  # Bundle name (e.g., "Google Chrome.app")
    path: Optional[str] = None  # Full path (e.g., "/Applications/Google Chrome.app")
    bundle_id: Optional[str] = None  # CFBundleIdentifier (e.g., "com.google.Chrome")
    is_running: bool = False
    window_count: int = 0
    confidence: float = 0.0
    resolution_method: str = "none"  # spotlight, cache, alias, fuzzy
    query_time_ms: float = 0.0


# =============================================================================
# v67.0: SPOTLIGHT CACHE ENTRY
# =============================================================================

@dataclass
class SpotlightCacheEntry:
    """Cached app information from Spotlight."""
    app_name: str
    bundle_name: str
    path: str
    bundle_id: Optional[str]
    created_at: datetime = field(default_factory=datetime.now)

    def is_expired(self, ttl_seconds: int = 3600) -> bool:
        """Check if cache entry has expired."""
        return datetime.now() > self.created_at + timedelta(seconds=ttl_seconds)


# =============================================================================
# v67.0: APP LIBRARY SINGLETON
# =============================================================================

class AppLibrary:
    """
    v67.0 CEREBRO PROTOCOL: Dynamic App Awareness via Spotlight Integration.

    This singleton provides JARVIS with real-time knowledge of all installed
    applications on macOS without any hardcoding.

    Architecture:
    1. Query Cache (fastest) - Recently resolved apps
    2. Spotlight Index (fast) - macOS mdfind query
    3. Fuzzy Search (slower) - When Spotlight returns multiple matches
    4. Yabai Integration - Verify running state and window count
    """

    _instance: Optional['AppLibrary'] = None
    _lock = asyncio.Lock()

    def __new__(cls) -> 'AppLibrary':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True

        # Cache for resolved apps (user_input → SpotlightCacheEntry)
        self._resolution_cache: Dict[str, SpotlightCacheEntry] = {}

        # Full app catalog (populated on first query or refresh)
        self._app_catalog: Dict[str, SpotlightCacheEntry] = {}
        self._catalog_last_refresh: Optional[datetime] = None
        self._catalog_ttl_seconds: int = 3600  # Refresh catalog hourly

        # Common variations/aliases (still useful as fast path)
        # These are DYNAMIC - populated from Spotlight, not hardcoded
        self._learned_aliases: Dict[str, str] = {}

        # Stats
        self._stats = {
            "cache_hits": 0,
            "spotlight_queries": 0,
            "fuzzy_fallbacks": 0,
            "total_queries": 0,
            "avg_query_time_ms": 0.0
        }

        logger.info("[v67.0] 🧠 CEREBRO: AppLibrary initialized")

    # =========================================================================
    # PRIMARY API: Resolve App Name
    # =========================================================================

    async def resolve_app_name_async(
        self,
        user_input: str,
        include_running_status: bool = True,
        threshold: float = 0.7
    ) -> AppResolutionResult:
        """
        v67.0: Resolve a user-provided app name to its actual system identity.

        This is the primary entry point for app resolution. It uses a multi-tier
        approach for maximum speed and accuracy:

        1. Cache lookup (O(1), ~0.1ms)
        2. Spotlight query (O(1), ~10-50ms)
        3. Fuzzy catalog search (O(N), ~100ms)

        Args:
            user_input: What the user said (e.g., "Chrome", "VS Code")
            include_running_status: Also check if app is running via Yabai
            threshold: Minimum confidence for fuzzy matching

        Returns:
            AppResolutionResult with full app details
        """
        start_time = time.time()
        self._stats["total_queries"] += 1

        if not user_input or not user_input.strip():
            return AppResolutionResult(found=False, resolution_method="invalid_input")

        normalized_input = user_input.strip().lower()

        # =====================================================================
        # TIER 1: Cache Lookup (Fastest)
        # =====================================================================
        if normalized_input in self._resolution_cache:
            entry = self._resolution_cache[normalized_input]
            if not entry.is_expired(self._catalog_ttl_seconds):
                self._stats["cache_hits"] += 1
                result = AppResolutionResult(
                    found=True,
                    app_name=entry.app_name,
                    bundle_name=entry.bundle_name,
                    path=entry.path,
                    bundle_id=entry.bundle_id,
                    confidence=1.0,
                    resolution_method="cache",
                    query_time_ms=(time.time() - start_time) * 1000
                )

                if include_running_status:
                    result.is_running, result.window_count = await self._check_running_status_async(entry.app_name)

                logger.debug(f"[v67.0] Cache hit: '{user_input}' → '{entry.app_name}'")
                return result

        # =====================================================================
        # TIER 2: Direct Spotlight Query (Fast)
        # =====================================================================
        spotlight_result = await self._query_spotlight_async(user_input)

        if spotlight_result:
            self._stats["spotlight_queries"] += 1

            # Cache the result
            cache_entry = SpotlightCacheEntry(
                app_name=spotlight_result["app_name"],
                bundle_name=spotlight_result["bundle_name"],
                path=spotlight_result["path"],
                bundle_id=spotlight_result.get("bundle_id")
            )
            self._resolution_cache[normalized_input] = cache_entry

            result = AppResolutionResult(
                found=True,
                app_name=spotlight_result["app_name"],
                bundle_name=spotlight_result["bundle_name"],
                path=spotlight_result["path"],
                bundle_id=spotlight_result.get("bundle_id"),
                confidence=spotlight_result.get("confidence", 0.95),
                resolution_method="spotlight",
                query_time_ms=(time.time() - start_time) * 1000
            )

            if include_running_status:
                result.is_running, result.window_count = await self._check_running_status_async(result.app_name)

            logger.info(f"[v67.0] 🎯 Spotlight found: '{user_input}' → '{result.app_name}'")
            return result

        # =====================================================================
        # TIER 3: Fuzzy Catalog Search (Slower but comprehensive)
        # =====================================================================
        await self._ensure_catalog_fresh_async()

        fuzzy_result = self._fuzzy_search_catalog(user_input, threshold)

        if fuzzy_result:
            self._stats["fuzzy_fallbacks"] += 1

            # Cache the result
            cache_entry = SpotlightCacheEntry(
                app_name=fuzzy_result["app_name"],
                bundle_name=fuzzy_result["bundle_name"],
                path=fuzzy_result["path"],
                bundle_id=fuzzy_result.get("bundle_id")
            )
            self._resolution_cache[normalized_input] = cache_entry

            # Learn this alias for faster future lookups
            self._learned_aliases[normalized_input] = fuzzy_result["app_name"]

            result = AppResolutionResult(
                found=True,
                app_name=fuzzy_result["app_name"],
                bundle_name=fuzzy_result["bundle_name"],
                path=fuzzy_result["path"],
                bundle_id=fuzzy_result.get("bundle_id"),
                confidence=fuzzy_result.get("confidence", 0.8),
                resolution_method="fuzzy_catalog",
                query_time_ms=(time.time() - start_time) * 1000
            )

            if include_running_status:
                result.is_running, result.window_count = await self._check_running_status_async(result.app_name)

            logger.info(f"[v67.0] 🔍 Fuzzy catalog match: '{user_input}' → '{result.app_name}'")
            return result

        # =====================================================================
        # NOT FOUND
        # =====================================================================
        query_time_ms = (time.time() - start_time) * 1000
        logger.info(f"[v67.0] ❌ App not found: '{user_input}' (searched in {query_time_ms:.1f}ms)")

        return AppResolutionResult(
            found=False,
            resolution_method="not_found",
            query_time_ms=query_time_ms
        )

    # =========================================================================
    # SPOTLIGHT INTEGRATION
    # =========================================================================

    async def _query_spotlight_async(self, app_name: str) -> Optional[Dict[str, Any]]:
        """
        Query macOS Spotlight for an application.

        Uses mdfind with kMDItemKind == 'Application' for O(1) lookup.
        """
        try:
            # Build mdfind query - case insensitive search
            # kMDItemDisplayName is the user-visible name
            # kMDItemKind == 'Application' filters to only apps
            query = f"kMDItemKind == 'Application' && kMDItemDisplayName == '*{app_name}*'c"

            proc = await asyncio.create_subprocess_exec(
                'mdfind', query,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)

            if proc.returncode != 0:
                logger.debug(f"[v67.0] mdfind error: {stderr.decode()}")
                return None

            paths = stdout.decode().strip().split('\n')
            paths = [p for p in paths if p.endswith('.app')]

            if not paths:
                return None

            # Find best match from results
            best_path = None
            best_confidence = 0.0

            for path in paths:
                bundle_name = os.path.basename(path)
                display_name = bundle_name.replace('.app', '')

                # Calculate confidence based on how well it matches
                confidence = self._calculate_match_confidence(app_name, display_name)

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_path = path

            if best_path and best_confidence >= 0.5:
                bundle_name = os.path.basename(best_path)
                display_name = bundle_name.replace('.app', '')

                # Try to get bundle ID
                bundle_id = await self._get_bundle_id_async(best_path)

                return {
                    "app_name": display_name,
                    "bundle_name": bundle_name,
                    "path": best_path,
                    "bundle_id": bundle_id,
                    "confidence": best_confidence
                }

            return None

        except asyncio.TimeoutError:
            logger.warning("[v67.0] Spotlight query timed out")
            return None
        except Exception as e:
            logger.error(f"[v67.0] Spotlight query error: {e}")
            return None

    async def _get_bundle_id_async(self, app_path: str) -> Optional[str]:
        """Get the CFBundleIdentifier from an app's Info.plist."""
        try:
            plist_path = os.path.join(app_path, "Contents", "Info.plist")
            if not os.path.exists(plist_path):
                return None

            proc = await asyncio.create_subprocess_exec(
                'defaults', 'read', plist_path, 'CFBundleIdentifier',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)

            if proc.returncode == 0:
                return stdout.decode().strip()
            return None

        except Exception:
            return None

    # =========================================================================
    # APP CATALOG (Full System Scan)
    # =========================================================================

    async def _ensure_catalog_fresh_async(self):
        """Ensure the app catalog is up to date."""
        if self._catalog_last_refresh is None:
            await self._refresh_catalog_async()
        elif datetime.now() > self._catalog_last_refresh + timedelta(seconds=self._catalog_ttl_seconds):
            await self._refresh_catalog_async()

    async def _refresh_catalog_async(self):
        """
        Refresh the full app catalog from Spotlight.

        This is a heavier operation that scans all installed apps.
        """
        logger.info("[v67.0] 🔄 Refreshing app catalog from Spotlight...")
        start_time = time.time()

        try:
            # Query all applications
            query = "kMDItemKind == 'Application'"

            proc = await asyncio.create_subprocess_exec(
                'mdfind', query,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30.0)

            if proc.returncode != 0:
                logger.warning("[v67.0] Failed to refresh app catalog")
                return

            paths = stdout.decode().strip().split('\n')
            paths = [p for p in paths if p.endswith('.app')]

            new_catalog: Dict[str, SpotlightCacheEntry] = {}

            for path in paths:
                bundle_name = os.path.basename(path)
                display_name = bundle_name.replace('.app', '')
                normalized = display_name.lower()

                new_catalog[normalized] = SpotlightCacheEntry(
                    app_name=display_name,
                    bundle_name=bundle_name,
                    path=path,
                    bundle_id=None  # Skip bundle ID for catalog refresh (too slow)
                )

            self._app_catalog = new_catalog
            self._catalog_last_refresh = datetime.now()

            elapsed = (time.time() - start_time) * 1000
            logger.info(f"[v67.0] ✅ Catalog refreshed: {len(new_catalog)} apps in {elapsed:.0f}ms")

        except Exception as e:
            logger.error(f"[v67.0] Catalog refresh error: {e}")

    def _fuzzy_search_catalog(self, user_input: str, threshold: float) -> Optional[Dict[str, Any]]:
        """Fuzzy search the app catalog."""
        if not self._app_catalog:
            return None

        normalized = user_input.lower().strip()

        # Quick exact match
        if normalized in self._app_catalog:
            entry = self._app_catalog[normalized]
            return {
                "app_name": entry.app_name,
                "bundle_name": entry.bundle_name,
                "path": entry.path,
                "bundle_id": entry.bundle_id,
                "confidence": 1.0
            }

        # Fuzzy search
        best_match = None
        best_confidence = 0.0

        for catalog_name, entry in self._app_catalog.items():
            confidence = self._calculate_match_confidence(normalized, catalog_name)

            if confidence > best_confidence and confidence >= threshold:
                best_confidence = confidence
                best_match = entry

        if best_match:
            return {
                "app_name": best_match.app_name,
                "bundle_name": best_match.bundle_name,
                "path": best_match.path,
                "bundle_id": best_match.bundle_id,
                "confidence": best_confidence
            }

        return None

    def _calculate_match_confidence(self, user_input: str, app_name: str) -> float:
        """Calculate match confidence between user input and app name."""
        user_lower = user_input.lower().strip()
        app_lower = app_name.lower().strip()

        # Remove common prefixes for comparison
        for prefix in ['google ', 'apple ', 'microsoft ', 'mozilla ', 'adobe ']:
            if app_lower.startswith(prefix):
                app_lower_stripped = app_lower[len(prefix):]
                if user_lower == app_lower_stripped:
                    return 0.98
                if user_lower in app_lower_stripped:
                    return 0.95

        # Exact match
        if user_lower == app_lower:
            return 1.0

        # Substring match
        if user_lower in app_lower:
            return 0.9
        if app_lower in user_lower:
            return 0.85

        # Word match
        user_words = set(user_lower.split())
        app_words = set(app_lower.split())
        if user_words & app_words:  # Intersection
            return 0.75

        # Character overlap (basic fuzzy)
        matching = sum(1 for c in user_lower if c in app_lower)
        ratio = matching / max(len(user_lower), 1)

        return ratio * 0.7  # Scale down character overlap

    # =========================================================================
    # RUNNING STATUS CHECK (Yabai Integration)
    # =========================================================================

    async def _check_running_status_async(self, app_name: str) -> Tuple[bool, int]:
        """
        Check if an app is running and count its windows via Yabai.

        Returns:
            Tuple of (is_running, window_count)
        """
        try:
            yabai_path = os.getenv("YABAI_PATH", "/opt/homebrew/bin/yabai")

            proc = await asyncio.create_subprocess_exec(
                yabai_path, "-m", "query", "--windows",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3.0)

            if proc.returncode != 0:
                return False, 0

            windows = json.loads(stdout.decode())

            # Count windows matching this app
            app_lower = app_name.lower()
            matching_windows = [
                w for w in windows
                if app_lower in w.get("app", "").lower()
            ]

            return len(matching_windows) > 0, len(matching_windows)

        except Exception as e:
            logger.debug(f"[v67.0] Running status check failed: {e}")
            return False, 0

    async def is_app_running_async(self, app_name: str) -> bool:
        """Check if an app is currently running."""
        is_running, _ = await self._check_running_status_async(app_name)
        return is_running

    async def get_running_apps_async(self) -> List[str]:
        """Get list of all currently running apps with windows."""
        try:
            yabai_path = os.getenv("YABAI_PATH", "/opt/homebrew/bin/yabai")

            proc = await asyncio.create_subprocess_exec(
                yabai_path, "-m", "query", "--windows",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3.0)

            if proc.returncode != 0:
                return []

            windows = json.loads(stdout.decode())

            # Get unique app names
            return list(set(w.get("app", "") for w in windows if w.get("app")))

        except Exception:
            return []

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get library statistics."""
        return {
            **self._stats,
            "cache_size": len(self._resolution_cache),
            "catalog_size": len(self._app_catalog),
            "learned_aliases": len(self._learned_aliases),
            "catalog_age_seconds": (
                (datetime.now() - self._catalog_last_refresh).total_seconds()
                if self._catalog_last_refresh else None
            )
        }

    def clear_cache(self):
        """Clear the resolution cache."""
        self._resolution_cache.clear()
        logger.info("[v67.0] Resolution cache cleared")

    async def refresh_async(self):
        """Force refresh the app catalog."""
        await self._refresh_catalog_async()


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

_app_library_instance: Optional[AppLibrary] = None


def get_app_library() -> AppLibrary:
    """Get the singleton AppLibrary instance."""
    global _app_library_instance
    if _app_library_instance is None:
        _app_library_instance = AppLibrary()
    return _app_library_instance


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def resolve_app_name(user_input: str) -> AppResolutionResult:
    """Convenience function to resolve an app name."""
    library = get_app_library()
    return await library.resolve_app_name_async(user_input)


async def is_app_installed(app_name: str) -> bool:
    """Check if an app is installed on the system."""
    library = get_app_library()
    result = await library.resolve_app_name_async(app_name, include_running_status=False)
    return result.found


async def is_app_running(app_name: str) -> bool:
    """Check if an app is currently running."""
    library = get_app_library()
    return await library.is_app_running_async(app_name)
