"""
Cache Registry v1.0 — Unified monitoring for all cache instances.

Provides a singleton registry where all cache subsystems can register
themselves for centralized stats, memory pressure response, and health
monitoring.

Usage:
    from backend.utils.cache_registry import get_cache_registry

    registry = get_cache_registry()

    # Register a cache (any object with get_stats() and optional clear())
    registry.register("voice_biometric", my_cache)

    # Get unified stats
    stats = registry.get_all_stats()

    # Respond to memory pressure
    freed = await registry.evict_under_pressure(target_mb=100)
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger("jarvis.cache_registry")


# ---------------------------------------------------------------------------
# Protocol — what a "cache" must look like to be registered
# ---------------------------------------------------------------------------

@runtime_checkable
class CacheLike(Protocol):
    """Minimum interface for a registerable cache."""

    def get_stats(self) -> Dict[str, Any]: ...


@runtime_checkable
class ClearableCacheLike(CacheLike, Protocol):
    """Cache that also supports clearing."""

    async def clear(self) -> None: ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class CacheRegistry:
    """
    Singleton registry for all cache instances in the system.

    Thread-safe. Caches register themselves at init time and the
    registry provides a single pane of glass for stats + eviction.
    """

    _instance: Optional["CacheRegistry"] = None
    _init_lock = threading.Lock()

    def __init__(self) -> None:
        self._caches: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._registered_at: Dict[str, float] = {}

    @classmethod
    def get_instance(cls) -> "CacheRegistry":
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # -- Registration -------------------------------------------------------

    def register(self, name: str, cache: Any) -> None:
        """
        Register a cache instance.

        The *cache* must have a ``get_stats() -> dict`` method.
        If it also has ``clear()`` (sync or async), it can participate
        in memory-pressure eviction.
        """
        with self._lock:
            if name in self._caches:
                logger.debug(f"[CacheRegistry] Re-registering cache: {name}")
            self._caches[name] = cache
            self._registered_at[name] = time.time()
            logger.debug(
                f"[CacheRegistry] Registered cache '{name}' "
                f"(type={type(cache).__name__})"
            )

    def unregister(self, name: str) -> None:
        """Remove a cache from the registry."""
        with self._lock:
            self._caches.pop(name, None)
            self._registered_at.pop(name, None)

    # -- Stats --------------------------------------------------------------

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return stats from every registered cache."""
        result: Dict[str, Dict[str, Any]] = {}
        with self._lock:
            caches = dict(self._caches)

        for name, cache in caches.items():
            try:
                stats = cache.get_stats()
                stats["_registered_at"] = self._registered_at.get(name, 0)
                stats["_type"] = type(cache).__name__
                result[name] = stats
            except Exception as exc:
                result[name] = {"error": str(exc)}
        return result

    def get_summary(self) -> Dict[str, Any]:
        """Return high-level summary across all caches."""
        all_stats = self.get_all_stats()
        total_size_mb = 0.0
        total_entries = 0
        total_evictions = 0
        for stats in all_stats.values():
            total_size_mb += stats.get("size_mb", 0.0)
            total_entries += stats.get("entries", stats.get("size", 0))
            total_evictions += stats.get("eviction_count", stats.get("evictions", 0))
        return {
            "cache_count": len(all_stats),
            "total_size_mb": round(total_size_mb, 2),
            "total_entries": total_entries,
            "total_evictions": total_evictions,
            "caches": list(all_stats.keys()),
        }

    # -- Memory pressure ----------------------------------------------------

    async def evict_under_pressure(self, target_mb: float = 50.0) -> float:
        """
        Ask all clearable caches to evict entries.

        Returns approximate MB freed (best-effort estimate).
        """
        freed_mb = 0.0
        with self._lock:
            caches = dict(self._caches)

        for name, cache in caches.items():
            if freed_mb >= target_mb:
                break
            try:
                before = cache.get_stats()
                before_mb = before.get("size_mb", 0.0)

                # Try async clear first, then sync
                if hasattr(cache, "clear"):
                    import asyncio
                    clear_fn = cache.clear
                    if asyncio.iscoroutinefunction(clear_fn):
                        await clear_fn()
                    else:
                        clear_fn()

                after = cache.get_stats()
                after_mb = after.get("size_mb", 0.0)
                delta = before_mb - after_mb
                if delta > 0:
                    freed_mb += delta
                    logger.info(
                        f"[CacheRegistry] Cleared cache '{name}': "
                        f"freed ~{delta:.1f}MB"
                    )
            except Exception as exc:
                logger.warning(
                    f"[CacheRegistry] Failed to clear cache '{name}': {exc}"
                )

        logger.info(
            f"[CacheRegistry] Memory pressure response: "
            f"freed ~{freed_mb:.1f}MB across {len(caches)} caches"
        )
        return freed_mb

    @property
    def cache_count(self) -> int:
        return len(self._caches)

    def __repr__(self) -> str:
        return f"CacheRegistry(caches={len(self._caches)})"


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def get_cache_registry() -> CacheRegistry:
    """Get the singleton cache registry."""
    return CacheRegistry.get_instance()
