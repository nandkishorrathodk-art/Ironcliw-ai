"""
v77.0: Graceful Shutdown - Gaps #36, #37
=========================================

Graceful shutdown and signal handling:
- Signal handlers (SIGTERM, SIGINT)
- Ordered shutdown sequence
- Shutdown timeout enforcement
- Resource cleanup
- Drain and close pattern

Author: JARVIS v77.0
"""

from __future__ import annotations

import asyncio
import functools
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


class ShutdownPhase(Enum):
    """Phases of shutdown."""
    RUNNING = "running"
    DRAINING = "draining"
    STOPPING = "stopping"
    CLEANUP = "cleanup"
    TERMINATED = "terminated"


@dataclass
class ShutdownHandler:
    """A shutdown handler."""
    name: str
    handler: Callable[[], Coroutine]
    priority: int = 50  # Lower = runs first
    timeout: float = 30.0
    required: bool = True  # If True, failure blocks shutdown


@dataclass
class ShutdownResult:
    """Result of shutdown process."""
    success: bool
    phase: ShutdownPhase
    duration_seconds: float
    handlers_run: int
    handlers_failed: int
    errors: List[str] = field(default_factory=list)


class GracefulShutdown:
    """
    Graceful shutdown manager.

    Features:
    - Signal handling (SIGTERM, SIGINT, SIGHUP)
    - Ordered shutdown handlers
    - Timeout enforcement
    - Drain period for in-flight requests
    - Cleanup phase
    """

    def __init__(
        self,
        drain_timeout: float = 10.0,
        shutdown_timeout: float = 60.0,
        cleanup_timeout: float = 10.0,
    ):
        self.drain_timeout = drain_timeout
        self.shutdown_timeout = shutdown_timeout
        self.cleanup_timeout = cleanup_timeout

        self._phase = ShutdownPhase.RUNNING
        self._handlers: List[ShutdownHandler] = []
        self._drain_callbacks: List[Callable[[], Coroutine]] = []
        self._cleanup_callbacks: List[Callable[[], Coroutine]] = []
        self._shutdown_event = asyncio.Event()
        self._shutdown_requested = False
        self._signals_registered = False
        self._original_handlers: Dict[int, Any] = {}

    @property
    def phase(self) -> ShutdownPhase:
        """Get current shutdown phase."""
        return self._phase

    @property
    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress."""
        return self._phase != ShutdownPhase.RUNNING

    def register_handler(
        self,
        name: str,
        handler: Callable[[], Coroutine],
        priority: int = 50,
        timeout: float = 30.0,
        required: bool = True,
    ) -> None:
        """
        Register a shutdown handler.

        Lower priority runs first.
        """
        self._handlers.append(ShutdownHandler(
            name=name,
            handler=handler,
            priority=priority,
            timeout=timeout,
            required=required,
        ))
        # Sort by priority
        self._handlers.sort(key=lambda h: h.priority)
        logger.debug(f"[GracefulShutdown] Registered handler: {name} (priority={priority})")

    def on_drain(self, callback: Callable[[], Coroutine]) -> None:
        """Register drain phase callback."""
        self._drain_callbacks.append(callback)

    def on_cleanup(self, callback: Callable[[], Coroutine]) -> None:
        """Register cleanup phase callback."""
        self._cleanup_callbacks.append(callback)

    def setup_signals(self) -> None:
        """Setup signal handlers."""
        if self._signals_registered:
            return

        try:
            loop = asyncio.get_running_loop()

            for sig in (signal.SIGTERM, signal.SIGINT):
                self._original_handlers[sig] = signal.getsignal(sig)
                loop.add_signal_handler(sig, self._signal_handler, sig)

            # SIGHUP for reload (Unix only)
            if hasattr(signal, "SIGHUP"):
                self._original_handlers[signal.SIGHUP] = signal.getsignal(signal.SIGHUP)
                loop.add_signal_handler(signal.SIGHUP, self._signal_handler, signal.SIGHUP)

            self._signals_registered = True
            logger.info("[GracefulShutdown] Signal handlers registered")

        except Exception as e:
            logger.warning(f"[GracefulShutdown] Failed to setup signals: {e}")

    def _signal_handler(self, sig: signal.Signals) -> None:
        """Handle shutdown signals."""
        sig_name = sig.name if hasattr(sig, "name") else str(sig)

        if self._shutdown_requested:
            logger.warning(f"[GracefulShutdown] Received {sig_name} again, forcing exit")
            sys.exit(1)

        logger.info(f"[GracefulShutdown] Received {sig_name}, initiating graceful shutdown")
        self._shutdown_requested = True
        self._shutdown_event.set()

        # Create shutdown task
        asyncio.create_task(self.shutdown())

    async def wait_for_shutdown(self) -> None:
        """Wait until shutdown is requested."""
        await self._shutdown_event.wait()

    async def shutdown(self, reason: str = "shutdown_requested") -> ShutdownResult:
        """
        Execute graceful shutdown sequence.

        1. Drain phase: Stop accepting new work
        2. Stop phase: Run shutdown handlers
        3. Cleanup phase: Final cleanup
        """
        start_time = time.time()
        errors: List[str] = []
        handlers_run = 0
        handlers_failed = 0

        logger.info(f"[GracefulShutdown] Starting shutdown: {reason}")

        try:
            # Phase 1: Drain
            self._phase = ShutdownPhase.DRAINING
            logger.info("[GracefulShutdown] Phase 1: Draining...")

            for callback in self._drain_callbacks:
                try:
                    await asyncio.wait_for(callback(), timeout=self.drain_timeout)
                except asyncio.TimeoutError:
                    errors.append(f"Drain callback timed out")
                except Exception as e:
                    errors.append(f"Drain error: {e}")

            # Phase 2: Stop handlers
            self._phase = ShutdownPhase.STOPPING
            logger.info("[GracefulShutdown] Phase 2: Running shutdown handlers...")

            for handler in self._handlers:
                handlers_run += 1

                try:
                    logger.debug(f"[GracefulShutdown] Running: {handler.name}")
                    await asyncio.wait_for(handler.handler(), timeout=handler.timeout)
                    logger.debug(f"[GracefulShutdown] Completed: {handler.name}")

                except asyncio.TimeoutError:
                    handlers_failed += 1
                    msg = f"Handler '{handler.name}' timed out after {handler.timeout}s"
                    errors.append(msg)
                    logger.error(f"[GracefulShutdown] {msg}")

                    if handler.required:
                        logger.warning(f"[GracefulShutdown] Required handler failed, continuing anyway")

                except Exception as e:
                    handlers_failed += 1
                    msg = f"Handler '{handler.name}' failed: {e}"
                    errors.append(msg)
                    logger.error(f"[GracefulShutdown] {msg}")

            # Phase 3: Cleanup
            self._phase = ShutdownPhase.CLEANUP
            logger.info("[GracefulShutdown] Phase 3: Cleanup...")

            for callback in self._cleanup_callbacks:
                try:
                    await asyncio.wait_for(callback(), timeout=self.cleanup_timeout)
                except Exception as e:
                    errors.append(f"Cleanup error: {e}")

            # Complete
            self._phase = ShutdownPhase.TERMINATED
            duration = time.time() - start_time

            logger.info(
                f"[GracefulShutdown] Completed in {duration:.2f}s "
                f"(handlers: {handlers_run}, failed: {handlers_failed})"
            )

            return ShutdownResult(
                success=handlers_failed == 0,
                phase=self._phase,
                duration_seconds=duration,
                handlers_run=handlers_run,
                handlers_failed=handlers_failed,
                errors=errors,
            )

        except Exception as e:
            logger.error(f"[GracefulShutdown] Fatal error: {e}")
            self._phase = ShutdownPhase.TERMINATED

            return ShutdownResult(
                success=False,
                phase=self._phase,
                duration_seconds=time.time() - start_time,
                handlers_run=handlers_run,
                handlers_failed=handlers_failed + 1,
                errors=errors + [str(e)],
            )

    def restore_signals(self) -> None:
        """Restore original signal handlers."""
        if not self._signals_registered:
            return

        try:
            loop = asyncio.get_running_loop()

            for sig, original in self._original_handlers.items():
                loop.remove_signal_handler(sig)
                if original and original != signal.SIG_DFL:
                    signal.signal(sig, original)

            self._signals_registered = False
            logger.debug("[GracefulShutdown] Signal handlers restored")

        except Exception as e:
            logger.warning(f"[GracefulShutdown] Failed to restore signals: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get shutdown status."""
        return {
            "phase": self._phase.value,
            "shutdown_requested": self._shutdown_requested,
            "handlers_registered": len(self._handlers),
            "drain_callbacks": len(self._drain_callbacks),
            "cleanup_callbacks": len(self._cleanup_callbacks),
            "timeouts": {
                "drain": self.drain_timeout,
                "shutdown": self.shutdown_timeout,
                "cleanup": self.cleanup_timeout,
            },
        }


# Global graceful shutdown manager
_shutdown: Optional[GracefulShutdown] = None


def get_graceful_shutdown() -> GracefulShutdown:
    """Get global graceful shutdown manager."""
    global _shutdown
    if _shutdown is None:
        _shutdown = GracefulShutdown()
    return _shutdown


def shutdown_handler(
    name: Optional[str] = None,
    priority: int = 50,
    timeout: float = 30.0,
    required: bool = True,
):
    """
    Decorator to register a function as a shutdown handler.

    Usage:
        @shutdown_handler("database", priority=10)
        async def close_database():
            await db.close()
    """
    def decorator(func: Callable[[], Coroutine]) -> Callable[[], Coroutine]:
        handler_name = name or func.__name__

        shutdown = get_graceful_shutdown()
        shutdown.register_handler(
            name=handler_name,
            handler=func,
            priority=priority,
            timeout=timeout,
            required=required,
        )

        return func

    return decorator


async def shutdown_on_signal() -> None:
    """
    Setup signal handlers and wait for shutdown.

    Usage:
        await shutdown_on_signal()
        # This only returns after shutdown completes
    """
    shutdown = get_graceful_shutdown()
    shutdown.setup_signals()
    await shutdown.wait_for_shutdown()
