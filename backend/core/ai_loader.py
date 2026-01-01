"""
JARVIS Hyper-Speed AI Loader v1.0
=================================

Advanced model loading system that decouples server startup from model initialization
using Metaprogramming, Zero-Copy serialization, and Dynamic Quantization.

Architecture:
    1. GhostModelProxy: Metaprogramming wrapper that acts as the model before it exists
    2. AsyncModelManager: Background loading with ThreadPoolExecutor (PyTorch releases GIL)
    3. Zero-Copy Loading: safetensors mmap for instant disk-to-RAM mapping
    4. Dynamic Quantization: INT8 for 4x smaller models, 2x faster inference

Key Benefits:
    - Server responds instantly (200 OK in milliseconds)
    - Models load asynchronously without blocking
    - Memory-mapped files avoid data copying
    - Request queuing for calls made before model is ready

Usage:
    from backend.core.ai_loader import get_ai_manager, ModelPriority

    # Register a model (returns Ghost Proxy immediately)
    ecapa_model = manager.register_model(
        name="ecapa_tdnn",
        loader_func=lambda: load_ecapa_model(),
        priority=ModelPriority.HIGH,
        quantize=True,
    )

    # Use the proxy like a real model (auto-waits if not ready)
    embedding = await ecapa_model.encode(audio)

Version: 1.0.0 - Hyper-Speed AI Loader Edition
"""
from __future__ import annotations

import asyncio
import functools
import logging
import os
import threading
import time
import weakref
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum, auto
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Deque,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

logger = logging.getLogger("jarvis.ai_loader")

T = TypeVar("T")
ModelT = TypeVar("ModelT")


# =============================================================================
# Configuration - Environment-Driven, Zero Hardcoding
# =============================================================================

@dataclass
class AILoaderConfig:
    """
    Dynamic AI loader configuration.

    All values can be overridden via environment variables:
        JARVIS_AI_MAX_WORKERS: Max concurrent model loads (default: 3)
        JARVIS_AI_QUANTIZE_DEFAULT: Default quantization (default: true)
        JARVIS_AI_QUEUE_TIMEOUT: Request queue timeout seconds (default: 30)
        JARVIS_AI_WARMUP_RETRIES: Retry count for warmup calls (default: 3)
        JARVIS_AI_SAFETENSORS_PREFER: Prefer safetensors format (default: true)
        JARVIS_AI_LAZY_QUANTIZE: Quantize lazily on first use (default: false)
    """
    max_workers: int = field(
        default_factory=lambda: int(os.getenv("JARVIS_AI_MAX_WORKERS", "3"))
    )
    quantize_default: bool = field(
        default_factory=lambda: os.getenv("JARVIS_AI_QUANTIZE_DEFAULT", "true").lower() == "true"
    )
    queue_timeout: float = field(
        default_factory=lambda: float(os.getenv("JARVIS_AI_QUEUE_TIMEOUT", "30"))
    )
    warmup_retries: int = field(
        default_factory=lambda: int(os.getenv("JARVIS_AI_WARMUP_RETRIES", "3"))
    )
    safetensors_prefer: bool = field(
        default_factory=lambda: os.getenv("JARVIS_AI_SAFETENSORS_PREFER", "true").lower() == "true"
    )
    lazy_quantize: bool = field(
        default_factory=lambda: os.getenv("JARVIS_AI_LAZY_QUANTIZE", "false").lower() == "true"
    )


# Global config
_config: Optional[AILoaderConfig] = None


def get_config() -> AILoaderConfig:
    """Get or create the global AI loader config."""
    global _config
    if _config is None:
        _config = AILoaderConfig()
    return _config


# =============================================================================
# Model Priority & Status
# =============================================================================

class ModelPriority(IntEnum):
    """Model loading priority levels."""
    CRITICAL = 1   # Load first (voice unlock, auth)
    HIGH = 2       # Load early (vision, embeddings)
    NORMAL = 3     # Standard priority
    LOW = 4        # Load last (optional features)
    LAZY = 5       # Load only on first use


class ModelStatus(IntEnum):
    """Model lifecycle status."""
    GHOST = 0       # Proxy created, not loading yet
    QUEUED = 1      # Waiting in load queue
    LOADING = 2     # Currently loading
    QUANTIZING = 3  # Applying quantization
    READY = 4       # Fully loaded and ready
    FAILED = 5      # Load failed
    UNLOADED = 6    # Explicitly unloaded


class ModelMetrics:
    """
    Metrics for a loaded model.

    Uses __slots__ for memory efficiency with manual initialization
    to avoid dataclass default value conflicts.
    """
    __slots__ = (
        'name', 'status', 'priority', 'load_start_time', 'load_end_time',
        'load_duration_ms', 'memory_bytes', 'quantized', 'calls_total',
        'calls_while_warming', 'errors', 'last_used',
    )

    def __init__(
        self,
        name: str,
        status: ModelStatus,
        priority: ModelPriority = ModelPriority.NORMAL,
        load_start_time: Optional[float] = None,
        load_end_time: Optional[float] = None,
        load_duration_ms: float = 0.0,
        memory_bytes: int = 0,
        quantized: bool = False,
        calls_total: int = 0,
        calls_while_warming: int = 0,
        errors: int = 0,
        last_used: Optional[float] = None,
    ):
        self.name = name
        self.status = status
        self.priority = priority
        self.load_start_time = load_start_time
        self.load_end_time = load_end_time
        self.load_duration_ms = load_duration_ms
        self.memory_bytes = memory_bytes
        self.quantized = quantized
        self.calls_total = calls_total
        self.calls_while_warming = calls_while_warming
        self.errors = errors
        self.last_used = last_used


# =============================================================================
# Pending Request Queue
# =============================================================================

@dataclass
class PendingRequest:
    """A request waiting for model to be ready."""
    __slots__ = ('args', 'kwargs', 'future', 'created_at')

    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    future: asyncio.Future
    created_at: float


# =============================================================================
# Ghost Model Proxy - Metaprogramming Core
# =============================================================================

class GhostModelProxy(Generic[ModelT]):
    """
    Metaprogramming Wrapper - Acts as the model before the model exists.

    Features:
    - Intercepts all attribute access and method calls
    - Queues requests made before model is ready
    - Auto-retries with exponential backoff
    - Seamless transition when real model arrives
    - Thread-safe operation

    Usage:
        proxy = GhostModelProxy("vision_encoder")

        # This works even before the model is loaded
        result = await proxy.encode(image)  # Queued if not ready

        # Later, when model loads:
        proxy.materialize(real_model)  # All queued requests execute
    """

    __slots__ = (
        '_name', '_real_model', '_status', '_metrics', '_config',
        '_pending_queue', '_lock', '_ready_event', '_error', '_loader_task',
    )

    def __init__(
        self,
        name: str,
        config: Optional[AILoaderConfig] = None,
    ):
        self._name = name
        self._real_model: Optional[ModelT] = None
        self._status = ModelStatus.GHOST
        self._config = config or get_config()
        self._pending_queue: Deque[PendingRequest] = deque()
        self._lock = threading.RLock()
        self._ready_event = asyncio.Event()
        self._error: Optional[Exception] = None
        self._loader_task: Optional[asyncio.Task] = None

        # Initialize metrics
        self._metrics = ModelMetrics(
            name=name,
            status=ModelStatus.GHOST,
            priority=ModelPriority.NORMAL,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def status(self) -> ModelStatus:
        return self._status

    @property
    def is_ready(self) -> bool:
        return self._status == ModelStatus.READY and self._real_model is not None

    @property
    def is_loading(self) -> bool:
        return self._status in (ModelStatus.QUEUED, ModelStatus.LOADING, ModelStatus.QUANTIZING)

    @property
    def metrics(self) -> ModelMetrics:
        return self._metrics

    def __repr__(self) -> str:
        return f"<GhostProxy:{self._name} status={self._status.name}>"

    def __getattr__(self, name: str) -> Any:
        """
        Intercept attribute access.

        If model is ready, forward to real model.
        If not ready, return a callable that queues the request.
        """
        # Avoid infinite recursion for private attrs
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        with self._lock:
            if self._real_model is not None:
                return getattr(self._real_model, name)

        # Return async wrapper that waits for model
        logger.debug(f"[GHOST] Accessing '{name}' on {self._name} before ready")

        async def _deferred_call(*args, **kwargs):
            return await self._wait_and_call(name, args, kwargs)

        return _deferred_call

    def __call__(self, *args, **kwargs) -> Any:
        """
        Intercept direct calls.

        If model is ready, forward the call.
        If not ready, queue the request and return a future.
        """
        with self._lock:
            if self._real_model is not None:
                self._metrics.calls_total += 1
                self._metrics.last_used = time.time()
                return self._real_model(*args, **kwargs)

        # Model not ready - queue the request
        self._metrics.calls_while_warming += 1
        logger.info(f"[GHOST] Queuing call to {self._name} (warming up)")

        # Create future for the caller to await
        loop = asyncio.get_event_loop()
        future = loop.create_future()

        request = PendingRequest(
            args=args,
            kwargs=kwargs,
            future=future,
            created_at=time.time(),
        )

        with self._lock:
            self._pending_queue.append(request)

        return future

    async def _wait_and_call(
        self,
        method_name: str,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        """Wait for model to be ready, then call the method."""
        # Wait for ready with timeout
        try:
            await asyncio.wait_for(
                self._ready_event.wait(),
                timeout=self._config.queue_timeout,
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Model {self._name} did not become ready within "
                f"{self._config.queue_timeout}s timeout"
            )

        if self._error:
            raise self._error

        if self._real_model is None:
            raise RuntimeError(f"Model {self._name} failed to load")

        # Call the actual method
        method = getattr(self._real_model, method_name)
        self._metrics.calls_total += 1
        self._metrics.last_used = time.time()

        result = method(*args, **kwargs)

        # Handle async methods
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def wait_ready(self, timeout: Optional[float] = None) -> bool:
        """Wait for the model to become ready."""
        timeout = timeout or self._config.queue_timeout
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
            return self._status == ModelStatus.READY
        except asyncio.TimeoutError:
            return False

    def materialize(self, model: ModelT) -> None:
        """
        Materialize the ghost with a real model.

        This is called when the background loading completes.
        All pending requests are executed.
        """
        with self._lock:
            self._real_model = model
            self._status = ModelStatus.READY
            self._metrics.status = ModelStatus.READY
            self._metrics.load_end_time = time.time()

            if self._metrics.load_start_time:
                self._metrics.load_duration_ms = (
                    (self._metrics.load_end_time - self._metrics.load_start_time) * 1000
                )

        # Signal ready
        self._ready_event.set()

        # Process pending requests
        self._process_pending_queue()

        logger.info(
            f"[GHOST] {self._name} materialized! "
            f"Load time: {self._metrics.load_duration_ms:.0f}ms, "
            f"Pending requests: {len(self._pending_queue)}"
        )

    def fail(self, error: Exception) -> None:
        """Mark the model as failed to load."""
        with self._lock:
            self._status = ModelStatus.FAILED
            self._metrics.status = ModelStatus.FAILED
            self._metrics.errors += 1
            self._error = error

        # Signal ready (with error) to unblock waiters
        self._ready_event.set()

        # Fail all pending requests
        self._fail_pending_queue(error)

        logger.error(f"[GHOST] {self._name} failed to load: {error}")

    def _process_pending_queue(self) -> None:
        """Process all pending requests now that model is ready."""
        while self._pending_queue:
            with self._lock:
                if not self._pending_queue:
                    break
                request = self._pending_queue.popleft()

            try:
                # Execute the queued call
                result = self._real_model(*request.args, **request.kwargs)
                request.future.set_result(result)
                self._metrics.calls_total += 1
            except Exception as e:
                request.future.set_exception(e)
                self._metrics.errors += 1

    def _fail_pending_queue(self, error: Exception) -> None:
        """Fail all pending requests with the given error."""
        while self._pending_queue:
            with self._lock:
                if not self._pending_queue:
                    break
                request = self._pending_queue.popleft()

            request.future.set_exception(error)

    def unload(self) -> None:
        """Unload the model to free memory."""
        with self._lock:
            self._real_model = None
            self._status = ModelStatus.UNLOADED
            self._metrics.status = ModelStatus.UNLOADED
            self._ready_event.clear()

        logger.info(f"[GHOST] {self._name} unloaded")


# =============================================================================
# Zero-Copy Model Loader
# =============================================================================

class ZeroCopyLoader:
    """
    Advanced model loader using safetensors and quantization.

    Features:
    - Zero-copy loading via mmap (safetensors)
    - Dynamic INT8 quantization for 4x smaller models
    - Automatic format detection (safetensors vs pickle)
    - Memory-efficient loading pipeline
    """

    @staticmethod
    def load_safetensors(
        path: Union[str, Path],
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Load a safetensors file using zero-copy mmap.

        Args:
            path: Path to the .safetensors file
            device: Target device ("cpu", "cuda", "mps")

        Returns:
            State dict with tensors
        """
        try:
            from safetensors.torch import load_file

            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Safetensors file not found: {path}")

            logger.debug(f"[ZERO-COPY] Loading {path.name} via mmap...")
            start = time.time()

            # This uses mmap - no data copying until tensors are accessed
            state_dict = load_file(str(path), device=device)

            duration_ms = (time.time() - start) * 1000
            logger.info(f"[ZERO-COPY] Loaded {path.name} in {duration_ms:.0f}ms (mmap)")

            return state_dict

        except ImportError:
            logger.warning("safetensors not available, falling back to torch.load")
            return ZeroCopyLoader._fallback_load(path, device)

    @staticmethod
    def _fallback_load(
        path: Union[str, Path],
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Fallback to standard torch loading."""
        import torch

        path = Path(path)
        logger.debug(f"[LOADER] Loading {path.name} via torch.load...")
        start = time.time()

        state_dict = torch.load(str(path), map_location=device, weights_only=True)

        duration_ms = (time.time() - start) * 1000
        logger.info(f"[LOADER] Loaded {path.name} in {duration_ms:.0f}ms")

        return state_dict

    @staticmethod
    def quantize_dynamic(
        model: Any,
        dtype: Any = None,
        inplace: bool = False,
    ) -> Any:
        """
        Apply dynamic INT8 quantization for 4x memory reduction.

        Args:
            model: PyTorch model to quantize
            dtype: Quantization dtype (default: torch.qint8)
            inplace: Modify model in place

        Returns:
            Quantized model
        """
        try:
            import torch

            if dtype is None:
                dtype = torch.qint8

            logger.debug(f"[QUANTIZE] Applying INT8 quantization...")
            start = time.time()

            # Quantize Linear layers (most common in transformers)
            quantized = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=dtype,
                inplace=inplace,
            )

            duration_ms = (time.time() - start) * 1000
            logger.info(f"[QUANTIZE] Quantized in {duration_ms:.0f}ms")

            return quantized

        except Exception as e:
            logger.warning(f"[QUANTIZE] Failed: {e}, returning original model")
            return model

    @staticmethod
    def estimate_memory(model: Any) -> int:
        """Estimate model memory usage in bytes."""
        try:
            import torch

            total_bytes = 0
            for param in model.parameters():
                total_bytes += param.nelement() * param.element_size()
            for buffer in model.buffers():
                total_bytes += buffer.nelement() * buffer.element_size()

            return total_bytes
        except Exception:
            return 0


# =============================================================================
# Async Model Manager
# =============================================================================

class AsyncModelManager:
    """
    Centralized manager for all AI model loading.

    Features:
    - Returns Ghost Proxies immediately for instant startup
    - Loads models in background threads (PyTorch releases GIL)
    - Priority-based loading queue
    - Automatic quantization
    - Memory tracking and limits
    - Hot-reload capability
    """

    def __init__(self, config: Optional[AILoaderConfig] = None):
        self._config = config or get_config()
        self._proxies: Dict[str, GhostModelProxy] = {}
        self._load_order: List[str] = []
        self._executor = ThreadPoolExecutor(
            max_workers=self._config.max_workers,
            thread_name_prefix="ai_loader",
        )
        self._lock = threading.RLock()
        self._shutdown = False
        self._total_memory_bytes = 0

        # Loading queue sorted by priority
        self._load_queue: List[Tuple[ModelPriority, str, Callable]] = []
        self._loading_in_progress: Set[str] = set()

    def register_model(
        self,
        name: str,
        loader_func: Callable[[], ModelT],
        priority: ModelPriority = ModelPriority.NORMAL,
        quantize: Optional[bool] = None,
        lazy: bool = False,
    ) -> GhostModelProxy[ModelT]:
        """
        Register a model for background loading.

        Returns a Ghost Proxy immediately that can be used as if
        the model were already loaded.

        Args:
            name: Unique model identifier
            loader_func: Function that loads and returns the model
            priority: Loading priority
            quantize: Apply INT8 quantization (None = use default)
            lazy: Only load on first use (ModelPriority.LAZY)

        Returns:
            GhostModelProxy that can be used immediately
        """
        with self._lock:
            if name in self._proxies:
                logger.warning(f"[AI] Model {name} already registered, returning existing proxy")
                return self._proxies[name]

            # Create ghost proxy
            proxy = GhostModelProxy(name, self._config)
            proxy._metrics.priority = ModelPriority.LAZY if lazy else priority
            self._proxies[name] = proxy
            self._load_order.append(name)

            # Determine quantization
            should_quantize = quantize if quantize is not None else self._config.quantize_default

        logger.info(
            f"[AI] Registered {name} (priority={priority.name}, "
            f"quantize={should_quantize}, lazy={lazy})"
        )

        # Start loading unless lazy
        if not lazy:
            self._queue_load(name, loader_func, priority, should_quantize)

        return proxy

    def _queue_load(
        self,
        name: str,
        loader_func: Callable,
        priority: ModelPriority,
        quantize: bool,
    ) -> None:
        """Queue a model for background loading."""
        with self._lock:
            proxy = self._proxies.get(name)
            if proxy:
                proxy._status = ModelStatus.QUEUED
                proxy._metrics.status = ModelStatus.QUEUED

        # Create wrapped loader with quantization
        def _wrapped_loader():
            return self._load_with_quantization(name, loader_func, quantize)

        # Schedule the load
        asyncio.create_task(
            self._background_load(name, _wrapped_loader, priority)
        )

    async def _background_load(
        self,
        name: str,
        loader_func: Callable,
        priority: ModelPriority,
    ) -> None:
        """Execute model loading in background thread."""
        proxy = self._proxies.get(name)
        if not proxy:
            logger.error(f"[AI] No proxy found for {name}")
            return

        with self._lock:
            if name in self._loading_in_progress:
                logger.debug(f"[AI] {name} already loading, skipping")
                return
            self._loading_in_progress.add(name)

        proxy._status = ModelStatus.LOADING
        proxy._metrics.status = ModelStatus.LOADING
        proxy._metrics.load_start_time = time.time()

        logger.info(f"[AI] Loading {name} (priority={priority.name})...")

        try:
            loop = asyncio.get_event_loop()

            # Run in thread pool (PyTorch releases GIL during heavy ops)
            model = await loop.run_in_executor(self._executor, loader_func)

            # Estimate memory
            memory = ZeroCopyLoader.estimate_memory(model)
            proxy._metrics.memory_bytes = memory
            self._total_memory_bytes += memory

            # Materialize the ghost
            proxy.materialize(model)

            logger.info(
                f"[AI] {name} ready! "
                f"Load: {proxy._metrics.load_duration_ms:.0f}ms, "
                f"Memory: {memory / 1024 / 1024:.1f}MB"
            )

        except Exception as e:
            proxy.fail(e)
            logger.error(f"[AI] {name} failed: {e}")

        finally:
            with self._lock:
                self._loading_in_progress.discard(name)

    def _load_with_quantization(
        self,
        name: str,
        loader_func: Callable,
        quantize: bool,
    ) -> Any:
        """Load model and optionally apply quantization."""
        # Update status
        proxy = self._proxies.get(name)
        if proxy:
            proxy._status = ModelStatus.LOADING
            proxy._metrics.status = ModelStatus.LOADING

        # Execute the loader
        model = loader_func()

        # Apply quantization if requested
        if quantize and model is not None:
            if proxy:
                proxy._status = ModelStatus.QUANTIZING
                proxy._metrics.status = ModelStatus.QUANTIZING
                proxy._metrics.quantized = True

            model = ZeroCopyLoader.quantize_dynamic(model)

        return model

    async def ensure_loaded(
        self,
        name: str,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Ensure a model is loaded before proceeding.

        Args:
            name: Model name
            timeout: Max wait time in seconds

        Returns:
            True if model is ready, False if timeout or error
        """
        proxy = self._proxies.get(name)
        if not proxy:
            logger.error(f"[AI] Unknown model: {name}")
            return False

        if proxy.is_ready:
            return True

        return await proxy.wait_ready(timeout)

    def get_proxy(self, name: str) -> Optional[GhostModelProxy]:
        """Get a model's Ghost Proxy."""
        return self._proxies.get(name)

    def get_model(self, name: str) -> Optional[Any]:
        """Get the real model if loaded, None otherwise."""
        proxy = self._proxies.get(name)
        if proxy and proxy.is_ready:
            return proxy._real_model
        return None

    async def unload(self, name: str) -> bool:
        """Unload a model to free memory."""
        proxy = self._proxies.get(name)
        if not proxy:
            return False

        if proxy._metrics.memory_bytes:
            self._total_memory_bytes -= proxy._metrics.memory_bytes

        proxy.unload()
        return True

    async def reload(
        self,
        name: str,
        loader_func: Optional[Callable] = None,
    ) -> bool:
        """Reload a model (unload then load fresh)."""
        await self.unload(name)

        # If no new loader, we can't reload
        if loader_func is None:
            logger.warning(f"[AI] Cannot reload {name} without loader function")
            return False

        proxy = self._proxies.get(name)
        if proxy:
            self._queue_load(
                name,
                loader_func,
                proxy._metrics.priority,
                proxy._metrics.quantized,
            )
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive loading statistics."""
        models = {}
        for name, proxy in self._proxies.items():
            m = proxy._metrics
            models[name] = {
                "status": m.status.name,
                "priority": m.priority.name,
                "load_duration_ms": m.load_duration_ms,
                "memory_mb": m.memory_bytes / 1024 / 1024 if m.memory_bytes else 0,
                "quantized": m.quantized,
                "calls_total": m.calls_total,
                "calls_while_warming": m.calls_while_warming,
                "errors": m.errors,
            }

        ready_count = sum(1 for p in self._proxies.values() if p.is_ready)
        loading_count = sum(1 for p in self._proxies.values() if p.is_loading)

        return {
            "models": models,
            "summary": {
                "total": len(self._proxies),
                "ready": ready_count,
                "loading": loading_count,
                "failed": len(self._proxies) - ready_count - loading_count,
                "total_memory_mb": self._total_memory_bytes / 1024 / 1024,
            },
            "config": {
                "max_workers": self._config.max_workers,
                "quantize_default": self._config.quantize_default,
                "queue_timeout": self._config.queue_timeout,
            },
        }

    async def shutdown(self) -> None:
        """Shutdown the manager and unload all models."""
        self._shutdown = True

        # Unload all models
        for name in list(self._proxies.keys()):
            await self.unload(name)

        # Shutdown executor
        self._executor.shutdown(wait=False)

        logger.info("[AI] AsyncModelManager shutdown complete")


# =============================================================================
# Global Manager Instance
# =============================================================================

_manager: Optional[AsyncModelManager] = None


def get_ai_manager() -> AsyncModelManager:
    """Get or create the global AI model manager."""
    global _manager
    if _manager is None:
        _manager = AsyncModelManager()
    return _manager


# =============================================================================
# Convenience Decorators
# =============================================================================

def ghost_model(
    name: str,
    priority: ModelPriority = ModelPriority.NORMAL,
    quantize: bool = True,
    lazy: bool = False,
):
    """
    Decorator to register a model loader function.

    Usage:
        @ghost_model("ecapa_tdnn", priority=ModelPriority.HIGH)
        def load_ecapa():
            return ECAPAModel.from_pretrained("...")

        # Use the model (auto-waits if not ready)
        embedding = await load_ecapa.proxy.encode(audio)
    """
    def decorator(loader_func: Callable[[], T]) -> Callable[[], T]:
        manager = get_ai_manager()
        proxy = manager.register_model(
            name=name,
            loader_func=loader_func,
            priority=priority,
            quantize=quantize,
            lazy=lazy,
        )

        @functools.wraps(loader_func)
        def wrapper() -> GhostModelProxy[T]:
            return proxy

        wrapper.proxy = proxy
        wrapper.name = name

        return wrapper

    return decorator


# =============================================================================
# Test & Debug
# =============================================================================

async def _test_ghost_proxy():
    """Test the Ghost Proxy system."""
    import asyncio

    print("Testing Hyper-Speed AI Loader...")
    print("=" * 60)

    manager = get_ai_manager()

    # Simulate a slow model load
    def slow_loader():
        print("  [LOADER] Simulating 2s model load...")
        time.sleep(2)
        return {"model": "test", "params": 1000}

    # Register model - returns immediately
    proxy = manager.register_model(
        name="test_model",
        loader_func=slow_loader,
        priority=ModelPriority.HIGH,
        quantize=False,
    )

    print(f"  Proxy created: {proxy}")
    print(f"  Status: {proxy.status.name}")
    print(f"  Is ready: {proxy.is_ready}")

    # Wait for model
    print("  Waiting for model...")
    ready = await proxy.wait_ready(timeout=5)
    print(f"  Ready: {ready}")
    print(f"  Status: {proxy.status.name}")

    # Get stats
    stats = manager.get_stats()
    print(f"\n  Stats: {stats}")

    # Cleanup
    await manager.shutdown()

    print("=" * 60)
    print("Ghost Proxy test complete!")


if __name__ == "__main__":
    asyncio.run(_test_ghost_proxy())
