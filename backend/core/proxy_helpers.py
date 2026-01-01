"""
JARVIS Proxy Helpers v1.0
=========================

Utilities for safely accessing Ghost Model Proxies throughout the application.

These helpers provide:
- Safe proxy retrieval with graceful degradation
- Warming state detection and handling
- HTTP-friendly error responses for FastAPI routes
- Async-aware waiting with configurable timeouts

Usage in FastAPI routes:
    from core.proxy_helpers import get_model_proxy, require_model_ready

    @router.post("/analyze")
    async def analyze_image(request: Request):
        # Option 1: Get proxy and handle warming state
        vision = get_model_proxy(request.app.state, "vision_analyzer")
        if not vision.is_ready:
            raise HTTPException(503, "Vision system warming up...")

        # Option 2: Require model to be ready (waits or raises)
        vision = await require_model_ready(request.app.state, "vision_analyzer")
        return await vision.analyze(image)

Version: 1.0.0 - Ghost Proxy Utilities
"""

import asyncio
import logging
from typing import Any, Optional, TypeVar, Union
from enum import Enum

logger = logging.getLogger("jarvis.proxy_helpers")

# Type variable for generic proxy typing
T = TypeVar("T")


class ProxyState(Enum):
    """States a Ghost Proxy can be in."""
    NOT_REGISTERED = "not_registered"
    GHOST = "ghost"
    QUEUED = "queued"
    LOADING = "loading"
    QUANTIZING = "quantizing"
    READY = "ready"
    FAILED = "failed"
    UNLOADED = "unloaded"


class ProxyNotReadyError(Exception):
    """Raised when a proxy is accessed but the model isn't ready."""

    def __init__(
        self,
        model_name: str,
        state: ProxyState,
        message: Optional[str] = None,
        retry_after: Optional[float] = None,
    ):
        self.model_name = model_name
        self.state = state
        self.retry_after = retry_after
        self.message = message or f"Model '{model_name}' is {state.value}"
        super().__init__(self.message)


class ProxyNotFoundError(Exception):
    """Raised when a proxy is not registered."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__(f"Model '{model_name}' is not registered")


def get_model_proxy(
    app_state: Any,
    model_name: str,
    raise_if_missing: bool = True,
) -> Optional[Any]:
    """
    Safely retrieve a Ghost Proxy from app state.

    Args:
        app_state: FastAPI app.state object
        model_name: Name of the model/proxy to retrieve
        raise_if_missing: If True, raises ProxyNotFoundError if not found

    Returns:
        The Ghost Proxy if found, None if not found and raise_if_missing=False

    Raises:
        ProxyNotFoundError: If model not registered and raise_if_missing=True

    Example:
        vision = get_model_proxy(request.app.state, "vision_analyzer")
        if vision.is_ready:
            result = vision.analyze(image)
    """
    proxy = getattr(app_state, model_name, None)

    if proxy is None:
        # Check if it's in the voice_model_proxies dict
        voice_proxies = getattr(app_state, 'voice_model_proxies', {})
        proxy = voice_proxies.get(model_name)

    if proxy is None:
        # Check if it's in the vision_model_proxies dict
        vision_proxies = getattr(app_state, 'vision_model_proxies', {})
        proxy = vision_proxies.get(model_name)

    if proxy is None:
        # Check if it's in the intelligence_proxies dict
        intel_proxies = getattr(app_state, 'intelligence_proxies', {})
        proxy = intel_proxies.get(model_name)

    if proxy is None and raise_if_missing:
        raise ProxyNotFoundError(model_name)

    return proxy


def get_proxy_state(proxy: Any) -> ProxyState:
    """
    Get the current state of a Ghost Proxy.

    Args:
        proxy: A GhostModelProxy instance

    Returns:
        ProxyState enum value
    """
    if proxy is None:
        return ProxyState.NOT_REGISTERED

    # Check if it's a GhostModelProxy
    if hasattr(proxy, 'status'):
        status = proxy.status
        # Map ModelStatus to ProxyState
        status_name = status.name if hasattr(status, 'name') else str(status)
        try:
            return ProxyState[status_name.upper()]
        except KeyError:
            return ProxyState.GHOST

    # If it's a real model (not a proxy), it's ready
    return ProxyState.READY


def is_proxy_ready(proxy: Any) -> bool:
    """
    Check if a Ghost Proxy is ready for use.

    Args:
        proxy: A GhostModelProxy instance or real model

    Returns:
        True if ready, False otherwise
    """
    if proxy is None:
        return False

    # Check for GhostModelProxy
    if hasattr(proxy, 'is_ready'):
        return proxy.is_ready

    # If it's a real model (not a proxy), it's ready
    return True


def is_proxy_loading(proxy: Any) -> bool:
    """
    Check if a Ghost Proxy is currently loading.

    Args:
        proxy: A GhostModelProxy instance

    Returns:
        True if loading/queued/quantizing, False otherwise
    """
    if proxy is None:
        return False

    if hasattr(proxy, 'is_loading'):
        return proxy.is_loading

    return False


async def wait_for_proxy(
    proxy: Any,
    timeout: Optional[float] = None,
    poll_interval: float = 0.1,
) -> bool:
    """
    Wait for a Ghost Proxy to become ready.

    Args:
        proxy: A GhostModelProxy instance
        timeout: Max seconds to wait (None = use proxy's default)
        poll_interval: Seconds between ready checks

    Returns:
        True if proxy became ready, False if timeout
    """
    if proxy is None:
        return False

    if hasattr(proxy, 'wait_ready'):
        return await proxy.wait_ready(timeout=timeout)

    # If it's already a real model, it's ready
    return True


async def require_model_ready(
    app_state: Any,
    model_name: str,
    timeout: Optional[float] = 30.0,
    raise_on_timeout: bool = True,
) -> Any:
    """
    Get a model proxy and wait for it to be ready.

    This is the recommended way to access models in FastAPI routes
    when you need the model to be fully loaded.

    Args:
        app_state: FastAPI app.state object
        model_name: Name of the model/proxy to retrieve
        timeout: Max seconds to wait for model to be ready
        raise_on_timeout: If True, raises ProxyNotReadyError on timeout

    Returns:
        The ready model/proxy

    Raises:
        ProxyNotFoundError: If model not registered
        ProxyNotReadyError: If model not ready after timeout

    Example:
        @router.post("/transcribe")
        async def transcribe_audio(request: Request, audio: UploadFile):
            whisper = await require_model_ready(request.app.state, "whisper_model")
            return await whisper.transcribe(audio)
    """
    proxy = get_model_proxy(app_state, model_name)

    if is_proxy_ready(proxy):
        return proxy

    # Wait for the proxy to become ready
    ready = await wait_for_proxy(proxy, timeout=timeout)

    if not ready and raise_on_timeout:
        state = get_proxy_state(proxy)
        raise ProxyNotReadyError(
            model_name=model_name,
            state=state,
            message=f"Model '{model_name}' not ready after {timeout}s (state: {state.value})",
            retry_after=5.0,  # Suggest client retry after 5s
        )

    return proxy


def get_all_proxy_stats(app_state: Any) -> dict:
    """
    Get status of all registered proxies.

    Args:
        app_state: FastAPI app.state object

    Returns:
        Dict with proxy names and their states
    """
    stats = {}

    # Check voice proxies
    voice_proxies = getattr(app_state, 'voice_model_proxies', {})
    for name, proxy in voice_proxies.items():
        stats[name] = {
            "category": "voice",
            "state": get_proxy_state(proxy).value,
            "ready": is_proxy_ready(proxy),
            "loading": is_proxy_loading(proxy),
        }

    # Check vision proxies
    vision_proxies = getattr(app_state, 'vision_model_proxies', {})
    for name, proxy in vision_proxies.items():
        stats[name] = {
            "category": "vision",
            "state": get_proxy_state(proxy).value,
            "ready": is_proxy_ready(proxy),
            "loading": is_proxy_loading(proxy),
        }

    # Check intelligence proxies
    intel_proxies = getattr(app_state, 'intelligence_proxies', {})
    for name, proxy in intel_proxies.items():
        stats[name] = {
            "category": "intelligence",
            "state": get_proxy_state(proxy).value,
            "ready": is_proxy_ready(proxy),
            "loading": is_proxy_loading(proxy),
        }

    # Check individual app.state attributes
    known_models = [
        ("ecapa_model", "voice"),
        ("voice_unlock_classifier", "voice"),
        ("vad_model", "voice"),
        ("vision_analyzer", "vision"),
        ("display_monitor", "vision"),
        ("neural_mesh", "intelligence"),
        ("hybrid_orchestrator", "intelligence"),
    ]

    for model_name, category in known_models:
        if model_name not in stats:
            proxy = getattr(app_state, model_name, None)
            if proxy is not None:
                stats[model_name] = {
                    "category": category,
                    "state": get_proxy_state(proxy).value,
                    "ready": is_proxy_ready(proxy),
                    "loading": is_proxy_loading(proxy),
                }

    return stats


# FastAPI integration helpers

def create_warming_response(model_name: str, retry_after: float = 5.0) -> dict:
    """
    Create a standard "warming up" response for API endpoints.

    Args:
        model_name: Name of the model that's warming up
        retry_after: Suggested retry delay in seconds

    Returns:
        Dict suitable for FastAPI JSONResponse
    """
    return {
        "status": "warming_up",
        "message": f"The {model_name} is initializing. Please retry shortly.",
        "model": model_name,
        "retry_after": retry_after,
    }


def http_exception_for_proxy_error(error: Union[ProxyNotReadyError, ProxyNotFoundError]):
    """
    Convert a proxy error to an appropriate HTTPException.

    Args:
        error: ProxyNotReadyError or ProxyNotFoundError

    Returns:
        HTTPException with appropriate status code

    Usage:
        try:
            model = await require_model_ready(app.state, "vision")
        except (ProxyNotReadyError, ProxyNotFoundError) as e:
            raise http_exception_for_proxy_error(e)
    """
    # Import here to avoid circular imports
    from fastapi import HTTPException

    if isinstance(error, ProxyNotFoundError):
        return HTTPException(
            status_code=500,
            detail={
                "error": "model_not_registered",
                "model": error.model_name,
                "message": str(error),
            }
        )

    if isinstance(error, ProxyNotReadyError):
        headers = {}
        if error.retry_after:
            headers["Retry-After"] = str(int(error.retry_after))

        return HTTPException(
            status_code=503,
            detail={
                "error": "model_warming_up",
                "model": error.model_name,
                "state": error.state.value,
                "message": str(error),
                "retry_after": error.retry_after,
            },
            headers=headers,
        )

    return HTTPException(status_code=500, detail=str(error))


# Decorator for routes that require a model

def requires_model(model_name: str, timeout: float = 30.0):
    """
    Decorator that ensures a model is ready before route execution.

    Args:
        model_name: Name of the required model
        timeout: Max seconds to wait for model

    Usage:
        @router.post("/vision/analyze")
        @requires_model("vision_analyzer")
        async def analyze(request: Request):
            vision = request.app.state.vision_analyzer
            return await vision.analyze(...)
    """
    def decorator(func):
        import functools

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Find the request object
            request = None
            for arg in args:
                if hasattr(arg, 'app') and hasattr(arg.app, 'state'):
                    request = arg
                    break

            if request is None:
                for key, val in kwargs.items():
                    if hasattr(val, 'app') and hasattr(val.app, 'state'):
                        request = val
                        break

            if request is None:
                raise RuntimeError("Could not find Request object in route arguments")

            try:
                await require_model_ready(request.app.state, model_name, timeout=timeout)
            except (ProxyNotReadyError, ProxyNotFoundError) as e:
                raise http_exception_for_proxy_error(e)

            return await func(*args, **kwargs)

        return wrapper
    return decorator
