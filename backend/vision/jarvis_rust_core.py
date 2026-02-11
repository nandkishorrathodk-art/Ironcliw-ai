"""
JARVIS Rust Core Python integration.

This module resolves the real Rust extension first, and only loads the local
Python stub when no extension is importable. Stub mode is explicit and
observable via `RUST_AVAILABLE`/`RUST_REASON`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import importlib
import logging
import sys

logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
_STUB_DIR = _THIS_DIR / "jarvis-rust-core"
_STUB_FILE = (_STUB_DIR / "jarvis_rust_core.py").resolve()
_REQUIRED_SYMBOLS = ("RustRuntimeManager", "RustAdvancedMemoryPool", "RustImageProcessor")
_OPTIONAL_SYMBOLS = ("RustMemoryPool", "quantize_model_weights")


def _module_origin(module: Any) -> str:
    origin = getattr(module, "__file__", None)
    if origin is None:
        return "<builtin>"
    try:
        return str(Path(origin).resolve())
    except Exception:
        return str(origin)


def _is_stub_module(module: Any) -> bool:
    if module is None:
        return False
    if getattr(module, "__rust_available__", None) is False:
        return True
    origin = getattr(module, "__file__", None)
    if origin is None:
        return False
    try:
        return Path(origin).resolve() == _STUB_FILE
    except Exception:
        return False


def _strip_stub_path() -> list[Tuple[int, str]]:
    removed: list[Tuple[int, str]] = []
    stub_path = str(_STUB_DIR.resolve())
    for idx in range(len(sys.path) - 1, -1, -1):
        candidate = sys.path[idx]
        try:
            resolved = str(Path(candidate).resolve())
        except Exception:
            resolved = candidate
        if resolved == stub_path:
            removed.append((idx, sys.path.pop(idx)))
    removed.reverse()
    return removed


def _restore_paths(entries: list[Tuple[int, str]]) -> None:
    for idx, value in entries:
        if idx < 0 or idx > len(sys.path):
            sys.path.append(value)
        else:
            sys.path.insert(idx, value)


def _required_symbols_available(module: Any) -> bool:
    return all(hasattr(module, symbol) for symbol in _REQUIRED_SYMBOLS)


def _compute_capabilities(module: Any) -> Dict[str, bool]:
    symbols = list(_REQUIRED_SYMBOLS) + list(_OPTIONAL_SYMBOLS)
    return {symbol: hasattr(module, symbol) for symbol in symbols}


def _load_real_module() -> Tuple[Optional[Any], Optional[str]]:
    existing = sys.modules.get("jarvis_rust_core")
    existing_was_stub = _is_stub_module(existing)
    if existing is not None and not existing_was_stub:
        return existing, None

    if existing_was_stub:
        sys.modules.pop("jarvis_rust_core", None)

    removed_entries = _strip_stub_path()
    try:
        module = importlib.import_module("jarvis_rust_core")
        if _is_stub_module(module):
            return None, f"Loaded stub module from {_module_origin(module)}"
        return module, None
    except ImportError as exc:
        return None, str(exc)
    finally:
        _restore_paths(removed_entries)
        if "jarvis_rust_core" not in sys.modules and existing is not None:
            sys.modules["jarvis_rust_core"] = existing


def _load_stub_module() -> Tuple[Optional[Any], Optional[str]]:
    if not _STUB_FILE.exists():
        return None, "Stub file not found"

    inserted = False
    stub_path = str(_STUB_DIR)
    if stub_path not in sys.path:
        sys.path.insert(0, stub_path)
        inserted = True

    try:
        module = importlib.import_module("jarvis_rust_core")
        return module, None
    except ImportError as exc:
        return None, str(exc)
    finally:
        if inserted and sys.path and sys.path[0] == stub_path:
            sys.path.pop(0)


def _load_module() -> Tuple[Optional[Any], bool, str, Dict[str, bool]]:
    module, error = _load_real_module()
    if module is not None:
        capabilities = _compute_capabilities(module)
        if _required_symbols_available(module):
            logger.info("Rust core loaded from %s", _module_origin(module))
            return module, True, "loaded_real_module", capabilities
        missing = [symbol for symbol, present in capabilities.items() if symbol in _REQUIRED_SYMBOLS and not present]
        reason = f"real module missing required symbols: {', '.join(missing)}"
        logger.warning("Rust module loaded but incompatible: %s", reason)
        return module, False, reason, capabilities

    stub_module, stub_error = _load_stub_module()
    if stub_module is not None:
        reason = "loaded_python_stub"
        logger.warning("Rust extension unavailable (%s); using Python stub from %s", error or "unknown error", _module_origin(stub_module))
        return stub_module, False, reason, _compute_capabilities(stub_module)

    reason = f"rust_unavailable: {error or stub_error or 'unknown import error'}"
    logger.warning("Rust core not available: %s", reason)
    return None, False, reason, {}


jrc, RUST_AVAILABLE, RUST_REASON, RUST_CAPABILITIES = _load_module()
RUST_MODULE_ORIGIN = _module_origin(jrc) if jrc is not None else "unloaded"

# Global references to Rust components
_runtime_manager = None
_memory_pool = None
_advanced_pool = None
_image_processor = None


def initialize_rust_runtime(config: Dict[str, Any]) -> None:
    """
    Initialize Rust runtime components that are present in the loaded module.
    """
    global _runtime_manager, _memory_pool, _advanced_pool, _image_processor

    _runtime_manager = None
    _memory_pool = None
    _advanced_pool = None
    _image_processor = None

    if not RUST_AVAILABLE or jrc is None:
        logger.warning("Rust runtime initialization skipped (%s)", RUST_REASON)
        return

    initialized_components = []

    try:
        if hasattr(jrc, "RustRuntimeManager"):
            _runtime_manager = jrc.RustRuntimeManager(
                worker_threads=config.get("worker_threads", 4),
                enable_cpu_affinity=config.get("enable_cpu_affinity", True),
            )
            initialized_components.append("runtime_manager")

        if hasattr(jrc, "RustMemoryPool"):
            _memory_pool = jrc.RustMemoryPool()
            initialized_components.append("memory_pool")

        if hasattr(jrc, "RustAdvancedMemoryPool"):
            _advanced_pool = jrc.RustAdvancedMemoryPool()
            initialized_components.append("advanced_memory_pool")
            if _memory_pool is None:
                _memory_pool = _advanced_pool

        if hasattr(jrc, "RustImageProcessor"):
            _image_processor = jrc.RustImageProcessor()
            initialized_components.append("image_processor")

        if not initialized_components:
            logger.warning("Rust module loaded but no expected runtime components were initialized")
            return

        logger.info("Rust runtime initialized: %s", ", ".join(initialized_components))

        if _runtime_manager is not None and hasattr(_runtime_manager, "stats"):
            try:
                logger.info("Rust runtime stats: %s", _runtime_manager.stats())
            except Exception as exc:
                logger.debug("Rust runtime stats unavailable: %s", exc)

    except Exception as exc:
        logger.error("Failed to initialize Rust runtime components: %s", exc)
        raise


def get_rust_runtime():
    """Get the global Rust runtime manager."""
    return _runtime_manager


def get_rust_memory_pool():
    """Get the global Rust memory pool."""
    return _memory_pool


def get_rust_advanced_pool():
    """Get the global advanced memory pool with leak detection."""
    return _advanced_pool


def get_rust_image_processor():
    """Get the global Rust image processor."""
    return _image_processor


def is_rust_available() -> bool:
    """Check whether Rust module is loaded and at least one component is initialized."""
    return RUST_AVAILABLE and any(
        component is not None
        for component in (_runtime_manager, _memory_pool, _advanced_pool, _image_processor)
    )


def process_image_with_rust(image_array):
    """
    Process an image with Rust acceleration when available.
    """
    if _image_processor is not None and hasattr(_image_processor, "process_numpy_image"):
        try:
            return _image_processor.process_numpy_image(image_array)
        except Exception as exc:
            logger.warning("Rust image processing failed: %s", exc)

    return image_array


def allocate_rust_buffer(size: int):
    """
    Allocate a buffer using Rust memory pool if available.
    """
    if _memory_pool is not None and hasattr(_memory_pool, "allocate"):
        try:
            return _memory_pool.allocate(size)
        except Exception as exc:
            logger.warning("Rust buffer allocation failed: %s", exc)

    return bytearray(size)


def quantize_weights_with_rust(weights):
    """
    Quantize model weights using Rust if available.
    """
    if (
        not RUST_AVAILABLE
        or jrc is None
        or not RUST_CAPABILITIES.get("quantize_model_weights", False)
    ):
        try:
            return weights.astype("int8")
        except Exception:
            return weights

    try:
        return jrc.quantize_model_weights(weights)
    except Exception as exc:
        logger.warning("Rust quantization failed: %s", exc)
        try:
            return weights.astype("int8")
        except Exception:
            return weights
