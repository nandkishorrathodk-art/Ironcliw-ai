"""
Ironcliw Backend Utilities
========================

This package provides utility modules for the Ironcliw backend, including:
- Environment configuration utilities for type-safe env var parsing
- Async I/O utilities for non-blocking operations (generic wrappers)
- Async startup utilities for non-blocking startup operations
- Async lock wrappers for cross-repo locking
- Audio processing utilities
- Model loading and caching utilities
- Various helper functions

Usage:
    from backend.utils import (
        # Environment configuration utilities
        get_env_str,
        get_env_int,
        get_env_bool,
        EnvConfig,
        # Generic async I/O utilities
        run_sync,
        path_exists,
        read_file,
        run_subprocess,
        # Startup-specific utilities
        async_process_wait,
        async_subprocess_run,
        async_check_port,
        StartupFileLock,
        CrossRepoLockManager,
    )
"""

# Environment configuration utilities - type-safe env var parsing
from backend.utils.env_config import (
    get_env_str,
    get_env_optional_str,
    get_env_int,
    get_env_float,
    get_env_bool,
    get_env_list,
    EnvConfig,
)

# Generic async I/O utilities - type-safe wrappers for blocking operations
from backend.utils.async_io import (
    run_sync,
    path_exists,
    read_file,
    run_subprocess,
)

# Async startup utilities - run blocking ops without blocking the event loop
from backend.utils.async_startup import (
    # Process wait
    async_process_wait,
    async_psutil_wait,
    # Subprocess
    async_subprocess_run,
    SubprocessResult,
    # Socket checks
    async_check_port,
    async_check_unix_socket,
    # File I/O
    async_file_read,
    async_file_write,
    async_json_read,
    async_json_write,
    # Executor management
    shutdown_startup_executor,
    # Dedicated executor (for testing/advanced use)
    _STARTUP_EXECUTOR,
)

# Async lock wrapper - non-blocking file locks with stale detection
from backend.utils.async_lock_wrapper import (
    StartupFileLock,
    CrossRepoLockManager,
    MAX_LOCK_TIMEOUT,
    MIN_LOCK_TIMEOUT,
    DEFAULT_LOCK_TIMEOUT,
    STALE_LOCK_RETRY_TIMEOUT,
)


__all__ = [
    # === Environment Configuration Utilities ===
    "get_env_str",
    "get_env_optional_str",
    "get_env_int",
    "get_env_float",
    "get_env_bool",
    "get_env_list",
    "EnvConfig",
    # === Generic Async I/O Utilities ===
    "run_sync",
    "path_exists",
    "read_file",
    "run_subprocess",
    # === Async Startup Utilities ===
    # Process wait
    "async_process_wait",
    "async_psutil_wait",
    # Subprocess
    "async_subprocess_run",
    "SubprocessResult",
    # Socket checks
    "async_check_port",
    "async_check_unix_socket",
    # File I/O
    "async_file_read",
    "async_file_write",
    "async_json_read",
    "async_json_write",
    # Executor management
    "shutdown_startup_executor",
    "_STARTUP_EXECUTOR",
    # === Async Lock Wrapper ===
    "StartupFileLock",
    "CrossRepoLockManager",
    "MAX_LOCK_TIMEOUT",
    "MIN_LOCK_TIMEOUT",
    "DEFAULT_LOCK_TIMEOUT",
    "STALE_LOCK_RETRY_TIMEOUT",
]
