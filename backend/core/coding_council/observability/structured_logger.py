"""
v77.0: Structured Logger - Gap #28
===================================

Structured logging with rich context:
- JSON-formatted structured logs
- Contextual fields propagation
- Log levels with filtering
- Async-safe logging
- Log rotation and retention

Author: Ironcliw v77.0
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO, Union

# Context variable for log context
_log_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "log_context",
    default={}
)


class LogLevel(IntEnum):
    """Log levels matching Python logging."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class LogContext:
    """Context for structured logging."""
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    operation: Optional[str] = None
    component: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.trace_id:
            result["trace_id"] = self.trace_id
        if self.span_id:
            result["span_id"] = self.span_id
        if self.operation:
            result["operation"] = self.operation
        if self.component:
            result["component"] = self.component
        if self.user_id:
            result["user_id"] = self.user_id
        if self.session_id:
            result["session_id"] = self.session_id
        if self.request_id:
            result["request_id"] = self.request_id
        result.update(self.extra)
        return result

    def merge(self, other: "LogContext") -> "LogContext":
        """Merge with another context."""
        return LogContext(
            trace_id=other.trace_id or self.trace_id,
            span_id=other.span_id or self.span_id,
            operation=other.operation or self.operation,
            component=other.component or self.component,
            user_id=other.user_id or self.user_id,
            session_id=other.session_id or self.session_id,
            request_id=other.request_id or self.request_id,
            extra={**self.extra, **other.extra},
        )


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logs."""

    def __init__(
        self,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_logger: bool = True,
        include_location: bool = True,
        pretty: bool = False,
    ):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_logger = include_logger
        self.include_location = include_location
        self.pretty = pretty

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {}

        # Timestamp
        if self.include_timestamp:
            log_entry["timestamp"] = datetime.utcfromtimestamp(record.created).isoformat() + "Z"

        # Level
        if self.include_level:
            log_entry["level"] = record.levelname

        # Logger name
        if self.include_logger:
            log_entry["logger"] = record.name

        # Message
        log_entry["message"] = record.getMessage()

        # Location
        if self.include_location:
            log_entry["location"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
            }

        # Context from context var
        ctx = _log_context.get()
        if ctx:
            log_entry["context"] = ctx

        # Extra fields from record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "message", "asctime"
            ):
                extra_fields[key] = value

        if extra_fields:
            log_entry["extra"] = extra_fields

        # Exception info
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info) if all(record.exc_info) else None,
            }

        indent = 2 if self.pretty else None
        return json.dumps(log_entry, default=str, indent=indent)


class StructuredLogger:
    """
    Structured logger with context propagation.

    Features:
    - JSON-formatted logs
    - Context propagation via contextvars
    - Multiple output handlers
    - Log level filtering
    - Async-safe
    """

    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        context: Optional[LogContext] = None,
    ):
        self.name = name
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._context = context or LogContext()
        self._handlers: List[logging.Handler] = []

    def add_console_handler(
        self,
        level: LogLevel = LogLevel.DEBUG,
        stream: TextIO = sys.stderr,
        pretty: bool = False,
    ) -> None:
        """Add console output handler."""
        handler = logging.StreamHandler(stream)
        handler.setLevel(level)
        handler.setFormatter(StructuredFormatter(pretty=pretty))
        self._logger.addHandler(handler)
        self._handlers.append(handler)

    def add_file_handler(
        self,
        filepath: Union[str, Path],
        level: LogLevel = LogLevel.DEBUG,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
    ) -> None:
        """Add rotating file handler."""
        from logging.handlers import RotatingFileHandler

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        handler = RotatingFileHandler(
            str(path),
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        handler.setLevel(level)
        handler.setFormatter(StructuredFormatter())
        self._logger.addHandler(handler)
        self._handlers.append(handler)

    def with_context(self, **kwargs) -> "StructuredLogger":
        """Create a new logger with additional context."""
        new_context = LogContext(**kwargs)
        merged = self._context.merge(new_context)

        child = StructuredLogger(
            name=self.name,
            level=self._logger.level,
            context=merged,
        )
        child._logger = self._logger  # Share handlers
        return child

    def _log(
        self,
        level: int,
        msg: str,
        *args,
        exc_info: bool = False,
        **kwargs,
    ) -> None:
        """Internal log method."""
        # Merge context into context var
        ctx = _log_context.get().copy()
        ctx.update(self._context.to_dict())
        ctx.update(kwargs)

        token = _log_context.set(ctx)
        try:
            self._logger.log(level, msg, *args, exc_info=exc_info)
        finally:
            _log_context.reset(token)

    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log debug message."""
        self._log(LogLevel.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        """Log info message."""
        self._log(LogLevel.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log warning message."""
        self._log(LogLevel.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args, exc_info: bool = True, **kwargs) -> None:
        """Log error message."""
        self._log(LogLevel.ERROR, msg, *args, exc_info=exc_info, **kwargs)

    def critical(self, msg: str, *args, exc_info: bool = True, **kwargs) -> None:
        """Log critical message."""
        self._log(LogLevel.CRITICAL, msg, *args, exc_info=exc_info, **kwargs)

    def exception(self, msg: str, *args, **kwargs) -> None:
        """Log exception with traceback."""
        self._log(LogLevel.ERROR, msg, *args, exc_info=True, **kwargs)

    def bind(self, **kwargs) -> "StructuredLogger":
        """Bind additional context (alias for with_context)."""
        return self.with_context(**kwargs)


def set_log_context(**kwargs) -> contextvars.Token:
    """
    Set log context for current async context.

    Usage:
        token = set_log_context(request_id="123")
        try:
            # logs will include request_id
            ...
        finally:
            reset_log_context(token)
    """
    ctx = _log_context.get().copy()
    ctx.update(kwargs)
    return _log_context.set(ctx)


def reset_log_context(token: contextvars.Token) -> None:
    """Reset log context to previous state."""
    _log_context.reset(token)


def get_log_context() -> Dict[str, Any]:
    """Get current log context."""
    return _log_context.get().copy()


# Global loggers registry
_loggers: Dict[str, StructuredLogger] = {}


def get_logger(
    name: str,
    level: Optional[LogLevel] = None,
    add_console: bool = True,
) -> StructuredLogger:
    """
    Get or create a structured logger.

    Usage:
        logger = get_logger(__name__)
        logger.info("Hello", user_id="123")
    """
    if name not in _loggers:
        logger = StructuredLogger(name, level or LogLevel.INFO)
        if add_console:
            logger.add_console_handler()
        _loggers[name] = logger

    return _loggers[name]


def configure_root_logger(
    level: LogLevel = LogLevel.INFO,
    log_file: Optional[Union[str, Path]] = None,
    console: bool = True,
    pretty: bool = False,
) -> StructuredLogger:
    """
    Configure the root logger for the application.

    Usage:
        configure_root_logger(
            level=LogLevel.DEBUG,
            log_file="~/.jarvis/logs/coding_council.log",
        )
    """
    logger = StructuredLogger("coding_council", level)

    if console:
        logger.add_console_handler(pretty=pretty)

    if log_file:
        logger.add_file_handler(Path(log_file).expanduser())

    _loggers["root"] = logger
    _loggers["coding_council"] = logger

    return logger
