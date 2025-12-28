"""
JARVIS Real-Time Log Monitor v1.0
==================================

Intelligent real-time log monitoring with voice narration integration.

Features:
- Real-time log pattern detection (watches logs as they're written)
- Intelligent error aggregation and severity classification
- Automatic voice narration of critical issues
- Smart throttling (prevents voice spam)
- Proactive health monitoring and diagnostics
- Performance anomaly detection
- Trend analysis (error rate increasing/decreasing)
- Integration with startup narrator and runtime voice system

Architecture:
    LogMonitor (background task)
      ├─> Watches log files via inotify/polling
      ├─> Analyzes patterns in real-time
      ├─> Classifies severity (CRITICAL, HIGH, MEDIUM, LOW)
      ├─> Smart throttling (prevents spam)
      └─> Triggers voice narrator for important events

Usage:
    from backend.core.logging.realtime_log_monitor import (
        get_log_monitor,
        LogMonitorConfig,
    )

    # Start monitoring
    monitor = await get_log_monitor(narrator=voice_narrator)
    await monitor.start()

    # Monitor will automatically:
    # - Watch all log files in ~/.jarvis/logs/
    # - Detect error patterns
    # - Announce critical issues via voice
    # - Provide health summaries

    # Get current status
    health = await monitor.get_health_status()
    print(health)

    # Stop monitoring
    await monitor.stop()
"""

import asyncio
import json
import os
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Set
import time
import logging

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LogMonitorConfig:
    """Configuration for real-time log monitor."""

    # Log directory to monitor
    log_dir: Path = field(default_factory=lambda: Path.home() / ".jarvis" / "logs")

    # Polling interval for log changes (seconds)
    poll_interval: float = 2.0

    # Error thresholds for voice alerts
    critical_error_threshold: int = 3  # Alert if >3 same errors in window
    high_error_threshold: int = 5      # Alert if >5 same errors in window
    error_rate_threshold: float = 10.0  # Alert if >10 errors/minute

    # Time windows
    error_window_minutes: int = 5
    performance_window_minutes: int = 10

    # Voice narration throttling
    min_seconds_between_alerts: float = 30.0  # Don't spam voice alerts
    critical_always_announces: bool = True    # Critical errors always announced

    # Performance thresholds
    slow_operation_threshold_ms: float = 1000.0
    very_slow_operation_threshold_ms: float = 5000.0

    # Health check
    enable_health_monitoring: bool = True
    health_check_interval_seconds: float = 60.0

    @staticmethod
    def from_env() -> "LogMonitorConfig":
        """Load configuration from environment variables."""
        config = LogMonitorConfig()

        if log_dir := os.getenv("JARVIS_LOG_DIR"):
            config.log_dir = Path(log_dir)

        if poll_interval := os.getenv("JARVIS_LOG_MONITOR_POLL_INTERVAL"):
            config.poll_interval = float(poll_interval)

        if threshold := os.getenv("JARVIS_LOG_MONITOR_CRITICAL_THRESHOLD"):
            config.critical_error_threshold = int(threshold)

        config.enable_health_monitoring = os.getenv(
            "JARVIS_LOG_MONITOR_HEALTH", "true"
        ).lower() == "true"

        return config


# =============================================================================
# SEVERITY CLASSIFICATION
# =============================================================================

class Severity:
    """Log issue severity levels."""
    CRITICAL = "CRITICAL"  # System-threatening, immediate attention
    HIGH = "HIGH"          # Significant issues, user should know
    MEDIUM = "MEDIUM"      # Notable but not urgent
    LOW = "LOW"            # Informational


@dataclass
class LogIssue:
    """Represents a detected log issue."""
    severity: str
    category: str  # e.g., "error_storm", "performance_degradation", "component_failure"
    message: str
    count: int
    first_seen: datetime
    last_seen: datetime
    affected_modules: Set[str] = field(default_factory=set)
    sample_log: Optional[Dict[str, Any]] = None
    should_announce: bool = True


# =============================================================================
# PATTERN DETECTOR
# =============================================================================

class PatternDetector:
    """
    Detects patterns in log stream.

    Patterns detected:
    - Error storms (many errors in short time)
    - Repeated failures (same error multiple times)
    - Component failures (specific module consistently failing)
    - Performance degradation (operations getting slower)
    - Startup failures (errors during initialization)
    """

    def __init__(self, config: LogMonitorConfig):
        self.config = config

        # Error tracking
        self.errors: Deque[Dict[str, Any]] = deque(maxlen=1000)
        self.error_signatures: Dict[str, List[datetime]] = defaultdict(list)

        # Performance tracking
        self.operation_times: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=100)
        )

        # Component health
        self.component_errors: Dict[str, int] = Counter()
        self.component_last_success: Dict[str, datetime] = {}

    def analyze_log_entry(self, entry: Dict[str, Any]) -> Optional[LogIssue]:
        """
        Analyze a single log entry for patterns.

        Returns:
            LogIssue if pattern detected, None otherwise
        """
        now = datetime.now()
        level = entry.get("level", "INFO")

        # Track errors
        if level in ("ERROR", "CRITICAL"):
            self.errors.append(entry)
            module = entry.get("module", "unknown")

            # Create error signature
            exception = entry.get("exception", {})
            error_type = exception.get("type", "LoggedError")
            error_msg = exception.get("message", entry.get("message", ""))[:100]
            signature = f"{error_type}:{error_msg}"

            # Track signature occurrences
            self.error_signatures[signature].append(now)

            # Track component health
            self.component_errors[module] += 1

            # Clean old signatures (older than window)
            cutoff = now - timedelta(minutes=self.config.error_window_minutes)
            self.error_signatures[signature] = [
                t for t in self.error_signatures[signature] if t > cutoff
            ]

            # Check for repeated error pattern
            recent_count = len(self.error_signatures[signature])

            if recent_count >= self.config.critical_error_threshold:
                return LogIssue(
                    severity=Severity.CRITICAL if level == "CRITICAL" else Severity.HIGH,
                    category="repeated_error",
                    message=f"{error_type} occurred {recent_count} times in {self.config.error_window_minutes} minutes: {error_msg}",
                    count=recent_count,
                    first_seen=min(self.error_signatures[signature]),
                    last_seen=max(self.error_signatures[signature]),
                    affected_modules={module},
                    sample_log=entry,
                )

        # Track performance
        context = entry.get("context", {})
        if "duration_ms" in context and "operation" in context:
            operation = context["operation"]
            duration = context["duration_ms"]

            self.operation_times[operation].append(duration)

            # Check for very slow operation
            if duration > self.config.very_slow_operation_threshold_ms:
                return LogIssue(
                    severity=Severity.HIGH,
                    category="very_slow_operation",
                    message=f"Operation '{operation}' took {duration:.0f}ms (threshold: {self.config.very_slow_operation_threshold_ms:.0f}ms)",
                    count=1,
                    first_seen=now,
                    last_seen=now,
                    affected_modules={entry.get("module", "unknown")},
                    sample_log=entry,
                )

            # Check for performance degradation (trending slower)
            if len(self.operation_times[operation]) >= 20:
                times = list(self.operation_times[operation])
                recent_avg = sum(times[-5:]) / 5
                baseline_avg = sum(times[:5]) / 5

                # If recent average is 2x baseline, alert
                if recent_avg > baseline_avg * 2 and recent_avg > self.config.slow_operation_threshold_ms:
                    return LogIssue(
                        severity=Severity.MEDIUM,
                        category="performance_degradation",
                        message=f"Operation '{operation}' is getting slower: {baseline_avg:.0f}ms → {recent_avg:.0f}ms",
                        count=1,
                        first_seen=now,
                        last_seen=now,
                        affected_modules={entry.get("module", "unknown")},
                        sample_log=entry,
                    )

        # Track successful operations
        if level in ("INFO", "DEBUG") and "success" in entry.get("message", "").lower():
            module = entry.get("module", "unknown")
            self.component_last_success[module] = now

        return None

    def get_error_rate(self) -> float:
        """Get current error rate (errors per minute)."""
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)

        recent_errors = sum(
            1 for e in self.errors
            if datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00")) > cutoff
        )

        return recent_errors  # Already per minute

    def get_component_health(self) -> Dict[str, str]:
        """Get health status of each component."""
        now = datetime.now()
        health = {}

        for module, error_count in self.component_errors.items():
            last_success = self.component_last_success.get(module)

            if error_count >= 10:
                health[module] = "CRITICAL"
            elif error_count >= 5:
                health[module] = "DEGRADED"
            elif last_success and (now - last_success) < timedelta(minutes=5):
                health[module] = "HEALTHY"
            else:
                health[module] = "UNKNOWN"

        return health


# =============================================================================
# REAL-TIME LOG MONITOR
# =============================================================================

class RealTimeLogMonitor:
    """
    Real-time log monitor with intelligent pattern detection and voice narration.

    Features:
    - Watches log files for new entries
    - Detects error patterns and performance issues
    - Announces critical issues via voice narrator
    - Smart throttling to prevent spam
    - Health monitoring and diagnostics
    """

    def __init__(
        self,
        config: LogMonitorConfig,
        narrator: Optional[Callable] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config
        self.narrator = narrator  # Voice narrator callback
        self.logger = logger or logging.getLogger(__name__)

        # Pattern detection
        self.detector = PatternDetector(config)

        # File tracking
        self.file_positions: Dict[Path, int] = {}

        # Voice throttling
        self.last_announcement_time: Optional[datetime] = None
        self.announced_signatures: Set[str] = set()

        # Monitoring control
        self.running = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.health_check_task: Optional[asyncio.Task] = None

        # Statistics
        self.stats = {
            "total_logs_analyzed": 0,
            "issues_detected": 0,
            "voice_announcements": 0,
            "start_time": None,
        }

    async def start(self) -> None:
        """Start real-time monitoring."""
        if self.running:
            self.logger.warning("[LogMonitor] Already running")
            return

        self.running = True
        self.stats["start_time"] = datetime.now()

        # Initialize file positions (start at end of existing files)
        log_files = list(self.config.log_dir.glob("*.jsonl"))
        for log_file in log_files:
            if log_file.exists():
                self.file_positions[log_file] = log_file.stat().st_size

        self.logger.info(
            f"[LogMonitor] Started monitoring {len(log_files)} log files",
            log_dir=str(self.config.log_dir),
        )

        # Start monitoring tasks
        self.monitor_task = asyncio.create_task(self._monitor_loop())

        if self.config.enable_health_monitoring:
            self.health_check_task = asyncio.create_task(self._health_check_loop())

    async def stop(self) -> None:
        """Stop monitoring."""
        self.running = False

        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass

        self.logger.info("[LogMonitor] Stopped monitoring")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                # Check all log files for new entries
                log_files = list(self.config.log_dir.glob("*.jsonl"))

                for log_file in log_files:
                    if not log_file.exists():
                        continue

                    # Get current file size
                    current_size = log_file.stat().st_size
                    last_position = self.file_positions.get(log_file, 0)

                    # If file grew, read new content
                    if current_size > last_position:
                        await self._process_new_logs(log_file, last_position, current_size)
                        self.file_positions[log_file] = current_size

                    # If file shrunk (rotation), reset position
                    elif current_size < last_position:
                        self.file_positions[log_file] = 0

                await asyncio.sleep(self.config.poll_interval)

            except Exception as e:
                self.logger.error(f"[LogMonitor] Monitor loop error: {e}", exc_info=True)
                await asyncio.sleep(self.config.poll_interval)

    async def _process_new_logs(
        self,
        log_file: Path,
        start_pos: int,
        end_pos: int,
    ) -> None:
        """Process new log entries from file."""
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                f.seek(start_pos)
                content = f.read(end_pos - start_pos)

            # Parse each line as JSON
            for line in content.strip().split("\n"):
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                    self.stats["total_logs_analyzed"] += 1

                    # Analyze for patterns
                    issue = self.detector.analyze_log_entry(entry)

                    if issue:
                        self.stats["issues_detected"] += 1
                        await self._handle_issue(issue)

                except json.JSONDecodeError:
                    # Skip malformed lines
                    pass

        except Exception as e:
            self.logger.error(
                f"[LogMonitor] Error processing logs from {log_file}: {e}",
                exc_info=True,
            )

    async def _handle_issue(self, issue: LogIssue) -> None:
        """Handle detected issue (potentially announce via voice)."""
        # Create signature for deduplication
        signature = f"{issue.category}:{issue.message[:50]}"

        # Check if we should announce
        should_announce = self._should_announce(issue, signature)

        if should_announce:
            await self._announce_issue(issue)
            self.announced_signatures.add(signature)
            self.last_announcement_time = datetime.now()
            self.stats["voice_announcements"] += 1

        # Log the issue (always)
        log_method = self.logger.critical if issue.severity == Severity.CRITICAL else self.logger.warning
        log_method(
            f"[LogMonitor] Detected {issue.severity} issue: {issue.message}",
            category=issue.category,
            severity=issue.severity,
            count=issue.count,
            affected_modules=list(issue.affected_modules),
            announced=should_announce,
        )

    def _should_announce(self, issue: LogIssue, signature: str) -> bool:
        """Determine if issue should be announced via voice."""
        # Critical errors always announced (if configured)
        if issue.severity == Severity.CRITICAL and self.config.critical_always_announces:
            return True

        # Check if already announced
        if signature in self.announced_signatures:
            return False

        # Check throttling
        if self.last_announcement_time:
            elapsed = (datetime.now() - self.last_announcement_time).total_seconds()
            if elapsed < self.config.min_seconds_between_alerts:
                return False

        # Only announce HIGH or CRITICAL
        return issue.severity in (Severity.CRITICAL, Severity.HIGH)

    async def _announce_issue(self, issue: LogIssue) -> None:
        """Announce issue via voice narrator."""
        if not self.narrator:
            return

        # Generate announcement message
        if issue.severity == Severity.CRITICAL:
            prefix = "Critical alert:"
        elif issue.severity == Severity.HIGH:
            prefix = "Important notice:"
        else:
            prefix = "Attention:"

        # Simplify message for speech
        message = self._simplify_for_speech(issue.message)
        announcement = f"{prefix} {message}"

        # Call narrator
        try:
            if asyncio.iscoroutinefunction(self.narrator):
                await self.narrator(announcement)
            else:
                # Run sync narrator in thread pool
                await asyncio.get_event_loop().run_in_executor(
                    None, self.narrator, announcement
                )
        except Exception as e:
            self.logger.error(
                f"[LogMonitor] Failed to announce via narrator: {e}",
                exc_info=True,
            )

    def _simplify_for_speech(self, message: str) -> str:
        """Simplify technical message for voice speech."""
        # Replace technical terms with speakable versions
        replacements = {
            "ms": "milliseconds",
            "->": "to",
            ">=": "greater than or equal to",
            "<=": "less than or equal to",
            ">": "greater than",
            "<": "less than",
        }

        result = message
        for old, new in replacements.items():
            result = result.replace(old, new)

        return result

    async def _health_check_loop(self) -> None:
        """Periodic health check and summary."""
        while self.running:
            try:
                await asyncio.sleep(self.config.health_check_interval_seconds)

                # Get health status
                health = await self.get_health_status()

                # Check for degraded components
                degraded = [
                    module
                    for module, status in health["component_health"].items()
                    if status in ("CRITICAL", "DEGRADED")
                ]

                if degraded and self.narrator:
                    # Announce degraded components
                    message = f"Health check: {len(degraded)} component{'s' if len(degraded) > 1 else ''} degraded: {', '.join(degraded[:3])}"
                    await self._announce_issue(
                        LogIssue(
                            severity=Severity.MEDIUM,
                            category="health_check",
                            message=message,
                            count=len(degraded),
                            first_seen=datetime.now(),
                            last_seen=datetime.now(),
                            affected_modules=set(degraded),
                        )
                    )

            except Exception as e:
                self.logger.error(
                    f"[LogMonitor] Health check error: {e}",
                    exc_info=True,
                )

    async def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        component_health = self.detector.get_component_health()
        error_rate = self.detector.get_error_rate()

        # Overall health
        critical_count = sum(1 for s in component_health.values() if s == "CRITICAL")
        degraded_count = sum(1 for s in component_health.values() if s == "DEGRADED")

        if critical_count > 0:
            overall = "CRITICAL"
        elif degraded_count > 0:
            overall = "DEGRADED"
        elif error_rate > self.config.error_rate_threshold:
            overall = "STRESSED"
        else:
            overall = "HEALTHY"

        return {
            "overall_health": overall,
            "component_health": component_health,
            "error_rate_per_minute": error_rate,
            "statistics": self.stats,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        uptime = None
        if self.stats["start_time"]:
            uptime = (datetime.now() - self.stats["start_time"]).total_seconds()

        return {
            **self.stats,
            "uptime_seconds": uptime,
            "running": self.running,
        }


# =============================================================================
# GLOBAL SINGLETON
# =============================================================================

_global_monitor: Optional[RealTimeLogMonitor] = None


async def get_log_monitor(
    config: Optional[LogMonitorConfig] = None,
    narrator: Optional[Callable] = None,
    logger: Optional[logging.Logger] = None,
) -> RealTimeLogMonitor:
    """
    Get or create global log monitor instance.

    Args:
        config: Optional configuration
        narrator: Optional voice narrator callback
        logger: Optional logger

    Returns:
        RealTimeLogMonitor instance
    """
    global _global_monitor

    if _global_monitor is None:
        _global_monitor = RealTimeLogMonitor(
            config=config or LogMonitorConfig.from_env(),
            narrator=narrator,
            logger=logger,
        )

    return _global_monitor


async def stop_global_monitor() -> None:
    """Stop global monitor if running."""
    global _global_monitor

    if _global_monitor:
        await _global_monitor.stop()
        _global_monitor = None
