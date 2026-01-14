"""
Model Versioning and Automatic Rollback System
===============================================

Provides intelligent model deployment with automatic rollback capabilities.

Features:
    - Semantic versioning for models
    - Deployment tracking and history
    - Automatic rollback based on metrics
    - Canary deployments with gradual rollout
    - Blue-green deployment support
    - Health-based promotion/demotion
    - A/B testing infrastructure
    - Version comparison and analysis

Theory:
    Model deployments are risky - a new version may perform worse than
    the previous one. This system provides:

    1. Safe deployment with automatic rollback on errors
    2. Gradual rollout (canary) to limit blast radius
    3. Metric-based health checking
    4. Quick rollback to known-good versions

Usage:
    manager = await get_model_version_manager()

    # Deploy new version
    await manager.deploy_version(
        model_name="classifier",
        version="2.1.0",
        deployment_type=DeploymentType.CANARY,
    )

    # Record metrics
    await manager.record_metric("classifier", "accuracy", 0.92)
    await manager.record_metric("classifier", "latency_p99", 150)

    # Automatic rollback if metrics degrade
    # Or manual rollback
    await manager.rollback("classifier", reason="Manual override")

Author: Trinity System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine, Deque, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("ModelVersioning")


# =============================================================================
# Configuration
# =============================================================================

# Rollback thresholds
ERROR_RATE_THRESHOLD = float(os.getenv("MODEL_ERROR_RATE_THRESHOLD", "0.05"))  # 5%
LATENCY_INCREASE_THRESHOLD = float(os.getenv("MODEL_LATENCY_THRESHOLD", "1.5"))  # 50% increase
ACCURACY_DROP_THRESHOLD = float(os.getenv("MODEL_ACCURACY_DROP", "0.02"))  # 2% drop

# Canary deployment
CANARY_INITIAL_PERCENTAGE = float(os.getenv("CANARY_INITIAL_PERCENT", "5.0"))
CANARY_INCREMENT = float(os.getenv("CANARY_INCREMENT", "10.0"))
CANARY_PROMOTION_INTERVAL = int(os.getenv("CANARY_PROMOTION_INTERVAL", "300"))  # 5 minutes

# Observation windows
WARM_UP_PERIOD = int(os.getenv("MODEL_WARMUP_PERIOD", "60"))  # seconds
METRIC_WINDOW = int(os.getenv("MODEL_METRIC_WINDOW", "300"))  # 5 minutes

# Redis configuration
VERSION_REDIS_PREFIX = os.getenv("VERSION_REDIS_PREFIX", "model_version:")


# =============================================================================
# Enums and Data Classes
# =============================================================================

class DeploymentType(Enum):
    """Types of model deployment."""
    IMMEDIATE = "immediate"      # Instant cutover
    CANARY = "canary"           # Gradual rollout
    BLUE_GREEN = "blue_green"   # Side-by-side
    SHADOW = "shadow"           # Test without serving


class DeploymentStatus(Enum):
    """Status of a deployment."""
    PENDING = "pending"
    WARMING_UP = "warming_up"
    ACTIVE = "active"
    PROMOTED = "promoted"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class RollbackReason(Enum):
    """Reasons for rollback."""
    HIGH_ERROR_RATE = "high_error_rate"
    LATENCY_DEGRADATION = "latency_degradation"
    ACCURACY_DROP = "accuracy_drop"
    HEALTH_CHECK_FAILED = "health_check_failed"
    MANUAL = "manual"
    TIMEOUT = "timeout"


@dataclass
class SemanticVersion:
    """Semantic version representation."""
    major: int
    minor: int
    patch: int
    prerelease: str = ""
    build: str = ""

    @classmethod
    def parse(cls, version_str: str) -> "SemanticVersion":
        """Parse a version string."""
        pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.]+))?(?:\+([a-zA-Z0-9.]+))?$'
        match = re.match(pattern, version_str)

        if not match:
            raise ValueError(f"Invalid version format: {version_str}")

        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3)),
            prerelease=match.group(4) or "",
            build=match.group(5) or "",
        )

    def __str__(self) -> str:
        """Convert to string."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version

    def __lt__(self, other: "SemanticVersion") -> bool:
        """Compare versions."""
        if (self.major, self.minor, self.patch) != (other.major, other.minor, other.patch):
            return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
        # Pre-release versions are less than release versions
        if self.prerelease and not other.prerelease:
            return True
        if not self.prerelease and other.prerelease:
            return False
        return self.prerelease < other.prerelease


@dataclass
class ModelVersion:
    """A model version with metadata."""
    model_name: str
    version: str
    created_at: float = field(default_factory=time.time)
    deployed_at: Optional[float] = None
    promoted_at: Optional[float] = None
    rolled_back_at: Optional[float] = None
    status: DeploymentStatus = DeploymentStatus.PENDING
    deployment_type: DeploymentType = DeploymentType.IMMEDIATE
    traffic_percentage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics_baseline: Dict[str, float] = field(default_factory=dict)

    @property
    def semantic_version(self) -> SemanticVersion:
        """Get parsed semantic version."""
        return SemanticVersion.parse(self.version)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "version": self.version,
            "created_at": self.created_at,
            "deployed_at": self.deployed_at,
            "promoted_at": self.promoted_at,
            "rolled_back_at": self.rolled_back_at,
            "status": self.status.value,
            "deployment_type": self.deployment_type.value,
            "traffic_percentage": self.traffic_percentage,
            "metadata": self.metadata,
            "metrics_baseline": self.metrics_baseline,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        """Create from dictionary."""
        return cls(
            model_name=data["model_name"],
            version=data["version"],
            created_at=data.get("created_at", time.time()),
            deployed_at=data.get("deployed_at"),
            promoted_at=data.get("promoted_at"),
            rolled_back_at=data.get("rolled_back_at"),
            status=DeploymentStatus(data.get("status", "pending")),
            deployment_type=DeploymentType(data.get("deployment_type", "immediate")),
            traffic_percentage=data.get("traffic_percentage", 0.0),
            metadata=data.get("metadata", {}),
            metrics_baseline=data.get("metrics_baseline", {}),
        )


@dataclass
class MetricDataPoint:
    """A metric data point."""
    value: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class RollbackEvent:
    """A rollback event record."""
    model_name: str
    from_version: str
    to_version: str
    reason: RollbackReason
    timestamp: float = field(default_factory=time.time)
    details: str = ""
    metrics_at_rollback: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# Metric Tracker for Version
# =============================================================================

class VersionMetricTracker:
    """
    Tracks metrics for a model version.

    Used for health monitoring and rollback decisions.
    """

    def __init__(self, window_size: int = METRIC_WINDOW):
        self._window_size = window_size
        self._metrics: Dict[str, Deque[MetricDataPoint]] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self._request_count = 0
        self._error_count = 0

    def record(self, metric_name: str, value: float) -> None:
        """Record a metric value."""
        self._metrics[metric_name].append(MetricDataPoint(value=value))

    def record_request(self, success: bool) -> None:
        """Record a request outcome."""
        self._request_count += 1
        if not success:
            self._error_count += 1

    def get_average(self, metric_name: str) -> Optional[float]:
        """Get average value in the window."""
        points = self._get_recent_points(metric_name)
        if not points:
            return None
        return sum(p.value for p in points) / len(points)

    def get_percentile(self, metric_name: str, percentile: float) -> Optional[float]:
        """Get percentile value (e.g., p99 = 99)."""
        points = self._get_recent_points(metric_name)
        if not points:
            return None
        values = sorted(p.value for p in points)
        idx = int(len(values) * percentile / 100)
        return values[min(idx, len(values) - 1)]

    def get_error_rate(self) -> float:
        """Get current error rate."""
        if self._request_count == 0:
            return 0.0
        return self._error_count / self._request_count

    def _get_recent_points(self, metric_name: str) -> List[MetricDataPoint]:
        """Get points within the time window."""
        cutoff = time.time() - self._window_size
        return [
            p for p in self._metrics[metric_name]
            if p.timestamp > cutoff
        ]

    def get_all_metrics(self) -> Dict[str, float]:
        """Get all current metric averages."""
        result = {}
        for metric_name in self._metrics:
            avg = self.get_average(metric_name)
            if avg is not None:
                result[metric_name] = avg
        result["error_rate"] = self.get_error_rate()
        result["request_count"] = self._request_count
        return result


# =============================================================================
# Model Version Manager
# =============================================================================

class ModelVersionManager:
    """
    Manages model versions with automatic rollback.

    Features:
    - Version tracking and history
    - Deployment management
    - Automatic rollback on degradation
    - Canary deployments
    - Metric-based promotion
    """

    def __init__(
        self,
        redis_client: Optional[Any] = None,
    ):
        self._redis = redis_client

        # Version state per model
        self._versions: Dict[str, Dict[str, ModelVersion]] = defaultdict(dict)
        self._active_versions: Dict[str, str] = {}  # model_name -> version
        self._previous_versions: Dict[str, str] = {}  # For rollback

        # Metrics per version
        self._version_metrics: Dict[str, Dict[str, VersionMetricTracker]] = defaultdict(dict)

        # Rollback history
        self._rollback_history: Deque[RollbackEvent] = deque(maxlen=100)

        # Canary state
        self._canary_tasks: Dict[str, asyncio.Task] = {}

        # Callbacks
        self._rollback_callbacks: List[Callable[[RollbackEvent], Coroutine[Any, Any, None]]] = []
        self._promotion_callbacks: List[Callable[[ModelVersion], Coroutine[Any, Any, None]]] = []

        # Background tasks
        self._running = False
        self._health_check_task: Optional[asyncio.Task] = None

        logger.info("ModelVersionManager initialized")

    async def start(self) -> None:
        """Start background health checking."""
        self._running = True
        self._health_check_task = asyncio.create_task(
            self._health_check_loop(),
            name="version_health_check",
        )
        logger.info("ModelVersionManager started")

    async def stop(self) -> None:
        """Stop background tasks."""
        self._running = False

        # Cancel canary tasks
        for task in self._canary_tasks.values():
            task.cancel()

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        logger.info("ModelVersionManager stopped")

    def on_rollback(
        self,
        callback: Callable[[RollbackEvent], Coroutine[Any, Any, None]],
    ) -> None:
        """Register a rollback callback."""
        self._rollback_callbacks.append(callback)

    def on_promotion(
        self,
        callback: Callable[[ModelVersion], Coroutine[Any, Any, None]],
    ) -> None:
        """Register a promotion callback."""
        self._promotion_callbacks.append(callback)

    async def register_version(
        self,
        model_name: str,
        version: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelVersion:
        """
        Register a new model version.

        Does not deploy - just registers for tracking.
        """
        # Validate version format
        SemanticVersion.parse(version)

        model_version = ModelVersion(
            model_name=model_name,
            version=version,
            metadata=metadata or {},
        )

        self._versions[model_name][version] = model_version
        self._version_metrics[model_name][version] = VersionMetricTracker()

        # Persist to Redis
        await self._save_version(model_version)

        logger.info(f"Registered version {version} for model {model_name}")
        return model_version

    async def deploy_version(
        self,
        model_name: str,
        version: str,
        deployment_type: DeploymentType = DeploymentType.IMMEDIATE,
        baseline_metrics: Optional[Dict[str, float]] = None,
    ) -> ModelVersion:
        """
        Deploy a model version.

        Args:
            model_name: Model name
            version: Version to deploy
            deployment_type: How to deploy
            baseline_metrics: Expected baseline metrics for comparison
        """
        if version not in self._versions[model_name]:
            await self.register_version(model_name, version)

        model_version = self._versions[model_name][version]
        model_version.deployment_type = deployment_type
        model_version.deployed_at = time.time()
        model_version.status = DeploymentStatus.WARMING_UP

        if baseline_metrics:
            model_version.metrics_baseline = baseline_metrics

        # Store previous version for rollback
        if model_name in self._active_versions:
            self._previous_versions[model_name] = self._active_versions[model_name]

        if deployment_type == DeploymentType.IMMEDIATE:
            # Instant cutover
            model_version.traffic_percentage = 100.0
            model_version.status = DeploymentStatus.ACTIVE
            self._active_versions[model_name] = version

        elif deployment_type == DeploymentType.CANARY:
            # Start canary rollout
            model_version.traffic_percentage = CANARY_INITIAL_PERCENTAGE
            model_version.status = DeploymentStatus.ACTIVE

            # Start canary promotion task
            task = asyncio.create_task(
                self._canary_promotion_loop(model_name, version),
                name=f"canary_{model_name}_{version}",
            )
            self._canary_tasks[f"{model_name}:{version}"] = task

        elif deployment_type == DeploymentType.BLUE_GREEN:
            # Run side-by-side (traffic goes to both)
            model_version.traffic_percentage = 0.0  # Shadow mode initially
            model_version.status = DeploymentStatus.ACTIVE

        elif deployment_type == DeploymentType.SHADOW:
            # Shadow mode - receives copies but doesn't serve
            model_version.traffic_percentage = 0.0
            model_version.status = DeploymentStatus.ACTIVE

        await self._save_version(model_version)
        logger.info(f"Deployed {model_name} version {version} ({deployment_type.value})")

        return model_version

    async def promote_version(
        self,
        model_name: str,
        version: str,
    ) -> bool:
        """
        Promote a version to 100% traffic.

        Used for canary/blue-green deployments.
        """
        if version not in self._versions[model_name]:
            logger.error(f"Version {version} not found for {model_name}")
            return False

        model_version = self._versions[model_name][version]

        # Cancel canary task if running
        task_key = f"{model_name}:{version}"
        if task_key in self._canary_tasks:
            self._canary_tasks[task_key].cancel()
            del self._canary_tasks[task_key]

        # Set previous active to 0%
        if model_name in self._active_versions:
            prev_version = self._active_versions[model_name]
            if prev_version != version and prev_version in self._versions[model_name]:
                self._versions[model_name][prev_version].traffic_percentage = 0.0
                self._previous_versions[model_name] = prev_version

        # Promote new version
        model_version.traffic_percentage = 100.0
        model_version.promoted_at = time.time()
        model_version.status = DeploymentStatus.PROMOTED
        self._active_versions[model_name] = version

        await self._save_version(model_version)

        # Notify callbacks
        for callback in self._promotion_callbacks:
            try:
                await callback(model_version)
            except Exception as e:
                logger.error(f"Promotion callback error: {e}")

        logger.info(f"Promoted {model_name} to version {version}")
        return True

    async def rollback(
        self,
        model_name: str,
        reason: RollbackReason = RollbackReason.MANUAL,
        details: str = "",
    ) -> bool:
        """
        Rollback to the previous version.
        """
        if model_name not in self._active_versions:
            logger.error(f"No active version for {model_name}")
            return False

        current_version = self._active_versions[model_name]

        # Find version to rollback to
        if model_name not in self._previous_versions:
            # Try to find a promoted version
            for version, mv in self._versions[model_name].items():
                if version != current_version and mv.status == DeploymentStatus.PROMOTED:
                    self._previous_versions[model_name] = version
                    break

        if model_name not in self._previous_versions:
            logger.error(f"No previous version to rollback to for {model_name}")
            return False

        previous_version = self._previous_versions[model_name]

        # Record rollback event
        current_mv = self._versions[model_name][current_version]
        metrics_tracker = self._version_metrics[model_name].get(current_version)
        current_metrics = metrics_tracker.get_all_metrics() if metrics_tracker else {}

        rollback_event = RollbackEvent(
            model_name=model_name,
            from_version=current_version,
            to_version=previous_version,
            reason=reason,
            details=details,
            metrics_at_rollback=current_metrics,
        )
        self._rollback_history.append(rollback_event)

        # Update versions
        current_mv.status = DeploymentStatus.ROLLED_BACK
        current_mv.rolled_back_at = time.time()
        current_mv.traffic_percentage = 0.0

        # Cancel canary if running
        task_key = f"{model_name}:{current_version}"
        if task_key in self._canary_tasks:
            self._canary_tasks[task_key].cancel()
            del self._canary_tasks[task_key]

        # Restore previous version
        previous_mv = self._versions[model_name][previous_version]
        previous_mv.traffic_percentage = 100.0
        previous_mv.status = DeploymentStatus.ACTIVE
        self._active_versions[model_name] = previous_version

        await self._save_version(current_mv)
        await self._save_version(previous_mv)

        # Notify callbacks
        for callback in self._rollback_callbacks:
            try:
                await callback(rollback_event)
            except Exception as e:
                logger.error(f"Rollback callback error: {e}")

        logger.warning(
            f"Rolled back {model_name} from {current_version} to {previous_version} "
            f"(reason: {reason.value})"
        )
        return True

    async def record_metric(
        self,
        model_name: str,
        metric_name: str,
        value: float,
        version: Optional[str] = None,
    ) -> None:
        """
        Record a metric for a model version.

        If version not specified, records for active version.
        """
        if version is None:
            version = self._active_versions.get(model_name)
            if version is None:
                return

        if version not in self._version_metrics[model_name]:
            self._version_metrics[model_name][version] = VersionMetricTracker()

        self._version_metrics[model_name][version].record(metric_name, value)

    async def record_request(
        self,
        model_name: str,
        success: bool,
        version: Optional[str] = None,
    ) -> None:
        """Record a request outcome."""
        if version is None:
            version = self._active_versions.get(model_name)
            if version is None:
                return

        if version not in self._version_metrics[model_name]:
            self._version_metrics[model_name][version] = VersionMetricTracker()

        self._version_metrics[model_name][version].record_request(success)

    def get_active_version(self, model_name: str) -> Optional[ModelVersion]:
        """Get the currently active version for a model."""
        version = self._active_versions.get(model_name)
        if version and version in self._versions[model_name]:
            return self._versions[model_name][version]
        return None

    def get_traffic_split(self, model_name: str) -> Dict[str, float]:
        """Get current traffic split for a model."""
        result = {}
        for version, mv in self._versions[model_name].items():
            if mv.traffic_percentage > 0:
                result[version] = mv.traffic_percentage
        return result

    def should_route_to_version(self, model_name: str, version: str) -> bool:
        """
        Check if a request should be routed to a specific version.

        Uses traffic percentage for probabilistic routing.
        """
        if version not in self._versions[model_name]:
            return False

        mv = self._versions[model_name][version]
        if mv.traffic_percentage >= 100.0:
            return True
        if mv.traffic_percentage <= 0.0:
            return False

        import random
        return random.random() * 100 < mv.traffic_percentage

    async def _canary_promotion_loop(
        self,
        model_name: str,
        version: str,
    ) -> None:
        """Background task to gradually promote canary."""
        try:
            # Wait for warmup
            await asyncio.sleep(WARM_UP_PERIOD)

            model_version = self._versions[model_name][version]
            model_version.status = DeploymentStatus.ACTIVE

            while model_version.traffic_percentage < 100.0:
                # Check health before promoting
                if not await self._check_version_health(model_name, version):
                    logger.warning(f"Canary {model_name}:{version} failed health check")
                    await self.rollback(
                        model_name,
                        reason=RollbackReason.HEALTH_CHECK_FAILED,
                        details="Canary health check failed",
                    )
                    return

                # Increase traffic
                new_percentage = min(
                    model_version.traffic_percentage + CANARY_INCREMENT,
                    100.0,
                )
                model_version.traffic_percentage = new_percentage

                # Decrease traffic on previous version
                if model_name in self._previous_versions:
                    prev_version = self._previous_versions[model_name]
                    if prev_version in self._versions[model_name]:
                        prev_mv = self._versions[model_name][prev_version]
                        prev_mv.traffic_percentage = max(0.0, 100.0 - new_percentage)

                logger.info(f"Canary {model_name}:{version} at {new_percentage:.1f}%")
                await self._save_version(model_version)

                # Wait before next promotion
                await asyncio.sleep(CANARY_PROMOTION_INTERVAL)

            # Fully promoted
            await self.promote_version(model_name, version)

        except asyncio.CancelledError:
            logger.info(f"Canary promotion cancelled for {model_name}:{version}")
        except Exception as e:
            logger.error(f"Canary promotion error: {e}")
            await self.rollback(
                model_name,
                reason=RollbackReason.HEALTH_CHECK_FAILED,
                details=str(e),
            )

    async def _check_version_health(
        self,
        model_name: str,
        version: str,
    ) -> bool:
        """Check if a version is healthy based on metrics."""
        if version not in self._version_metrics[model_name]:
            return True  # No metrics = assume healthy

        tracker = self._version_metrics[model_name][version]
        model_version = self._versions[model_name][version]

        # Check error rate
        error_rate = tracker.get_error_rate()
        if error_rate > ERROR_RATE_THRESHOLD:
            logger.warning(f"{model_name}:{version} error rate {error_rate:.2%} > threshold")
            return False

        # Check against baseline metrics
        for metric_name, baseline in model_version.metrics_baseline.items():
            current = tracker.get_average(metric_name)
            if current is None:
                continue

            # For latency-like metrics (higher is worse)
            if "latency" in metric_name.lower() or "time" in metric_name.lower():
                if current > baseline * LATENCY_INCREASE_THRESHOLD:
                    logger.warning(
                        f"{model_name}:{version} {metric_name} {current:.2f} > "
                        f"baseline {baseline:.2f}"
                    )
                    return False

            # For accuracy-like metrics (lower is worse)
            if "accuracy" in metric_name.lower() or "score" in metric_name.lower():
                if current < baseline - ACCURACY_DROP_THRESHOLD:
                    logger.warning(
                        f"{model_name}:{version} {metric_name} {current:.2f} < "
                        f"baseline {baseline:.2f}"
                    )
                    return False

        return True

    async def _health_check_loop(self) -> None:
        """Background loop to check all active versions."""
        while self._running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                for model_name, version in list(self._active_versions.items()):
                    if not await self._check_version_health(model_name, version):
                        # Automatic rollback
                        tracker = self._version_metrics[model_name].get(version)
                        error_rate = tracker.get_error_rate() if tracker else 0.0

                        if error_rate > ERROR_RATE_THRESHOLD:
                            reason = RollbackReason.HIGH_ERROR_RATE
                        else:
                            reason = RollbackReason.HEALTH_CHECK_FAILED

                        await self.rollback(model_name, reason=reason)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(30)

    async def _save_version(self, model_version: ModelVersion) -> None:
        """Save version to Redis."""
        if not self._redis:
            return

        try:
            key = f"{VERSION_REDIS_PREFIX}{model_version.model_name}:{model_version.version}"
            await self._redis.set(key, json.dumps(model_version.to_dict()))
        except Exception as e:
            logger.warning(f"Failed to save version to Redis: {e}")

    def get_version_history(
        self,
        model_name: str,
        limit: int = 10,
    ) -> List[ModelVersion]:
        """Get version history for a model."""
        versions = list(self._versions[model_name].values())
        versions.sort(key=lambda v: v.created_at, reverse=True)
        return versions[:limit]

    def get_rollback_history(
        self,
        model_name: Optional[str] = None,
        limit: int = 10,
    ) -> List[RollbackEvent]:
        """Get rollback history."""
        history = list(self._rollback_history)
        if model_name:
            history = [r for r in history if r.model_name == model_name]
        history.sort(key=lambda r: r.timestamp, reverse=True)
        return history[:limit]

    def get_metrics(self) -> Dict[str, Any]:
        """Get manager metrics."""
        return {
            "models_tracked": len(self._versions),
            "active_versions": dict(self._active_versions),
            "canary_deployments": len(self._canary_tasks),
            "total_rollbacks": len(self._rollback_history),
            "recent_rollbacks": [
                {
                    "model": r.model_name,
                    "from": r.from_version,
                    "to": r.to_version,
                    "reason": r.reason.value,
                }
                for r in list(self._rollback_history)[-3:]
            ],
        }


# =============================================================================
# Global Factory
# =============================================================================

_manager_instance: Optional[ModelVersionManager] = None
_manager_lock = asyncio.Lock()


async def get_model_version_manager(
    redis_client: Optional[Any] = None,
) -> ModelVersionManager:
    """Get or create the global ModelVersionManager instance."""
    global _manager_instance

    async with _manager_lock:
        if _manager_instance is None:
            _manager_instance = ModelVersionManager(redis_client=redis_client)
            await _manager_instance.start()

        return _manager_instance


async def shutdown_model_version_manager() -> None:
    """Shutdown the global manager."""
    global _manager_instance

    if _manager_instance:
        await _manager_instance.stop()
        _manager_instance = None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ModelVersionManager",
    "ModelVersion",
    "SemanticVersion",
    "DeploymentType",
    "DeploymentStatus",
    "RollbackReason",
    "RollbackEvent",
    "VersionMetricTracker",
    "get_model_version_manager",
    "shutdown_model_version_manager",
]
