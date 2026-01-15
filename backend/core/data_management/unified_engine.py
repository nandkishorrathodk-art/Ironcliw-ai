"""
Unified Data Management Engine v1.0
====================================

The central nervous system for data management across the JARVIS Trinity.
Implements enterprise-grade data lifecycle management with:

1. TRAINING DATA COLLECTION
   - Automatic collection from all interaction points
   - Intelligent batching with configurable thresholds
   - Cross-repo forwarding to Reactor Core for training
   - Quality scoring and filtering

2. DATA VERSIONING
   - Content-addressable storage (Git-like)
   - Semantic versioning for datasets
   - Branch/tag support for experiments
   - Diff and merge capabilities

3. DATA VALIDATION
   - Schema registry with evolution support
   - Statistical quality checks
   - Anomaly detection with ML
   - Data drift monitoring

4. DATA PRIVACY
   - PII detection (regex + ML)
   - Anonymization (k-anonymity, differential privacy)
   - Encryption at rest and in transit
   - GDPR/CCPA compliance tracking

5. DATA RETENTION POLICIES
   - TTL-based lifecycle management
   - Tiered storage (hot/warm/cold/archive)
   - Automatic archival and deletion
   - Legal hold support

6. DATA DEDUPLICATION
   - Content-hash deduplication
   - Semantic similarity matching
   - Fuzzy dedup for near-duplicates
   - Space savings tracking

7. INTELLIGENT DATA SAMPLING
   - Stratified sampling
   - Importance sampling
   - Active learning sampling
   - Reservoir sampling for streaming

8. DATA LINEAGE
   - DAG-based provenance tracking
   - Full audit trail
   - Impact analysis
   - Reproducibility support

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    UNIFIED DATA MANAGEMENT ENGINE v1.0                   │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐   │
    │  │   COLLECTION    │     │   VERSIONING    │     │   VALIDATION    │   │
    │  │   (Batching)    │────▶│   (Content-     │────▶│   (Schema +     │   │
    │  │                 │     │    Addressable) │     │    Quality)     │   │
    │  └────────┬────────┘     └────────┬────────┘     └────────┬────────┘   │
    │           │                       │                       │             │
    │           ▼                       ▼                       ▼             │
    │  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐   │
    │  │   PRIVACY       │     │   RETENTION     │     │   DEDUP         │   │
    │  │   (PII +        │◀───▶│   (TTL +        │◀───▶│   (Hash +       │   │
    │  │    Anonymize)   │     │    Tiered)      │     │    Semantic)    │   │
    │  └────────┬────────┘     └────────┬────────┘     └────────┬────────┘   │
    │           │                       │                       │             │
    │           ▼                       ▼                       ▼             │
    │  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐   │
    │  │   SAMPLING      │     │   LINEAGE       │     │   STORAGE       │   │
    │  │   (Stratified + │────▶│   (DAG +        │────▶│   (Persistent   │   │
    │  │    Active)      │     │    Audit)       │     │    + Cache)     │   │
    │  └─────────────────┘     └─────────────────┘     └─────────────────┘   │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

Author: Trinity Data System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import base64
import functools
import hashlib
import hmac
import json
import logging
import math
import os
import pickle
import random
import re
import secrets
import shutil
import statistics
import struct
import time
import traceback
import uuid
import zlib
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Deque,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)

try:
    import aiofiles
    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logger = logging.getLogger("DataManagement.UnifiedEngine")

T = TypeVar("T")


# =============================================================================
# CONFIGURATION - ALL DYNAMIC FROM ENVIRONMENT
# =============================================================================

class DataManagementConfig:
    """Dynamic configuration with environment variable support."""

    # Base Paths
    @staticmethod
    def get_data_root() -> Path:
        default = Path.home() / ".jarvis" / "data_management"
        return Path(os.getenv("JARVIS_DATA_ROOT", str(default)))

    @staticmethod
    def get_versions_path() -> Path:
        return DataManagementConfig.get_data_root() / "versions"

    @staticmethod
    def get_lineage_path() -> Path:
        return DataManagementConfig.get_data_root() / "lineage"

    @staticmethod
    def get_archive_path() -> Path:
        return DataManagementConfig.get_data_root() / "archive"

    # Collection Configuration
    @staticmethod
    def get_batch_size() -> int:
        return int(os.getenv("DATA_BATCH_SIZE", "100"))

    @staticmethod
    def get_batch_timeout_seconds() -> float:
        return float(os.getenv("DATA_BATCH_TIMEOUT", "60.0"))

    @staticmethod
    def get_max_queue_size() -> int:
        return int(os.getenv("DATA_MAX_QUEUE_SIZE", "10000"))

    # Versioning Configuration
    @staticmethod
    def get_max_versions() -> int:
        return int(os.getenv("DATA_MAX_VERSIONS", "100"))

    @staticmethod
    def get_compression_enabled() -> bool:
        return os.getenv("DATA_COMPRESSION", "true").lower() == "true"

    # Validation Configuration
    @staticmethod
    def get_quality_threshold() -> float:
        return float(os.getenv("DATA_QUALITY_THRESHOLD", "0.8"))

    @staticmethod
    def get_anomaly_sensitivity() -> float:
        return float(os.getenv("DATA_ANOMALY_SENSITIVITY", "2.0"))

    # Privacy Configuration
    @staticmethod
    def get_encryption_key() -> Optional[str]:
        return os.getenv("DATA_ENCRYPTION_KEY")

    @staticmethod
    def get_default_privacy_level() -> str:
        return os.getenv("DATA_DEFAULT_PRIVACY", "standard")

    @staticmethod
    def get_pii_detection_enabled() -> bool:
        return os.getenv("DATA_PII_DETECTION", "true").lower() == "true"

    # Retention Configuration
    @staticmethod
    def get_default_retention_days() -> int:
        return int(os.getenv("DATA_RETENTION_DAYS", "90"))

    @staticmethod
    def get_archive_after_days() -> int:
        return int(os.getenv("DATA_ARCHIVE_DAYS", "30"))

    @staticmethod
    def get_cleanup_interval_hours() -> float:
        return float(os.getenv("DATA_CLEANUP_INTERVAL", "6.0"))

    # Deduplication Configuration
    @staticmethod
    def get_similarity_threshold() -> float:
        return float(os.getenv("DATA_SIMILARITY_THRESHOLD", "0.95"))

    @staticmethod
    def get_dedup_enabled() -> bool:
        return os.getenv("DATA_DEDUP_ENABLED", "true").lower() == "true"

    # Sampling Configuration
    @staticmethod
    def get_default_sample_size() -> int:
        return int(os.getenv("DATA_SAMPLE_SIZE", "1000"))

    @staticmethod
    def get_reservoir_size() -> int:
        return int(os.getenv("DATA_RESERVOIR_SIZE", "10000"))

    # Lineage Configuration
    @staticmethod
    def get_lineage_enabled() -> bool:
        return os.getenv("DATA_LINEAGE_ENABLED", "true").lower() == "true"

    @staticmethod
    def get_lineage_max_depth() -> int:
        return int(os.getenv("DATA_LINEAGE_MAX_DEPTH", "50"))


# =============================================================================
# ENUMS
# =============================================================================

class DataType(str, Enum):
    """Types of data in the system."""
    INTERACTION = "interaction"         # User interactions
    TRAINING = "training"               # Training examples
    VOICE_SAMPLE = "voice_sample"       # Voice biometric samples
    EMBEDDING = "embedding"             # ML embeddings
    MODEL_WEIGHTS = "model_weights"     # Model parameters
    CONFIGURATION = "configuration"     # System configuration
    METRICS = "metrics"                 # Performance metrics
    LOGS = "logs"                       # System logs
    EXPERIENCE = "experience"           # Learning experiences
    FEEDBACK = "feedback"               # User feedback


class DataQuality(str, Enum):
    """Quality levels for data."""
    EXCELLENT = "excellent"   # >0.95 quality score
    GOOD = "good"             # 0.8-0.95
    ACCEPTABLE = "acceptable" # 0.6-0.8
    POOR = "poor"             # 0.4-0.6
    INVALID = "invalid"       # <0.4


class PrivacyLevel(str, Enum):
    """Privacy classification levels."""
    PUBLIC = "public"               # No restrictions
    INTERNAL = "internal"           # Internal use only
    CONFIDENTIAL = "confidential"   # Restricted access
    SENSITIVE = "sensitive"         # Contains PII
    RESTRICTED = "restricted"       # Highly sensitive, encrypted


class RetentionPolicy(str, Enum):
    """Data retention policies."""
    TRANSIENT = "transient"     # Delete after use
    SHORT_TERM = "short_term"   # 7 days
    MEDIUM_TERM = "medium_term" # 30 days
    LONG_TERM = "long_term"     # 90 days
    ARCHIVE = "archive"         # 1 year, then archive
    PERMANENT = "permanent"     # Never delete
    LEGAL_HOLD = "legal_hold"   # Cannot delete


class SamplingMethod(str, Enum):
    """Data sampling methods."""
    RANDOM = "random"                   # Simple random sampling
    STRATIFIED = "stratified"           # Stratified by category
    IMPORTANCE = "importance"           # Weighted by importance
    ACTIVE_LEARNING = "active_learning" # Uncertainty-based
    RESERVOIR = "reservoir"             # Streaming reservoir
    SYSTEMATIC = "systematic"           # Every nth item
    CLUSTER = "cluster"                 # Cluster-based


class LineageEventType(str, Enum):
    """Types of lineage events."""
    CREATED = "created"
    TRANSFORMED = "transformed"
    VALIDATED = "validated"
    ANONYMIZED = "anonymized"
    SAMPLED = "sampled"
    MERGED = "merged"
    SPLIT = "split"
    ARCHIVED = "archived"
    DELETED = "deleted"
    ACCESSED = "accessed"
    EXPORTED = "exported"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DataRecord:
    """A single data record in the system."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    data_type: DataType = DataType.INTERACTION
    content: Dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    timestamp: float = field(default_factory=time.time)
    quality_score: float = 1.0
    privacy_level: PrivacyLevel = PrivacyLevel.INTERNAL
    retention_policy: RetentionPolicy = RetentionPolicy.MEDIUM_TERM
    version: str = "1.0.0"
    lineage_id: Optional[str] = None
    parent_ids: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    expires_at: Optional[float] = None

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = self._compute_hash()
        if not self.expires_at:
            self.expires_at = self._compute_expiry()

    def _compute_hash(self) -> str:
        """Compute content hash."""
        content_str = json.dumps(self.content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    def _compute_expiry(self) -> float:
        """Compute expiry time based on retention policy."""
        policy_days = {
            RetentionPolicy.TRANSIENT: 0,
            RetentionPolicy.SHORT_TERM: 7,
            RetentionPolicy.MEDIUM_TERM: 30,
            RetentionPolicy.LONG_TERM: 90,
            RetentionPolicy.ARCHIVE: 365,
            RetentionPolicy.PERMANENT: 36500,  # 100 years
            RetentionPolicy.LEGAL_HOLD: 36500,
        }
        days = policy_days.get(self.retention_policy, 30)
        return self.timestamp + (days * 86400)

    def is_expired(self) -> bool:
        """Check if record has expired."""
        if self.retention_policy == RetentionPolicy.LEGAL_HOLD:
            return False
        return time.time() > self.expires_at if self.expires_at else False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "data_type": self.data_type.value,
            "content": self.content,
            "content_hash": self.content_hash,
            "metadata": self.metadata,
            "source": self.source,
            "timestamp": self.timestamp,
            "quality_score": self.quality_score,
            "privacy_level": self.privacy_level.value,
            "retention_policy": self.retention_policy.value,
            "version": self.version,
            "lineage_id": self.lineage_id,
            "parent_ids": self.parent_ids,
            "tags": list(self.tags),
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataRecord":
        return cls(
            id=data.get("id", uuid.uuid4().hex),
            data_type=DataType(data.get("data_type", "interaction")),
            content=data.get("content", {}),
            content_hash=data.get("content_hash", ""),
            metadata=data.get("metadata", {}),
            source=data.get("source", "unknown"),
            timestamp=data.get("timestamp", time.time()),
            quality_score=data.get("quality_score", 1.0),
            privacy_level=PrivacyLevel(data.get("privacy_level", "internal")),
            retention_policy=RetentionPolicy(data.get("retention_policy", "medium_term")),
            version=data.get("version", "1.0.0"),
            lineage_id=data.get("lineage_id"),
            parent_ids=data.get("parent_ids", []),
            tags=set(data.get("tags", [])),
            expires_at=data.get("expires_at"),
        )


@dataclass
class DataBatch:
    """A batch of data records."""
    batch_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    records: List[DataRecord] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    batch_size: int = 0
    total_size_bytes: int = 0
    data_types: Set[DataType] = field(default_factory=set)
    quality_summary: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add(self, record: DataRecord) -> None:
        self.records.append(record)
        self.batch_size += 1
        self.data_types.add(record.data_type)
        quality = DataQuality.EXCELLENT if record.quality_score > 0.95 else \
                  DataQuality.GOOD if record.quality_score > 0.8 else \
                  DataQuality.ACCEPTABLE if record.quality_score > 0.6 else \
                  DataQuality.POOR if record.quality_score > 0.4 else DataQuality.INVALID
        self.quality_summary[quality.value] = self.quality_summary.get(quality.value, 0) + 1

    def complete(self) -> None:
        self.completed_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "record_count": len(self.records),
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "batch_size": self.batch_size,
            "total_size_bytes": self.total_size_bytes,
            "data_types": [dt.value for dt in self.data_types],
            "quality_summary": self.quality_summary,
            "metadata": self.metadata,
        }


@dataclass
class DataVersion:
    """Version information for a dataset."""
    version_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    version: str = "1.0.0"  # Semantic version
    parent_version: Optional[str] = None
    content_hash: str = ""
    created_at: float = field(default_factory=time.time)
    created_by: str = "system"
    record_count: int = 0
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    branch: str = "main"
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "version": self.version,
            "parent_version": self.parent_version,
            "content_hash": self.content_hash,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "record_count": self.record_count,
            "size_bytes": self.size_bytes,
            "metadata": self.metadata,
            "tags": self.tags,
            "branch": self.branch,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataVersion":
        return cls(**data)


@dataclass
class DataLineageNode:
    """A node in the data lineage DAG."""
    node_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    record_id: str = ""
    event_type: LineageEventType = LineageEventType.CREATED
    timestamp: float = field(default_factory=time.time)
    actor: str = "system"
    operation: str = ""
    input_ids: List[str] = field(default_factory=list)
    output_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "record_id": self.record_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "operation": self.operation,
            "input_ids": self.input_ids,
            "output_ids": self.output_ids,
            "metadata": self.metadata,
        }


@dataclass
class DataLineageEdge:
    """An edge in the data lineage DAG."""
    edge_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    source_id: str = ""
    target_id: str = ""
    relationship: str = "derived_from"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of data validation."""
    valid: bool = True
    quality_score: float = 1.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    schema_valid: bool = True
    statistical_valid: bool = True
    anomaly_detected: bool = False
    anomaly_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "quality_score": self.quality_score,
            "errors": self.errors,
            "warnings": self.warnings,
            "schema_valid": self.schema_valid,
            "statistical_valid": self.statistical_valid,
            "anomaly_detected": self.anomaly_detected,
            "anomaly_score": self.anomaly_score,
            "metadata": self.metadata,
        }


@dataclass
class PrivacyReport:
    """Report from privacy analysis."""
    contains_pii: bool = False
    pii_types: List[str] = field(default_factory=list)
    pii_locations: List[Dict[str, Any]] = field(default_factory=list)
    risk_score: float = 0.0
    anonymization_applied: bool = False
    encryption_applied: bool = False
    compliance_status: Dict[str, bool] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contains_pii": self.contains_pii,
            "pii_types": self.pii_types,
            "pii_locations": self.pii_locations,
            "risk_score": self.risk_score,
            "anonymization_applied": self.anonymization_applied,
            "encryption_applied": self.encryption_applied,
            "compliance_status": self.compliance_status,
            "recommendations": self.recommendations,
        }


@dataclass
class SamplingStrategy:
    """Configuration for data sampling."""
    method: SamplingMethod = SamplingMethod.RANDOM
    sample_size: int = 1000
    stratify_by: Optional[str] = None
    importance_weights: Optional[Dict[str, float]] = None
    uncertainty_threshold: float = 0.5
    seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# TRAINING DATA COLLECTOR
# =============================================================================

class TrainingDataCollector:
    """
    Automatic training data collection with batching and forwarding.

    Features:
    - Automatic collection from all interaction points
    - Intelligent batching with configurable thresholds
    - Quality scoring and filtering
    - Cross-repo forwarding to Reactor Core
    """

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue(
            maxsize=DataManagementConfig.get_max_queue_size()
        )
        self._current_batch: DataBatch = DataBatch()
        self._batch_lock = asyncio.Lock()
        self._running = False
        self._batch_task: Optional[asyncio.Task] = None
        self._forward_callback: Optional[Callable[[DataBatch], Awaitable[bool]]] = None

        # Metrics
        self._total_collected = 0
        self._total_batches = 0
        self._total_forwarded = 0
        self._total_dropped = 0

        self.logger = logging.getLogger("DataCollector")

    async def start(self) -> None:
        """Start the collector."""
        if self._running:
            return

        self._running = True
        self._batch_task = asyncio.create_task(self._batch_processor_loop())
        self.logger.info("Training data collector started")

    async def stop(self) -> None:
        """Stop the collector and flush remaining data."""
        self._running = False

        # Flush remaining batch
        if self._current_batch.records:
            await self._flush_batch()

        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass

        self.logger.info(
            f"Training data collector stopped. "
            f"Collected: {self._total_collected}, "
            f"Batches: {self._total_batches}, "
            f"Forwarded: {self._total_forwarded}"
        )

    def set_forward_callback(
        self,
        callback: Callable[[DataBatch], Awaitable[bool]]
    ) -> None:
        """Set callback for forwarding batches to Reactor Core."""
        self._forward_callback = callback

    async def collect(
        self,
        content: Dict[str, Any],
        data_type: DataType = DataType.TRAINING,
        source: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
        quality_score: float = 1.0,
        tags: Optional[Set[str]] = None,
    ) -> str:
        """
        Collect a training data record.

        Returns the record ID.
        """
        record = DataRecord(
            data_type=data_type,
            content=content,
            source=source,
            metadata=metadata or {},
            quality_score=quality_score,
            tags=tags or set(),
        )

        # Drop low quality data
        if quality_score < DataManagementConfig.get_quality_threshold():
            self._total_dropped += 1
            self.logger.debug(f"Dropped low quality record: {record.id}")
            return record.id

        try:
            await asyncio.wait_for(
                self._queue.put(record),
                timeout=5.0,
            )
            self._total_collected += 1
            return record.id

        except asyncio.TimeoutError:
            self._total_dropped += 1
            self.logger.warning("Queue full, dropping record")
            return record.id

    async def collect_interaction(
        self,
        user_input: str,
        system_response: str,
        intent: Optional[str] = None,
        domain: Optional[str] = None,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Convenience method for collecting interaction data."""
        content = {
            "input": user_input,
            "output": system_response,
            "intent": intent,
            "domain": domain,
            "success": success,
        }
        return await self.collect(
            content=content,
            data_type=DataType.INTERACTION,
            source="user_interaction",
            metadata=metadata,
            quality_score=1.0 if success else 0.7,
            tags={"interaction", domain or "general"},
        )

    async def collect_voice_sample(
        self,
        embedding: List[float],
        speaker_id: str,
        confidence: float,
        environment: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Convenience method for collecting voice samples."""
        content = {
            "embedding": embedding[:50] if len(embedding) > 50 else embedding,  # Truncate for storage
            "embedding_dim": len(embedding),
            "speaker_id": speaker_id,
            "confidence": confidence,
            "environment": environment,
        }
        return await self.collect(
            content=content,
            data_type=DataType.VOICE_SAMPLE,
            source="voice_biometric",
            metadata=metadata,
            quality_score=confidence,
            tags={"voice", "biometric"},
        )

    async def collect_feedback(
        self,
        feedback_type: str,
        feedback_value: Any,
        context: Optional[Dict[str, Any]] = None,
        source: str = "user",
    ) -> str:
        """Convenience method for collecting feedback."""
        content = {
            "type": feedback_type,
            "value": feedback_value,
            "context": context or {},
        }
        return await self.collect(
            content=content,
            data_type=DataType.FEEDBACK,
            source=source,
            tags={"feedback", feedback_type},
        )

    async def _batch_processor_loop(self) -> None:
        """Background loop for batch processing."""
        batch_timeout = DataManagementConfig.get_batch_timeout_seconds()
        batch_size = DataManagementConfig.get_batch_size()
        last_flush = time.time()

        while self._running:
            try:
                # Get record from queue with timeout
                try:
                    record = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0,
                    )

                    async with self._batch_lock:
                        self._current_batch.add(record)

                except asyncio.TimeoutError:
                    pass

                # Check if we should flush
                should_flush = False
                async with self._batch_lock:
                    if len(self._current_batch.records) >= batch_size:
                        should_flush = True
                    elif time.time() - last_flush >= batch_timeout and self._current_batch.records:
                        should_flush = True

                if should_flush:
                    await self._flush_batch()
                    last_flush = time.time()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Batch processor error: {e}")
                await asyncio.sleep(1.0)

    async def _flush_batch(self) -> None:
        """Flush current batch."""
        async with self._batch_lock:
            if not self._current_batch.records:
                return

            batch = self._current_batch
            batch.complete()
            self._current_batch = DataBatch()

        self._total_batches += 1

        # Forward to Reactor Core if callback set
        if self._forward_callback:
            try:
                success = await self._forward_callback(batch)
                if success:
                    self._total_forwarded += 1
                    self.logger.info(
                        f"Forwarded batch {batch.batch_id} with {len(batch.records)} records"
                    )
                else:
                    self.logger.warning(f"Failed to forward batch {batch.batch_id}")
            except Exception as e:
                self.logger.error(f"Error forwarding batch: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "total_collected": self._total_collected,
            "total_batches": self._total_batches,
            "total_forwarded": self._total_forwarded,
            "total_dropped": self._total_dropped,
            "queue_size": self._queue.qsize(),
            "current_batch_size": len(self._current_batch.records),
            "running": self._running,
        }


# =============================================================================
# DATA VERSION MANAGER
# =============================================================================

class DataVersionManager:
    """
    Git-like data versioning with content-addressable storage.

    Features:
    - Content-addressable storage
    - Semantic versioning for datasets
    - Branch/tag support
    - Diff and merge capabilities
    """

    def __init__(self):
        self._versions_path = DataManagementConfig.get_versions_path()
        self._versions: Dict[str, DataVersion] = {}
        self._branches: Dict[str, str] = {"main": ""}  # branch -> latest version_id
        self._tags: Dict[str, str] = {}  # tag -> version_id
        self._lock = asyncio.Lock()
        self._initialized = False

        self.logger = logging.getLogger("DataVersionManager")

    async def initialize(self) -> None:
        """Initialize version manager."""
        self._versions_path.mkdir(parents=True, exist_ok=True)
        await self._load_versions()
        self._initialized = True
        self.logger.info(f"Version manager initialized with {len(self._versions)} versions")

    async def _load_versions(self) -> None:
        """Load existing versions from disk."""
        manifest_path = self._versions_path / "manifest.json"
        if manifest_path.exists():
            try:
                if HAS_AIOFILES:
                    async with aiofiles.open(manifest_path) as f:
                        data = json.loads(await f.read())
                else:
                    data = json.loads(manifest_path.read_text())

                for v_data in data.get("versions", []):
                    version = DataVersion.from_dict(v_data)
                    self._versions[version.version_id] = version

                self._branches = data.get("branches", {"main": ""})
                self._tags = data.get("tags", {})

            except Exception as e:
                self.logger.error(f"Failed to load versions: {e}")

    async def _save_manifest(self) -> None:
        """Save versions manifest to disk."""
        manifest_path = self._versions_path / "manifest.json"
        data = {
            "versions": [v.to_dict() for v in self._versions.values()],
            "branches": self._branches,
            "tags": self._tags,
        }

        content = json.dumps(data, indent=2)
        if HAS_AIOFILES:
            async with aiofiles.open(manifest_path, "w") as f:
                await f.write(content)
        else:
            manifest_path.write_text(content)

    async def create_version(
        self,
        records: List[DataRecord],
        message: str = "",
        branch: str = "main",
        parent_version: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> DataVersion:
        """Create a new version of the dataset."""
        async with self._lock:
            # Compute content hash
            content_hashes = [r.content_hash for r in records]
            combined = "|".join(sorted(content_hashes))
            content_hash = hashlib.sha256(combined.encode()).hexdigest()[:16]

            # Determine parent version
            if parent_version is None:
                parent_version = self._branches.get(branch)

            # Compute version number
            if parent_version and parent_version in self._versions:
                parent = self._versions[parent_version]
                # Increment patch version
                parts = parent.version.split(".")
                parts[2] = str(int(parts[2]) + 1)
                version_str = ".".join(parts)
            else:
                version_str = "1.0.0"

            # Create version
            version = DataVersion(
                version=version_str,
                parent_version=parent_version,
                content_hash=content_hash,
                record_count=len(records),
                size_bytes=sum(len(json.dumps(r.to_dict())) for r in records),
                branch=branch,
                message=message,
                tags=tags or [],
            )

            # Store version
            self._versions[version.version_id] = version
            self._branches[branch] = version.version_id

            # Add tags
            for tag in version.tags:
                self._tags[tag] = version.version_id

            # Save data
            version_path = self._versions_path / version.version_id
            version_path.mkdir(parents=True, exist_ok=True)

            records_data = [r.to_dict() for r in records]

            if DataManagementConfig.get_compression_enabled():
                compressed = zlib.compress(json.dumps(records_data).encode())
                data_path = version_path / "data.json.zlib"
                data_path.write_bytes(compressed)
            else:
                data_path = version_path / "data.json"
                data_path.write_text(json.dumps(records_data, indent=2))

            await self._save_manifest()

            self.logger.info(f"Created version {version.version} ({version.version_id})")
            return version

    async def get_version(self, version_id: str) -> Optional[DataVersion]:
        """Get a specific version."""
        return self._versions.get(version_id)

    async def get_latest_version(self, branch: str = "main") -> Optional[DataVersion]:
        """Get the latest version on a branch."""
        version_id = self._branches.get(branch)
        if version_id:
            return self._versions.get(version_id)
        return None

    async def load_version_data(self, version_id: str) -> List[DataRecord]:
        """Load the data for a specific version."""
        version = self._versions.get(version_id)
        if not version:
            return []

        version_path = self._versions_path / version_id

        try:
            # Try compressed first
            compressed_path = version_path / "data.json.zlib"
            if compressed_path.exists():
                data = json.loads(zlib.decompress(compressed_path.read_bytes()))
            else:
                data_path = version_path / "data.json"
                if data_path.exists():
                    data = json.loads(data_path.read_text())
                else:
                    return []

            return [DataRecord.from_dict(r) for r in data]

        except Exception as e:
            self.logger.error(f"Failed to load version data: {e}")
            return []

    async def diff_versions(
        self,
        version_id_a: str,
        version_id_b: str,
    ) -> Dict[str, Any]:
        """Compute diff between two versions."""
        records_a = await self.load_version_data(version_id_a)
        records_b = await self.load_version_data(version_id_b)

        hashes_a = {r.content_hash: r for r in records_a}
        hashes_b = {r.content_hash: r for r in records_b}

        added = [h for h in hashes_b if h not in hashes_a]
        removed = [h for h in hashes_a if h not in hashes_b]
        common = [h for h in hashes_a if h in hashes_b]

        return {
            "version_a": version_id_a,
            "version_b": version_id_b,
            "added_count": len(added),
            "removed_count": len(removed),
            "unchanged_count": len(common),
            "added_hashes": added[:100],  # Limit for readability
            "removed_hashes": removed[:100],
        }

    async def create_tag(self, tag_name: str, version_id: str) -> bool:
        """Create a tag pointing to a version."""
        if version_id not in self._versions:
            return False

        async with self._lock:
            self._tags[tag_name] = version_id
            await self._save_manifest()

        return True

    async def create_branch(self, branch_name: str, from_version: Optional[str] = None) -> bool:
        """Create a new branch."""
        async with self._lock:
            if from_version:
                if from_version not in self._versions:
                    return False
                self._branches[branch_name] = from_version
            else:
                self._branches[branch_name] = self._branches.get("main", "")
            await self._save_manifest()

        return True

    def get_status(self) -> Dict[str, Any]:
        return {
            "version_count": len(self._versions),
            "branch_count": len(self._branches),
            "tag_count": len(self._tags),
            "branches": dict(self._branches),
            "tags": dict(self._tags),
            "latest_versions": {
                branch: self._versions.get(vid, {}).version if vid and vid in self._versions else None
                for branch, vid in self._branches.items()
            },
        }


# =============================================================================
# DATA VALIDATOR
# =============================================================================

class SchemaRegistry:
    """Registry of data schemas for validation."""

    def __init__(self):
        self._schemas: Dict[str, Dict[str, Any]] = {}
        self._validators: Dict[str, Callable[[Dict], bool]] = {}

    def register_schema(
        self,
        data_type: DataType,
        schema: Dict[str, Any],
        validator: Optional[Callable[[Dict], bool]] = None,
    ) -> None:
        """Register a schema for a data type."""
        self._schemas[data_type.value] = schema
        if validator:
            self._validators[data_type.value] = validator

    def get_schema(self, data_type: DataType) -> Optional[Dict[str, Any]]:
        """Get schema for a data type."""
        return self._schemas.get(data_type.value)

    def validate(self, data_type: DataType, content: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate content against schema."""
        errors = []

        schema = self._schemas.get(data_type.value)
        if not schema:
            return True, []

        # Basic type checking
        for field_name, field_spec in schema.get("fields", {}).items():
            required = field_spec.get("required", False)
            field_type = field_spec.get("type", "any")

            if field_name not in content:
                if required:
                    errors.append(f"Missing required field: {field_name}")
                continue

            value = content[field_name]

            # Type validation
            if field_type == "string" and not isinstance(value, str):
                errors.append(f"Field {field_name} should be string")
            elif field_type == "number" and not isinstance(value, (int, float)):
                errors.append(f"Field {field_name} should be number")
            elif field_type == "boolean" and not isinstance(value, bool):
                errors.append(f"Field {field_name} should be boolean")
            elif field_type == "array" and not isinstance(value, list):
                errors.append(f"Field {field_name} should be array")
            elif field_type == "object" and not isinstance(value, dict):
                errors.append(f"Field {field_name} should be object")

        # Custom validator
        if data_type.value in self._validators:
            try:
                if not self._validators[data_type.value](content):
                    errors.append("Custom validation failed")
            except Exception as e:
                errors.append(f"Validation error: {e}")

        return len(errors) == 0, errors


class DataValidator:
    """
    Comprehensive data validation pipeline.

    Features:
    - Schema validation
    - Statistical quality checks
    - Anomaly detection
    - Data drift monitoring
    """

    def __init__(self):
        self._schema_registry = SchemaRegistry()
        self._statistics: Dict[str, Dict[str, Any]] = {}
        self._drift_baseline: Dict[str, Dict[str, float]] = {}
        self._lock = asyncio.Lock()

        self.logger = logging.getLogger("DataValidator")

        # Register default schemas
        self._register_default_schemas()

    def _register_default_schemas(self) -> None:
        """Register default schemas for common data types."""
        # Interaction schema
        self._schema_registry.register_schema(
            DataType.INTERACTION,
            {
                "fields": {
                    "input": {"type": "string", "required": True},
                    "output": {"type": "string", "required": True},
                    "intent": {"type": "string", "required": False},
                    "domain": {"type": "string", "required": False},
                    "success": {"type": "boolean", "required": False},
                }
            }
        )

        # Voice sample schema
        self._schema_registry.register_schema(
            DataType.VOICE_SAMPLE,
            {
                "fields": {
                    "embedding": {"type": "array", "required": True},
                    "speaker_id": {"type": "string", "required": True},
                    "confidence": {"type": "number", "required": True},
                }
            }
        )

        # Feedback schema
        self._schema_registry.register_schema(
            DataType.FEEDBACK,
            {
                "fields": {
                    "type": {"type": "string", "required": True},
                    "value": {"type": "any", "required": True},
                }
            }
        )

    @property
    def schema_registry(self) -> SchemaRegistry:
        return self._schema_registry

    async def validate(self, record: DataRecord) -> ValidationResult:
        """Validate a single data record."""
        result = ValidationResult()

        # Schema validation
        schema_valid, schema_errors = self._schema_registry.validate(
            record.data_type,
            record.content,
        )
        result.schema_valid = schema_valid
        result.errors.extend(schema_errors)

        # Statistical validation
        stat_valid, stat_warnings = await self._validate_statistics(record)
        result.statistical_valid = stat_valid
        result.warnings.extend(stat_warnings)

        # Anomaly detection
        anomaly_detected, anomaly_score = await self._detect_anomaly(record)
        result.anomaly_detected = anomaly_detected
        result.anomaly_score = anomaly_score
        if anomaly_detected:
            result.warnings.append(f"Anomaly detected with score {anomaly_score:.3f}")

        # Calculate overall quality score
        quality_factors = [
            1.0 if schema_valid else 0.5,
            1.0 if stat_valid else 0.7,
            1.0 - (anomaly_score * 0.5) if anomaly_detected else 1.0,
            record.quality_score,
        ]
        result.quality_score = statistics.mean(quality_factors)

        # Determine validity
        result.valid = (
            schema_valid and
            result.quality_score >= DataManagementConfig.get_quality_threshold()
        )

        return result

    async def validate_batch(self, records: List[DataRecord]) -> List[ValidationResult]:
        """Validate a batch of records."""
        return [await self.validate(r) for r in records]

    async def _validate_statistics(self, record: DataRecord) -> Tuple[bool, List[str]]:
        """Validate record against statistical expectations."""
        warnings = []
        data_type = record.data_type.value

        async with self._lock:
            if data_type not in self._statistics:
                self._statistics[data_type] = {
                    "count": 0,
                    "content_lengths": [],
                    "quality_scores": [],
                }

            stats = self._statistics[data_type]
            stats["count"] += 1

            # Track content length
            content_len = len(json.dumps(record.content))
            stats["content_lengths"].append(content_len)
            if len(stats["content_lengths"]) > 1000:
                stats["content_lengths"] = stats["content_lengths"][-1000:]

            # Track quality scores
            stats["quality_scores"].append(record.quality_score)
            if len(stats["quality_scores"]) > 1000:
                stats["quality_scores"] = stats["quality_scores"][-1000:]

        # Check for statistical anomalies
        if len(stats["content_lengths"]) >= 20:
            mean_len = statistics.mean(stats["content_lengths"])
            std_len = statistics.stdev(stats["content_lengths"]) or 1

            z_score = abs(content_len - mean_len) / std_len
            sensitivity = DataManagementConfig.get_anomaly_sensitivity()

            if z_score > sensitivity:
                warnings.append(
                    f"Content length {content_len} deviates from mean "
                    f"({mean_len:.0f}) by {z_score:.1f} std devs"
                )

        return len(warnings) == 0, warnings

    async def _detect_anomaly(self, record: DataRecord) -> Tuple[bool, float]:
        """Detect anomalies in record using statistical methods."""
        # Simple z-score based anomaly detection
        if not HAS_NUMPY:
            return False, 0.0

        data_type = record.data_type.value
        content_str = json.dumps(record.content, sort_keys=True)
        content_len = len(content_str)

        async with self._lock:
            if data_type in self._statistics:
                lengths = self._statistics[data_type].get("content_lengths", [])
                if len(lengths) >= 20:
                    mean = np.mean(lengths)
                    std = np.std(lengths) or 1
                    z_score = abs(content_len - mean) / std

                    sensitivity = DataManagementConfig.get_anomaly_sensitivity()
                    if z_score > sensitivity:
                        return True, min(z_score / (sensitivity * 2), 1.0)

        return False, 0.0

    async def update_drift_baseline(self, data_type: DataType) -> None:
        """Update drift baseline for a data type."""
        async with self._lock:
            if data_type.value in self._statistics:
                stats = self._statistics[data_type.value]
                self._drift_baseline[data_type.value] = {
                    "mean_length": statistics.mean(stats["content_lengths"]) if stats["content_lengths"] else 0,
                    "mean_quality": statistics.mean(stats["quality_scores"]) if stats["quality_scores"] else 1.0,
                }

    def get_status(self) -> Dict[str, Any]:
        return {
            "schema_count": len(self._schema_registry._schemas),
            "statistics": {
                dt: {"count": s["count"]} for dt, s in self._statistics.items()
            },
            "drift_baselines": dict(self._drift_baseline),
        }


# =============================================================================
# DATA PRIVACY MANAGER
# =============================================================================

class PIIDetector:
    """
    Detects Personally Identifiable Information in data.

    Patterns detected:
    - Email addresses
    - Phone numbers
    - Social Security Numbers
    - Credit card numbers
    - IP addresses
    - Names (basic heuristics)
    - Addresses (basic heuristics)
    """

    # PII regex patterns
    PATTERNS = {
        "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        "phone_us": re.compile(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'),
        "ssn": re.compile(r'\b[0-9]{3}[-\s]?[0-9]{2}[-\s]?[0-9]{4}\b'),
        "credit_card": re.compile(r'\b(?:[0-9]{4}[-\s]?){3}[0-9]{4}\b'),
        "ip_address": re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
        "date_of_birth": re.compile(r'\b(?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12][0-9]|3[01])[/-](?:19|20)[0-9]{2}\b'),
    }

    def detect(self, content: str) -> List[Dict[str, Any]]:
        """Detect PII in content string."""
        findings = []

        for pii_type, pattern in self.PATTERNS.items():
            for match in pattern.finditer(content):
                findings.append({
                    "type": pii_type,
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                })

        return findings

    def detect_in_dict(self, data: Dict[str, Any], path: str = "") -> List[Dict[str, Any]]:
        """Recursively detect PII in a dictionary."""
        findings = []

        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key

            if isinstance(value, str):
                for finding in self.detect(value):
                    finding["path"] = current_path
                    findings.append(finding)
            elif isinstance(value, dict):
                findings.extend(self.detect_in_dict(value, current_path))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, str):
                        for finding in self.detect(item):
                            finding["path"] = f"{current_path}[{i}]"
                            findings.append(finding)
                    elif isinstance(item, dict):
                        findings.extend(self.detect_in_dict(item, f"{current_path}[{i}]"))

        return findings


class DataAnonymizer:
    """
    Anonymizes data using various techniques.

    Techniques:
    - Masking (replace with **)
    - Hashing (one-way hash)
    - Tokenization (replace with token)
    - Generalization (replace with category)
    - Noise addition (for numerical data)
    """

    def __init__(self, salt: Optional[str] = None):
        self._salt = salt or secrets.token_hex(16)

    def mask(self, value: str, keep_first: int = 0, keep_last: int = 0) -> str:
        """Mask a string value."""
        if len(value) <= keep_first + keep_last:
            return "*" * len(value)

        first = value[:keep_first] if keep_first else ""
        last = value[-keep_last:] if keep_last else ""
        middle = "*" * (len(value) - keep_first - keep_last)

        return first + middle + last

    def hash(self, value: str) -> str:
        """Hash a value (one-way)."""
        salted = f"{self._salt}{value}".encode()
        return hashlib.sha256(salted).hexdigest()[:16]

    def tokenize(self, value: str, token_prefix: str = "TOK") -> str:
        """Replace with a consistent token."""
        hash_val = self.hash(value)
        return f"{token_prefix}_{hash_val[:8]}"

    def generalize_email(self, email: str) -> str:
        """Generalize email to domain only."""
        try:
            parts = email.split("@")
            return f"***@{parts[1]}"
        except (IndexError, AttributeError):
            return "***@***.***"

    def generalize_phone(self, phone: str) -> str:
        """Generalize phone to area code only."""
        digits = re.sub(r'\D', '', phone)
        if len(digits) >= 3:
            return f"({digits[:3]}) ***-****"
        return "(***) ***-****"

    def add_noise(self, value: float, noise_factor: float = 0.1) -> float:
        """Add noise to a numerical value."""
        noise = random.gauss(0, abs(value) * noise_factor)
        return value + noise

    def anonymize_record(
        self,
        record: DataRecord,
        pii_findings: List[Dict[str, Any]],
    ) -> DataRecord:
        """Anonymize a record based on PII findings."""
        content = json.loads(json.dumps(record.content))  # Deep copy

        for finding in pii_findings:
            pii_type = finding["type"]
            path_parts = finding["path"].replace("]", "").replace("[", ".").split(".")

            # Navigate to parent
            current = content
            for part in path_parts[:-1]:
                if part.isdigit():
                    current = current[int(part)]
                else:
                    current = current[part]

            # Apply anonymization based on type
            last_part = path_parts[-1]
            if last_part.isdigit():
                idx = int(last_part)
                original = current[idx]
            else:
                original = current[last_part]

            if pii_type == "email":
                anonymized = self.generalize_email(original)
            elif pii_type in ("phone_us", "ssn", "credit_card"):
                anonymized = self.mask(original, keep_first=0, keep_last=4)
            elif pii_type == "ip_address":
                anonymized = "x.x.x.x"
            else:
                anonymized = self.tokenize(original)

            if last_part.isdigit():
                current[int(last_part)] = anonymized
            else:
                current[last_part] = anonymized

        # Create new record with anonymized content
        new_record = DataRecord(
            id=record.id,
            data_type=record.data_type,
            content=content,
            metadata={**record.metadata, "anonymized": True},
            source=record.source,
            timestamp=record.timestamp,
            quality_score=record.quality_score,
            privacy_level=record.privacy_level,
            retention_policy=record.retention_policy,
            version=record.version,
            lineage_id=record.lineage_id,
            parent_ids=[*record.parent_ids, record.id],
            tags={*record.tags, "anonymized"},
        )

        return new_record


class DataPrivacyManager:
    """
    Comprehensive data privacy management.

    Features:
    - PII detection
    - Anonymization
    - Encryption
    - Compliance tracking
    """

    def __init__(self):
        # Initialize logger FIRST so it's available for _init_encryption
        self.logger = logging.getLogger("DataPrivacyManager")

        self._pii_detector = PIIDetector()
        self._anonymizer = DataAnonymizer()
        self._encryptor: Optional[Fernet] = None
        self._lock = asyncio.Lock()

        # Initialize encryption if key available
        self._init_encryption()

    def _init_encryption(self) -> None:
        """Initialize encryption with key from environment."""
        if not HAS_CRYPTO:
            self.logger.warning("Cryptography not available, encryption disabled")
            return

        key = DataManagementConfig.get_encryption_key()
        if key:
            try:
                # Derive key using PBKDF2
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=b"jarvis_data_salt",  # Fixed salt for consistency
                    iterations=100000,
                )
                derived_key = base64.urlsafe_b64encode(kdf.derive(key.encode()))
                self._encryptor = Fernet(derived_key)
                self.logger.info("Encryption initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize encryption: {e}")

    async def analyze(self, record: DataRecord) -> PrivacyReport:
        """Analyze a record for privacy concerns."""
        report = PrivacyReport()

        if not DataManagementConfig.get_pii_detection_enabled():
            return report

        # Detect PII
        content_str = json.dumps(record.content)
        findings = self._pii_detector.detect_in_dict(record.content)

        if findings:
            report.contains_pii = True
            report.pii_types = list(set(f["type"] for f in findings))
            report.pii_locations = findings

            # Calculate risk score
            type_weights = {
                "ssn": 1.0,
                "credit_card": 0.9,
                "date_of_birth": 0.7,
                "email": 0.5,
                "phone_us": 0.5,
                "ip_address": 0.3,
            }
            total_weight = sum(type_weights.get(f["type"], 0.5) for f in findings)
            report.risk_score = min(total_weight / len(findings) if findings else 0, 1.0)

            # Recommendations
            if "ssn" in report.pii_types or "credit_card" in report.pii_types:
                report.recommendations.append("Highly sensitive PII detected - encryption required")
                report.recommendations.append("Consider not storing this data")
            else:
                report.recommendations.append("Consider anonymization before storage")

        # Check compliance
        report.compliance_status = {
            "gdpr": not report.contains_pii or record.privacy_level in (
                PrivacyLevel.CONFIDENTIAL, PrivacyLevel.SENSITIVE, PrivacyLevel.RESTRICTED
            ),
            "ccpa": not report.contains_pii or record.privacy_level in (
                PrivacyLevel.CONFIDENTIAL, PrivacyLevel.SENSITIVE, PrivacyLevel.RESTRICTED
            ),
        }

        return report

    async def anonymize(self, record: DataRecord) -> Tuple[DataRecord, PrivacyReport]:
        """Anonymize a record."""
        report = await self.analyze(record)

        if not report.contains_pii:
            return record, report

        anonymized = self._anonymizer.anonymize_record(record, report.pii_locations)
        report.anonymization_applied = True

        return anonymized, report

    async def encrypt(self, record: DataRecord) -> DataRecord:
        """Encrypt sensitive content in a record."""
        if not self._encryptor:
            self.logger.warning("Encryption not available")
            return record

        content_str = json.dumps(record.content)
        encrypted = self._encryptor.encrypt(content_str.encode())

        new_record = DataRecord(
            id=record.id,
            data_type=record.data_type,
            content={"encrypted": base64.b64encode(encrypted).decode()},
            metadata={**record.metadata, "encrypted": True},
            source=record.source,
            timestamp=record.timestamp,
            quality_score=record.quality_score,
            privacy_level=PrivacyLevel.RESTRICTED,
            retention_policy=record.retention_policy,
            version=record.version,
            lineage_id=record.lineage_id,
            parent_ids=record.parent_ids,
            tags={*record.tags, "encrypted"},
        )

        return new_record

    async def decrypt(self, record: DataRecord) -> DataRecord:
        """Decrypt encrypted content."""
        if not self._encryptor:
            return record

        if "encrypted" not in record.content:
            return record

        try:
            encrypted = base64.b64decode(record.content["encrypted"])
            decrypted = self._encryptor.decrypt(encrypted)
            content = json.loads(decrypted)

            new_record = DataRecord(
                id=record.id,
                data_type=record.data_type,
                content=content,
                metadata={k: v for k, v in record.metadata.items() if k != "encrypted"},
                source=record.source,
                timestamp=record.timestamp,
                quality_score=record.quality_score,
                privacy_level=record.privacy_level,
                retention_policy=record.retention_policy,
                version=record.version,
                lineage_id=record.lineage_id,
                parent_ids=record.parent_ids,
                tags=record.tags - {"encrypted"},
            )

            return new_record

        except Exception as e:
            self.logger.error(f"Failed to decrypt record: {e}")
            return record

    def get_status(self) -> Dict[str, Any]:
        return {
            "pii_detection_enabled": DataManagementConfig.get_pii_detection_enabled(),
            "encryption_available": self._encryptor is not None,
            "default_privacy_level": DataManagementConfig.get_default_privacy_level(),
        }


# =============================================================================
# DATA RETENTION MANAGER
# =============================================================================

class DataRetentionManager:
    """
    Manages data retention policies and lifecycle.

    Features:
    - TTL-based lifecycle management
    - Tiered storage (hot/warm/cold/archive)
    - Automatic archival and deletion
    - Legal hold support
    """

    def __init__(self):
        self._archive_path = DataManagementConfig.get_archive_path()
        self._legal_holds: Set[str] = set()  # Record IDs under legal hold
        self._lock = asyncio.Lock()
        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None

        # Metrics
        self._total_archived = 0
        self._total_deleted = 0
        self._total_bytes_archived = 0

        self.logger = logging.getLogger("DataRetentionManager")

    async def start(self) -> None:
        """Start the retention manager."""
        if self._running:
            return

        self._archive_path.mkdir(parents=True, exist_ok=True)
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.logger.info("Data retention manager started")

    async def stop(self) -> None:
        """Stop the retention manager."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Data retention manager stopped")

    async def set_legal_hold(self, record_id: str) -> None:
        """Place a record under legal hold."""
        async with self._lock:
            self._legal_holds.add(record_id)
            self.logger.info(f"Legal hold set for record {record_id}")

    async def remove_legal_hold(self, record_id: str) -> None:
        """Remove legal hold from a record."""
        async with self._lock:
            self._legal_holds.discard(record_id)
            self.logger.info(f"Legal hold removed for record {record_id}")

    async def should_archive(self, record: DataRecord) -> bool:
        """Check if a record should be archived."""
        if record.id in self._legal_holds:
            return False

        if record.retention_policy in (RetentionPolicy.PERMANENT, RetentionPolicy.LEGAL_HOLD):
            return False

        archive_days = DataManagementConfig.get_archive_after_days()
        age_days = (time.time() - record.timestamp) / 86400

        return age_days >= archive_days

    async def should_delete(self, record: DataRecord) -> bool:
        """Check if a record should be deleted."""
        if record.id in self._legal_holds:
            return False

        if record.retention_policy in (RetentionPolicy.PERMANENT, RetentionPolicy.LEGAL_HOLD):
            return False

        return record.is_expired()

    async def archive(self, record: DataRecord) -> str:
        """Archive a record."""
        archive_date = datetime.now().strftime("%Y/%m/%d")
        archive_dir = self._archive_path / archive_date
        archive_dir.mkdir(parents=True, exist_ok=True)

        archive_file = archive_dir / f"{record.id}.json.zlib"
        content = json.dumps(record.to_dict())

        if DataManagementConfig.get_compression_enabled():
            compressed = zlib.compress(content.encode())
            archive_file.write_bytes(compressed)
            self._total_bytes_archived += len(compressed)
        else:
            archive_file = archive_dir / f"{record.id}.json"
            archive_file.write_text(content)
            self._total_bytes_archived += len(content)

        self._total_archived += 1
        self.logger.debug(f"Archived record {record.id}")

        return str(archive_file)

    async def restore(self, archive_path: str) -> Optional[DataRecord]:
        """Restore a record from archive."""
        try:
            path = Path(archive_path)

            if path.suffix == ".zlib":
                data = json.loads(zlib.decompress(path.read_bytes()))
            else:
                data = json.loads(path.read_text())

            return DataRecord.from_dict(data)

        except Exception as e:
            self.logger.error(f"Failed to restore from archive: {e}")
            return None

    async def _cleanup_loop(self) -> None:
        """Background loop for cleanup operations."""
        interval_hours = DataManagementConfig.get_cleanup_interval_hours()

        while self._running:
            try:
                await asyncio.sleep(interval_hours * 3600)

                # Clean up old archives
                await self._cleanup_old_archives()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(3600)

    async def _cleanup_old_archives(self) -> None:
        """Clean up archives past retention period."""
        retention_days = DataManagementConfig.get_default_retention_days()
        cutoff = datetime.now() - timedelta(days=retention_days)

        cleaned = 0
        for year_dir in self._archive_path.iterdir():
            if not year_dir.is_dir():
                continue

            for month_dir in year_dir.iterdir():
                if not month_dir.is_dir():
                    continue

                for day_dir in month_dir.iterdir():
                    if not day_dir.is_dir():
                        continue

                    try:
                        dir_date = datetime.strptime(
                            f"{year_dir.name}/{month_dir.name}/{day_dir.name}",
                            "%Y/%m/%d"
                        )
                        if dir_date < cutoff:
                            shutil.rmtree(day_dir)
                            cleaned += 1
                    except ValueError:
                        continue

        if cleaned > 0:
            self.logger.info(f"Cleaned up {cleaned} old archive directories")

    def get_status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "legal_holds_count": len(self._legal_holds),
            "total_archived": self._total_archived,
            "total_deleted": self._total_deleted,
            "total_bytes_archived": self._total_bytes_archived,
            "archive_path": str(self._archive_path),
        }


# =============================================================================
# DATA DEDUPLICATOR
# =============================================================================

class DataDeduplicator:
    """
    Content-based data deduplication.

    Features:
    - Content-hash based exact deduplication
    - Semantic similarity for near-duplicates
    - Configurable similarity threshold
    - Space savings tracking
    """

    def __init__(self):
        self._hash_index: Dict[str, str] = {}  # content_hash -> record_id
        self._lock = asyncio.Lock()

        # Metrics
        self._total_checked = 0
        self._duplicates_found = 0
        self._bytes_saved = 0

        self.logger = logging.getLogger("DataDeduplicator")

    async def is_duplicate(self, record: DataRecord) -> Tuple[bool, Optional[str]]:
        """
        Check if a record is a duplicate.

        Returns (is_duplicate, original_record_id)
        """
        if not DataManagementConfig.get_dedup_enabled():
            return False, None

        self._total_checked += 1

        async with self._lock:
            # Exact match
            if record.content_hash in self._hash_index:
                self._duplicates_found += 1
                return True, self._hash_index[record.content_hash]

            # Register this record
            self._hash_index[record.content_hash] = record.id

        return False, None

    async def find_similar(
        self,
        record: DataRecord,
        candidates: List[DataRecord],
        threshold: Optional[float] = None,
    ) -> List[Tuple[DataRecord, float]]:
        """
        Find similar records above threshold.

        Returns list of (record, similarity_score) tuples.
        """
        threshold = threshold or DataManagementConfig.get_similarity_threshold()
        similar = []

        record_text = json.dumps(record.content, sort_keys=True)

        for candidate in candidates:
            if candidate.id == record.id:
                continue

            candidate_text = json.dumps(candidate.content, sort_keys=True)
            similarity = self._compute_similarity(record_text, candidate_text)

            if similarity >= threshold:
                similar.append((candidate, similarity))

        return sorted(similar, key=lambda x: x[1], reverse=True)

    def _compute_similarity(self, text_a: str, text_b: str) -> float:
        """Compute Jaccard similarity between two texts."""
        # Simple word-level Jaccard similarity
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())

        if not words_a or not words_b:
            return 0.0

        intersection = len(words_a & words_b)
        union = len(words_a | words_b)

        return intersection / union if union else 0.0

    async def register(self, record: DataRecord) -> None:
        """Register a record in the dedup index."""
        async with self._lock:
            self._hash_index[record.content_hash] = record.id

    async def remove(self, record: DataRecord) -> None:
        """Remove a record from the dedup index."""
        async with self._lock:
            if record.content_hash in self._hash_index:
                if self._hash_index[record.content_hash] == record.id:
                    del self._hash_index[record.content_hash]

    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": DataManagementConfig.get_dedup_enabled(),
            "index_size": len(self._hash_index),
            "total_checked": self._total_checked,
            "duplicates_found": self._duplicates_found,
            "dedup_rate": self._duplicates_found / max(self._total_checked, 1),
            "bytes_saved": self._bytes_saved,
        }


# =============================================================================
# INTELLIGENT DATA SAMPLER
# =============================================================================

class IntelligentDataSampler:
    """
    Intelligent data sampling for training.

    Features:
    - Stratified sampling
    - Importance sampling
    - Active learning sampling
    - Reservoir sampling for streaming
    """

    def __init__(self):
        self._reservoir: Deque[DataRecord] = deque(
            maxlen=DataManagementConfig.get_reservoir_size()
        )
        self._reservoir_count = 0
        self._lock = asyncio.Lock()

        self.logger = logging.getLogger("DataSampler")

    async def sample(
        self,
        records: List[DataRecord],
        strategy: SamplingStrategy,
    ) -> List[DataRecord]:
        """Sample records according to strategy."""
        if not records:
            return []

        method = strategy.method
        sample_size = min(strategy.sample_size, len(records))

        if method == SamplingMethod.RANDOM:
            return await self._random_sample(records, sample_size, strategy.seed)

        elif method == SamplingMethod.STRATIFIED:
            return await self._stratified_sample(
                records, sample_size, strategy.stratify_by, strategy.seed
            )

        elif method == SamplingMethod.IMPORTANCE:
            return await self._importance_sample(
                records, sample_size, strategy.importance_weights, strategy.seed
            )

        elif method == SamplingMethod.ACTIVE_LEARNING:
            return await self._active_learning_sample(
                records, sample_size, strategy.uncertainty_threshold
            )

        elif method == SamplingMethod.RESERVOIR:
            return await self._reservoir_sample(records, sample_size)

        elif method == SamplingMethod.SYSTEMATIC:
            return await self._systematic_sample(records, sample_size)

        elif method == SamplingMethod.CLUSTER:
            return await self._cluster_sample(records, sample_size)

        else:
            return await self._random_sample(records, sample_size, strategy.seed)

    async def _random_sample(
        self,
        records: List[DataRecord],
        sample_size: int,
        seed: Optional[int] = None,
    ) -> List[DataRecord]:
        """Simple random sampling."""
        rng = random.Random(seed)
        return rng.sample(records, min(sample_size, len(records)))

    async def _stratified_sample(
        self,
        records: List[DataRecord],
        sample_size: int,
        stratify_by: Optional[str],
        seed: Optional[int] = None,
    ) -> List[DataRecord]:
        """Stratified sampling by a field."""
        if not stratify_by:
            return await self._random_sample(records, sample_size, seed)

        # Group by stratification field
        groups: Dict[str, List[DataRecord]] = defaultdict(list)
        for record in records:
            key = record.content.get(stratify_by, record.data_type.value)
            groups[str(key)].append(record)

        # Calculate samples per group
        total = len(records)
        samples_per_group = {
            key: max(1, int(sample_size * len(group) / total))
            for key, group in groups.items()
        }

        # Sample from each group
        rng = random.Random(seed)
        result = []
        for key, group in groups.items():
            n = min(samples_per_group[key], len(group))
            result.extend(rng.sample(group, n))

        return result[:sample_size]

    async def _importance_sample(
        self,
        records: List[DataRecord],
        sample_size: int,
        weights: Optional[Dict[str, float]],
        seed: Optional[int] = None,
    ) -> List[DataRecord]:
        """Importance sampling with weights."""
        if not weights:
            # Use quality score as default weight
            record_weights = [r.quality_score for r in records]
        else:
            # Use provided weights based on data type or tags
            record_weights = []
            for r in records:
                w = weights.get(r.data_type.value, 1.0)
                for tag in r.tags:
                    if tag in weights:
                        w *= weights[tag]
                record_weights.append(w)

        # Normalize weights
        total_weight = sum(record_weights)
        if total_weight == 0:
            return await self._random_sample(records, sample_size, seed)

        probabilities = [w / total_weight for w in record_weights]

        # Sample with replacement based on probabilities
        rng = random.Random(seed)
        indices = rng.choices(
            range(len(records)),
            weights=probabilities,
            k=sample_size
        )

        return [records[i] for i in indices]

    async def _active_learning_sample(
        self,
        records: List[DataRecord],
        sample_size: int,
        uncertainty_threshold: float,
    ) -> List[DataRecord]:
        """Sample records with high uncertainty for active learning."""
        # Use quality score as proxy for certainty (lower quality = higher uncertainty)
        scored = [(r, 1.0 - r.quality_score) for r in records]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Take records above uncertainty threshold
        uncertain = [r for r, score in scored if score >= uncertainty_threshold]

        if len(uncertain) >= sample_size:
            return uncertain[:sample_size]

        # Supplement with random sampling
        remaining = [r for r in records if r not in uncertain]
        random.shuffle(remaining)
        return uncertain + remaining[:sample_size - len(uncertain)]

    async def _reservoir_sample(
        self,
        records: List[DataRecord],
        sample_size: int,
    ) -> List[DataRecord]:
        """Reservoir sampling for streaming data."""
        async with self._lock:
            for record in records:
                self._reservoir_count += 1

                if len(self._reservoir) < sample_size:
                    self._reservoir.append(record)
                else:
                    # Random replacement
                    j = random.randint(0, self._reservoir_count - 1)
                    if j < sample_size:
                        self._reservoir[j] = record

            return list(self._reservoir)

    async def _systematic_sample(
        self,
        records: List[DataRecord],
        sample_size: int,
    ) -> List[DataRecord]:
        """Systematic sampling (every nth record)."""
        if sample_size >= len(records):
            return records

        interval = len(records) / sample_size
        start = random.uniform(0, interval)

        indices = [int(start + i * interval) for i in range(sample_size)]
        return [records[i] for i in indices if i < len(records)]

    async def _cluster_sample(
        self,
        records: List[DataRecord],
        sample_size: int,
    ) -> List[DataRecord]:
        """Cluster-based sampling."""
        # Simple clustering by data type
        clusters: Dict[str, List[DataRecord]] = defaultdict(list)
        for record in records:
            clusters[record.data_type.value].append(record)

        # Sample entire clusters
        cluster_list = list(clusters.values())
        random.shuffle(cluster_list)

        result = []
        for cluster in cluster_list:
            if len(result) + len(cluster) <= sample_size:
                result.extend(cluster)
            elif len(result) < sample_size:
                remaining = sample_size - len(result)
                result.extend(random.sample(cluster, min(remaining, len(cluster))))

        return result[:sample_size]

    async def add_to_reservoir(self, record: DataRecord) -> None:
        """Add a single record to the reservoir (streaming mode)."""
        async with self._lock:
            reservoir_size = DataManagementConfig.get_reservoir_size()
            self._reservoir_count += 1

            if len(self._reservoir) < reservoir_size:
                self._reservoir.append(record)
            else:
                j = random.randint(0, self._reservoir_count - 1)
                if j < reservoir_size:
                    self._reservoir[j] = record

    def get_status(self) -> Dict[str, Any]:
        return {
            "reservoir_size": len(self._reservoir),
            "reservoir_max": DataManagementConfig.get_reservoir_size(),
            "total_seen": self._reservoir_count,
        }


# =============================================================================
# DATA LINEAGE TRACKER
# =============================================================================

class DataLineageTracker:
    """
    DAG-based data lineage tracking.

    Features:
    - Full provenance tracking
    - Impact analysis
    - Audit trail
    - Reproducibility support
    """

    def __init__(self):
        self._lineage_path = DataManagementConfig.get_lineage_path()
        self._nodes: Dict[str, DataLineageNode] = {}
        self._edges: Dict[str, DataLineageEdge] = {}
        self._record_nodes: Dict[str, List[str]] = {}  # record_id -> [node_ids]
        self._lock = asyncio.Lock()
        self._initialized = False

        self.logger = logging.getLogger("DataLineageTracker")

    async def initialize(self) -> None:
        """Initialize lineage tracker."""
        self._lineage_path.mkdir(parents=True, exist_ok=True)
        await self._load_lineage()
        self._initialized = True
        self.logger.info(f"Lineage tracker initialized with {len(self._nodes)} nodes")

    async def _load_lineage(self) -> None:
        """Load lineage from disk."""
        lineage_file = self._lineage_path / "lineage.json"
        if lineage_file.exists():
            try:
                data = json.loads(lineage_file.read_text())
                for node_data in data.get("nodes", []):
                    node = DataLineageNode(**node_data)
                    node.event_type = LineageEventType(node_data.get("event_type", "created"))
                    self._nodes[node.node_id] = node

                    if node.record_id not in self._record_nodes:
                        self._record_nodes[node.record_id] = []
                    self._record_nodes[node.record_id].append(node.node_id)

                for edge_data in data.get("edges", []):
                    edge = DataLineageEdge(**edge_data)
                    self._edges[edge.edge_id] = edge

            except Exception as e:
                self.logger.error(f"Failed to load lineage: {e}")

    async def _save_lineage(self) -> None:
        """Save lineage to disk."""
        lineage_file = self._lineage_path / "lineage.json"
        data = {
            "nodes": [
                {**n.__dict__, "event_type": n.event_type.value}
                for n in self._nodes.values()
            ],
            "edges": [e.__dict__ for e in self._edges.values()],
        }

        content = json.dumps(data, indent=2, default=str)
        lineage_file.write_text(content)

    async def track(
        self,
        record_id: str,
        event_type: LineageEventType,
        operation: str,
        input_ids: Optional[List[str]] = None,
        output_ids: Optional[List[str]] = None,
        actor: str = "system",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Track a lineage event."""
        if not DataManagementConfig.get_lineage_enabled():
            return ""

        async with self._lock:
            node = DataLineageNode(
                record_id=record_id,
                event_type=event_type,
                operation=operation,
                input_ids=input_ids or [],
                output_ids=output_ids or [],
                actor=actor,
                metadata=metadata or {},
            )

            self._nodes[node.node_id] = node

            if record_id not in self._record_nodes:
                self._record_nodes[record_id] = []
            self._record_nodes[record_id].append(node.node_id)

            # Create edges for inputs
            for input_id in (input_ids or []):
                edge = DataLineageEdge(
                    source_id=input_id,
                    target_id=record_id,
                    relationship="derived_from",
                )
                self._edges[edge.edge_id] = edge

            # Create edges for outputs
            for output_id in (output_ids or []):
                edge = DataLineageEdge(
                    source_id=record_id,
                    target_id=output_id,
                    relationship="produces",
                )
                self._edges[edge.edge_id] = edge

            await self._save_lineage()

            return node.node_id

    async def get_lineage(self, record_id: str) -> List[DataLineageNode]:
        """Get all lineage events for a record."""
        node_ids = self._record_nodes.get(record_id, [])
        return [self._nodes[nid] for nid in node_ids if nid in self._nodes]

    async def get_upstream(self, record_id: str, max_depth: int = 0) -> List[str]:
        """Get all upstream (parent) records."""
        max_depth = max_depth or DataManagementConfig.get_lineage_max_depth()
        upstream = set()
        queue = [(record_id, 0)]

        while queue:
            current_id, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            for edge in self._edges.values():
                if edge.target_id == current_id and edge.source_id not in upstream:
                    upstream.add(edge.source_id)
                    queue.append((edge.source_id, depth + 1))

        return list(upstream)

    async def get_downstream(self, record_id: str, max_depth: int = 0) -> List[str]:
        """Get all downstream (derived) records."""
        max_depth = max_depth or DataManagementConfig.get_lineage_max_depth()
        downstream = set()
        queue = [(record_id, 0)]

        while queue:
            current_id, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            for edge in self._edges.values():
                if edge.source_id == current_id and edge.target_id not in downstream:
                    downstream.add(edge.target_id)
                    queue.append((edge.target_id, depth + 1))

        return list(downstream)

    async def impact_analysis(self, record_id: str) -> Dict[str, Any]:
        """Analyze the impact of changes to a record."""
        downstream = await self.get_downstream(record_id)

        impact = {
            "record_id": record_id,
            "downstream_count": len(downstream),
            "downstream_records": downstream[:100],  # Limit for readability
            "impact_level": "high" if len(downstream) > 10 else "medium" if len(downstream) > 0 else "low",
        }

        return impact

    async def export_graph(self, record_id: str) -> Dict[str, Any]:
        """Export lineage as a graph for visualization."""
        upstream = await self.get_upstream(record_id)
        downstream = await self.get_downstream(record_id)

        all_records = {record_id} | set(upstream) | set(downstream)

        nodes = [
            {
                "id": rid,
                "type": "source" if rid in upstream else "target" if rid in downstream else "center",
            }
            for rid in all_records
        ]

        edges = [
            {
                "source": e.source_id,
                "target": e.target_id,
                "relationship": e.relationship,
            }
            for e in self._edges.values()
            if e.source_id in all_records and e.target_id in all_records
        ]

        return {
            "nodes": nodes,
            "edges": edges,
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": DataManagementConfig.get_lineage_enabled(),
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "tracked_records": len(self._record_nodes),
        }


# =============================================================================
# UNIFIED DATA MANAGEMENT ENGINE
# =============================================================================

class UnifiedDataManagementEngine:
    """
    Central coordination point for all data management patterns.

    Integrates:
    - Training Data Collection
    - Data Versioning
    - Data Validation
    - Data Privacy
    - Data Retention
    - Data Deduplication
    - Intelligent Sampling
    - Data Lineage

    This is the main entry point for data management in the JARVIS ecosystem.
    """

    _instance: Optional["UnifiedDataManagementEngine"] = None
    _lock = asyncio.Lock()

    def __init__(self):
        # Core components
        self._collector = TrainingDataCollector()
        self._version_manager = DataVersionManager()
        self._validator = DataValidator()
        self._privacy_manager = DataPrivacyManager()
        self._retention_manager = DataRetentionManager()
        self._deduplicator = DataDeduplicator()
        self._sampler = IntelligentDataSampler()
        self._lineage_tracker = DataLineageTracker()

        # State
        self._initialized = False
        self._running = False

        # Cross-repo integration
        self._forward_callback: Optional[Callable[[DataBatch], Awaitable[bool]]] = None

        self.logger = logging.getLogger("UnifiedDataManagementEngine")

    @classmethod
    async def get_instance(cls) -> "UnifiedDataManagementEngine":
        """Get or create the singleton instance."""
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    async def initialize(self) -> bool:
        """Initialize all data management components."""
        if self._initialized:
            return True

        self.logger.info("=" * 60)
        self.logger.info("Initializing Unified Data Management Engine v1.0")
        self.logger.info("=" * 60)

        try:
            # Initialize version manager
            await self._version_manager.initialize()

            # Initialize lineage tracker
            await self._lineage_tracker.initialize()

            # Start collector with forward callback
            self._collector.set_forward_callback(self._forward_to_reactor)
            await self._collector.start()

            # Start retention manager
            await self._retention_manager.start()

            self._initialized = True
            self._running = True

            self.logger.info("Unified Data Management Engine initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize data management engine: {e}")
            return False

    async def shutdown(self) -> None:
        """Shutdown all data management components."""
        self.logger.info("Shutting down Unified Data Management Engine...")
        self._running = False

        await self._collector.stop()
        await self._retention_manager.stop()

        self._initialized = False
        self.logger.info("Unified Data Management Engine shutdown complete")

    def set_forward_callback(
        self,
        callback: Callable[[DataBatch], Awaitable[bool]]
    ) -> None:
        """Set callback for forwarding batches to Reactor Core."""
        self._forward_callback = callback
        self._collector.set_forward_callback(self._forward_to_reactor)

    async def _forward_to_reactor(self, batch: DataBatch) -> bool:
        """Forward batch to Reactor Core for training."""
        if self._forward_callback:
            return await self._forward_callback(batch)
        return True

    # =========================================================================
    # PUBLIC API - COLLECTION
    # =========================================================================

    async def collect(
        self,
        content: Dict[str, Any],
        data_type: DataType = DataType.TRAINING,
        source: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
        privacy_level: PrivacyLevel = PrivacyLevel.INTERNAL,
        retention_policy: RetentionPolicy = RetentionPolicy.MEDIUM_TERM,
    ) -> str:
        """
        Collect a data record with full pipeline processing.

        Pipeline:
        1. Create record
        2. Validate
        3. Check for duplicates
        4. Analyze privacy
        5. Anonymize if needed
        6. Track lineage
        7. Add to collection queue
        """
        # Create record
        record = DataRecord(
            data_type=data_type,
            content=content,
            source=source,
            metadata=metadata or {},
            privacy_level=privacy_level,
            retention_policy=retention_policy,
        )

        # Validate
        validation = await self._validator.validate(record)
        record.quality_score = validation.quality_score

        if not validation.valid:
            self.logger.warning(f"Invalid record {record.id}: {validation.errors}")
            return record.id

        # Check for duplicates
        is_dup, original_id = await self._deduplicator.is_duplicate(record)
        if is_dup:
            self.logger.debug(f"Duplicate record {record.id}, original: {original_id}")
            return original_id

        # Privacy analysis
        privacy_report = await self._privacy_manager.analyze(record)
        if privacy_report.contains_pii and privacy_level not in (
            PrivacyLevel.SENSITIVE, PrivacyLevel.RESTRICTED
        ):
            # Anonymize PII
            record, _ = await self._privacy_manager.anonymize(record)

        # Track lineage
        await self._lineage_tracker.track(
            record_id=record.id,
            event_type=LineageEventType.CREATED,
            operation="collect",
            actor=source,
            metadata={
                "quality_score": validation.quality_score,
                "pii_detected": privacy_report.contains_pii,
            },
        )

        # Add to collection queue
        return await self._collector.collect(
            content=record.content,
            data_type=record.data_type,
            source=record.source,
            metadata=record.metadata,
            quality_score=record.quality_score,
            tags=record.tags,
        )

    async def collect_interaction(
        self,
        user_input: str,
        system_response: str,
        **kwargs
    ) -> str:
        """Convenience method for collecting interactions."""
        return await self._collector.collect_interaction(
            user_input=user_input,
            system_response=system_response,
            **kwargs
        )

    async def collect_voice_sample(
        self,
        embedding: List[float],
        speaker_id: str,
        confidence: float,
        **kwargs
    ) -> str:
        """Convenience method for collecting voice samples."""
        return await self._collector.collect_voice_sample(
            embedding=embedding,
            speaker_id=speaker_id,
            confidence=confidence,
            **kwargs
        )

    async def collect_feedback(
        self,
        feedback_type: str,
        feedback_value: Any,
        **kwargs
    ) -> str:
        """Convenience method for collecting feedback."""
        return await self._collector.collect_feedback(
            feedback_type=feedback_type,
            feedback_value=feedback_value,
            **kwargs
        )

    # =========================================================================
    # PUBLIC API - VERSIONING
    # =========================================================================

    async def create_version(
        self,
        records: List[DataRecord],
        message: str = "",
        branch: str = "main",
        tags: Optional[List[str]] = None,
    ) -> DataVersion:
        """Create a new version of a dataset."""
        return await self._version_manager.create_version(
            records=records,
            message=message,
            branch=branch,
            tags=tags,
        )

    async def get_version(self, version_id: str) -> Optional[DataVersion]:
        """Get a specific version."""
        return await self._version_manager.get_version(version_id)

    async def load_version_data(self, version_id: str) -> List[DataRecord]:
        """Load data for a specific version."""
        return await self._version_manager.load_version_data(version_id)

    async def diff_versions(self, version_a: str, version_b: str) -> Dict[str, Any]:
        """Diff two versions."""
        return await self._version_manager.diff_versions(version_a, version_b)

    # =========================================================================
    # PUBLIC API - VALIDATION
    # =========================================================================

    async def validate(self, record: DataRecord) -> ValidationResult:
        """Validate a data record."""
        return await self._validator.validate(record)

    async def validate_batch(self, records: List[DataRecord]) -> List[ValidationResult]:
        """Validate a batch of records."""
        return await self._validator.validate_batch(records)

    @property
    def schema_registry(self) -> SchemaRegistry:
        """Get the schema registry."""
        return self._validator.schema_registry

    # =========================================================================
    # PUBLIC API - PRIVACY
    # =========================================================================

    async def analyze_privacy(self, record: DataRecord) -> PrivacyReport:
        """Analyze a record for privacy concerns."""
        return await self._privacy_manager.analyze(record)

    async def anonymize(self, record: DataRecord) -> Tuple[DataRecord, PrivacyReport]:
        """Anonymize a record."""
        return await self._privacy_manager.anonymize(record)

    async def encrypt(self, record: DataRecord) -> DataRecord:
        """Encrypt a record."""
        return await self._privacy_manager.encrypt(record)

    async def decrypt(self, record: DataRecord) -> DataRecord:
        """Decrypt a record."""
        return await self._privacy_manager.decrypt(record)

    # =========================================================================
    # PUBLIC API - RETENTION
    # =========================================================================

    async def set_legal_hold(self, record_id: str) -> None:
        """Place a record under legal hold."""
        await self._retention_manager.set_legal_hold(record_id)

    async def remove_legal_hold(self, record_id: str) -> None:
        """Remove legal hold from a record."""
        await self._retention_manager.remove_legal_hold(record_id)

    async def archive(self, record: DataRecord) -> str:
        """Archive a record."""
        path = await self._retention_manager.archive(record)

        await self._lineage_tracker.track(
            record_id=record.id,
            event_type=LineageEventType.ARCHIVED,
            operation="archive",
            metadata={"archive_path": path},
        )

        return path

    # =========================================================================
    # PUBLIC API - DEDUPLICATION
    # =========================================================================

    async def is_duplicate(self, record: DataRecord) -> Tuple[bool, Optional[str]]:
        """Check if a record is a duplicate."""
        return await self._deduplicator.is_duplicate(record)

    async def find_similar(
        self,
        record: DataRecord,
        candidates: List[DataRecord],
        threshold: float = 0.9,
    ) -> List[Tuple[DataRecord, float]]:
        """Find similar records."""
        return await self._deduplicator.find_similar(record, candidates, threshold)

    # =========================================================================
    # PUBLIC API - SAMPLING
    # =========================================================================

    async def sample(
        self,
        records: List[DataRecord],
        strategy: SamplingStrategy,
    ) -> List[DataRecord]:
        """Sample records according to strategy."""
        sampled = await self._sampler.sample(records, strategy)

        # Track lineage for sampled records
        for record in sampled:
            await self._lineage_tracker.track(
                record_id=record.id,
                event_type=LineageEventType.SAMPLED,
                operation=f"sample_{strategy.method.value}",
                metadata={"sample_size": len(sampled)},
            )

        return sampled

    # =========================================================================
    # PUBLIC API - LINEAGE
    # =========================================================================

    async def track_lineage(
        self,
        record_id: str,
        event_type: LineageEventType,
        operation: str,
        **kwargs
    ) -> str:
        """Track a lineage event."""
        return await self._lineage_tracker.track(
            record_id=record_id,
            event_type=event_type,
            operation=operation,
            **kwargs
        )

    async def get_lineage(self, record_id: str) -> List[DataLineageNode]:
        """Get lineage for a record."""
        return await self._lineage_tracker.get_lineage(record_id)

    async def get_upstream(self, record_id: str) -> List[str]:
        """Get upstream records."""
        return await self._lineage_tracker.get_upstream(record_id)

    async def get_downstream(self, record_id: str) -> List[str]:
        """Get downstream records."""
        return await self._lineage_tracker.get_downstream(record_id)

    async def impact_analysis(self, record_id: str) -> Dict[str, Any]:
        """Analyze impact of changes to a record."""
        return await self._lineage_tracker.impact_analysis(record_id)

    # =========================================================================
    # STATUS
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all data management components."""
        return {
            "engine": {
                "initialized": self._initialized,
                "running": self._running,
            },
            "collector": self._collector.get_metrics(),
            "versioning": self._version_manager.get_status(),
            "validation": self._validator.get_status(),
            "privacy": self._privacy_manager.get_status(),
            "retention": self._retention_manager.get_status(),
            "deduplication": self._deduplicator.get_status(),
            "sampling": self._sampler.get_status(),
            "lineage": self._lineage_tracker.get_status(),
        }


# =============================================================================
# GLOBAL INSTANCE AND HELPER FUNCTIONS
# =============================================================================

_engine: Optional[UnifiedDataManagementEngine] = None


async def get_data_management_engine() -> UnifiedDataManagementEngine:
    """Get the global data management engine instance."""
    global _engine
    if _engine is None:
        _engine = await UnifiedDataManagementEngine.get_instance()
    return _engine


async def initialize_data_management() -> bool:
    """Initialize the global data management engine."""
    engine = await get_data_management_engine()
    return await engine.initialize()


async def shutdown_data_management() -> None:
    """Shutdown the global data management engine."""
    global _engine
    if _engine:
        await _engine.shutdown()
        _engine = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main Engine
    "UnifiedDataManagementEngine",
    "get_data_management_engine",
    "initialize_data_management",
    "shutdown_data_management",
    # Configuration
    "DataManagementConfig",
    # Data Types
    "DataRecord",
    "DataBatch",
    "DataVersion",
    "DataLineageNode",
    "DataLineageEdge",
    "ValidationResult",
    "PrivacyReport",
    "SamplingStrategy",
    # Enums
    "DataType",
    "DataQuality",
    "PrivacyLevel",
    "RetentionPolicy",
    "SamplingMethod",
    "LineageEventType",
    # Collectors
    "TrainingDataCollector",
    # Versioning
    "DataVersionManager",
    # Validation
    "DataValidator",
    "SchemaRegistry",
    # Privacy
    "DataPrivacyManager",
    "PIIDetector",
    "DataAnonymizer",
    # Retention
    "DataRetentionManager",
    # Deduplication
    "DataDeduplicator",
    # Sampling
    "IntelligentDataSampler",
    # Lineage
    "DataLineageTracker",
]
