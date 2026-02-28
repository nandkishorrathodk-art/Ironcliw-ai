"""
Cross-Repository Data Bridge v1.0
=================================

Provides seamless data synchronization and streaming across the Ironcliw Trinity:
- Ironcliw (Body) - Execution and interaction data
- Ironcliw Prime (Mind) - Reasoning and planning data
- Reactor Core (Learning) - Training and model data

Features:
- Bi-directional data streaming with async queues
- Conflict resolution using CRDTs and vector clocks
- Priority-based data routing
- Automatic retry with exponential backoff
- Data transformation pipelines
- Real-time sync status monitoring
- Bandwidth-aware throttling

Author: Trinity Data System
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)
from functools import wraps
import uuid

logger = logging.getLogger(__name__)

# Type variables
T = TypeVar("T")
DataT = TypeVar("DataT", bound="CrossRepoDataPacket")


# =============================================================================
# ENUMS
# =============================================================================


class DataFlowDirection(Enum):
    """Direction of data flow between repos."""
    Ironcliw_TO_PRIME = "jarvis_to_prime"
    PRIME_TO_Ironcliw = "prime_to_jarvis"
    Ironcliw_TO_REACTOR = "jarvis_to_reactor"
    REACTOR_TO_Ironcliw = "reactor_to_jarvis"
    PRIME_TO_REACTOR = "prime_to_reactor"
    REACTOR_TO_PRIME = "reactor_to_prime"
    BROADCAST = "broadcast"  # All repos

    @property
    def source(self) -> str:
        """Get source repo."""
        if self == DataFlowDirection.BROADCAST:
            return "all"
        return self.value.split("_to_")[0]

    @property
    def target(self) -> str:
        """Get target repo."""
        if self == DataFlowDirection.BROADCAST:
            return "all"
        return self.value.split("_to_")[1]


class SyncMode(Enum):
    """Synchronization mode."""
    REAL_TIME = auto()       # Immediate sync
    BATCH = auto()           # Batch sync at intervals
    ON_DEMAND = auto()       # Sync when requested
    EVENTUAL = auto()        # Eventually consistent
    STRONG = auto()          # Strong consistency (blocking)


class SyncState(Enum):
    """State of sync operation."""
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    RETRYING = auto()
    CANCELLED = auto()


class DataPriority(Enum):
    """Priority levels for data sync."""
    CRITICAL = 0    # Immediate sync, blocks other operations
    HIGH = 1        # Next in queue
    NORMAL = 2      # Standard processing
    LOW = 3         # Background sync
    BATCH = 4       # Wait for batch window


class ConflictResolution(Enum):
    """Conflict resolution strategies."""
    LAST_WRITE_WINS = auto()
    FIRST_WRITE_WINS = auto()
    MERGE = auto()
    MANUAL = auto()
    CRDT = auto()
    VECTOR_CLOCK = auto()


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class CrossRepoDataConfig:
    """Configuration for cross-repo data operations."""

    # Sync settings
    default_sync_mode: SyncMode = SyncMode.EVENTUAL
    batch_interval_seconds: float = float(os.getenv("DATA_BATCH_INTERVAL", "5.0"))
    max_batch_size: int = int(os.getenv("DATA_MAX_BATCH_SIZE", "100"))

    # Retry settings
    max_retries: int = int(os.getenv("DATA_SYNC_MAX_RETRIES", "5"))
    base_retry_delay: float = float(os.getenv("DATA_SYNC_RETRY_DELAY", "1.0"))
    max_retry_delay: float = float(os.getenv("DATA_SYNC_MAX_RETRY_DELAY", "60.0"))

    # Queue settings
    max_queue_size: int = int(os.getenv("DATA_QUEUE_SIZE", "10000"))
    queue_timeout: float = float(os.getenv("DATA_QUEUE_TIMEOUT", "30.0"))

    # Bandwidth settings
    max_bandwidth_mbps: float = float(os.getenv("DATA_MAX_BANDWIDTH_MBPS", "100.0"))
    throttle_threshold: float = float(os.getenv("DATA_THROTTLE_THRESHOLD", "0.8"))

    # Conflict resolution
    default_conflict_resolution: ConflictResolution = ConflictResolution.LAST_WRITE_WINS

    # Persistence
    persistence_path: str = os.getenv(
        "DATA_PERSISTENCE_PATH",
        "/tmp/jarvis_data_bridge"
    )
    enable_persistence: bool = os.getenv("DATA_ENABLE_PERSISTENCE", "true").lower() == "true"

    # Monitoring
    metrics_enabled: bool = os.getenv("DATA_METRICS_ENABLED", "true").lower() == "true"
    health_check_interval: float = float(os.getenv("DATA_HEALTH_CHECK_INTERVAL", "10.0"))


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class VectorTimestamp:
    """Vector clock timestamp for causality tracking."""

    clocks: Dict[str, int] = field(default_factory=dict)

    def increment(self, node_id: str) -> "VectorTimestamp":
        """Increment clock for node."""
        new_clocks = self.clocks.copy()
        new_clocks[node_id] = new_clocks.get(node_id, 0) + 1
        return VectorTimestamp(clocks=new_clocks)

    def merge(self, other: "VectorTimestamp") -> "VectorTimestamp":
        """Merge with another vector clock."""
        all_nodes = set(self.clocks.keys()) | set(other.clocks.keys())
        merged = {}
        for node in all_nodes:
            merged[node] = max(
                self.clocks.get(node, 0),
                other.clocks.get(node, 0)
            )
        return VectorTimestamp(clocks=merged)

    def happens_before(self, other: "VectorTimestamp") -> bool:
        """Check if this timestamp happens before other."""
        at_least_one_less = False
        for node in set(self.clocks.keys()) | set(other.clocks.keys()):
            self_val = self.clocks.get(node, 0)
            other_val = other.clocks.get(node, 0)
            if self_val > other_val:
                return False
            if self_val < other_val:
                at_least_one_less = True
        return at_least_one_less

    def concurrent_with(self, other: "VectorTimestamp") -> bool:
        """Check if timestamps are concurrent (neither happens before)."""
        return not self.happens_before(other) and not other.happens_before(self)

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return self.clocks.copy()

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "VectorTimestamp":
        """Create from dictionary."""
        return cls(clocks=data.copy())


@dataclass
class CrossRepoDataPacket:
    """A packet of data for cross-repo transfer."""

    # Identity
    packet_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Source and destination
    source_repo: str = "jarvis"
    target_repo: str = "prime"
    direction: DataFlowDirection = DataFlowDirection.Ironcliw_TO_PRIME

    # Data
    data_type: str = "generic"
    payload: Dict[str, Any] = field(default_factory=dict)
    payload_hash: str = ""

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    priority: DataPriority = DataPriority.NORMAL
    ttl_seconds: Optional[float] = None

    # Sync tracking
    sync_mode: SyncMode = SyncMode.EVENTUAL
    vector_timestamp: VectorTimestamp = field(default_factory=VectorTimestamp)

    # Retry tracking
    attempt_count: int = 0
    last_error: Optional[str] = None

    # Lineage
    parent_packet_id: Optional[str] = None
    correlation_id: Optional[str] = None

    def __post_init__(self):
        """Compute hash after init."""
        if not self.payload_hash and self.payload:
            self.payload_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute hash of payload."""
        content = json.dumps(self.payload, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def is_expired(self) -> bool:
        """Check if packet has expired."""
        if self.ttl_seconds is None:
            return False
        elapsed = (datetime.utcnow() - self.created_at).total_seconds()
        return elapsed > self.ttl_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "packet_id": self.packet_id,
            "source_repo": self.source_repo,
            "target_repo": self.target_repo,
            "direction": self.direction.value,
            "data_type": self.data_type,
            "payload": self.payload,
            "payload_hash": self.payload_hash,
            "created_at": self.created_at.isoformat(),
            "priority": self.priority.value,
            "ttl_seconds": self.ttl_seconds,
            "sync_mode": self.sync_mode.name,
            "vector_timestamp": self.vector_timestamp.to_dict(),
            "attempt_count": self.attempt_count,
            "last_error": self.last_error,
            "parent_packet_id": self.parent_packet_id,
            "correlation_id": self.correlation_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CrossRepoDataPacket":
        """Create from dictionary."""
        return cls(
            packet_id=data.get("packet_id", str(uuid.uuid4())),
            source_repo=data.get("source_repo", "jarvis"),
            target_repo=data.get("target_repo", "prime"),
            direction=DataFlowDirection(data.get("direction", "jarvis_to_prime")),
            data_type=data.get("data_type", "generic"),
            payload=data.get("payload", {}),
            payload_hash=data.get("payload_hash", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            priority=DataPriority(data.get("priority", 2)),
            ttl_seconds=data.get("ttl_seconds"),
            sync_mode=SyncMode[data.get("sync_mode", "EVENTUAL")],
            vector_timestamp=VectorTimestamp.from_dict(data.get("vector_timestamp", {})),
            attempt_count=data.get("attempt_count", 0),
            last_error=data.get("last_error"),
            parent_packet_id=data.get("parent_packet_id"),
            correlation_id=data.get("correlation_id"),
        )


@dataclass
class DataSyncState:
    """State of a data sync operation."""

    sync_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    packet_id: str = ""
    state: SyncState = SyncState.PENDING

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Progress
    bytes_transferred: int = 0
    total_bytes: int = 0

    # Results
    success: bool = False
    error_message: Optional[str] = None
    retries: int = 0

    # Ack tracking
    acked_by: Set[str] = field(default_factory=set)

    @property
    def progress_percent(self) -> float:
        """Get sync progress percentage."""
        if self.total_bytes == 0:
            return 0.0 if self.state == SyncState.PENDING else 100.0
        return (self.bytes_transferred / self.total_bytes) * 100.0

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get sync duration."""
        if self.started_at is None:
            return None
        end = self.completed_at or datetime.utcnow()
        return (end - self.started_at).total_seconds()


@dataclass
class SyncMetrics:
    """Metrics for sync operations."""

    # Counters
    packets_sent: int = 0
    packets_received: int = 0
    packets_failed: int = 0
    packets_retried: int = 0

    # Bytes
    bytes_sent: int = 0
    bytes_received: int = 0

    # Timing
    total_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0

    # Queues
    queue_depth: int = 0
    peak_queue_depth: int = 0

    # By direction
    by_direction: Dict[str, int] = field(default_factory=dict)

    # By data type
    by_data_type: Dict[str, int] = field(default_factory=dict)

    @property
    def avg_latency_ms(self) -> float:
        """Get average latency."""
        total = self.packets_sent + self.packets_received
        if total == 0:
            return 0.0
        return self.total_latency_ms / total

    @property
    def success_rate(self) -> float:
        """Get success rate."""
        total = self.packets_sent
        if total == 0:
            return 1.0
        return (total - self.packets_failed) / total


# =============================================================================
# DATA TRANSFORMERS
# =============================================================================


class DataTransformer(ABC):
    """Base class for data transformers."""

    @abstractmethod
    async def transform(
        self,
        packet: CrossRepoDataPacket,
        direction: DataFlowDirection
    ) -> CrossRepoDataPacket:
        """Transform a data packet."""
        pass

    @abstractmethod
    def can_transform(self, data_type: str) -> bool:
        """Check if transformer handles this data type."""
        pass


class CompressionTransformer(DataTransformer):
    """Compress/decompress data for transfer."""

    def __init__(self, compression_level: int = 6):
        self.compression_level = compression_level
        self._supported_types = {"training_data", "model_weights", "embeddings"}

    async def transform(
        self,
        packet: CrossRepoDataPacket,
        direction: DataFlowDirection
    ) -> CrossRepoDataPacket:
        """Compress data for transfer."""
        import zlib

        if "compressed" not in packet.payload:
            # Compress
            data_bytes = json.dumps(packet.payload.get("data", {})).encode()
            compressed = zlib.compress(data_bytes, self.compression_level)

            new_payload = {
                "compressed": True,
                "original_size": len(data_bytes),
                "compressed_size": len(compressed),
                "data": compressed.hex(),
            }
            packet.payload = new_payload

        return packet

    def can_transform(self, data_type: str) -> bool:
        """Check if we should compress this data type."""
        return data_type in self._supported_types


class EncryptionTransformer(DataTransformer):
    """Encrypt/decrypt sensitive data for transfer."""

    def __init__(self):
        self._supported_types = {"credentials", "pii", "sensitive"}
        # In production, use proper key management
        self._key = os.getenv("DATA_ENCRYPTION_KEY", "default_key_change_me")

    async def transform(
        self,
        packet: CrossRepoDataPacket,
        direction: DataFlowDirection
    ) -> CrossRepoDataPacket:
        """Encrypt data for transfer."""
        # Simple XOR encryption for demo - use proper crypto in production
        import base64

        if "encrypted" not in packet.payload:
            data_str = json.dumps(packet.payload.get("data", {}))
            key_bytes = self._key.encode()
            data_bytes = data_str.encode()

            encrypted = bytes([
                data_bytes[i] ^ key_bytes[i % len(key_bytes)]
                for i in range(len(data_bytes))
            ])

            packet.payload = {
                "encrypted": True,
                "data": base64.b64encode(encrypted).decode(),
            }

        return packet

    def can_transform(self, data_type: str) -> bool:
        """Check if we should encrypt this data type."""
        return data_type in self._supported_types


class SchemaTransformer(DataTransformer):
    """Transform data between different schema versions."""

    def __init__(self):
        self._migrations: Dict[Tuple[str, str], Callable] = {}

    def register_migration(
        self,
        from_version: str,
        to_version: str,
        migration_fn: Callable[[Dict], Dict]
    ):
        """Register a schema migration."""
        self._migrations[(from_version, to_version)] = migration_fn

    async def transform(
        self,
        packet: CrossRepoDataPacket,
        direction: DataFlowDirection
    ) -> CrossRepoDataPacket:
        """Apply schema migrations."""
        source_version = packet.payload.get("schema_version", "1.0")
        target_version = packet.payload.get("target_schema_version", source_version)

        if source_version != target_version:
            key = (source_version, target_version)
            if key in self._migrations:
                packet.payload["data"] = self._migrations[key](
                    packet.payload.get("data", {})
                )
                packet.payload["schema_version"] = target_version

        return packet

    def can_transform(self, data_type: str) -> bool:
        """All types can be schema-transformed."""
        return True


# =============================================================================
# PRIORITY QUEUE
# =============================================================================


class PriorityDataQueue:
    """Priority queue for data packets."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._queues: Dict[DataPriority, asyncio.Queue] = {
            priority: asyncio.Queue(maxsize=max_size // 5)
            for priority in DataPriority
        }
        self._size = 0
        self._lock = asyncio.Lock()

    async def put(
        self,
        packet: CrossRepoDataPacket,
        timeout: Optional[float] = None
    ) -> bool:
        """Add packet to queue."""
        async with self._lock:
            if self._size >= self.max_size:
                # Evict lowest priority if full
                for priority in reversed(DataPriority):
                    if not self._queues[priority].empty():
                        try:
                            self._queues[priority].get_nowait()
                            self._size -= 1
                            break
                        except asyncio.QueueEmpty:
                            continue
                else:
                    return False

            queue = self._queues[packet.priority]
            try:
                if timeout:
                    await asyncio.wait_for(queue.put(packet), timeout)
                else:
                    await queue.put(packet)
                self._size += 1
                return True
            except asyncio.TimeoutError:
                return False

    async def get(self, timeout: Optional[float] = None) -> Optional[CrossRepoDataPacket]:
        """Get highest priority packet."""
        # Check queues in priority order
        for priority in DataPriority:
            queue = self._queues[priority]
            try:
                if not queue.empty():
                    packet = queue.get_nowait()
                    async with self._lock:
                        self._size -= 1
                    return packet
            except asyncio.QueueEmpty:
                continue

        # If all empty, wait on critical queue
        if timeout:
            try:
                packet = await asyncio.wait_for(
                    self._queues[DataPriority.CRITICAL].get(),
                    timeout
                )
                async with self._lock:
                    self._size -= 1
                return packet
            except asyncio.TimeoutError:
                return None

        return None

    @property
    def size(self) -> int:
        """Get total queue size."""
        return self._size

    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        return {
            "total_size": self._size,
            "by_priority": {
                priority.name: self._queues[priority].qsize()
                for priority in DataPriority
            }
        }


# =============================================================================
# CONFLICT RESOLVER
# =============================================================================


class ConflictResolver:
    """Resolves conflicts between concurrent data updates."""

    def __init__(self, default_strategy: ConflictResolution = ConflictResolution.LAST_WRITE_WINS):
        self.default_strategy = default_strategy
        self._custom_resolvers: Dict[str, Callable] = {}

    def register_resolver(
        self,
        data_type: str,
        resolver: Callable[[CrossRepoDataPacket, CrossRepoDataPacket], CrossRepoDataPacket]
    ):
        """Register custom resolver for data type."""
        self._custom_resolvers[data_type] = resolver

    async def resolve(
        self,
        packet1: CrossRepoDataPacket,
        packet2: CrossRepoDataPacket,
        strategy: Optional[ConflictResolution] = None
    ) -> CrossRepoDataPacket:
        """Resolve conflict between two packets."""
        strategy = strategy or self.default_strategy

        # Check for custom resolver
        if packet1.data_type in self._custom_resolvers:
            return self._custom_resolvers[packet1.data_type](packet1, packet2)

        if strategy == ConflictResolution.LAST_WRITE_WINS:
            return packet1 if packet1.created_at >= packet2.created_at else packet2

        elif strategy == ConflictResolution.FIRST_WRITE_WINS:
            return packet1 if packet1.created_at <= packet2.created_at else packet2

        elif strategy == ConflictResolution.VECTOR_CLOCK:
            if packet1.vector_timestamp.happens_before(packet2.vector_timestamp):
                return packet2
            elif packet2.vector_timestamp.happens_before(packet1.vector_timestamp):
                return packet1
            else:
                # Concurrent - fall back to merge
                return await self._merge_packets(packet1, packet2)

        elif strategy == ConflictResolution.MERGE:
            return await self._merge_packets(packet1, packet2)

        else:
            # Default to last write
            return packet1 if packet1.created_at >= packet2.created_at else packet2

    async def _merge_packets(
        self,
        packet1: CrossRepoDataPacket,
        packet2: CrossRepoDataPacket
    ) -> CrossRepoDataPacket:
        """Merge two packets."""
        # Deep merge payloads
        merged_payload = self._deep_merge(packet1.payload, packet2.payload)

        # Create merged packet
        merged = CrossRepoDataPacket(
            source_repo=packet1.source_repo,
            target_repo=packet1.target_repo,
            direction=packet1.direction,
            data_type=packet1.data_type,
            payload=merged_payload,
            priority=min(packet1.priority, packet2.priority, key=lambda p: p.value),
            sync_mode=packet1.sync_mode,
            vector_timestamp=packet1.vector_timestamp.merge(packet2.vector_timestamp),
            parent_packet_id=packet1.packet_id,
            correlation_id=packet1.correlation_id or packet2.correlation_id,
        )

        return merged

    def _deep_merge(self, dict1: Dict, dict2: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            elif key in result and isinstance(result[key], list) and isinstance(value, list):
                # Merge lists, deduplicate
                result[key] = list(set(result[key] + value))
            else:
                result[key] = value
        return result


# =============================================================================
# BANDWIDTH THROTTLER
# =============================================================================


class BandwidthThrottler:
    """Throttles data transfer to respect bandwidth limits."""

    def __init__(
        self,
        max_bandwidth_mbps: float = 100.0,
        window_seconds: float = 1.0
    ):
        self.max_bandwidth_bps = max_bandwidth_mbps * 1_000_000 / 8  # Bytes per second
        self.window_seconds = window_seconds
        self._bytes_in_window: List[Tuple[float, int]] = []
        self._lock = asyncio.Lock()

    async def acquire(self, num_bytes: int) -> float:
        """Acquire permission to transfer bytes. Returns wait time."""
        async with self._lock:
            now = time.time()

            # Clean old entries
            cutoff = now - self.window_seconds
            self._bytes_in_window = [
                (t, b) for t, b in self._bytes_in_window
                if t > cutoff
            ]

            # Calculate current usage
            current_bytes = sum(b for _, b in self._bytes_in_window)

            # Check if we need to wait
            if current_bytes + num_bytes > self.max_bandwidth_bps * self.window_seconds:
                # Calculate wait time
                excess = (current_bytes + num_bytes) - (self.max_bandwidth_bps * self.window_seconds)
                wait_time = excess / self.max_bandwidth_bps
                return wait_time

            # Record transfer
            self._bytes_in_window.append((now, num_bytes))
            return 0.0

    def get_usage_percent(self) -> float:
        """Get current bandwidth usage percentage."""
        now = time.time()
        cutoff = now - self.window_seconds
        current_bytes = sum(
            b for t, b in self._bytes_in_window
            if t > cutoff
        )
        return (current_bytes / (self.max_bandwidth_bps * self.window_seconds)) * 100


# =============================================================================
# CROSS-REPO DATA BRIDGE
# =============================================================================


class CrossRepoDataBridge:
    """Main bridge for cross-repo data synchronization."""

    def __init__(self, config: Optional[CrossRepoDataConfig] = None):
        self.config = config or CrossRepoDataConfig()
        self._node_id = os.getenv("Ironcliw_NODE_ID", f"node_{uuid.uuid4().hex[:8]}")

        # Queues
        self._outbound_queue = PriorityDataQueue(self.config.max_queue_size)
        self._inbound_queue = PriorityDataQueue(self.config.max_queue_size)

        # Components
        self._transformers: List[DataTransformer] = [
            CompressionTransformer(),
            EncryptionTransformer(),
            SchemaTransformer(),
        ]
        self._conflict_resolver = ConflictResolver(self.config.default_conflict_resolution)
        self._throttler = BandwidthThrottler(self.config.max_bandwidth_mbps)

        # State
        self._vector_clock = VectorTimestamp()
        self._sync_states: Dict[str, DataSyncState] = {}
        self._metrics = SyncMetrics()

        # Handlers
        self._data_handlers: Dict[str, Callable] = {}
        self._direction_handlers: Dict[DataFlowDirection, List[Callable]] = defaultdict(list)

        # Tasks
        self._running = False
        self._tasks: List[asyncio.Task] = []

        # Persistence
        if self.config.enable_persistence:
            Path(self.config.persistence_path).mkdir(parents=True, exist_ok=True)

        logger.info(f"CrossRepoDataBridge initialized with node_id={self._node_id}")

    async def start(self):
        """Start the data bridge."""
        if self._running:
            return

        self._running = True

        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._outbound_processor()),
            asyncio.create_task(self._inbound_processor()),
            asyncio.create_task(self._batch_processor()),
            asyncio.create_task(self._health_monitor()),
        ]

        # Load persisted state
        await self._load_state()

        logger.info("CrossRepoDataBridge started")

    async def stop(self):
        """Stop the data bridge."""
        if not self._running:
            return

        self._running = False

        # Cancel tasks
        for task in self._tasks:
            task.cancel()

        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        # Persist state
        await self._save_state()

        logger.info("CrossRepoDataBridge stopped")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def send(
        self,
        data: Dict[str, Any],
        data_type: str,
        direction: DataFlowDirection,
        priority: DataPriority = DataPriority.NORMAL,
        sync_mode: Optional[SyncMode] = None,
        ttl_seconds: Optional[float] = None,
        correlation_id: Optional[str] = None,
    ) -> str:
        """Send data to another repo."""
        # Increment vector clock
        self._vector_clock = self._vector_clock.increment(self._node_id)

        # Create packet
        packet = CrossRepoDataPacket(
            source_repo=direction.source,
            target_repo=direction.target,
            direction=direction,
            data_type=data_type,
            payload={"data": data},
            priority=priority,
            sync_mode=sync_mode or self.config.default_sync_mode,
            ttl_seconds=ttl_seconds,
            vector_timestamp=self._vector_clock,
            correlation_id=correlation_id or str(uuid.uuid4()),
        )

        # Apply transformers
        for transformer in self._transformers:
            if transformer.can_transform(data_type):
                packet = await transformer.transform(packet, direction)

        # Create sync state
        sync_state = DataSyncState(
            packet_id=packet.packet_id,
            total_bytes=len(json.dumps(packet.payload)),
        )
        self._sync_states[packet.packet_id] = sync_state

        # Handle based on sync mode
        if sync_mode == SyncMode.STRONG:
            # Synchronous send
            success = await self._send_immediate(packet)
            if not success:
                raise RuntimeError(f"Failed to send packet {packet.packet_id}")
        else:
            # Queue for async send
            await self._outbound_queue.put(packet)

        # Update metrics
        self._metrics.packets_sent += 1
        self._metrics.by_direction[direction.value] = \
            self._metrics.by_direction.get(direction.value, 0) + 1
        self._metrics.by_data_type[data_type] = \
            self._metrics.by_data_type.get(data_type, 0) + 1

        return packet.packet_id

    async def receive(self, timeout: Optional[float] = None) -> Optional[CrossRepoDataPacket]:
        """Receive next inbound packet."""
        return await self._inbound_queue.get(timeout)

    def register_handler(
        self,
        data_type: str,
        handler: Callable[[CrossRepoDataPacket], Coroutine[Any, Any, None]]
    ):
        """Register handler for data type."""
        self._data_handlers[data_type] = handler

    def register_direction_handler(
        self,
        direction: DataFlowDirection,
        handler: Callable[[CrossRepoDataPacket], Coroutine[Any, Any, None]]
    ):
        """Register handler for data direction."""
        self._direction_handlers[direction].append(handler)

    async def get_sync_state(self, packet_id: str) -> Optional[DataSyncState]:
        """Get sync state for a packet."""
        return self._sync_states.get(packet_id)

    def get_metrics(self) -> SyncMetrics:
        """Get current metrics."""
        self._metrics.queue_depth = self._outbound_queue.size + self._inbound_queue.size
        self._metrics.peak_queue_depth = max(
            self._metrics.peak_queue_depth,
            self._metrics.queue_depth
        )
        return self._metrics

    def get_vector_clock(self) -> VectorTimestamp:
        """Get current vector clock."""
        return self._vector_clock

    # -------------------------------------------------------------------------
    # Internal Processing
    # -------------------------------------------------------------------------

    async def _outbound_processor(self):
        """Process outbound packets."""
        while self._running:
            try:
                packet = await self._outbound_queue.get(timeout=1.0)
                if packet is None:
                    continue

                # Check expiry
                if packet.is_expired():
                    logger.warning(f"Packet {packet.packet_id} expired, dropping")
                    self._metrics.packets_failed += 1
                    continue

                # Throttle if needed
                packet_size = len(json.dumps(packet.payload))
                wait_time = await self._throttler.acquire(packet_size)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

                # Send with retry
                success = await self._send_with_retry(packet)

                if success:
                    sync_state = self._sync_states.get(packet.packet_id)
                    if sync_state:
                        sync_state.state = SyncState.COMPLETED
                        sync_state.completed_at = datetime.utcnow()
                        sync_state.success = True
                else:
                    self._metrics.packets_failed += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in outbound processor: {e}")

    async def _inbound_processor(self):
        """Process inbound packets."""
        while self._running:
            try:
                packet = await self._inbound_queue.get(timeout=1.0)
                if packet is None:
                    continue

                # Update vector clock
                self._vector_clock = self._vector_clock.merge(packet.vector_timestamp)
                self._vector_clock = self._vector_clock.increment(self._node_id)

                # Dispatch to handlers
                await self._dispatch_packet(packet)

                self._metrics.packets_received += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in inbound processor: {e}")

    async def _batch_processor(self):
        """Process batched packets."""
        while self._running:
            try:
                await asyncio.sleep(self.config.batch_interval_seconds)

                # Collect batch packets
                batch: List[CrossRepoDataPacket] = []
                while len(batch) < self.config.max_batch_size:
                    packet = await self._outbound_queue.get(timeout=0.1)
                    if packet is None:
                        break
                    if packet.sync_mode == SyncMode.BATCH:
                        batch.append(packet)
                    else:
                        # Put back non-batch packets
                        await self._outbound_queue.put(packet)

                if batch:
                    await self._send_batch(batch)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")

    async def _health_monitor(self):
        """Monitor health of connections."""
        while self._running:
            try:
                await asyncio.sleep(self.config.health_check_interval)

                # Log metrics
                metrics = self.get_metrics()
                logger.debug(
                    f"DataBridge health: sent={metrics.packets_sent}, "
                    f"received={metrics.packets_received}, "
                    f"failed={metrics.packets_failed}, "
                    f"queue={metrics.queue_depth}"
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")

    async def _send_immediate(self, packet: CrossRepoDataPacket) -> bool:
        """Send packet immediately (synchronous)."""
        sync_state = self._sync_states.get(packet.packet_id)
        if sync_state:
            sync_state.state = SyncState.IN_PROGRESS
            sync_state.started_at = datetime.utcnow()

        try:
            # In a real implementation, this would send to the actual target
            # For now, simulate successful send
            await self._simulate_send(packet)

            self._metrics.bytes_sent += len(json.dumps(packet.payload))
            return True

        except Exception as e:
            logger.error(f"Failed to send packet {packet.packet_id}: {e}")
            if sync_state:
                sync_state.error_message = str(e)
            return False

    async def _send_with_retry(self, packet: CrossRepoDataPacket) -> bool:
        """Send packet with exponential backoff retry."""
        delay = self.config.base_retry_delay

        for attempt in range(self.config.max_retries):
            packet.attempt_count = attempt + 1

            sync_state = self._sync_states.get(packet.packet_id)
            if sync_state:
                sync_state.state = SyncState.IN_PROGRESS if attempt == 0 else SyncState.RETRYING
                sync_state.retries = attempt

            success = await self._send_immediate(packet)
            if success:
                return True

            self._metrics.packets_retried += 1

            # Exponential backoff with jitter
            jitter = delay * 0.1 * (2 * asyncio.get_running_loop().time() % 1 - 0.5)
            await asyncio.sleep(delay + jitter)
            delay = min(delay * 2, self.config.max_retry_delay)

        # Move to dead letter queue
        if sync_state:
            sync_state.state = SyncState.FAILED
            sync_state.error_message = f"Failed after {self.config.max_retries} retries"

        return False

    async def _send_batch(self, batch: List[CrossRepoDataPacket]) -> bool:
        """Send batch of packets."""
        logger.debug(f"Sending batch of {len(batch)} packets")

        # Group by direction
        by_direction: Dict[DataFlowDirection, List[CrossRepoDataPacket]] = defaultdict(list)
        for packet in batch:
            by_direction[packet.direction].append(packet)

        # Send each group
        success = True
        for direction, packets in by_direction.items():
            for packet in packets:
                if not await self._send_with_retry(packet):
                    success = False

        return success

    async def _simulate_send(self, packet: CrossRepoDataPacket):
        """Simulate sending a packet (for testing/demo)."""
        # Simulate network latency
        latency = 0.01 + (hash(packet.packet_id) % 50) / 1000
        await asyncio.sleep(latency)

        self._metrics.total_latency_ms += latency * 1000
        self._metrics.min_latency_ms = min(self._metrics.min_latency_ms, latency * 1000)
        self._metrics.max_latency_ms = max(self._metrics.max_latency_ms, latency * 1000)

    async def _dispatch_packet(self, packet: CrossRepoDataPacket):
        """Dispatch packet to registered handlers."""
        # Type handler
        if packet.data_type in self._data_handlers:
            try:
                await self._data_handlers[packet.data_type](packet)
            except Exception as e:
                logger.error(f"Error in handler for {packet.data_type}: {e}")

        # Direction handlers
        for handler in self._direction_handlers.get(packet.direction, []):
            try:
                await handler(packet)
            except Exception as e:
                logger.error(f"Error in direction handler: {e}")

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    async def _save_state(self):
        """Save state to disk."""
        if not self.config.enable_persistence:
            return

        state_file = Path(self.config.persistence_path) / "bridge_state.json"
        state = {
            "vector_clock": self._vector_clock.to_dict(),
            "metrics": {
                "packets_sent": self._metrics.packets_sent,
                "packets_received": self._metrics.packets_received,
                "packets_failed": self._metrics.packets_failed,
            }
        }

        try:
            with open(state_file, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    async def _load_state(self):
        """Load state from disk."""
        if not self.config.enable_persistence:
            return

        state_file = Path(self.config.persistence_path) / "bridge_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, 'r') as f:
                state = json.load(f)

            self._vector_clock = VectorTimestamp.from_dict(state.get("vector_clock", {}))
            metrics = state.get("metrics", {})
            self._metrics.packets_sent = metrics.get("packets_sent", 0)
            self._metrics.packets_received = metrics.get("packets_received", 0)
            self._metrics.packets_failed = metrics.get("packets_failed", 0)

        except Exception as e:
            logger.error(f"Failed to load state: {e}")


# =============================================================================
# GLOBAL INSTANCE MANAGEMENT
# =============================================================================

_cross_repo_bridge: Optional[CrossRepoDataBridge] = None
_bridge_lock = asyncio.Lock()


async def get_cross_repo_data_bridge() -> CrossRepoDataBridge:
    """Get or create the global data bridge instance."""
    global _cross_repo_bridge

    async with _bridge_lock:
        if _cross_repo_bridge is None:
            _cross_repo_bridge = CrossRepoDataBridge()
            await _cross_repo_bridge.start()
        return _cross_repo_bridge


async def initialize_cross_repo_data(
    config: Optional[CrossRepoDataConfig] = None
) -> CrossRepoDataBridge:
    """Initialize the cross-repo data bridge."""
    global _cross_repo_bridge

    async with _bridge_lock:
        if _cross_repo_bridge is not None:
            await _cross_repo_bridge.stop()

        _cross_repo_bridge = CrossRepoDataBridge(config)
        await _cross_repo_bridge.start()

        logger.info("Cross-repo data bridge initialized")
        return _cross_repo_bridge


async def shutdown_cross_repo_data():
    """Shutdown the cross-repo data bridge."""
    global _cross_repo_bridge

    async with _bridge_lock:
        if _cross_repo_bridge is not None:
            await _cross_repo_bridge.stop()
            _cross_repo_bridge = None
            logger.info("Cross-repo data bridge shutdown")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "CrossRepoDataConfig",
    # Enums
    "DataFlowDirection",
    "SyncMode",
    "SyncState",
    "DataPriority",
    "ConflictResolution",
    # Data Structures
    "VectorTimestamp",
    "CrossRepoDataPacket",
    "DataSyncState",
    "SyncMetrics",
    # Transformers
    "DataTransformer",
    "CompressionTransformer",
    "EncryptionTransformer",
    "SchemaTransformer",
    # Queue
    "PriorityDataQueue",
    # Conflict Resolution
    "ConflictResolver",
    # Throttling
    "BandwidthThrottler",
    # Main Bridge
    "CrossRepoDataBridge",
    # Global Functions
    "get_cross_repo_data_bridge",
    "initialize_cross_repo_data",
    "shutdown_cross_repo_data",
]
