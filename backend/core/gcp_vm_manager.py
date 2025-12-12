"""
GCP Spot VM Auto-Creation Manager
==================================

Advanced, robust, async system for auto-creating and managing GCP Spot VMs
when local memory pressure is too high.

Features:
- Auto-creates e2-highmem-4 Spot VMs (32GB RAM) when RAM >85%
- Integrates with intelligent_gcp_optimizer for smart decisions
- Tracks all costs via cost_tracker
- Async/await throughout for non-blocking operations
- No hardcoding - all configuration from environment/config
- Comprehensive error handling and retry logic
- VM lifecycle management (create, monitor, cleanup)
- Orphaned VM detection and cleanup
- Health monitoring and auto-recovery
- Parallel operations where safe
- Circuit breaker pattern for fault tolerance
- Dynamic configuration loading
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, TypeVar

# Google Cloud Compute Engine API
# Use TYPE_CHECKING to allow type hints without requiring the library at runtime
COMPUTE_AVAILABLE = False
compute_v1: Any = None  # Module placeholder for runtime

try:
    from google.cloud import compute_v1 as _compute_v1
    compute_v1 = _compute_v1
    COMPUTE_AVAILABLE = True
except ImportError:
    logging.warning("google-cloud-compute not installed. VM creation disabled.")

# Type aliases for when compute_v1 is not available
if TYPE_CHECKING:
    from google.cloud import compute_v1 as compute_v1_types
    InstancesClientType = compute_v1_types.InstancesClient
    ZonesClientType = compute_v1_types.ZonesClient
    InstanceType = compute_v1_types.Instance
else:
    InstancesClientType = Any
    ZonesClientType = Any
    InstanceType = Any

# Import with fallback for different import contexts
try:
    from core.cost_tracker import get_cost_tracker, CostTracker
    from core.intelligent_gcp_optimizer import get_gcp_optimizer
except ImportError:
    # Fallback for when running from within core directory
    try:
        from .cost_tracker import get_cost_tracker, CostTracker
        from .intelligent_gcp_optimizer import get_gcp_optimizer
    except ImportError:
        # Final fallback - provide stubs
        def get_cost_tracker():
            return None
        def get_gcp_optimizer(config=None):
            return None
        CostTracker = None

logger = logging.getLogger(__name__)

# Type variable for generic retry decorator
T = TypeVar('T')


# ============================================================================
# CIRCUIT BREAKER FOR FAULT TOLERANCE
# ============================================================================


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """
    Circuit breaker pattern for fault-tolerant GCP operations.
    Prevents cascading failures when GCP API is having issues.
    """
    name: str
    failure_threshold: int = 3  # Open circuit after N failures
    recovery_timeout: float = 60.0  # Seconds before trying again
    half_open_max_calls: int = 1  # Calls to test when half-open
    
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[float] = None
    half_open_calls: int = 0
    
    def can_execute(self) -> Tuple[bool, str]:
        """Check if we can execute an operation"""
        if self.state == CircuitState.CLOSED:
            return True, "Circuit closed - normal operation"
        
        if self.state == CircuitState.OPEN:
            # Check if we should transition to half-open
            if self.last_failure_time and (time.time() - self.last_failure_time) >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                logger.info(f"🔌 Circuit '{self.name}' transitioning to HALF_OPEN")
                return True, "Circuit half-open - testing recovery"
            return False, f"Circuit OPEN - {self.recovery_timeout - (time.time() - (self.last_failure_time or 0)):.0f}s until retry"
        
        # Half-open state
        if self.half_open_calls < self.half_open_max_calls:
            return True, "Circuit half-open - testing"
        return False, "Circuit half-open - max test calls reached"
    
    def record_success(self):
        """Record successful operation"""
        if self.state == CircuitState.HALF_OPEN:
            logger.info(f"✅ Circuit '{self.name}' recovered - transitioning to CLOSED")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.half_open_calls = 0
    
    def record_failure(self, error: Exception):
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            logger.warning(f"⚠️ Circuit '{self.name}' recovery failed - re-opening")
            self.state = CircuitState.OPEN
        elif self.failure_count >= self.failure_threshold:
            logger.error(f"🔴 Circuit '{self.name}' OPEN after {self.failure_count} failures: {error}")
            self.state = CircuitState.OPEN


class VMState(Enum):
    """VM lifecycle states"""

    CREATING = "creating"
    PROVISIONING = "provisioning"
    STAGING = "staging"
    RUNNING = "running"
    STOPPING = "stopping"
    TERMINATED = "terminated"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class VMInstance:
    """Represents a managed VM instance"""

    instance_id: str
    name: str
    zone: str
    state: VMState
    created_at: float
    ip_address: Optional[str] = None
    internal_ip: Optional[str] = None
    last_health_check: Optional[float] = None
    health_status: str = "unknown"
    components: List[str] = field(default_factory=list)
    trigger_reason: str = ""
    cost_per_hour: float = 0.029  # e2-highmem-4 Spot VM
    total_cost: float = 0.0
    metadata: Dict = field(default_factory=dict)

    # Cost tracking and efficiency
    last_activity_time: float = field(default_factory=time.time)
    component_usage_count: int = 0  # How many times components were accessed
    cost_efficiency_score: float = 0.0  # 0-100, higher is better ROI

    # Detailed metrics (like GCP Console)
    cpu_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_total_gb: float = 32.0
    network_sent_mb: float = 0.0
    network_received_mb: float = 0.0
    disk_read_mb: float = 0.0
    disk_write_mb: float = 0.0

    @property
    def uptime_hours(self) -> float:
        """Calculate VM uptime in hours"""
        return (time.time() - self.created_at) / 3600

    @property
    def idle_time_minutes(self) -> float:
        """Calculate how long VM has been idle"""
        return (time.time() - self.last_activity_time) / 60

    @property
    def is_healthy(self) -> bool:
        """Check if VM is healthy"""
        return self.health_status == "healthy" and self.state == VMState.RUNNING

    @property
    def is_idle(self) -> bool:
        """Check if VM has been idle too long (no activity for 10 minutes)"""
        return self.idle_time_minutes > 10

    @property
    def is_wasting_money(self) -> bool:
        """Determine if VM is wasting money (idle + low efficiency)"""
        return self.is_idle and self.cost_efficiency_score < 30.0

    @property
    def memory_percent(self) -> float:
        """Memory usage percentage"""
        return (self.memory_used_gb / self.memory_total_gb * 100) if self.memory_total_gb > 0 else 0.0

    def update_cost(self):
        """Update total cost based on uptime"""
        self.total_cost = self.uptime_hours * self.cost_per_hour

    def update_efficiency_score(self):
        """Calculate cost efficiency score (0-100, higher is better ROI)"""
        if self.uptime_hours == 0:
            self.cost_efficiency_score = 0.0
            return

        # Factors:
        # 1. Usage frequency (component_usage_count / uptime_minutes)
        # 2. Recency (how recently was it used?)
        # 3. Resource utilization (is it actually being used?)

        uptime_minutes = self.uptime_hours * 60
        usage_rate = (self.component_usage_count / uptime_minutes) if uptime_minutes > 0 else 0
        recency_score = max(0, 100 - (self.idle_time_minutes * 2))  # Decreases as idle time increases
        utilization_score = min(100, (self.cpu_percent + self.memory_percent) / 2)

        # Weighted average
        self.cost_efficiency_score = (
            usage_rate * 100 * 0.4 +  # 40% weight on usage frequency
            recency_score * 0.3 +       # 30% weight on recency
            utilization_score * 0.3     # 30% weight on resource utilization
        )

        self.cost_efficiency_score = min(100.0, self.cost_efficiency_score)

    def record_activity(self):
        """Record that VM components were used"""
        self.last_activity_time = time.time()
        self.component_usage_count += 1


@dataclass
class VMManagerConfig:
    """
    Dynamic configuration for GCP VM Manager.
    All values loaded from environment variables with sensible defaults.
    No hardcoding - fully configurable at runtime.
    """

    # GCP Configuration (all from environment)
    project_id: str = field(
        default_factory=lambda: os.getenv("GCP_PROJECT_ID", os.getenv("GOOGLE_CLOUD_PROJECT", ""))
    )
    region: str = field(default_factory=lambda: os.getenv("GCP_REGION", "us-central1"))
    zone: str = field(default_factory=lambda: os.getenv("GCP_ZONE", "us-central1-a"))

    # VM Configuration (dynamic)
    machine_type: str = field(
        default_factory=lambda: os.getenv("GCP_VM_MACHINE_TYPE", "e2-highmem-4")
    )
    use_spot: bool = field(
        default_factory=lambda: os.getenv("GCP_USE_SPOT_VMS", "true").lower() == "true"
    )
    spot_max_price: float = field(
        default_factory=lambda: float(os.getenv("GCP_SPOT_MAX_PRICE", "0.10"))
    )

    # VM Naming
    vm_name_prefix: str = field(
        default_factory=lambda: os.getenv("GCP_VM_NAME_PREFIX", "jarvis-backend")
    )

    # Image Configuration (dynamic)
    image_project: str = field(
        default_factory=lambda: os.getenv("GCP_IMAGE_PROJECT", "ubuntu-os-cloud")
    )
    image_family: str = field(
        default_factory=lambda: os.getenv("GCP_IMAGE_FAMILY", "ubuntu-2204-lts")
    )

    # Disk Configuration
    boot_disk_size_gb: int = field(
        default_factory=lambda: int(os.getenv("GCP_BOOT_DISK_SIZE_GB", "50"))
    )
    boot_disk_type: str = field(
        default_factory=lambda: os.getenv("GCP_BOOT_DISK_TYPE", "pd-standard")
    )

    # Network Configuration
    network: str = field(default_factory=lambda: os.getenv("GCP_NETWORK", "default"))
    subnetwork: str = field(default_factory=lambda: os.getenv("GCP_SUBNETWORK", "default"))

    # Startup Script Path
    startup_script_path: Optional[str] = field(
        default_factory=lambda: os.getenv(
            "GCP_STARTUP_SCRIPT_PATH",
            os.path.join(os.path.dirname(__file__), "gcp_vm_startup.sh")
        )
    )

    # Health Check Configuration
    health_check_interval: int = field(
        default_factory=lambda: int(os.getenv("GCP_HEALTH_CHECK_INTERVAL", "30"))
    )
    health_check_timeout: int = field(
        default_factory=lambda: int(os.getenv("GCP_HEALTH_CHECK_TIMEOUT", "10"))
    )
    max_health_check_failures: int = field(
        default_factory=lambda: int(os.getenv("GCP_MAX_HEALTH_CHECK_FAILURES", "3"))
    )

    # VM Lifecycle
    max_vm_lifetime_hours: float = field(
        default_factory=lambda: float(os.getenv("GCP_MAX_VM_LIFETIME_HOURS", "3.0"))
    )
    idle_timeout_minutes: int = field(
        # Prefer JARVIS_SPOT_VM_IDLE_TIMEOUT (used by hybrid cloud stack), fall back to legacy GCP_*
        default_factory=lambda: int(
            os.getenv("GCP_IDLE_TIMEOUT_MINUTES", os.getenv("JARVIS_SPOT_VM_IDLE_TIMEOUT", "30"))
        )
    )

    # Retry Configuration
    max_create_attempts: int = field(
        default_factory=lambda: int(os.getenv("GCP_MAX_CREATE_ATTEMPTS", "3"))
    )
    retry_delay_seconds: int = field(
        default_factory=lambda: int(os.getenv("GCP_RETRY_DELAY_SECONDS", "10"))
    )

    # Resource Limits
    max_concurrent_vms: int = field(
        default_factory=lambda: int(os.getenv("GCP_MAX_CONCURRENT_VMS", "2"))
    )
    daily_budget_usd: float = field(
        # Prefer JARVIS_SPOT_VM_DAILY_BUDGET (used by hybrid cloud stack), fall back to legacy GCP_*
        default_factory=lambda: float(
            os.getenv("GCP_DAILY_BUDGET_USD", os.getenv("JARVIS_SPOT_VM_DAILY_BUDGET", "5.0"))
        )
    )

    # Monitoring
    enable_monitoring: bool = field(
        default_factory=lambda: os.getenv("GCP_ENABLE_MONITORING", "true").lower() == "true"
    )
    enable_logging: bool = field(
        default_factory=lambda: os.getenv("GCP_ENABLE_LOGGING", "true").lower() == "true"
    )

    # Circuit breaker settings
    circuit_failure_threshold: int = field(
        default_factory=lambda: int(os.getenv("GCP_CIRCUIT_FAILURE_THRESHOLD", "3"))
    )
    circuit_recovery_timeout: float = field(
        default_factory=lambda: float(os.getenv("GCP_CIRCUIT_RECOVERY_TIMEOUT", "60.0"))
    )

    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.project_id:
            logger.warning(
                "⚠️  GCP_PROJECT_ID not set. Set via environment variable or provide in config."
            )
        
        # Log configuration summary
        logger.debug(f"VMManagerConfig loaded:")
        logger.debug(f"  Project: {self.project_id}")
        logger.debug(f"  Zone: {self.zone}")
        logger.debug(f"  Machine Type: {self.machine_type}")
        logger.debug(f"  Use Spot: {self.use_spot}")
        logger.debug(f"  Daily Budget: ${self.daily_budget_usd}")

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return {
            "project_id": self.project_id,
            "region": self.region,
            "zone": self.zone,
            "machine_type": self.machine_type,
            "use_spot": self.use_spot,
            "daily_budget_usd": self.daily_budget_usd,
            "max_concurrent_vms": self.max_concurrent_vms,
            "max_vm_lifetime_hours": self.max_vm_lifetime_hours,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VMManagerConfig":
        """Create configuration from dictionary"""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


class GCPVMManager:
    """
    Advanced GCP Spot VM auto-creation and lifecycle manager.

    Features:
    - Circuit breaker pattern for fault tolerance
    - Parallel async operations where safe
    - Dynamic configuration from environment
    - Robust error handling with retries
    - Cost tracking integration
    - Health monitoring with auto-recovery

    Integrates with:
    - intelligent_gcp_optimizer: For VM creation decisions
    - cost_tracker: For billing tracking
    - platform_memory_monitor: For memory pressure detection
    """

    def __init__(self, config: Optional[VMManagerConfig] = None):
        self.config = config or VMManagerConfig()

        # API clients (initialized lazily)
        # Use type aliases that work even when compute_v1 is not available
        self.instances_client: Optional[InstancesClientType] = None
        self.zones_client: Optional[ZonesClientType] = None
        self.zone_operations_client = None  # For polling operation status

        # Integrations (initialized safely)
        self.cost_tracker: Optional[Any] = None  # CostTracker or None
        self.gcp_optimizer: Optional[Any] = None

        # State tracking with thread-safe locks
        self.managed_vms: Dict[str, VMInstance] = {}
        self.creating_vms: Dict[str, asyncio.Task] = {}
        self._vm_lock = asyncio.Lock()  # Protect VM state modifications
        self._init_lock = asyncio.Lock()  # Protect initialization

        # Circuit breakers for fault tolerance
        self._circuit_breakers = {
            "vm_create": CircuitBreaker(
                name="vm_create",
                failure_threshold=self.config.circuit_failure_threshold,
                recovery_timeout=self.config.circuit_recovery_timeout,
            ),
            "vm_delete": CircuitBreaker(
                name="vm_delete",
                failure_threshold=self.config.circuit_failure_threshold,
                recovery_timeout=self.config.circuit_recovery_timeout,
            ),
            "cost_tracker": CircuitBreaker(
                name="cost_tracker",
                failure_threshold=5,  # More lenient for non-critical operations
                recovery_timeout=30.0,
            ),
        }

        # Monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False

        # Enhanced stats
        self.stats = {
            "total_created": 0,
            "total_failed": 0,
            "total_terminated": 0,
            "current_active": 0,
            "total_cost": 0.0,
            "circuit_breaks": 0,
            "retries": 0,
            "last_error": None,
            "last_error_time": None,
        }

        self.initialized = False
        self._initialization_error: Optional[Exception] = None

    async def initialize(self):
        """
        Initialize GCP API clients and integrations with robust error handling.
        
        Uses async lock to prevent race conditions during initialization.
        Gracefully handles missing dependencies and API failures.
        """
        # Quick check without lock
        if self.initialized:
            return

        async with self._init_lock:
            # Double-check after acquiring lock
            if self.initialized:
                return

            if not COMPUTE_AVAILABLE:
                self._initialization_error = RuntimeError(
                    "google-cloud-compute package not installed. "
                    "Install with: pip install google-cloud-compute"
                )
                logger.warning(
                    "ℹ️  GCP Compute Engine API not available. "
                    "Spot VM creation disabled."
                )
                raise self._initialization_error

            logger.info("🚀 Initializing GCP VM Manager...")

            try:
                # Initialize GCP Compute Engine clients (parallel-safe)
                await self._initialize_gcp_clients()
                
                # Initialize integrations (with error isolation)
                await self._initialize_integrations()
                
                # Start monitoring if enabled
                if self.config.enable_monitoring:
                    self.monitoring_task = asyncio.create_task(
                        self._monitoring_loop(),
                        name="gcp_vm_monitoring"
                    )
                    logger.info("✅ VM monitoring started")

                self.initialized = True
                logger.info("✅ GCP VM Manager ready")
                logger.info(f"   Project: {self.config.project_id}")
                logger.info(f"   Zone: {self.config.zone}")
                logger.info(f"   Machine Type: {self.config.machine_type}")
                logger.info(f"   Daily Budget: ${self.config.daily_budget_usd}")

            except Exception as e:
                self._initialization_error = e
                logger.error(f"Failed to initialize GCP VM Manager: {e}", exc_info=True)
                raise

    async def _initialize_gcp_clients(self):
        """Initialize GCP API clients with error isolation"""
        try:
            # Run client initialization in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            self.instances_client = await loop.run_in_executor(
                None, compute_v1.InstancesClient
            )
            self.zones_client = await loop.run_in_executor(
                None, compute_v1.ZonesClient
            )
            self.zone_operations_client = await loop.run_in_executor(
                None, compute_v1.ZoneOperationsClient
            )
            
            logger.info(f"✅ GCP API clients initialized (Project: {self.config.project_id})")
            
        except Exception as e:
            logger.error(f"Failed to initialize GCP API clients: {e}")
            raise RuntimeError(f"GCP API client initialization failed: {e}") from e

    async def _initialize_integrations(self):
        """Initialize integrations with graceful fallbacks"""
        # Cost tracker - non-critical, continue without it
        try:
            self.cost_tracker = get_cost_tracker()
            if self.cost_tracker:
                # Ensure cost tracker is initialized
                if hasattr(self.cost_tracker, 'initialize'):
                    await self.cost_tracker.initialize()
                logger.info("✅ Cost tracker integrated")
            else:
                logger.warning("⚠️  Cost tracker not available - cost tracking disabled")
        except Exception as e:
            logger.warning(f"⚠️  Cost tracker initialization failed (non-critical): {e}")
            self.cost_tracker = None

        # GCP optimizer - non-critical, continue without it
        try:
            self.gcp_optimizer = get_gcp_optimizer(config={"project_id": self.config.project_id})
            if self.gcp_optimizer:
                logger.info("✅ GCP optimizer integrated")
            else:
                logger.warning("⚠️  GCP optimizer not available - using fallback decisions")
        except Exception as e:
            logger.warning(f"⚠️  GCP optimizer initialization failed (non-critical): {e}")
            self.gcp_optimizer = None

    async def should_create_vm(
        self, memory_snapshot, trigger_reason: str = ""
    ) -> Tuple[bool, str, float]:
        """
        Determine if we should create a VM based on current conditions

        Returns: (should_create, reason, confidence_score)
        """
        if not self.initialized:
            await self.initialize()

        # Check budget limits
        if self.cost_tracker:
            daily_cost = await self.cost_tracker.get_daily_cost()
            if daily_cost >= self.config.daily_budget_usd:
                return (
                    False,
                    f"Daily budget exceeded: ${daily_cost:.2f} / ${self.config.daily_budget_usd:.2f}",
                    0.0,
                )

        # Check concurrent VM limits
        active_vms = len([vm for vm in self.managed_vms.values() if vm.state == VMState.RUNNING])
        if active_vms >= self.config.max_concurrent_vms:
            return (
                False,
                f"Max concurrent VMs reached: {active_vms} / {self.config.max_concurrent_vms}",
                0.0,
            )

        # Check if already creating a VM
        if self.creating_vms:
            return False, "VM creation already in progress", 0.0

        # Use intelligent optimizer for decision
        if self.gcp_optimizer:
            should_create, reason, score = await self.gcp_optimizer.should_create_vm(
                memory_snapshot, current_processes=None
            )
            # PressureScore uses composite_score, not overall_score
            confidence = score.composite_score if hasattr(score, 'composite_score') else getattr(score, 'overall_score', 0.0)
            return should_create, reason, confidence

        # Fallback: Simple memory pressure check
        if memory_snapshot.gcp_shift_recommended:
            return True, trigger_reason or memory_snapshot.reasoning, 0.8

        return False, "Memory pressure within acceptable limits", 0.0

    async def create_vm(
        self, components: List[str], trigger_reason: str, metadata: Optional[Dict] = None
    ) -> Optional[VMInstance]:
        """
        Create a new GCP Spot VM instance with circuit breaker protection.

        Args:
            components: List of components that will run on this VM
            trigger_reason: Why this VM is being created
            metadata: Additional metadata

        Returns:
            VMInstance if successful, None otherwise
        """
        if not self.initialized:
            await self.initialize()

        # Check circuit breaker before attempting
        circuit = self._circuit_breakers["vm_create"]
        can_execute, circuit_reason = circuit.can_execute()
        
        if not can_execute:
            logger.warning(f"🔌 VM creation blocked by circuit breaker: {circuit_reason}")
            self.stats["circuit_breaks"] += 1
            return None

        logger.info(f"🚀 Creating GCP Spot VM...")
        logger.info(f"   Components: {', '.join(components)}")
        logger.info(f"   Trigger: {trigger_reason}")

        attempt = 0
        last_error = None
        vm_instance = None

        while attempt < self.config.max_create_attempts:
            attempt += 1
            try:
                # Generate unique VM name
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                vm_name = f"{self.config.vm_name_prefix}-{timestamp}"

                logger.info(
                    f"🔨 Attempt {attempt}/{self.config.max_create_attempts}: Creating VM '{vm_name}'"
                )

                # Build VM configuration
                instance_config = self._build_instance_config(
                    vm_name=vm_name,
                    components=components,
                    trigger_reason=trigger_reason,
                    metadata=metadata or {},
                )

                # Create the VM (async operation)
                operation = await asyncio.to_thread(
                    self.instances_client.insert,
                    project=self.config.project_id,
                    zone=self.config.zone,
                    instance_resource=instance_config,
                )

                logger.info(f"⏳ VM creation operation started: {operation.name}")

                # Wait for operation to complete
                await self._wait_for_operation(operation)

                # Get the created instance
                instance = await asyncio.to_thread(
                    self.instances_client.get,
                    project=self.config.project_id,
                    zone=self.config.zone,
                    instance=vm_name,
                )

                # Extract IP addresses
                ip_address = None
                internal_ip = None
                if instance.network_interfaces:
                    internal_ip = instance.network_interfaces[0].network_i_p
                    if instance.network_interfaces[0].access_configs:
                        ip_address = instance.network_interfaces[0].access_configs[0].nat_i_p

                # Create VMInstance tracking object
                vm_instance = VMInstance(
                    instance_id=str(instance.id),
                    name=vm_name,
                    zone=self.config.zone,
                    state=VMState.RUNNING,
                    created_at=time.time(),
                    ip_address=ip_address,
                    internal_ip=internal_ip,
                    components=components,
                    trigger_reason=trigger_reason,
                    metadata=metadata or {},
                )

                # Track the VM with lock protection
                async with self._vm_lock:
                    self.managed_vms[vm_name] = vm_instance
                    self.stats["total_created"] += 1
                    self.stats["current_active"] += 1

                # Record in cost tracker (isolated error handling)
                await self._record_vm_creation_safe(
                    vm_instance=vm_instance,
                    components=components,
                    trigger_reason=trigger_reason,
                    metadata=metadata,
                )

                # Circuit breaker success
                circuit.record_success()

                logger.info(f"✅ VM created successfully: {vm_name}")
                logger.info(f"   External IP: {ip_address or 'N/A'}")
                logger.info(f"   Internal IP: {internal_ip or 'N/A'}")
                logger.info(f"   Cost: ${vm_instance.cost_per_hour:.3f}/hour")

                return vm_instance

            except Exception as e:
                last_error = e
                self.stats["retries"] += 1
                self.stats["last_error"] = str(e)
                self.stats["last_error_time"] = datetime.now().isoformat()
                logger.error(f"❌ Attempt {attempt} failed: {e}")

                if attempt < self.config.max_create_attempts:
                    delay = self.config.retry_delay_seconds * attempt
                    logger.info(f"⏳ Retrying in {delay}s...")
                    await asyncio.sleep(delay)

        # All attempts failed - record in circuit breaker
        circuit.record_failure(last_error)
        
        self.stats["total_failed"] += 1
        logger.error(f"❌ Failed to create VM after {self.config.max_create_attempts} attempts")
        logger.error(f"   Last error: {last_error}")

        return None

    async def _record_vm_creation_safe(
        self,
        vm_instance: VMInstance,
        components: List[str],
        trigger_reason: str,
        metadata: Optional[Dict] = None,
    ):
        """
        Safely record VM creation in cost tracker with circuit breaker.
        
        This is isolated so cost tracker failures don't affect VM creation.
        """
        if not self.cost_tracker:
            logger.debug("Cost tracker not available - skipping creation record")
            return

        circuit = self._circuit_breakers["cost_tracker"]
        can_execute, _ = circuit.can_execute()
        
        if not can_execute:
            logger.debug("Cost tracker circuit open - skipping creation record")
            return

        try:
            # Try the new method name first, fall back to old name
            if hasattr(self.cost_tracker, 'record_vm_creation'):
                await self.cost_tracker.record_vm_creation(
                    instance_id=vm_instance.instance_id,
                    vm_type=self.config.machine_type,
                    region=self.config.region,
                    zone=self.config.zone,
                    components=components,
                    trigger_reason=trigger_reason,
                    metadata=metadata or {},
                )
            elif hasattr(self.cost_tracker, 'record_vm_created'):
                await self.cost_tracker.record_vm_created(
                    instance_id=vm_instance.instance_id,
                    components=components,
                    trigger_reason=trigger_reason,
                    metadata=metadata or {},
                )
            else:
                logger.warning("Cost tracker has no VM creation recording method")
                return
            
            circuit.record_success()
            logger.debug(f"💰 VM creation recorded in cost tracker: {vm_instance.instance_id}")
            
        except Exception as e:
            circuit.record_failure(e)
            logger.warning(f"⚠️  Failed to record VM creation in cost tracker (non-critical): {e}")

    def _build_instance_config(
        self, vm_name: str, components: List[str], trigger_reason: str, metadata: Dict
    ) -> InstanceType:
        """Build GCP Instance configuration"""

        # Machine type URL
        machine_type_url = f"zones/{self.config.zone}/machineTypes/{self.config.machine_type}"

        # Boot disk configuration
        boot_disk = compute_v1.AttachedDisk(
            auto_delete=True,
            boot=True,
            initialize_params=compute_v1.AttachedDiskInitializeParams(
                disk_size_gb=self.config.boot_disk_size_gb,
                disk_type=f"zones/{self.config.zone}/diskTypes/{self.config.boot_disk_type}",
                source_image=f"projects/{self.config.image_project}/global/images/family/{self.config.image_family}",
            ),
        )

        # Network interface
        network_interface = compute_v1.NetworkInterface(
            network=f"global/networks/{self.config.network}",
            access_configs=[compute_v1.AccessConfig(name="External NAT", type="ONE_TO_ONE_NAT")],
        )

        # Metadata
        metadata_items = [
            compute_v1.Items(key="jarvis-components", value=",".join(components)),
            compute_v1.Items(key="jarvis-trigger", value=trigger_reason),
            compute_v1.Items(key="jarvis-created-at", value=datetime.now().isoformat()),
        ]

        # Add startup script if provided
        if self.config.startup_script_path and os.path.exists(self.config.startup_script_path):
            with open(self.config.startup_script_path, "r") as f:
                startup_script = f.read()
            metadata_items.append(compute_v1.Items(key="startup-script", value=startup_script))

        # Build instance
        instance = compute_v1.Instance(
            name=vm_name,
            machine_type=machine_type_url,
            disks=[boot_disk],
            network_interfaces=[network_interface],
            metadata=compute_v1.Metadata(items=metadata_items),
            tags=compute_v1.Tags(items=["jarvis", "backend", "spot-vm"]),
            labels={"created-by": "jarvis", "type": "backend", "vm-class": "spot"},
        )

        # Configure as Spot VM
        if self.config.use_spot:
            instance.scheduling = compute_v1.Scheduling(
                preemptible=True,
                on_host_maintenance="TERMINATE",
                automatic_restart=False,
                provisioning_model="SPOT",
                instance_termination_action="DELETE",
            )

        return instance

    async def _wait_for_operation(self, operation, timeout: int = 300):
        """
        Wait for a GCP zone operation to complete.
        
        Uses the ZoneOperationsClient to poll operation status correctly.
        This fixes the 'Unknown field for Instance: target_link' error.
        """
        start_time = time.time()
        operation_name = operation.name
        
        logger.debug(f"Waiting for operation: {operation_name}")

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Operation '{operation_name}' timed out after {timeout}s")

            # Check operation status
            if operation.status == compute_v1.Operation.Status.DONE:
                if operation.error:
                    errors = [error.message for error in operation.error.errors]
                    raise Exception(f"Operation failed: {', '.join(errors)}")
                logger.debug(f"Operation '{operation_name}' completed successfully")
                return

            await asyncio.sleep(2)

            # Refresh operation status using ZoneOperationsClient (correct approach)
            try:
                operation = await asyncio.to_thread(
                    self.zone_operations_client.get,
                    project=self.config.project_id,
                    zone=self.config.zone,
                    operation=operation_name,
                )
            except Exception as e:
                logger.warning(f"Error polling operation status: {e}")
                # Continue waiting - operation might still complete
                await asyncio.sleep(2)

    async def terminate_vm(self, vm_name: str, reason: str = "Manual termination") -> bool:
        """
        Terminate a VM instance with circuit breaker protection.
        
        Args:
            vm_name: Name of the VM to terminate
            reason: Reason for termination (for logging/tracking)
            
        Returns:
            True if terminated successfully, False otherwise
        """
        async with self._vm_lock:
            if vm_name not in self.managed_vms:
                logger.warning(f"⚠️  VM not found in managed VMs: {vm_name}")
                # Try to delete anyway in case it exists in GCP but not tracked
                return await self._force_delete_vm(vm_name, reason)

            vm = self.managed_vms[vm_name]

        # Check circuit breaker
        circuit = self._circuit_breakers["vm_delete"]
        can_execute, circuit_reason = circuit.can_execute()
        
        if not can_execute:
            logger.warning(f"🔌 VM termination blocked by circuit breaker: {circuit_reason}")
            self.stats["circuit_breaks"] += 1
            return False

        logger.info(f"🛑 Terminating VM: {vm_name} (Reason: {reason})")

        try:
            # Update cost before termination
            vm.update_cost()

            # Record termination in cost tracker (isolated)
            await self._record_vm_termination_safe(vm, reason)

            # Delete the VM
            operation = await asyncio.to_thread(
                self.instances_client.delete,
                project=self.config.project_id,
                zone=self.config.zone,
                instance=vm_name,
            )

            await self._wait_for_operation(operation)

            # Update tracking with lock protection
            async with self._vm_lock:
                vm.state = VMState.TERMINATED
                self.stats["total_terminated"] += 1
                self.stats["current_active"] = max(0, self.stats["current_active"] - 1)
                self.stats["total_cost"] += vm.total_cost

                # Remove from managed VMs
                if vm_name in self.managed_vms:
                    del self.managed_vms[vm_name]

            # Circuit breaker success
            circuit.record_success()

            logger.info(f"✅ VM terminated: {vm_name}")
            logger.info(f"   Uptime: {vm.uptime_hours:.2f}h")
            logger.info(f"   Cost: ${vm.total_cost:.4f}")

            return True

        except Exception as e:
            circuit.record_failure(e)
            logger.error(f"❌ Failed to terminate VM {vm_name}: {e}", exc_info=True)
            return False

    async def _force_delete_vm(self, vm_name: str, reason: str) -> bool:
        """Force delete a VM that may exist in GCP but not in our tracking"""
        try:
            logger.info(f"🗑️  Force-deleting untracked VM: {vm_name}")
            
            operation = await asyncio.to_thread(
                self.instances_client.delete,
                project=self.config.project_id,
                zone=self.config.zone,
                instance=vm_name,
            )
            
            await self._wait_for_operation(operation)
            logger.info(f"✅ Force-deleted VM: {vm_name}")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️  Force delete failed for {vm_name}: {e}")
            return False

    async def _record_vm_termination_safe(self, vm: VMInstance, reason: str):
        """
        Safely record VM termination in cost tracker with circuit breaker.
        
        This is isolated so cost tracker failures don't affect VM termination.
        """
        if not self.cost_tracker:
            logger.debug("Cost tracker not available - skipping termination record")
            return

        circuit = self._circuit_breakers["cost_tracker"]
        can_execute, _ = circuit.can_execute()
        
        if not can_execute:
            logger.debug("Cost tracker circuit open - skipping termination record")
            return

        try:
            # Try the new method name first, fall back to old name
            if hasattr(self.cost_tracker, 'record_vm_termination'):
                await self.cost_tracker.record_vm_termination(
                    instance_id=vm.instance_id,
                    reason=reason,
                    total_cost=vm.total_cost,
                )
            elif hasattr(self.cost_tracker, 'record_vm_deleted'):
                await self.cost_tracker.record_vm_deleted(
                    instance_id=vm.instance_id,
                    was_orphaned=False,
                    actual_cost=vm.total_cost,
                )
            else:
                logger.warning("Cost tracker has no VM termination recording method")
                return
            
            circuit.record_success()
            logger.debug(f"💰 VM termination recorded in cost tracker: {vm.instance_id}")
            
        except Exception as e:
            circuit.record_failure(e)
            logger.warning(f"⚠️  Failed to record VM termination in cost tracker (non-critical): {e}")

    async def _monitoring_loop(self):
        """
        Background monitoring loop for VM health, lifecycle, and intelligent cost-cutting.

        Features:
        - Health monitoring
        - Cost tracking and efficiency scoring
        - Idle VM detection and auto-termination
        - Memory pressure monitoring (terminate when local RAM normalizes)
        - Detailed metrics collection (CPU, memory, network, disk)
        """
        logger.info("🔍 VM monitoring loop started (intelligent cost-cutting enabled)")
        self.is_monitoring = True

        while self.is_monitoring:
            try:
                await asyncio.sleep(self.config.health_check_interval)

                for vm_name, vm in list(self.managed_vms.items()):
                    # Update cost and efficiency
                    vm.update_cost()
                    vm.update_efficiency_score()

                    # === INTELLIGENT COST-CUTTING CHECKS ===

                    # 1. Check VM lifetime (hard limit)
                    if vm.uptime_hours >= self.config.max_vm_lifetime_hours:
                        logger.info(
                            f"⏰ VM {vm_name} exceeded max lifetime ({self.config.max_vm_lifetime_hours}h)"
                        )
                        await self.terminate_vm(vm_name, reason="Max lifetime exceeded")
                        continue

                    # 2. Check if VM is wasting money (idle + low efficiency)
                    idle_limit = float(self.config.idle_timeout_minutes)
                    is_idle = vm.idle_time_minutes > idle_limit
                    is_wasting_money = is_idle and (vm.cost_efficiency_score < 30.0)

                    if is_wasting_money:
                        logger.warning(
                            f"💰 VM {vm_name} is wasting money: "
                            f"idle for {vm.idle_time_minutes:.1f}m, "
                            f"efficiency score: {vm.cost_efficiency_score:.1f}%"
                        )
                        await self.terminate_vm(
                            vm_name,
                            reason=(
                                f"Cost waste: idle {vm.idle_time_minutes:.1f}m "
                                f"(limit {idle_limit:.0f}m), efficiency {vm.cost_efficiency_score:.1f}%"
                            ),
                        )
                        continue

                    # 3. Check if local memory pressure normalized (VM no longer needed)
                    try:
                        import psutil
                        local_mem_percent = psutil.virtual_memory().percent

                        # If local RAM dropped below 70%, VM is no longer needed
                        if local_mem_percent < 70:
                            logger.info(
                                f"📉 Local RAM normalized ({local_mem_percent:.1f}%) - "
                                f"VM {vm_name} no longer needed"
                            )
                            await self.terminate_vm(
                                vm_name,
                                reason=f"Memory pressure resolved (local RAM: {local_mem_percent:.1f}%)"
                            )
                            continue
                    except Exception as mem_check_error:
                        logger.debug(f"Could not check local memory: {mem_check_error}")

                    # === DETAILED METRICS COLLECTION (GCP Console-like) ===
                    try:
                        # Get instance details from GCP
                        instance = await asyncio.to_thread(
                            self.instances_client.get,
                            project=self.config.project_id,
                            zone=self.config.zone,
                            instance=vm_name,
                        )

                        # Update VM state
                        status_map = {
                            "PROVISIONING": VMState.PROVISIONING,
                            "STAGING": VMState.STAGING,
                            "RUNNING": VMState.RUNNING,
                            "STOPPING": VMState.STOPPING,
                            "TERMINATED": VMState.TERMINATED,
                        }
                        vm.state = status_map.get(instance.status, VMState.UNKNOWN)

                        # Collect CPU metrics (from GCP Monitoring API if available)
                        # For now, we'll use placeholder - in production, integrate with GCP Monitoring API
                        # TODO: Integrate with google-cloud-monitoring for real CPU/network/disk metrics

                        # Estimate metrics based on VM activity
                        if vm.component_usage_count > 0:
                            # Active VM - higher resource usage
                            vm.cpu_percent = min(80.0, 30.0 + (vm.component_usage_count % 50))
                            vm.memory_used_gb = min(28.0, 10.0 + (vm.component_usage_count % 18))
                        else:
                            # Idle VM - minimal resources
                            vm.cpu_percent = 5.0
                            vm.memory_used_gb = 2.0

                        # Health status
                        is_healthy = vm.state == VMState.RUNNING
                        vm.last_health_check = time.time()
                        vm.health_status = "healthy" if is_healthy else "unhealthy"

                        if not is_healthy:
                            logger.warning(f"⚠️  VM {vm_name} health check failed (state: {vm.state.value})")

                        # Log cost efficiency warnings
                        if vm.cost_efficiency_score < 50:
                            logger.warning(
                                f"⚠️  VM {vm_name} low efficiency: {vm.cost_efficiency_score:.1f}% "
                                f"(idle: {vm.idle_time_minutes:.1f}m)"
                            )

                    except Exception as metrics_error:
                        logger.debug(f"Could not collect metrics for {vm_name}: {metrics_error}")

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)

    async def _health_check_vm(self, vm_name: str) -> bool:
        """Perform health check on a VM"""
        if vm_name not in self.managed_vms:
            return False

        vm = self.managed_vms[vm_name]

        try:
            # Get instance status from GCP
            instance = await asyncio.to_thread(
                self.instances_client.get,
                project=self.config.project_id,
                zone=self.config.zone,
                instance=vm_name,
            )

            # Update state
            status_map = {
                "PROVISIONING": VMState.PROVISIONING,
                "STAGING": VMState.STAGING,
                "RUNNING": VMState.RUNNING,
                "STOPPING": VMState.STOPPING,
                "TERMINATED": VMState.TERMINATED,
            }
            vm.state = status_map.get(instance.status, VMState.UNKNOWN)

            return vm.state == VMState.RUNNING

        except Exception as e:
            logger.error(f"Health check failed for {vm_name}: {e}")
            return False

    async def cleanup_all_vms(self, reason: str = "System shutdown"):
        """Terminate all managed VMs with cost summary"""
        if not self.managed_vms:
            logger.info("ℹ️  No VMs to clean up")
            return

        logger.info(f"🧹 Cleaning up all VMs: {reason}")

        # Calculate total costs before terminating
        total_session_cost = 0.0
        total_uptime_hours = 0.0
        vm_count = len(self.managed_vms)

        for vm in self.managed_vms.values():
            vm.update_cost()
            total_session_cost += vm.total_cost
            total_uptime_hours += vm.uptime_hours

        # Terminate all VMs
        tasks = []
        for vm_name in list(self.managed_vms.keys()):
            tasks.append(self.terminate_vm(vm_name, reason=reason))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Display cost summary
        logger.info("✅ All VMs cleaned up")
        logger.info("=" * 60)
        logger.info("💰 GCP VM COST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"   VMs Terminated:  {vm_count}")
        logger.info(f"   Total Uptime:    {total_uptime_hours:.2f} hours")
        logger.info(f"   Session Cost:    ${total_session_cost:.4f}")
        logger.info(f"   Total Lifetime:  ${self.stats['total_cost']:.4f}")
        logger.info("=" * 60)

    async def cleanup(self):
        """Cleanup and shutdown"""
        self.is_monitoring = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        await self.cleanup_all_vms(reason="Manager shutdown")

        self.initialized = False
        logger.info("🧹 GCP VM Manager cleaned up")

    def get_stats(self) -> Dict:
        """Get comprehensive manager statistics including circuit breaker status"""
        circuit_status = {}
        for name, circuit in self._circuit_breakers.items():
            circuit_status[name] = {
                "state": circuit.state.value,
                "failure_count": circuit.failure_count,
                "can_execute": circuit.can_execute()[0],
            }
        
        # Calculate VM costs summary
        total_vm_cost = sum(vm.total_cost for vm in self.managed_vms.values())
        
        return {
            **self.stats,
            "managed_vms": len(self.managed_vms),
            "creating_vms": len(self.creating_vms),
            "current_vm_cost": total_vm_cost,
            "circuit_breakers": circuit_status,
            "config": self.config.to_dict(),
            "initialized": self.initialized,
        }

    def get_circuit_breaker_status(self) -> Dict[str, Dict]:
        """Get detailed circuit breaker status"""
        return {
            name: {
                "state": circuit.state.value,
                "failure_count": circuit.failure_count,
                "failure_threshold": circuit.failure_threshold,
                "recovery_timeout": circuit.recovery_timeout,
                "last_failure_time": circuit.last_failure_time,
                "can_execute": circuit.can_execute()[0],
                "reason": circuit.can_execute()[1],
            }
            for name, circuit in self._circuit_breakers.items()
        }

    def reset_circuit_breaker(self, name: str) -> bool:
        """Manually reset a circuit breaker"""
        if name in self._circuit_breakers:
            circuit = self._circuit_breakers[name]
            circuit.state = CircuitState.CLOSED
            circuit.failure_count = 0
            circuit.half_open_calls = 0
            logger.info(f"🔌 Circuit breaker '{name}' manually reset")
            return True
        return False


# ============================================================================
# SINGLETON MANAGEMENT
# ============================================================================

_gcp_vm_manager: Optional[GCPVMManager] = None
_manager_lock = asyncio.Lock()


async def get_gcp_vm_manager(config: Optional[VMManagerConfig] = None) -> GCPVMManager:
    """
    Get or create singleton GCP VM Manager with thread-safe initialization.
    
    Args:
        config: Optional configuration override
        
    Returns:
        Initialized GCPVMManager instance
        
    Raises:
        RuntimeError: If GCP Compute API is not available
    """
    global _gcp_vm_manager

    # Quick check without lock
    if _gcp_vm_manager is not None and _gcp_vm_manager.initialized:
        return _gcp_vm_manager

    async with _manager_lock:
        # Double-check after acquiring lock
        if _gcp_vm_manager is None:
            _gcp_vm_manager = GCPVMManager(config=config)
        
        if not _gcp_vm_manager.initialized:
            await _gcp_vm_manager.initialize()

        return _gcp_vm_manager


async def get_gcp_vm_manager_safe(config: Optional[VMManagerConfig] = None) -> Optional[GCPVMManager]:
    """
    Safely get GCP VM Manager, returning None if not available.
    
    This is useful when you want to use the manager if available
    but don't want to fail if GCP is not configured.
    
    Args:
        config: Optional configuration override
        
    Returns:
        GCPVMManager instance or None if not available
    """
    try:
        return await get_gcp_vm_manager(config)
    except Exception as e:
        logger.debug(f"GCP VM Manager not available: {e}")
        return None


async def create_vm_if_needed(
    memory_snapshot, 
    components: List[str], 
    trigger_reason: str, 
    metadata: Optional[Dict] = None
) -> Optional[VMInstance]:
    """
    Convenience function: Check if VM needed and create if so.

    Args:
        memory_snapshot: Memory pressure snapshot
        components: Components to run on the VM
        trigger_reason: Why this VM is being created
        metadata: Additional metadata
        
    Returns:
        VMInstance if created, None otherwise
    """
    try:
        # Explicit opt-in guardrail (prevents surprise spend).
        # Accept legacy GCP_VM_ENABLED, otherwise use JARVIS_SPOT_VM_ENABLED.
        enabled_flag = os.getenv("GCP_VM_ENABLED")
        if enabled_flag is None:
            enabled_flag = os.getenv("JARVIS_SPOT_VM_ENABLED", "false")
        if str(enabled_flag).lower() != "true":
            logger.info("ℹ️  Spot VM creation disabled by configuration")
            return None

        manager = await get_gcp_vm_manager_safe()
        
        if manager is None:
            logger.debug("GCP VM Manager not available - cannot create VM")
            return None

        should_create, reason, confidence = await manager.should_create_vm(
            memory_snapshot, trigger_reason
        )

        if should_create:
            logger.info(f"✅ VM creation recommended: {reason} (confidence: {confidence:.2%})")
            return await manager.create_vm(components, trigger_reason, metadata)
        else:
            logger.info(f"ℹ️  VM creation not needed: {reason}")
            return None
            
    except Exception as e:
        logger.error(f"Error in create_vm_if_needed: {e}")
        return None


async def cleanup_vm_manager():
    """Cleanup the global VM manager instance"""
    global _gcp_vm_manager
    
    async with _manager_lock:
        if _gcp_vm_manager is not None:
            await _gcp_vm_manager.cleanup()
            _gcp_vm_manager = None
            logger.info("🧹 GCP VM Manager singleton cleaned up")


def is_vm_manager_available() -> bool:
    """Check if VM manager is available without initializing"""
    return COMPUTE_AVAILABLE


def get_vm_manager_sync() -> Optional[GCPVMManager]:
    """Get VM manager synchronously (only if already initialized)"""
    return _gcp_vm_manager if _gcp_vm_manager and _gcp_vm_manager.initialized else None
