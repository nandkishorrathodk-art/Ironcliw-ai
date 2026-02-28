# Phase 3.0 Architecture Upgrade Plan - Enterprise-Grade Trinity Ecosystem

**Status**: ✅ Complete
**Version**: 3.0.0
**Date**: January 14, 2026

---

## 🎯 Objective

Transform the Trinity Ecosystem (Ironcliw, J-Prime, Reactor-Core) from a "good bones" implementation to **true Enterprise Grade** with:
- Dynamic service discovery (zero hardcoded ports)
- Self-healing process orchestration
- High-performance data pipelines
- Robust IPC mechanisms
- System hardening against all edge cases

---

## 📦 Components Being Upgraded

### ✅ COMPLETED: Service Registry System
**File**: `backend/core/service_registry.py` (NEW - 600+ lines)

**Features Implemented**:
- ✅ File-based service registry (`~/.jarvis/registry/services.json`)
- ✅ Atomic file operations with `fcntl` locking (thread/process safe)
- ✅ Automatic stale service cleanup (dead PIDs, timeout heartbeats)
- ✅ Dynamic service discovery (no hardcoded ports/URLs)
- ✅ Health tracking with heartbeat system
- ✅ Background cleanup task with configurable interval

**Key APIs**:
```python
registry = ServiceRegistry()

# Register service
await registry.register_service(
    service_name="jarvis-prime",
    pid=os.getpid(),
    port=8002,
    health_endpoint="/health"
)

# Discover service dynamically
service = await registry.discover_service("jarvis-prime")
url = f"http://{service.host}:{service.port}{service.health_endpoint}"

# Heartbeat to stay alive
await registry.heartbeat("jarvis-prime", status="healthy")

# Wait for service availability
service = await registry.wait_for_service("reactor-core", timeout=30.0)
```

---

### 🚧 IN PROGRESS: Cross-Repo Orchestrator v3.0
**File**: `backend/supervisor/cross_repo_startup_orchestrator.py` (UPGRADING from v1.0)

**New Features** (v3.0):

#### 1. Process Lifecycle Management
- ✅ Spawn processes with `asyncio.create_subprocess_exec` (non-blocking)
- ✅ Track PIDs and manage process lifecycle
- ✅ Graceful shutdown with SIGTERM → wait → SIGKILL
- ✅ Automatic zombie process cleanup

#### 2. Output Streaming
- ✅ Capture stdout/stderr from child processes
- ✅ Stream logs in real-time to main Ironcliw log
- ✅ Prefix each line with `[SERVICE]` for easy filtering
- ✅ Non-blocking async streaming

Example log output:
```
[Ironcliw] Starting system...
[J-PRIME] Loading model...
[J-PRIME] Model loaded in 2.3s
[REACTOR] Initializing training pipeline...
[REACTOR] Ready to accept jobs
```

#### 3. Auto-Healing with Exponential Backoff
- ✅ Detect dead processes via PID monitoring
- ✅ Automatically restart crashed services
- ✅ Exponential backoff: 1s → 2s → 4s → 8s → 16s (max 5 attempts)
- ✅ Alert on repeated failures

#### 4. Dynamic Service Discovery Integration
- ✅ Read service info from registry (no hardcoded ports)
- ✅ Register services on launch
- ✅ Update service status in registry
- ✅ Deregister on shutdown

#### 5. Health Monitoring
- ✅ Continuous PID alive checks
- ✅ HTTP health endpoint probing
- ✅ Heartbeat timeout detection
- ✅ Service status reporting

**Class Structure**:
```python
class ManagedProcess:
    """Represents a managed child process."""
    service_name: str
    process: asyncio.subprocess.Process
    restart_count: int
    last_restart: float
    output_stream_task: asyncio.Task
    health_monitor_task: asyncio.Task

class ProcessOrchestrator:
    """Enterprise-grade process lifecycle manager."""

    async def spawn_service(service_name, script_path, port) -> ManagedProcess
    async def monitor_process(managed_process) -> None
    async def stream_output(process, prefix) -> None
    async def restart_service(managed_process) -> bool
    async def shutdown_service(managed_process) -> None
    async def start_all_services() -> Dict[str, bool]
```

---

### 📋 PENDING: Advanced Training Coordinator v3.0
**File**: `backend/intelligence/advanced_training_coordinator.py` (UPGRADING from v2.0)

**New Features** (v3.0):

#### 1. Parallel Data Serialization
**Problem**: Large experience lists (1000s of items) block event loop during JSON serialization

**Solution**: `ProcessPoolExecutor` for CPU-bound operations
```python
from concurrent.futures import ProcessPoolExecutor

async def prepare_training_data(experiences: List[Experience]) -> str:
    """Serialize experiences in background process pool."""
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor() as pool:
        json_data = await loop.run_in_executor(
            pool,
            _serialize_experiences,  # CPU-bound function
            experiences
        )
    return json_data
```

#### 2. Drop-Box Protocol (Shared Memory Transport)
**Problem**: Sending 100MB JSON payloads over HTTP is slow and memory-intensive

**Solution**: Write dataset to shared file, send only path
```python
# Coordinator writes dataset
dropbox_dir = Path("~/.jarvis/bridge/training_staging")
job_file = dropbox_dir / f"{job_id}.json"
await asyncio.to_thread(job_file.write_text, json_data)

# Send only path to Reactor Core
await reactor_client.start_training(
    job_id=job_id,
    dataset_path=str(job_file)  # Not the data itself!
)

# Reactor Core reads locally
dataset = json.loads(Path(dataset_path).read_text())
```

**Benefits**:
- ✅ Zero HTTP overhead for large datasets
- ✅ Non-blocking file I/O
- ✅ Automatic cleanup after training

#### 3. Persistent State Machine
**Problem**: If Ironcliw crashes mid-training, no way to resume

**Solution**: SQLite state tracking
```python
import aiosqlite

class TrainingStateManager:
    """Persistent state for training jobs."""

    async def save_job_state(self, job_id, status, metadata):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO training_jobs VALUES (?, ?, ?)",
                (job_id, status, json.dumps(metadata))
            )
            await db.commit()

    async def resume_active_jobs(self):
        """On startup, reconnect to any active training jobs."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT job_id FROM training_jobs WHERE status='running'"
            ) as cursor:
                active_jobs = await cursor.fetchall()

        for (job_id,) in active_jobs:
            await self.reconnect_to_training_stream(job_id)
```

---

### 📋 PENDING: Reactor Core FastAPI Interface
**File**: `reactor-core/reactor_api_interface.py` (NEW - for external repo)

**Purpose**: Provide a drop-in FastAPI router that Reactor Core can import

**Features**:
```python
from fastapi import FastAPI, APIRouter
from backend.core.service_registry import register_current_service

router = APIRouter(prefix="/api")

@router.on_event("startup")
async def register_service():
    """Register Reactor Core in service registry on startup."""
    await register_current_service(
        service_name="reactor-core",
        port=8090,
        health_endpoint="/health"
    )

@router.post("/training/start")
async def start_training(request: TrainingRequest):
    """Start training using drop-box protocol."""
    # Read dataset from shared file
    dataset_path = Path(request.dataset_path)
    experiences = json.loads(dataset_path.read_text())

    # Start training
    job = await training_engine.start(experiences)

    # Clean up dataset file
    dataset_path.unlink()

    return {"job_id": job.id, "status": "started"}

@router.get("/training/stream/{job_id}")
async def stream_status(job_id: str):
    """Stream training status via SSE."""
    async def event_generator():
        async for status in training_engine.stream_status(job_id):
            yield f"event: status\ndata: {status.json()}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

---

### 📋 PENDING: System Hardening

#### 1. Race Condition Prevention
```python
# In run_supervisor.py startup
CRITICAL_DIRS = [
    Path.home() / ".jarvis" / "registry",
    Path.home() / ".jarvis" / "bridge" / "training_staging",
    Path.home() / ".jarvis" / "reactor" / "events",
]

for directory in CRITICAL_DIRS:
    directory.mkdir(parents=True, exist_ok=True)
```

#### 2. Graceful Shutdown Handlers
```python
import signal
import sys

class ProcessOrchestrator:
    def __init__(self):
        # ... existing init

        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Handle SIGINT/SIGTERM for graceful shutdown."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        asyncio.create_task(self.shutdown_all_services())

    async def shutdown_all_services(self):
        """Clean shutdown of all child processes."""
        for managed_process in self.processes.values():
            # Try graceful shutdown first (SIGTERM)
            managed_process.process.terminate()

            try:
                # Wait up to 10s for graceful shutdown
                await asyncio.wait_for(
                    managed_process.process.wait(),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                # Force kill if necessary (SIGKILL)
                managed_process.process.kill()
                await managed_process.process.wait()

        # Deregister all services from registry
        registry = get_service_registry()
        for service_name in self.processes.keys():
            await registry.deregister_service(service_name)

        sys.exit(0)
```

---

## 🔧 Configuration (Zero Hardcoding)

All configuration via environment variables:

```bash
# Service Registry
export Ironcliw_REGISTRY_DIR=~/.jarvis/registry
export Ironcliw_HEARTBEAT_TIMEOUT=60  # seconds
export Ironcliw_CLEANUP_INTERVAL=30  # seconds

# Process Orchestration
export JPRIME_REPO_PATH=~/Documents/repos/jarvis-prime
export REACTOR_REPO_PATH=~/Documents/repos/reactor-core
export JPRIME_SCRIPT=main.py
export REACTOR_SCRIPT=main.py
export AUTO_HEALING_ENABLED=true
export MAX_RESTART_ATTEMPTS=5
export RESTART_BACKOFF_BASE=1.0  # seconds

# Drop-Box Protocol
export TRAINING_DROPBOX_DIR=~/.jarvis/bridge/training_staging
export DROPBOX_CLEANUP_ENABLED=true

# State Persistence
export TRAINING_STATE_DB=~/.jarvis/training_state.db
export AUTO_RESUME_JOBS=true
```

---

## 📊 Implementation Progress

| Component | Status | Progress |
|-----------|--------|----------|
| Service Registry | ✅ Complete | 100% |
| Process Orchestrator v3.0 | ✅ Complete | 100% |
| Training Coordinator v3.0 | ✅ Complete | 100% |
| Reactor Core Interface | ✅ Complete | 100% |
| System Hardening | ✅ Complete | 100% |
| Trinity IPC Hub v4.0 | ✅ Complete | 100% |
| Trinity Bridge v4.0 | ✅ Complete | 100% |
| Integration Testing | ✅ Complete | 100% |

### Completed Files

1. **`backend/core/service_registry.py`** (~540 lines)
   - File-based service registry with atomic operations
   - Dynamic service discovery (zero hardcoded ports)
   - Automatic stale service cleanup
   - Health tracking with heartbeats

2. **`backend/supervisor/cross_repo_startup_orchestrator.py`** (~860 lines)
   - Enterprise-grade process lifecycle manager
   - Auto-healing with exponential backoff
   - Real-time output streaming with service prefixes
   - Graceful shutdown with SIGTERM escalation

3. **`backend/intelligence/advanced_training_coordinator.py`** (~1578 lines)
   - ProcessPoolExecutor for parallel data serialization
   - Drop-Box Protocol for large dataset transfer
   - Persistent State Machine with SQLite
   - Auto-resume on startup

4. **`backend/reactor/reactor_api_interface.py`** (~650 lines)
   - Drop-in FastAPI router for Reactor Core
   - Drop-Box Protocol support
   - SSE streaming for training status
   - Service Registry integration

5. **`backend/core/system_hardening.py`** (~580 lines)
   - Critical directory initialization
   - Graceful shutdown orchestration
   - Resource leak prevention
   - System health monitoring

6. **`backend/core/trinity_ipc_hub.py`** (~1800 lines)
   - All 10 communication channels for Trinity ecosystem
   - Circuit breaker for resilience
   - Reliable message queue with exactly-once delivery
   - Dead letter queue for failed messages
   - Event bus with multi-cast Pub/Sub
   - Cross-repo RPC layer
   - Model registry with metadata

7. **`backend/core/trinity_bridge.py`** (~500 lines)
   - Unified integration layer for single-command startup
   - Automatic service discovery and registration
   - Cross-repo health monitoring
   - Graceful degradation when repos unavailable

8. **`tests/integration/test_phase3_integration.py`** (~650 lines)
   - 20 comprehensive integration tests
   - Tests for all Phase 3 components
   - Tests for Trinity IPC Hub channels
   - Tests for circuit breaker behavior

---

## 🎯 Success Criteria

**When complete, the following must work**:

### 1. Zero Configuration Startup
```bash
cd ~/Documents/repos/Ironcliw-AI-Agent
python3 run_supervisor.py
```

Expected behavior:
- ✅ Ironcliw starts and registers itself
- ✅ Discovers J-Prime is not running, launches it automatically
- ✅ J-Prime registers itself in service registry
- ✅ Discovers Reactor-Core is not running, launches it
- ✅ Reactor-Core registers itself
- ✅ All services discovered dynamically (no hardcoded URLs)

### 2. Auto-Healing
```bash
# Kill J-Prime process manually
kill -9 <jprime_pid>
```

Expected behavior:
- ✅ Process monitor detects J-Prime death within 1 second
- ✅ Automatically restarts J-Prime
- ✅ J-Prime re-registers in service registry
- ✅ Training coordinator reconnects seamlessly

### 3. Output Streaming
```bash
tail -f logs/jarvis*.log
```

Expected output:
```
[Ironcliw] System starting...
[J-PRIME] Loading model from /path/to/model.gguf
[J-PRIME] Model loaded successfully (2.3s)
[J-PRIME] Listening on port 8002
[REACTOR] Initializing training pipeline...
[REACTOR] Watching ~/.jarvis/trinity/events for experiences
[REACTOR] Ready to accept training jobs
```

### 4. Graceful Shutdown
```bash
# Press Ctrl+C in run_supervisor.py terminal
```

Expected behavior:
- ✅ SIGINT caught by supervisor
- ✅ Sends SIGTERM to J-Prime and Reactor-Core
- ✅ Waits up to 10s for graceful shutdown
- ✅ All services deregister from registry
- ✅ No zombie processes left behind

### 5. Drop-Box Training
```bash
# Trigger training with 10,000 experiences
```

Expected behavior:
- ✅ Coordinator serializes experiences in background process pool (non-blocking)
- ✅ Writes 50MB dataset to `~/.jarvis/bridge/training_staging/job_123.json`
- ✅ Sends only path to Reactor Core (not 50MB over HTTP)
- ✅ Reactor Core reads file locally
- ✅ File automatically deleted after training starts

---

## 🚀 Next Steps

1. ✅ Complete Service Registry implementation
2. ✅ Complete Process Orchestrator v3.0
3. ✅ Upgrade Training Coordinator v3.0
4. ✅ Generate Reactor Core FastAPI interface
5. ✅ Implement system hardening
6. ✅ Create Trinity IPC Hub v4.0 (all 10 communication channels)
7. ✅ Create Trinity Bridge v4.0 (unified integration layer)
8. ✅ Integrate into run_supervisor.py
9. ✅ Integration testing complete (20/20 tests passing)

---

## 🧪 Integration Testing Checklist

### Service Registry Tests
- [x] Service registration and discovery
- [x] Stale service cleanup
- [x] Heartbeat functionality
- [x] Atomic file operations under concurrent access

### Process Orchestrator Tests
- [x] Process spawning with registry integration
- [x] Auto-healing after process crash
- [x] Output streaming verification
- [x] Graceful shutdown sequence

### Training Coordinator Tests
- [x] Drop-box protocol for large datasets
- [x] State persistence across restarts
- [x] Auto-resume of interrupted training
- [x] ProcessPoolExecutor performance

### Reactor Core Interface Tests
- [x] Training API endpoint functionality
- [x] SSE streaming correctness
- [x] Drop-box dataset loading
- [x] Health endpoint response

### System Hardening Tests
- [x] Critical directory creation
- [x] Signal handler behavior
- [x] Resource cleanup on shutdown
- [x] Health monitoring accuracy

### Trinity IPC Hub Tests
- [x] IPC Hub initialization with all 10 channels
- [x] Model registry operations
- [x] Event bus pub/sub
- [x] Message queue with ACK/NACK
- [x] Training data pipeline
- [x] Circuit breaker behavior

---

## 🎉 Phase 3.0 Complete!

All components have been implemented, tested, and integrated. The Trinity Ecosystem
now has enterprise-grade architecture with:

- **Zero hardcoded configuration** - all ports/URLs discovered dynamically
- **Auto-healing processes** - crashed services restart automatically
- **All 10 communication channels** - complete cross-repo communication
- **Circuit breaker resilience** - graceful degradation under failure
- **Single-command startup** - `python3 run_supervisor.py` starts everything

---

**End of Plan Document**
