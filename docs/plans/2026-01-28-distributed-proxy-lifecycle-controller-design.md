# Distributed Proxy Lifecycle Controller Design

**Date:** 2026-01-28
**Version:** 1.0.0
**Status:** Approved for Implementation
**Author:** JARVIS System

## Executive Summary

This document describes the design for a production-grade, distributed Cloud SQL proxy lifecycle management system that provides:

1. **Cross-repo leader election** - Only one repo manages the proxy
2. **Bulletproof startup orchestration** - Multi-stage verification before proceeding
3. **launchd persistence** - Proxy survives reboots and auto-restarts on crash
4. **Intelligent health monitoring** - Anomaly detection with predictive restart
5. **Unified observability** - Audit trails and diagnostics across all repos

## Problem Statement

The JARVIS ecosystem (JARVIS, Prime, Reactor Core) experiences intermittent "Connection refused" errors when:
- The Cloud SQL proxy dies during sleep/wake cycles
- Multiple repos race to start the proxy simultaneously
- Components attempt database connections before proxy is verified ready
- Recovery attempts fail silently, leaving the system in a degraded state

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    JARVIS Unified Supervisor (run_supervisor.py)             │
│                         Single Entry Point for Everything                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  Layer 1: CROSS-REPO LEADER ELECTION                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  DistributedProxyLeader (Raft-inspired consensus)                   │    │
│  │  • File-based lock with heartbeat (no external dependencies)        │    │
│  │  • Leader manages proxy lifecycle, followers observe                │    │
│  │  • Automatic leader failover on crash/timeout                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────────────────┤
│  Layer 2: STARTUP BARRIER & DEPENDENCY GRAPH                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  AsyncStartupBarrier (Blocks until proxy VERIFIED ready)            │    │
│  │  • Multi-stage verification: TCP → Auth → Query → Latency           │    │
│  │  • Dependency injection: Components declare CloudSQL dependency     │    │
│  │  • Parallel init of non-dependent components while waiting          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────────────────┤
│  Layer 3: INTELLIGENT PROXY LIFECYCLE                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  ProxyLifecycleController (State Machine)                           │    │
│  │  States: STOPPED → STARTING → VERIFYING → READY → DEGRADED → DEAD  │    │
│  │  • launchd service for macOS persistence                           │    │
│  │  • Predictive restart based on latency trends                       │    │
│  │  • Wake-from-sleep detection and proactive restart                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────────────────┤
│  Layer 4: HEALTH & OBSERVABILITY                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  UnifiedHealthAggregator                                            │    │
│  │  • Real-time metrics: latency, error rate, connection pool status  │    │
│  │  • Anomaly detection with Z-score analysis                         │    │
│  │  • Event sourcing for complete audit trail                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Specifications

### 1. Distributed Proxy Leader (Layer 1)

**Purpose:** Ensure only one repo manages the proxy lifecycle at any time.

**State Machine:**
```
FOLLOWER ──[election timeout]──> CANDIDATE
    ^                                │
    │                                │
    │              [win election]    │
    │                    ┌───────────┘
    │                    v
    └────────────── LEADER
         [heartbeat timeout]
```

**Election Protocol:**
1. On startup, check if leader exists (read state file)
2. If leader heartbeat fresh (<lease_duration), become FOLLOWER
3. If no leader or stale heartbeat, start ELECTION
4. Election: random backoff, then try to acquire file lock
5. Winner becomes LEADER, writes heartbeat, manages proxy
6. Losers become FOLLOWERS, monitor leader health
7. If leader dies, FOLLOWERS detect via stale heartbeat → new election

**Configuration (Environment Variables):**
| Variable | Default | Description |
|----------|---------|-------------|
| `PROXY_LEADER_HEARTBEAT_INTERVAL` | 5.0 | Seconds between heartbeats |
| `PROXY_LEADER_ELECTION_TIMEOUT_MIN` | 3.0 | Minimum election timeout |
| `PROXY_LEADER_ELECTION_TIMEOUT_MAX` | 10.0 | Maximum election timeout |
| `PROXY_LEADER_LEASE_DURATION` | 15.0 | Heartbeat staleness threshold |

**Edge Cases:**
- Split-brain prevention via file locking
- Clock drift resistance via monotonic time
- Zombie leader detection via heartbeat timeout
- Crash recovery via stale lock detection

### 2. Async Startup Barrier (Layer 2)

**Purpose:** Block CloudSQL-dependent components until proxy is VERIFIED ready.

**Multi-Stage Verification Pipeline:**
```
Stage 1: TCP Connect      → Is port 5432 accepting connections?
Stage 2: TLS Handshake    → Can we establish secure connection?
Stage 3: Authentication   → Are credentials valid?
Stage 4: Query Execution  → Does SELECT 1 succeed?
Stage 5: Latency Check    → Is response time acceptable?
```

**Dependency Declaration:**
```python
class DependencyType(Enum):
    CLOUDSQL = "cloudsql"
    FILESYSTEM = "filesystem"
    NETWORK = "network"
    GCP_CREDENTIALS = "gcp_creds"
    VOICE_ENGINE = "voice_engine"
    NONE = "none"

@dataclass
class ComponentManifest:
    name: str
    dependencies: Set[DependencyType]
    init_func: Callable[[], Awaitable[bool]]
    priority: int = 50
    timeout: float = 30.0
    required: bool = True
```

**Parallel Initialization Waves:**
```
Wave 0 (No deps):     Memory, Config, Logging      [PARALLEL]
Wave 1 (Filesystem):  Vision, Audio capture        [PARALLEL]
Wave 2 (Network):     API clients, WebSocket       [PARALLEL]
─── CLOUDSQL BARRIER ─── (Wait for verification)
Wave 3 (CloudSQL):    Voice Unlock, Neural Mesh    [PARALLEL]
Wave 4 (All ready):   Cross-repo bridges           [PARALLEL]
```

**Configuration:**
| Variable | Default | Description |
|----------|---------|-------------|
| `CLOUDSQL_ENSURE_READY_TIMEOUT` | 60.0 | Max wait for proxy ready |
| `CLOUDSQL_RETRY_BASE_DELAY` | 1.0 | Initial retry delay |
| `CLOUDSQL_RETRY_MAX_DELAY` | 10.0 | Maximum retry delay |
| `CLOUDSQL_VERIFICATION_STAGES` | 5 | Number of stages |

### 3. Proxy Lifecycle Controller (Layer 3)

**Purpose:** Manage proxy process with state machine, launchd persistence, and predictive health.

**State Machine:**
```python
class ProxyState(Enum):
    UNKNOWN = "unknown"
    STOPPED = "stopped"
    STARTING = "starting"
    VERIFYING = "verifying"
    READY = "ready"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    DEAD = "dead"

VALID_TRANSITIONS = {
    ProxyState.UNKNOWN: {ProxyState.STOPPED, ProxyState.STARTING, ProxyState.READY},
    ProxyState.STOPPED: {ProxyState.STARTING},
    ProxyState.STARTING: {ProxyState.VERIFYING, ProxyState.STOPPED, ProxyState.DEAD},
    ProxyState.VERIFYING: {ProxyState.READY, ProxyState.STARTING, ProxyState.DEAD},
    ProxyState.READY: {ProxyState.DEGRADED, ProxyState.STOPPED, ProxyState.RECOVERING},
    ProxyState.DEGRADED: {ProxyState.READY, ProxyState.RECOVERING, ProxyState.DEAD},
    ProxyState.RECOVERING: {ProxyState.STARTING, ProxyState.READY, ProxyState.DEAD},
    ProxyState.DEAD: {ProxyState.STARTING},
}
```

**launchd Service (macOS Persistence):**
- Location: `~/Library/LaunchAgents/com.jarvis.cloudsql-proxy.plist`
- KeepAlive: Restart on crash
- RunAtLoad: Start on login
- ThrottleInterval: Prevent restart loops
- Environment: GOOGLE_APPLICATION_CREDENTIALS

**Sleep/Wake Detection:**
- Native: Darwin notification center via PyObjC
- Fallback: Poll system logs for sleep/wake events
- Action: Proactive restart after wake (configurable delay)

**Predictive Restart (ML-lite):**
- Track latency with Exponential Moving Average (EMA)
- Calculate baseline during warmup period
- Detect degradation trend (EMA > baseline × threshold)
- Preemptive restart before complete failure

**Configuration:**
| Variable | Default | Description |
|----------|---------|-------------|
| `PROXY_WAKE_DELAY` | 2.0 | Delay after wake before restart |
| `PROXY_EMA_ALPHA` | 0.3 | EMA smoothing factor |
| `PROXY_DEGRADATION_THRESHOLD` | 3.0 | Multiplier for degradation detection |
| `PROXY_PREEMPTIVE_RESTART` | true | Enable predictive restart |
| `PROXY_RESTART_THROTTLE` | 30 | launchd restart throttle seconds |

### 4. Unified Health Aggregator (Layer 4)

**Purpose:** Cross-repo health coordination with anomaly detection and audit trails.

**Health Snapshot Schema:**
```python
@dataclass
class HealthSnapshot:
    timestamp: float
    correlation_id: str
    source_repo: str

    # Proxy health
    proxy_state: ProxyState
    proxy_pid: Optional[int]
    proxy_uptime_seconds: float

    # Connection health
    tcp_latency_ms: float
    db_latency_ms: float
    tls_latency_ms: float

    # Pool health
    pool_size: int
    pool_available: int
    pool_waiting: int

    # Anomaly indicators
    latency_zscore: float
    is_anomaly: bool
    anomaly_type: Optional[str]
```

**Anomaly Detection:**
- Sliding window statistics (configurable size)
- Robust baseline learning (median + MAD)
- Z-score calculation for each measurement
- Threshold-based anomaly flagging (|Z| > 2.5)

**Event Sourcing:**
- Immutable append-only log (JSONL format)
- Location: `~/.jarvis/health_events.jsonl`
- Auto-rotation on size threshold
- Query API for debugging

**Automated Diagnosis:**
- Pattern detection around failure time
- Latency degradation analysis
- Connection pool exhaustion detection
- Sleep/wake correlation

**Configuration:**
| Variable | Default | Description |
|----------|---------|-------------|
| `HEALTH_WINDOW_SIZE` | 100 | Sliding window samples |
| `BASELINE_MIN_SAMPLES` | 20 | Samples before baseline |
| `ANOMALY_ZSCORE_THRESHOLD` | 2.5 | Z-score for anomaly |
| `DIAGNOSIS_WINDOW_SECONDS` | 300 | Analysis window |

## Cross-Repo Integration

### Startup Sequence

```
Phase 0: LEADER ELECTION (2-5 seconds)
  └─ Check for existing leader, elect if needed

Phase 1: PROXY LIFECYCLE SETUP (Leader only, 5-15 seconds)
  └─ Install launchd, start proxy, register notifications

Phase 2: MULTI-STAGE VERIFICATION (All repos, 5-30 seconds)
  └─ TCP → TLS → Auth → Query → Latency

Phase 3: PARALLEL COMPONENT INIT (10-30 seconds)
  └─ Wave-based initialization with CloudSQL barrier

Phase 4: CROSS-REPO STARTUP (Leader only)
  └─ Start Prime and Reactor as subprocesses
```

### Shared State File

Location: `~/.jarvis/cross_repo/unified_state.json`

```json
{
    "version": "1.0",
    "last_updated": 1706500000.123,
    "leader": {
        "id": "jarvis-main:12345:1706499000",
        "repo": "jarvis",
        "heartbeat": 1706500000.100,
        "proxy_state": "READY"
    },
    "followers": [
        {"id": "jarvis-prime:12346:1706499500", "repo": "prime"},
        {"id": "reactor-core:12347:1706499600", "repo": "reactor"}
    ],
    "proxy": {
        "state": "READY",
        "pid": 68319,
        "port": 5432,
        "uptime_seconds": 3600,
        "latency_ms": 5.2,
        "launchd_managed": true
    }
}
```

## Implementation Plan

### Files to Create

| File | Purpose |
|------|---------|
| `backend/core/proxy/__init__.py` | Package exports |
| `backend/core/proxy/distributed_leader.py` | Leader election |
| `backend/core/proxy/lifecycle_controller.py` | State machine + launchd |
| `backend/core/proxy/startup_barrier.py` | Multi-stage verification |
| `backend/core/proxy/health_aggregator.py` | Observability |
| `backend/core/proxy/orchestrator.py` | Unified orchestrator |

### Files to Modify

| File | Changes |
|------|---------|
| `run_supervisor.py` | Integrate orchestrator in startup |
| `backend/intelligence/cloud_sql_connection_manager.py` | Use new lifecycle controller |

### Files to Generate (Runtime)

| File | Purpose |
|------|---------|
| `~/Library/LaunchAgents/com.jarvis.cloudsql-proxy.plist` | launchd service |
| `~/.jarvis/cross_repo/unified_state.json` | Cross-repo state |
| `~/.jarvis/cross_repo/proxy_leader.lock` | Leader election lock |
| `~/.jarvis/health_events.jsonl` | Audit trail |

## Testing Strategy

### Unit Tests
- State machine transitions
- Leader election protocol
- Verification stages
- Anomaly detection math

### Integration Tests
- Full startup sequence
- Leader failover
- Sleep/wake recovery
- Cross-repo coordination

### Chaos Tests
- Kill proxy mid-operation
- Simulate sleep/wake
- Network partition (localhost firewall)
- Credential expiration

## Rollout Plan

1. **Phase 1:** Deploy to development environment
2. **Phase 2:** Enable with feature flag (`PROXY_NEW_LIFECYCLE=true`)
3. **Phase 3:** Gradual rollout (monitor for regressions)
4. **Phase 4:** Full deployment, remove feature flag

## Success Metrics

- **MTTR (Mean Time To Recovery):** < 10 seconds
- **False positive anomalies:** < 5%
- **Startup reliability:** 99.9%
- **Zero "Connection refused" errors** during normal operation

## Appendix: Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `PROXY_LEADER_HEARTBEAT_INTERVAL` | 5.0 | Leader heartbeat interval |
| `PROXY_LEADER_LEASE_DURATION` | 15.0 | Heartbeat staleness threshold |
| `CLOUDSQL_ENSURE_READY_TIMEOUT` | 60.0 | Startup verification timeout |
| `PROXY_WAKE_DELAY` | 2.0 | Post-wake restart delay |
| `PROXY_EMA_ALPHA` | 0.3 | Latency EMA smoothing |
| `PROXY_DEGRADATION_THRESHOLD` | 3.0 | Degradation detection multiplier |
| `HEALTH_WINDOW_SIZE` | 100 | Statistics window size |
| `ANOMALY_ZSCORE_THRESHOLD` | 2.5 | Anomaly detection threshold |
| `CLOUDSQL_REQUIRED` | true | Fail startup if CloudSQL unavailable |

---

*Document approved for implementation on 2026-01-28*
