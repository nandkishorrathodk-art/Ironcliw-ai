# Reactor Core Pipeline Activation — Design Document

**Date:** 2026-02-15
**Status:** Approved
**Scope:** Cross-repo (Ironcliw-AI-Agent, reactor-core, jarvis-prime)

## Problem Statement

The Reactor Core training pipeline is 80% built but 0% connected. All three repos have matching schemas (v1.0 canonical `ExperienceEvent`), implemented ingestion endpoints, training pipelines, and deployment watchers — but zero training jobs have ever run. `~/.jarvis/reactor_state/jobs.json` is empty.

The infrastructure exists. The schemas match. The code is written. Nothing is plugged in.

## Root Cause: 6 Specific Disconnects

| # | Disconnect | Location | Root Cause |
|---|---|---|---|
| 1 | DataFlywheelManager bypasses ReactorCoreClient | `unified_supervisor.py:22779-22788` | Raw HTTP POST instead of client's `trigger_training()`. Ignores configurable thresholds, min interval, priority. |
| 2 | Health monitor ignores training readiness | `reactor_core_client.py:344-468` | Only checks HTTP 200. Reactor-Core returns `training_ready`, `phase`, `trinity_connected` — all ignored. |
| 3 | `get_experience_count()` never called | `reactor_core_client.py:721-736` | Fully implemented, never invoked. Auto-trigger uses hardcoded `batch_size * 10`. |
| 4 | Training jobs triggered but never polled | `reactor_core_client.py:634-654` | `get_training_job()` exists. No code calls it after triggering. |
| 5 | No deployment feedback (Prime to Reactor) | `jarvis_prime/docker/reactor_core_watcher.py` | Deploys GGUF but never writes success/failure back. |
| 6 | Endpoint path mismatches | `reactor_core_client.py` vs `reactor_core/api/server.py` | Client: `/api/experiences/stream`. Server: `/api/v1/experiences/stream`. Client: `/api/training/trigger`. Server: `/api/v1/train`. |

## Approach: Supervisor-Driven Activation

The unified_supervisor already discovers Reactor-Core, monitors its health, and has `ReactorCoreClient` with auto-trigger logic. We make the supervisor the active coordinator that drives the loop, leveraging existing advanced patterns:

- `CircuitBreaker` from `backend/kernel/circuit_breaker.py`
- `TraceContext` from `backend/neural_mesh/monitoring/trace_manager.py`
- `TrinityEventBus` from `backend/core/trinity_event_bus.py`
- `DeadManSwitch` / `ProbationStatus` from `backend/core/supervisor/rollback_manager.py`
- `ScoringEngine` from `reactor_core/distillation/scoring_engine.py`
- `Gatekeeper` / `ApprovalCriterion` from `reactor_core/eval/gatekeeper.py`
- `DataVersion` / `DataHash` from `reactor_core/data/versioning.py`
- `MemoryQuantizer` / `MemoryTier` from `backend/core/memory_quantizer.py`

## Design

### 1. Fix Endpoint Path Mismatches

Three corrections in `reactor_core_client.py`:

| Method | Current Path | Correct Path |
|---|---|---|
| `stream_experience()` (line 714) | `/api/experiences/stream` | `/api/v1/experiences/stream` |
| `trigger_training()` (line 579) | `/api/training/trigger` | `/api/v1/train` |
| `get_experience_count()` (line 732) | `/api/experiences/count` | `/api/v1/experiences/count` |

Audit all other `_request()` calls to verify paths match Reactor-Core's `server.py` route registrations.

### 2. Enrich Health Monitor with Training Readiness

In `reactor_core_client.py:344-468`, parse the JSON response body from `/health`:

- Store `training_ready` and `phase` on the client instance
- Expose `is_training_ready` property
- Log phase transitions
- Only attempt training triggers when `training_ready == True`

### 3. Intelligent Experience Scoring (Quality-Weighted Triggers)

Replace raw count threshold with weighted scoring:

| Experience Type | Weight | Rationale |
|---|---|---|
| `CORRECTION` | 10x | Highest DPO signal |
| `FEEDBACK` negative | 5x | Direct quality signal |
| `ERROR` | 3x | Failure mode data |
| Novel `task_type` | 3x | Capability expansion |
| Low confidence (<0.7) | 2x | Borderline = informative |
| Normal `INTERACTION` | 1x | Baseline |
| Near-duplicate | 0.1x | Minimal value |

Auto-trigger fires when `weighted_score >= REACTOR_CORE_WEIGHTED_THRESHOLD` (default 100).

Deduplication via `DataHash` on `(user_input, task_type)` tuples with bloom filter.

### 4. Wire DataFlywheelManager Through ReactorCoreClient

Replace raw HTTP block at `unified_supervisor.py:22773-22788` with call to `check_and_trigger_training()` (exists at `reactor_core_client.py:1127-1158`). This respects configurable thresholds, min interval, and priority.

### 5. Training Job Lifecycle Monitoring

After triggering, store `job_id` on the client. In health monitor loop:

- If job active: poll `get_training_job(job_id)`
- On completion: log, emit `TrinityEvent("training.completed")`, clear state
- On failure: log, record circuit breaker failure, reset state

### 6. Training Circuit Breaker

Use existing `CircuitBreaker` pattern:

- Circuit: `"reactor_training"`
- OPEN after 3 consecutive failures (job fail, gate reject, post-deploy rollback)
- Recovery timeout: 1 hour (doubles on re-failure, max 24h)
- HALF_OPEN: allow 1 test job with 50% data

### 7. Resource-Aware Training

Check `MemoryQuantizer` before training:

| Memory Tier | Behavior |
|---|---|
| ABUNDANT/OPTIMAL | Full training |
| ELEVATED | Reduced batch size (50%), cap experiences at 5000 |
| CONSTRAINED | Defer to Night Shift |
| CRITICAL/EMERGENCY | Abort |

Monitor during training every 30s. Checkpoint on CRITICAL, resume on recovery.

### 8. Pipeline Observability (Correlation IDs)

Every experience gets `correlation_id` from `TelemetryEmitter`. Events emitted via `TrinityEventBus` at each stage:

```
experience.emitted → experience.ingested → training.started →
training.completed → gate.evaluated → model.deployed →
probation.started → probation.committed | probation.rollback
```

All events share `correlation_id`. `causation_id` tracks causal chains.

### 9. Deployment Gate

Leverages existing `Gatekeeper` with `ApprovalCriterion` framework.

**v1 checks:**
- GGUF header valid (magic bytes + version)
- File size reasonable (>100MB, <expected for quant level)
- Model loads without error (llama-cpp-python)
- Generates non-empty, non-repetitive text on 5 fixed prompts

**v2 checks (after first loop proves out):**
- `ScoringEngine` multi-criteria quality >= 0.6
- Safety score >= 0.95
- No regression vs previous model (delta >= -0.05)
- Latency acceptable (>= 8 tok/s)

Returns `ApprovalDecision`: APPROVED, REJECTED, or PENDING_REVIEW.

### 10. Deployment Feedback Schema

Prime writes to `~/.jarvis/cross_repo/deployment_status.json`:

```json
{
    "model_id": "string",
    "deployment_status": "success | failed | rollback",
    "deployed_at": "ISO 8601",
    "previous_model": "string",
    "health_check_passed": true,
    "first_inference_latency_ms": 3200,
    "error": null,
    "reactor_job_id": "string"
}
```

### 11. Post-Deployment Probation

Leverages existing `DeadManSwitch` / `ProbationStatus`:

- 30-minute monitoring window (configurable `REACTOR_PROBATION_MINUTES`)
- Probes every 60s: latency, error rate, correction rate, memory stability
- `health_score >= 0.8` → COMMITTED
- `health_score < 0.5` → ROLLING_BACK (restore previous GGUF)
- Emergency rollback if error rate > 5x baseline at any probe

### 12. Graceful Experience Draining

Atomic snapshot on training trigger:
1. Lock `job_manager.experiences`
2. Copy + clear buffer
3. Write snapshot to `~/.jarvis/reactor/training_data/snapshot_{job_id}.jsonl`
4. Release lock
5. New experiences buffer for next job

Snapshot file = dataset version via `DataHash.from_file()`.

### 13. Model Lineage Tracking

Every training job appends to `~/.jarvis/reactor/models/lineage.jsonl`:

```json
{
    "model_id": "string",
    "model_hash": "sha256",
    "parent_model": "string",
    "training_method": "lora_sft",
    "training_job_id": "string",
    "dataset": {
        "hash": "sha256",
        "size": 847,
        "date_range": ["2026-02-10", "2026-02-15"],
        "source_distribution": {"jarvis_body": 612, "corrections": 37},
        "weighted_score": 142.5
    },
    "eval_scores": {"overall_quality": 0.82, "safety": 0.98},
    "gate_decision": "APPROVED",
    "deployed_at": "ISO 8601",
    "probation_result": "COMMITTED",
    "transformation_steps": ["DataVersion TransformationStep entries"]
}
```

### 14. Job Persistence (Survives Supervisor Restart)

- On job create/update: write to `~/.jarvis/reactor_state/jobs.json`
- On Reactor-Core startup: load existing jobs
- On supervisor reconnect: health monitor discovers in-progress/completed jobs

Reactor-Core owns execution. Supervisor monitors status.

## Files Changed

| Repo | File | Changes | Est. Lines |
|---|---|---|---|
| Ironcliw | `backend/clients/reactor_core_client.py` | Fix paths, enrich health, weighted scoring, job polling, circuit breaker, resource gating | ~250 |
| Ironcliw | `unified_supervisor.py` | Replace DataFlywheelManager raw HTTP with client call, subscribe to events | ~40 |
| Ironcliw | `backend/core/telemetry_emitter.py` | Add correlation_id, emit TrinityEvent on flush | ~30 |
| Reactor-Core | `reactor_core/api/server.py` | Persist jobs, load on startup, atomic snapshots, resource check | ~100 |
| Reactor-Core | `reactor_core/training/unified_pipeline.py` | Call DeploymentGate, emit events, memory monitoring, checkpoint | ~80 |
| Reactor-Core | `reactor_core/eval/gatekeeper.py` | Add GGUF-specific checks, wire multi-criteria eval | ~60 |
| Reactor-Core | New: `reactor_core/deployment/gate.py` | DeploymentGate orchestrator | ~100 |
| Reactor-Core | `reactor_core/data/versioning.py` | Model lineage writer, snapshot management | ~50 |
| Prime | `jarvis_prime/docker/reactor_core_watcher.py` | Write deployment_status.json, start probation, emit events | ~80 |
| Prime | `jarvis_prime/core/reactor_core_bridge.py` | Read probation result, write deployment feedback | ~40 |

**Total: ~830 lines across 3 repos, 9 existing files + 1 new file.**

## Execution Phases

**Phase 1 (Days 1-2): Verify the wiring**
- curl Reactor-Core on port 8090
- Manual experience POST to correct endpoint
- Verify ReactorCoreWatcher in Prime is running + watching correct directory

**Phase 2 (Days 3-4): Connect the core loop**
- Fix endpoint paths (disconnect #6)
- Wire DataFlywheelManager through client (disconnect #1)
- Enrich health monitor (disconnect #2)
- Add experience count checking (disconnect #3)
- Add job polling (disconnect #4)

**Phase 3 (Days 5-7): Close the loop + harden**
- Add deployment feedback (disconnect #5)
- Add DeploymentGate (smoke tests)
- Add job persistence
- Add experience snapshots
- End-to-end test

**Phase 4 (Week 2): Beef up**
- Training circuit breaker
- Resource-aware training
- Quality-weighted triggers
- Pipeline observability (correlation IDs + TrinityEventBus)
- Model lineage tracking

**Phase 5 (Week 3): Post-deployment safety**
- Probation / auto-rollback
- Deployment gate v2 (ScoringEngine + regression checks)
- Night Shift activation
