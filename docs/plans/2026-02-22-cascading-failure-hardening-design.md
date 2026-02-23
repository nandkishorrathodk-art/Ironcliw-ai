# Design: Cascading Failure Hardening (Scenarios A + B)

**Date:** 2026-02-22
**Status:** Approved
**Problem:** Two cascading failure modes remain after the emergency offload rewrite: (A) post-crisis permanent degradation — after emergency offload kills the LLM, nothing reloads it, and (B) startup memory cascade — phases 2-3 proceed without memory checks, subprocess exec can OOM.
**Goal:** Add post-crisis model recovery with hybrid lazy+background reload, and startup memory phase gates with per-subprocess admission control.

---

## Context

The v266.1 emergency offload rewrite replaced the SIGSTOP death spiral with clean model unloading. That fixes the *during-crisis* behavior. But two gaps remain:

**Scenario A (Post-crisis):** After the model is unloaded (or terminated), RAM drops, hysteresis disarms — and then nothing. The PRIME_LOCAL circuit breaker stays OPEN. The system falls through to CLAUDE for all inference. It stays degraded until manual restart. On a 16GB Mac where the model is the primary inference path, this means permanent latency increase and cloud API cost until reboot.

**Scenario B (Startup):** Memory decisions are made once before Phase 1. But Phases 2-3 consume 700MB-1.5GB (Docker, GCP controllers, backend imports). If RAM was borderline at the decision point, it's worse by Phase 2. `create_subprocess_exec` can fail with ENOMEM with no recovery path.

## Fix 1: Post-Crisis Model Recovery

**Location:** `backend/core/gcp_hybrid_prime_router.py` (monitoring loop + new state), `backend/intelligence/unified_model_serving.py` (circuit breaker reset)

### Recovery State Machine

After emergency offload ends (hysteresis disarms at line 1152):

```
Hysteresis disarms → _model_needs_recovery = True
                     _recovery_stable_since = 0

Each monitoring poll (1-5s):
  If _model_needs_recovery:
    Read tier from MemoryQuantizer
    If tier <= ELEVATED (< 75%):
      If _recovery_stable_since == 0:
        _recovery_stable_since = now  (start stability window)
      Elif now - _recovery_stable_since >= 30s:
        → Attempt background model reload
    Else (tier > ELEVATED):
      _recovery_stable_since = 0  (reset — not stable yet)

Background reload:
  model_serving = get_model_serving()
  success = await model_serving.load_model()
  If success:
    model_serving.reset_local_circuit_breaker()
    _model_needs_recovery = False
    _recovery_attempts = 0
  If failure:
    _recovery_attempts += 1
    _recovery_stable_since = 0  (wait for next stability window)
    If _recovery_attempts >= 3:
      _model_needs_recovery = False  (give up this cycle)
      Log: "Recovery failed after 3 attempts — staying on CLAUDE"
```

### Configuration

| Constant | Env Var | Default | Purpose |
|---|---|---|---|
| `RECOVERY_STABILITY_THRESHOLD_PERCENT` | `EMERGENCY_RECOVERY_THRESHOLD_PERCENT` | 75.0 | RAM % below which recovery can start |
| `RECOVERY_STABILITY_DURATION_SEC` | `EMERGENCY_RECOVERY_STABILITY_SEC` | 30.0 | Sustained seconds below threshold |
| `RECOVERY_MAX_ATTEMPTS` | `EMERGENCY_RECOVERY_MAX_ATTEMPTS` | 3 | Max reload attempts per crisis cycle |

### Circuit Breaker Reset

New method on `UnifiedModelServing`:

```python
def reset_local_circuit_breaker(self) -> None:
    """Reset PRIME_LOCAL circuit breaker after verified model reload."""
```

This directly sets the canonical breaker state to CLOSED with zero failure count. Only called after `load_model()` returns True — the model is verified loaded before the breaker opens.

### Hybrid Lazy + Background Warm-up

During recovery, the user's experience is:
1. First request after crisis → routed to CLAUDE (circuit breaker OPEN for LOCAL)
2. Background reload starts (triggered by stability window, not by request)
3. When model loads → circuit breaker closes → subsequent requests use LOCAL
4. If reload fails → CLAUDE continues indefinitely (graceful degradation)

No user-visible latency spike. No blocking. CLAUDE absorbs the gap.

## Fix 2: Startup Memory Phase Gates

**Location:** `unified_supervisor.py` (phase methods + new admission gate)

### Monotonic Mode Escalation

During startup, mode can only degrade (never recover upward):

```
local_full → local_optimized → sequential → cloud_first → cloud_only → minimal
```

Recovery upward only after `_startup_complete = True`. Implemented as a guard in `_reevaluate_startup_mode()` — if new severity < current severity and startup not complete, reject the change.

### Phase Boundary Gates

| Phase | Gate Logic |
|---|---|
| Phase 2 (Resources) | Call `_reevaluate_startup_mode("phase_2_resources")`. If escalated to `cloud_only`, set capability flags to `deferred` for Docker and local storage. Later phases check flags before assuming resource availability. |
| Phase 3 (Backend) | Call `_reevaluate_startup_mode("phase_3_backend")`. If RAM critically low + mode is `cloud_first`, set `JARVIS_BACKEND_MINIMAL=true` to load only control-plane services (health, routing, websocket, orchestration). |

### Capability Flags (Deferred Init)

Instead of silently skipping resources, set environment flags:

```
JARVIS_CAPABILITY_DOCKER=deferred     (not "false")
JARVIS_CAPABILITY_LOCAL_STORAGE=deferred
```

Downstream components branch on these flags. If a deferred capability is later needed, it's initialized on-demand with its own memory check. This prevents silent assumption violations.

### Per-Subprocess Admission Gate

New lightweight function:

```python
async def can_spawn_heavy_process(estimated_mb: int) -> bool:
```

Checks MemoryQuantizer tier + available RAM against the estimated subprocess cost. Called right before each `create_subprocess_exec` for backend, frontend, Docker. Returns False if spawning would push into CRITICAL/EMERGENCY tier.

Callers decide what to do on False:
- Backend: switch to in-process minimal mode
- Docker: defer initialization, set capability flag
- Frontend: defer to Phase 7 when more RAM is free

### Subprocess OOM Handler

Wrap `create_subprocess_exec` calls with `OSError` handling using `errno.ENOMEM` (not string matching):

```python
try:
    proc = await asyncio.create_subprocess_exec(...)
except OSError as e:
    if e.errno == errno.ENOMEM:
        # Force-escalate mode, return False
    raise
except MemoryError:
    # Same handling
```

On ENOMEM: log available RAM, force-escalate startup mode, return False. Caller uses cloud or minimal fallback.

---

## Files Modified

| File | Changes |
|---|---|
| `backend/core/gcp_hybrid_prime_router.py` | Recovery state variables, recovery monitoring logic in polling loop, recovery config constants |
| `backend/intelligence/unified_model_serving.py` | `reset_local_circuit_breaker()` method |
| `unified_supervisor.py` | Phase 2/3 gates, monotonic mode constraint, `can_spawn_heavy_process()`, subprocess OOM handler, capability flags |

**No new files.**

---

## Expected Impact

| Metric | Before | After |
|---|---|---|
| Post-crisis inference | Degraded to CLAUDE forever | Auto-recovers to LOCAL within ~60s of memory stabilizing |
| Startup OOM on Phase 2-3 | Crash, no recovery | Graceful mode escalation, deferred init |
| Subprocess ENOMEM | Unhandled OSError | Caught, mode escalated, cloud fallback |
| Mode oscillation during startup | Possible (bidirectional) | Prevented (monotonic degradation only) |
| Resource assumption violations | Silent (assumed Docker exists) | Explicit capability flags |

---

## Edge Cases

| Case | Handling |
|---|---|
| Double crisis (reload → RAM spikes again) | Hysteresis re-arms, new offload cycle. Recovery resets. |
| Model reload fails 3x | Gives up this cycle. Stays on CLAUDE. Next crisis cycle gets fresh 3 attempts. |
| GCP available during recovery | Still reload locally. Local is faster + cheaper. GCP was temporary. |
| RAM fluctuates during reload | Reload itself runs via `load_model()` which has its own headroom checks. If RAM spikes mid-load, load_model returns False. Recovery waits for next window. |
| Phase 2 deferred Docker needed in Phase 5 | Phase 5 checks capability flag, triggers on-demand Docker init with admission gate. |
| All modes exhausted (cloud_only + GCP unavailable) | `minimal` mode: health endpoint only, no inference. Logs critical alert. |
