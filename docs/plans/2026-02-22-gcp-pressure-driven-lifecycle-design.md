# Design: Resource-Pressure-Driven GCP Lifecycle

**Date:** 2026-02-22
**Status:** Approved
**Problem:** GCP costs spiking to $92+/mo (475% increase) due to eager VM creation on every startup, Invincible Node never auto-stopping, and no demand-awareness.
**Goal:** GCP resources created only under real memory/CPU pressure, stopped on session shutdown, with proper cost gating.

---

## Principles

1. **Pressure-driven, not eager** -- no GCP resources created at startup. Only when the local M1 16GB is genuinely under pressure.
2. **Single brain** -- enhance the existing `GCPHybridPrimeRouter`, don't create a parallel decision-maker.
3. **Cure the disease** -- centralize scattered guard logic, fix event loop blocking, fail-closed on budget errors.
4. **No new files** -- all changes are modifications to existing files.

---

## Architecture

### 1. VM Lifecycle State Machine (GCPHybridPrimeRouter)

Replace scattered boolean fields with a formal state machine:

```
IDLE -----(pressure detected)-----> TRIGGERING
TRIGGERING --(3/5 readings above)-> PROVISIONING
TRIGGERING --(pressure drops)-----> IDLE
PROVISIONING -(API call sent)-----> BOOTING
PROVISIONING -(API fails)---------> COOLING_DOWN
BOOTING ----(health check pass)---> ACTIVE
BOOTING ----(timeout/fail)--------> COOLING_DOWN
ACTIVE -----(pressure below 70%)-> COOLING_DOWN
COOLING_DOWN -(grace timer expires)> IDLE (VM auto-stops via idle timeout)
ACTIVE -----(session shutdown)----> STOPPING
STOPPING ---(VM stopped)----------> IDLE
Any state --(session shutdown)----> STOPPING
```

States replace these booleans: `_vm_provisioning_in_progress`, `_emergency_offload_active`, `_emergency_offload_released_at`, `_emergency_offload_cycle_count`, `_emergency_offload_hysteresis_armed`.

### 2. Hysteresis Bands

- **Trigger threshold:** 85% macOS-aware pressure (`CRITICAL_RAM_PERCENT`, already 85.0)
- **Release threshold:** 70% macOS-aware pressure (`GCP_RELEASE_RAM_PERCENT`, new, default 70.0)
- **Dead zone:** 70-85% -- no state transitions. VM stays in whatever state it's in.

### 3. Sustained Pressure Detection

Replace "sustained for 10s" (1-2 readings at 10s interval) with:

- Require **3 out of 5 consecutive readings** above trigger threshold
- The router's existing 1s fast-polling mode (RAM >60%) means 5 readings = 5 seconds
- The existing `PredictiveMemoryPlanner` (numpy polyfit over last 50 samples) is consulted: if predicted tier in 2 minutes is CRITICAL+, that counts as a "reading above threshold"
- The existing `_calculate_memory_rate()` 100MB/sec spike detection remains as an instant-trigger bypass (no N-of-M required)

### 4. MemoryQuantizer Integration

The router currently does its own RAM polling via psutil, independent of MemoryQuantizer. Enhancement:

- Register callback via `memory_quantizer.register_tier_change_callback()`
- MemoryQuantizer's macOS-aware `_calculate_tier_macos()` -- which uses `(wired + active + compressed) / total` -- becomes the authoritative pressure signal
- Router's own psutil polling serves as a fast-path secondary signal (1s vs MemoryQuantizer's 10s)
- Eliminates ambiguity of "which 85%" on M1 unified memory

### 5. Local Model Unload After GCP Takeover

When state machine transitions to ACTIVE:

1. Existing SIGSTOP fires (already implemented, pauses local LLM)
2. After 30s of confirmed GCP stability (3 health checks at 10s), escalate:
   - Call `UnifiedModelServing.stop()` to unload GGUF model from RAM
   - Reclaims the 4-8GB that caused the pressure
   - Sets `Ironcliw_GCP_OFFLOAD_ACTIVE=true`
3. If GCP becomes unhealthy later, `load_model()` re-loads locally (existing hot-swap)

This closes the feedback loop: pressure -> GCP -> unload -> pressure drops -> GCP can release.

---

## Remove Eager Pre-Warm

**Delete** from `unified_supervisor.py` (git history preserves):

- `early_invincible_node_prewarm` task block (~lines 63960-64157)
- `early_spot_vm_warm` task block (~lines 63519-63574)

No feature flag. If eager behavior is ever wanted, cherry-pick from history.

`GCPInstanceManager` still initializes in Phase 2 for orphan detection and state sync. Creates nothing.

---

## Centralize InvincibleGuard

Extract scattered guard logic (3 locations with inconsistent checks) into one method on `GCPVMManager`:

```python
class VMAction(Enum):
    STOP = "stop"
    DELETE = "delete"
    TERMINATE = "terminate"

def check_vm_protection(self, vm_name: str, action: VMAction) -> Tuple[bool, str]:
    """
    Returns (is_protected, reason). True = action BLOCKED.

    Detection criteria (unified, all 3 locations use this):
    1. VM name starts with configured static_instance_name (from config)
    2. Metadata vm_class == "invincible"
    3. GCP label vm-class == "invincible"

    Action awareness:
    - DELETE/TERMINATE: Always blocked for invincible VMs
    - STOP: Allowed when reason is "session_shutdown" or "supervisor_cleanup"
            AND gcp_session_lifecycle config is True. Blocked otherwise.
    """
```

All three existing guard locations call this single method.

---

## Extend terminate_vm() with Action Parameter

No separate `stop_vm()`. Existing pipeline handles both:

```python
async def terminate_vm(self, vm_name: str, reason: str = "",
                       action: VMAction = VMAction.DELETE) -> bool:
```

When `action=VMAction.STOP`:
- Guard: `check_vm_protection(vm_name, VMAction.STOP)` -- allows for session_shutdown
- GCP API: `instances_client.stop()` (not delete)
- State: `managed_vms[vm_name].status = VMStatus.STOPPED` (new enum value, not removed from dict)
- Cost: session end recorded, hourly accrual stops
- Dashboard: "VM stopped (session shutdown)"
- Monitoring loop: skips STOPPED VMs entirely

When `action=VMAction.DELETE`: existing behavior unchanged.

---

## Shutdown Integration

Position in `JarvisSystemKernel.cleanup()`: step 15 (existing GCP cleanup position).

- After: SSR shutdown (step 6), Trinity stop (step 12) -- all VM consumers are done
- Before: Frontend/backend process termination (step 16)

Replace `shutdown_orchestrator()` internals to call `terminate_vm(action=VMAction.STOP)` for each tracked VM.

**Crash recovery:** On next startup, `_sync_managed_vms_with_gcp()` discovers running VMs. No pressure at startup, so monitoring loop's idle timeout (10 min) STOPs them.

---

## MemoryQuantizer Event Loop Fix

In `backend/core/memory_quantizer.py`, replace synchronous subprocess:

```python
# Before (blocks event loop up to 2s):
result = subprocess.run(['memory_pressure'], capture_output=True, text=True, timeout=2)

# After (non-blocking):
proc = await asyncio.create_subprocess_exec(
    'memory_pressure',
    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
)
stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)
```

Same pattern as ECAPA v265.2 fix and ChromaDB/InfraOrch v265.2 fix.

---

## Cost Gating Fixes

### Fail Closed

```python
# Before (fail open):
except Exception:
    return True, "Budget check error (allowing)", details

# After (fail closed):
except Exception as e:
    return False, f"Budget check error (blocking): {e}", details
```

### Deduplicate Cost Accounting

`can_create_vm()` sums `get_cost_summary("day")` AND active session costs. Active sessions whose start time falls within the current day's DB query window are already counted. Fix: exclude overlapping sessions from the active sum.

### Circuit Breaker Integration

Already works: `_trigger_vm_provisioning()` -> `_gcp_controller.create_vm()` -> `gcp_vm_manager` circuit breaker (3 failures -> OPEN). Router transitions to COOLING_DOWN and falls back to Cloud Run / Claude API. No new code needed.

---

## Cloud SQL: No Lifecycle Toggling

Cloud SQL stays always-on. Removed from design because:
1. `gcloud sql instances patch` requires `cloudsql.instances.update` IAM permission
2. 60-90s cold start on every dev restart
3. Crash -> NEVER activation stuck with no recovery
4. Cost: ~$7-10/mo is acceptable; VM cost reduction is the big win

---

## STOPPED VM Cost Model

When VMs are STOPPED:
- Persistent disk (~80GB pd-standard): ~$3.20/mo
- Static IP (unattached): ~$7.30/mo
- Total idle: ~$10.50/mo

Tradeoff: $10.50/mo for 30s restart vs $0 with 3-5 min cold creation. Worth it for solo developer.

---

## .env.gcp Default Change

```
Ironcliw_INVINCIBLE_NODE_ENABLED=false
```

The Invincible Node is no longer eagerly created. The pressure-driven system in GCPHybridPrimeRouter handles VM lifecycle. The `Ironcliw_INVINCIBLE_NODE_ENABLED` flag becomes irrelevant to startup but is checked by the router when deciding which VM type to provision under pressure.

---

## Files Modified

| File | Changes |
|---|---|
| `backend/core/gcp_hybrid_prime_router.py` | VMLifecycleState enum, hysteresis bands, MemoryQuantizer callback, local model unload, 3-of-5 sustained detection |
| `backend/core/gcp_vm_manager.py` | Centralize InvincibleGuard into `check_vm_protection()`, extend `terminate_vm()` with action param, VMAction enum, VMStatus.STOPPED |
| `unified_supervisor.py` | Delete early_invincible_node_prewarm, delete early_spot_vm_warm, update cleanup() to use terminate_vm(action=STOP) |
| `backend/core/memory_quantizer.py` | Replace subprocess.run with asyncio.create_subprocess_exec |
| `backend/core/cost_tracker.py` | Fail-closed in can_create_vm(), deduplicate active session costs |
| `.env.gcp` | Ironcliw_INVINCIBLE_NODE_ENABLED=false |

**No new files.**

---

## Expected Cost Impact

| Resource | Before | After |
|---|---|---|
| Invincible Node VM (24/7 Spot) | ~$21/mo | ~$0-5/mo (only runs under pressure, stopped on shutdown) |
| Static IP (when VM stopped) | $7.30/mo | $7.30/mo (accepted tradeoff) |
| Persistent disk (when VM stopped) | $3.20/mo | $3.20/mo (accepted tradeoff) |
| Cloud SQL | $7-10/mo | $7-10/mo (kept always-on) |
| Spot VM churn (stall retries) | Variable | Eliminated (no eager creation) |
| **Total estimated** | **~$50-92/mo** | **~$20-30/mo** |
