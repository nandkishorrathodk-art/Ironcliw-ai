# Design: OOMBridge Fallback Hardening

**Date:** 2026-02-23
**Status:** Approved
**Problem:** When OOMBridge fails to initialize (import error, timeout, dependency failure), the fallback logic in `unified_supervisor.py` has a guard condition that prevents mode degradation when the current mode is `cloud_first` or `cloud_only`. The system stays in a cloud mode it can't execute, then parallel_initializer loads all components in parallel with 2.6GB RAM, causing OOM crash.
**Goal:** Fix the fallback guard, add retry, force sequential init under low RAM, and split desired vs effective mode so cloud recovery probing survives startup degradation.

---

## Context

The log shows:
```
⚠ [OOMBridge] OOM pre-flight unavailable:  (startup_mode=cloud_first, avail=2.6GB)
```

Three bugs form a cascade:

**Bug 1 (guard excludes cloud modes):** `unified_supervisor.py` line 64180-64182. When OOMBridge throws, the exception handler calls `_resolve_local_startup_mode_on_cloud_unavailable("cloud_only", 2.6)` which returns `"sequential"`. But the guard condition `_current_mode in ("local_full", "local_optimized", "sequential", "minimal")` is False when `_current_mode` is `"cloud_first"` — so the mode is never updated. The system stays in `cloud_first` with no bridge to make it work.

**Bug 2 (no retry):** The bridge import fails instantly (path resolution). The 3s timeout wraps a call that never executes. No retry is attempted.

**Bug 3 (parallel_initializer fail-open):** `parallel_initializer.py` line 810-811. When the OOM check also fails there, it's labeled "non-fatal" and all components (UAE, SAI, Learning DB) load in parallel with 2.6GB → OOM.

## Fix 1: Guard Condition — Severity-Only Comparison

**Location:** `unified_supervisor.py:64180-64182`

Replace the allowlist guard with a pure severity comparison:

```
Before:
  if (_current_mode in ("local_full", "local_optimized", "sequential", "minimal")
      and _mode_severity.get(_guard_mode, 0) > _mode_severity.get(_current_mode, 0)):

After:
  if _mode_severity.get(_guard_mode, 0) > _mode_severity.get(_current_mode, 0):
```

Wait — this would try to degrade `cloud_first`(sev=3) to `sequential`(sev=2) which is LOWER severity. That's wrong.

The real issue: `_resolve_local_startup_mode_on_cloud_unavailable("cloud_only", 2.6)` returns `"sequential"` (sev=2). But `cloud_first` is sev=3. So `_guard_mode`(2) < `_current_mode`(3) → no update.

The problem is deeper: the severity ordering treats cloud modes as "more severe" than sequential. But when the bridge is broken, `cloud_first` can't execute — it's not a valid mode anymore. The fix must force degradation to the **local** fallback regardless of cloud severity.

**Correct fix:** When OOMBridge fails, bypass the severity comparison entirely. The bridge is the only thing that makes cloud modes viable. Without it, cloud modes are broken — unconditionally degrade to the local fallback:

```python
except Exception as _oom_err:
    _available_gb = _read_available_memory_gb()
    _guard_mode = _resolve_local_startup_mode_on_cloud_unavailable(
        "cloud_only", available_gb=_available_gb,
    )
    # OOMBridge is broken — cloud modes can't execute without it.
    # Unconditionally degrade to local fallback.
    os.environ["Ironcliw_STARTUP_MEMORY_MODE"] = _guard_mode
```

With 2.6GB: `_guard_mode = "sequential"` (because `predicted_post_load = 2.6 - 4.6 = 0.0 < critical(2.0)`). Mode becomes `sequential`. Correct.

With 6GB: `_guard_mode = "local_full"` (plenty of RAM). Mode becomes `local_full`. Also correct — if RAM is abundant, local is fine even without the bridge.

## Fix 2: Quick Retry with Structured Timeout

**Location:** `unified_supervisor.py:64128-64163`

Wrap the existing call in a retry loop (max 2 attempts):

```
Attempt 1: 3s timeout (existing behavior)
On failure → Attempt 2: 5s timeout (retry)
On failure → Fall through to Fix 1 degradation
```

The retry is opportunistic. If the bridge import itself fails (not a timeout), the retry also fails instantly — no wasted time. Only helps when the failure was a transient timeout (e.g., MemoryQuantizer not ready yet).

## Fix 3: Parallel Initializer Fail-Closed Under Low RAM

**Location:** `backend/core/parallel_initializer.py:810-811`

When OOMBridge check fails AND available RAM < 4GB, set a `_force_sequential` flag. In the priority group loop, when this flag is set, initialize components one-at-a-time instead of using `asyncio.gather()`.

```python
except Exception as e:
    logger.warning(f"[OOM Prevention] Check failed: {e}")
    import psutil
    _avail_gb = psutil.virtual_memory().available / (1024**3)
    if _avail_gb < 4.0:
        self._force_sequential = True
        logger.warning(
            f"[OOM Prevention] Forcing sequential init "
            f"(bridge unavailable + {_avail_gb:.1f}GB available)"
        )
```

In the priority group execution loop, when `_force_sequential` is True:
- Initialize components one-at-a-time (no `asyncio.gather()`)
- Check available RAM between each component
- If RAM drops below critical (2.0GB), skip remaining non-essential components

The 4.0GB threshold matches OOMBridge's own `cloud_trigger_ram_gb` threshold — below this, parallel init is unsafe without coordination.

## Fix 4: Desired Mode vs Effective Mode

**Location:** `unified_supervisor.py` (mode-setting locations)

Two env vars, two purposes:

| Var | Purpose | Set when | Read by |
|---|---|---|---|
| `Ironcliw_STARTUP_DESIRED_MODE` | Operator intent / policy | Once, at initial mode decision | GCP probe eligibility, post-startup recovery |
| `Ironcliw_STARTUP_MEMORY_MODE` | Runtime safety mode (effective) | Monotonically degraded during startup | Phase gates, parallel_initializer, subprocess admission |

`Ironcliw_STARTUP_DESIRED_MODE` is set at two locations:
- Line 63728: `_startup_mem_mode` from IntelligentResourceOrchestrator
- Line 63702: fallback when ResourceOrchestrator fails

Never modified after initial set.

### GCP Probe Eligibility

**Location:** `unified_supervisor.py:64203` (GCP Availability Probe)

Change the probe eligibility check from `Ironcliw_STARTUP_MEMORY_MODE` to `Ironcliw_STARTUP_DESIRED_MODE`:

```python
# Before:
_mode = os.environ.get("Ironcliw_STARTUP_MEMORY_MODE", "local_full")
if _mode in ("cloud_first", "cloud_only"):
    # probe GCP

# After:
_desired = os.environ.get("Ironcliw_STARTUP_DESIRED_MODE", "local_full")
if _desired in ("cloud_first", "cloud_only"):
    # probe GCP — based on operator intent, not runtime degradation
```

This ensures cloud recovery probing survives startup degradation. If the operator intended cloud, the GCP probe runs regardless of whether the effective mode got degraded to `sequential`.

---

## Files Modified

| File | Changes |
|---|---|
| `unified_supervisor.py` | Fix 1 (guard condition), Fix 2 (retry), Fix 4 (desired_mode var + GCP probe eligibility) |
| `backend/core/parallel_initializer.py` | Fix 3 (fail-closed under low RAM, sequential fallback) |

**No new files.**

---

## Expected Impact

| Metric | Before | After |
|---|---|---|
| OOMBridge failure + 2.6GB RAM | Stays in `cloud_first`, parallel init → OOM crash | Degrades to `sequential`, components load one-at-a-time |
| GCP probe after degradation | Skipped (mode is `sequential`) | Runs (desired_mode is still `cloud_first`) |
| parallel_initializer without bridge | "Non-fatal", loads all in parallel | Fail-closed: forces sequential when <4GB |
| OOMBridge transient timeout | Instant failure, no retry | One retry with 5s budget |
| Post-startup cloud recovery | Blocked (mode degraded away from cloud) | Enabled (desired_mode preserved, pressure-driven lifecycle active) |

---

## Edge Cases

| Case | Handling |
|---|---|
| OOMBridge import fails instantly (not timeout) | Retry also fails instantly (<1ms). Falls through to degradation. No wasted time. |
| RAM is 6GB+ but bridge fails | `_guard_mode = "local_full"`. System runs locally with plenty of RAM. Correct. |
| RAM is 1.0GB (below critical) | `_guard_mode = "minimal"`. Minimal mode — health endpoint only. |
| GCP probe succeeds after degradation | GCP available for pressure-driven lifecycle. Background offload when needed. |
| desired_mode = cloud_first but GCP never available | System runs in sequential mode locally. Pressure-driven lifecycle cannot offload but doesn't crash. |
| parallel_initializer _force_sequential + component fails | Sequential init continues to next component. Failed component marked degraded, not fatal. |
| Bridge succeeds on retry | Normal flow resumes. No degradation applied. Best outcome. |
