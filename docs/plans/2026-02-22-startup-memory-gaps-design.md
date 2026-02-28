# Design: Startup Memory Gap Closure

**Date:** 2026-02-22
**Status:** Approved
**Problem:** Three gaps remain after the pressure-driven GCP lifecycle and cost/memory hardening were shipped. The `COMPONENT_UNLOAD` strategy is a no-op (zero escape valve when GCP disabled), model headroom is too tight for 16GB systems, and MemoryQuantizer monitoring doesn't cover early startup phases.
**Goal:** Close all three gaps with targeted fixes. No new infrastructure — the pressure-driven system is the correct architecture.

---

## Context

The pressure-driven GCP lifecycle (v266.0) covers Phases 4-7 where all heavy memory consumers live. When memory hits 85% sustained, the VM lifecycle state machine provisions a GCP VM and unloads the local model. The thrash cascade (v266.0) adds mmap detection and two-step response (downgrade model → GCP offload).

These three gaps are what remain:

## Fix 1: Wire COMPONENT_UNLOAD to Local Model Unload

**Location:** `backend/core/memory_quantizer.py:984-986`, `backend/intelligence/unified_model_serving.py`

**Problem:** `_apply_optimization_strategy(COMPONENT_UNLOAD)` at line 984 is a stub:
```python
elif strategy == OptimizationStrategy.COMPONENT_UNLOAD:
    logger.debug("Component unload requested")
```
When GCP is disabled (`GCP_ENABLED=false`), the pressure-driven system disables itself. CRITICAL/EMERGENCY tier fires `COMPONENT_UNLOAD` but nothing happens. The system has no escape valve — memory pressure builds with no relief.

**Fix:** Add unload callback registration to MemoryQuantizer (following the existing pattern of `register_tier_change_callback()` and `register_thrash_callback()`):

```python
def register_unload_callback(self, callback: Callable) -> None:
    """Register callback for COMPONENT_UNLOAD strategy.
    Callback receives current MemoryTier."""
    self._unload_callbacks.append(callback)
```

In `_apply_optimization_strategy(COMPONENT_UNLOAD)`, fire all registered callbacks with the current tier.

`UnifiedModelServing.start()` registers a callback that calls `stop()` (unloads the local LLM model, freeing 4-8GB). The callback only fires on CRITICAL/EMERGENCY tier, which is the correct threshold — at that point, having no local model is better than OOM.

**Why not DynamicComponentManager?** It has a full component registry, priority system, and async queue — all unnecessary complexity for unloading one component on one signal. A callback is simpler, more reliable, and consistent with the existing pattern.

## Fix 2: Bump Model Headroom Margins

**Location:** `backend/intelligence/unified_model_serving.py:706-716`

**Problem:** Current headroom values (0.75-1.5GB) are too tight on 16GB. After model loading, the frontend (500MB-1.2GB) and Docker (200-500MB) consume the headroom. The system ends up at 84-85% RAM — right at the pressure-driven trigger threshold — from normal operation, not a failure.

**Fix:** Increase defaults and make env-var-configurable:

| Tier | Before | After | Env Var |
|------|--------|-------|---------|
| ABUNDANT/OPTIMAL | 0.75GB | 1.5GB | `Ironcliw_MODEL_HEADROOM_RELAXED` |
| ELEVATED | 1.0GB | 2.0GB | `Ironcliw_MODEL_HEADROOM_NORMAL` |
| CONSTRAINED | 1.5GB | 2.5GB | `Ironcliw_MODEL_HEADROOM_TIGHT` |

Effect: `_select_best_model()` naturally picks one tier smaller (e.g., Q4 instead of Q8), trading model quality for system stability. The pressure-driven system can always offload to GCP if higher quality is needed.

## Fix 3: Start MemoryQuantizer Monitoring Earlier

**Location:** `unified_supervisor.py` (Phase 1 Preflight), `backend/core/memory_quantizer.py`

**Problem:** The `_monitor_loop()` (tier callbacks, thrash detection, pagein tracking) only starts when a consumer calls `start_monitoring()` during Phase 4+. Phases 0-2 are unmonitored. While nothing memory-heavy happens in those phases, Docker startup (Phase 2) can add 200-500MB on cold boot, and the monitoring data (baseline pagein rates, tier history) would be useful for Phase 4 decisions.

**Fix:** Add `get_memory_quantizer().start_monitoring()` call in Phase 1 (Preflight), after the singleton is created but before any heavy initialization. The monitoring loop is lightweight (one `vm_stat` + one `psutil` call every 1-10s) and has zero impact on startup performance.

---

## Files Modified

| File | Changes |
|---|---|
| `backend/core/memory_quantizer.py` | Wire `COMPONENT_UNLOAD` to callbacks, add `register_unload_callback()` |
| `backend/intelligence/unified_model_serving.py` | Register unload callback, bump headroom defaults, add env var overrides |
| `unified_supervisor.py` | Start MemoryQuantizer monitoring in Phase 1 |

**No new files.**

---

## Expected Impact

| Metric | Before | After |
|---|---|---|
| COMPONENT_UNLOAD on CRITICAL/EMERGENCY | No-op (stub) | Unloads local LLM (4-8GB freed) |
| GCP-disabled escape valve | None | Local model unload on CRITICAL tier |
| Model headroom on 16GB | 0.75-1.5GB (consumed by frontend/Docker) | 1.5-2.5GB (survives frontend+Docker) |
| Monitoring coverage | Phase 4+ | Phase 1+ |
| Baseline pagein data for Phase 4 | None | Available from Phase 1 |
