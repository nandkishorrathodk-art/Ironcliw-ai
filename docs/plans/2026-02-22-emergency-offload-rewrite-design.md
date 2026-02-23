# Design: Emergency Offload Path Rewrite

**Date:** 2026-02-22
**Status:** Approved
**Problem:** The SIGSTOP/SIGCONT emergency offload mechanism in GCPHybridPrimeRouter has three failure modes: deadlock risk (SIGSTOP on lock-holding processes), cost-gated death spiral (SIGSTOP → budget exceeded → SIGCONT → repeat → TERMINATE), and 120s cooldown gap. Root cause: SIGSTOP doesn't free memory — it freezes processes in place.
**Goal:** Replace SIGSTOP as the primary emergency response with clean model unloading via COMPONENT_UNLOAD callbacks. SIGSTOP becomes a one-shot last resort at 95%+ RAM only.

---

## Context

The v266.0 work just shipped COMPONENT_UNLOAD callbacks that cleanly unload the LLM model (4-8GB freed). This makes the SIGSTOP-first approach obsolete — clean unloading actually frees memory, while SIGSTOP just freezes it in place.

The SIGSTOP mechanism (v93.0) was built before the pressure-driven GCP lifecycle and memory quantizer existed. It was the only tool available. Now there's a better one.

## Fix 1: Rewrite Emergency Offload — Clean Unload First

**Location:** `backend/core/gcp_hybrid_prime_router.py`, `_emergency_offload()` (~line 1410)

**Current escalation:**
```
RAM 80% → SIGSTOP → try GCP → wait 60s → SIGCONT → repeat (3 cycles) → TERMINATE
```

**New escalation ladder:**
```
Step 1: RAM >= CRITICAL tier (85% sustained)
  → Fire COMPONENT_UNLOAD via MemoryQuantizer
  → Clean model unload frees 4-8GB
  → Wait 10-15s, re-read memory

Step 2: If RAM still high after verification
  → Attempt GCP VM provisioning (if available + budget allows)

Step 3: RAM >= 95% AND unload failed AND GCP unavailable
  → SIGSTOP (one-shot, JARVIS-owned PIDs only)
  → If no recovery in 60s → TERMINATE
  → No cycling, no SIGCONT → re-SIGSTOP loop
```

The threshold moves from 80% to CRITICAL tier (85% sustained). At 80%, the system is in normal stress on 16GB — unloading would hurt local availability unnecessarily.

## Fix 2: Post-Unload Verification Gate

**Location:** `backend/core/gcp_hybrid_prime_router.py`, new logic in `_emergency_offload()`

After firing COMPONENT_UNLOAD:
1. Wait 10-15s (configurable via `EMERGENCY_UNLOAD_VERIFY_DELAY_SEC`)
2. Re-read memory via psutil
3. If RAM dropped by >= 1GB (`EMERGENCY_UNLOAD_MIN_DROP_GB`), stop escalation
4. If not, proceed to GCP attempt
5. Only after GCP failure + RAM >= 95%, fall through to SIGSTOP

This prevents over-escalation when the clean unload worked but needs a moment to settle (gc.collect, mmap page reclamation, etc.).

## Fix 3: One-Shot SIGSTOP — Kill the Death Spiral

**Location:** `backend/core/gcp_hybrid_prime_router.py`, remove cycling logic

Root cause of the death spiral: `_emergency_offload_cycle_count` allows 3 SIGSTOP → SIGCONT → SIGSTOP cycles. Each cycle puts RAM right back since SIGSTOP doesn't free anything.

Changes:
- Remove `_emergency_offload_cycle_count` tracking
- Remove `EMERGENCY_OFFLOAD_MAX_CYCLES` configuration
- SIGSTOP fires once. If timeout (60s) expires without recovery → TERMINATE
- No SIGCONT → re-trigger path
- 120s cooldown only applies to SIGSTOP (which now rarely fires)

The `_emergency_offload_hysteresis_armed` flag stays — it's useful for preventing immediate re-trigger after any emergency action, not just SIGSTOP.

## Fix 4: SIGSTOP Scoped to JARVIS-Owned PIDs

**Location:** `backend/core/gcp_hybrid_prime_router.py`, `_pause_local_llm_processes()` (~line 1510)

Current behavior: scans all processes for patterns like `ollama`, `llama`, `transformers`, etc. This can hit unrelated processes.

New behavior:
1. First priority: use `_local_llm_pids` (tracked PIDs from PrimeLocalClient)
2. Only fall back to pattern-based scanning if no tracked PIDs available
3. Log which PIDs were frozen and the selection method used

## Fix 5: Update Impact Predictions

**Location:** `backend/core/memory_quantizer.py`, `predict_optimization_impact()`

| Strategy | Before | After | Rationale |
|---|---|---|---|
| EMERGENCY_CLEANUP | 500MB | 50MB | gc.collect(2) + flush — realistic |
| COMPONENT_UNLOAD | 300MB | 6000MB | Now actually unloads LLM model |

---

## Files Modified

| File | Changes |
|---|---|
| `backend/core/gcp_hybrid_prime_router.py` | Rewrite `_emergency_offload()`, add verification gate, remove cycling, scope SIGSTOP |
| `backend/core/memory_quantizer.py` | Update `predict_optimization_impact()` |

**No new files.**

---

## Expected Impact

| Metric | Before | After |
|---|---|---|
| Emergency response at CRITICAL | SIGSTOP (freezes memory) | Clean unload (frees 4-8GB) |
| Death spiral risk | 3-cycle SIGSTOP→SIGCONT loop → TERMINATE | Eliminated — unload frees memory |
| Cooldown gap | 120s unprotected | Only applies to SIGSTOP (rare) |
| SIGSTOP scope | Pattern-based (broad) | JARVIS-owned PIDs first |
| False positive unloads | At 80% RAM (normal stress) | At 85%+ sustained (CRITICAL) |
| SIGSTOP occurrence | Primary response (every emergency) | Last resort only (95%+ after unload fails) |
