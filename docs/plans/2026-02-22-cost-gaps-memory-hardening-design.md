# Design: Cost Gap Closure + LLM Memory Hardening

**Date:** 2026-02-22
**Status:** Approved
**Problem:** Two remaining cost leak vectors bypass the pressure-driven lifecycle we just shipped. Additionally, the LLM model loading path has no protection against stale RAM checks or mmap thrashing — both are theoretical risks that could re-trigger the cost spiral.
**Goal:** Close all cost leak paths. Add defensive memory hardening so mmap thrashing is detected and handled via a two-step cascade (downgrade model, then GCP offload).

---

## Part A: Close Remaining Cost Leak Vectors

### 1. `should_create_vm()` Fail-Open Wrapper

**Location:** `backend/core/gcp_vm_manager.py:4533-4534`

**Problem:** `can_create_vm()` in cost_tracker.py now correctly fails closed (commit `e0806a6f`). But `should_create_vm()` wraps it in `try/except Exception` that catches failures and allows VM creation anyway. This completely negates the fail-closed fix.

**Fix:**
```python
# Before (line 4534):
except Exception as e:
    logger.warning(f"⚠️ Budget check failed (allowing VM): {e}")

# After:
except Exception as e:
    logger.error(f"🚫 Budget check failed — blocking VM creation for safety: {e}")
    return (False, f"Budget check error (blocking): {e}", 0.0)
```

### 2. `ensure_static_vm_ready()` Budget Gate

**Location:** `backend/core/gcp_vm_manager.py:7287`

**Problem:** The Invincible Node creation path (`ensure_static_vm_ready()`) never calls `can_create_vm()` or `should_create_vm()`. It creates/starts VMs with zero budget awareness. This is likely responsible for $30-45/mo of the $92 bill.

**Fix:** Add a budget gate at the top of `ensure_static_vm_ready()`, before any VM creation or start operations:

```python
# Budget gate — check before creating or starting any VM
if self.cost_tracker and hasattr(self.cost_tracker, 'can_create_vm'):
    try:
        allowed, reason, details = await self.cost_tracker.can_create_vm()
        if not allowed:
            logger.warning(f"[VMLifecycle] ensure_static_vm_ready blocked by budget: {reason}")
            return (False, None, f"Budget exceeded: {reason}")
    except Exception as e:
        logger.error(f"[VMLifecycle] Budget check failed — blocking for safety: {e}")
        return (False, None, f"Budget check error: {e}")
```

**Scope exception:** When the VM already exists in STOPPED state and just needs starting (fast restart path), the budget check uses a lower threshold — starting a stopped VM costs near-zero compared to creating a new one. The gate only hard-blocks on `create` operations, not `start` of existing VMs.

---

## Part B: LLM Memory Hardening

### Principles

1. **Detect by signal, not symptom** — pageins/sec from `vm_stat` is the direct signal for mmap thrashing, not inference latency (which conflates with other delays).
2. **Two steps max** — one downgrade attempt, then GCP. No gradient search through the catalog.
3. **Application-level reservation** — reserve RAM in MemoryQuantizer's accounting before loading, preventing other components from consuming the headroom.
4. **No new files** — all changes in existing files.

### 3. Atomic Model Selection with Memory Reservation

**Location:** `backend/intelligence/unified_model_serving.py` (load_model), `backend/core/memory_quantizer.py`

**Problem:** `_select_best_model()` checks RAM, then 5-60s passes before `Llama()` allocates. Other startup phases consume RAM in the gap. The model may no longer fit when it actually loads.

**Fix:** After `_select_best_model()` picks a model, immediately register a memory reservation with MemoryQuantizer:

```python
# In load_model(), after model selection:
reservation_gb = effective_model_size_gb + kv_cache_gb + headroom_gb
self._memory_reservation_id = memory_quantizer.reserve_memory(
    reservation_gb, component="unified_model_serving"
)

# In MemoryQuantizer, the reservation is factored into tier calculations:
# effective_used = actual_used + sum(active_reservations)
# This makes all other components see the reserved headroom as consumed.

# On successful model load: release reservation (model itself holds the RAM)
# On failed load: release reservation, select smaller model
```

This is NOT an OS-level allocation. It's an accounting convention — MemoryQuantizer includes reservations in its `used_percent` calculation, which cascades to:
- GCPHybridPrimeRouter's pressure detection (sees higher pressure, doesn't trigger false GCP)
- Other startup phases' memory checks (see less available headroom)
- UnifiedModelServing's own pre-load validation (sees accurate picture)

**MemoryQuantizer API additions:**
```python
def reserve_memory(self, gb: float, component: str) -> str:
    """Reserve memory in accounting. Returns reservation ID."""

def release_reservation(self, reservation_id: str) -> None:
    """Release a memory reservation."""

def get_effective_used_percent(self) -> float:
    """Actual used + active reservations."""
```

### 4. Mmap Thrash Detection via `vm_stat` Pageins

**Location:** `backend/core/memory_quantizer.py`

**Problem:** When mmap-loaded models are under memory pressure, pages get evicted and every inference call triggers page faults. Latency degrades from 200ms to 30+ seconds per token. Nothing detects or reports this.

**Detection signal:** `vm_stat` pageins delta between samples. The MemoryQuantizer already runs async subprocess monitoring. Add a parallel `vm_stat` call.

**Implementation:**
```python
async def _get_pagein_rate(self) -> float:
    """Get pageins/sec from vm_stat delta between two samples."""
    proc = await asyncio.create_subprocess_exec(
        'vm_stat',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)
    # Parse "Pageins:" line, compute delta from last sample
    ...
```

**Thresholds (configurable via env vars):**
- `THRASH_PAGEIN_HEALTHY = 100` — pageins/sec below this is normal mmap behavior
- `THRASH_PAGEIN_WARNING = 500` — sustained 10s triggers THRASHING state
- `THRASH_PAGEIN_EMERGENCY = 2000` — immediate emergency (skip downgrade)
- `THRASH_SUSTAINED_SECONDS = 10` — how long warning-level must persist

**Callback mechanism:**
```python
def register_thrash_callback(self, callback: Callable[[str], Awaitable[None]]) -> None:
    """Register callback for thrash state changes.
    Callback receives: 'healthy', 'thrashing', or 'emergency'."""
```

**Integration with `_monitor_loop()`:** The existing monitoring loop (runs every 1-10s depending on tier) adds a `vm_stat` pagein rate check. When the rate crosses a threshold for the sustained period, callbacks fire.

### 5. Two-Step Cascade Response

**Location:** `backend/intelligence/unified_model_serving.py`

**Trigger:** MemoryQuantizer thrash callback fires with `'thrashing'` or `'emergency'`.

**Step 1 — Downgrade (on `thrashing`):**
1. Set `self._model_swapping = True` — signals PrimeRouter to queue/buffer requests
2. Call `self.stop()` to unload current model
3. Find the next smaller model in QUANT_CATALOG (one tier down by `quality_rank`)
4. Call `self.load_model(model_name=smaller_model)` with the reservation system
5. Wait 30s, re-check pagein rate
6. If recovered: set `self._model_swapping = False`, log downgrade, done
7. If still thrashing: proceed to Step 2

**Step 2 — GCP Offload (on continued thrashing or `emergency`):**
1. Signal GCPHybridPrimeRouter to enter TRIGGERING state (bypass N-of-M, instant)
2. The existing pressure-driven lifecycle handles PROVISIONING → BOOTING → ACTIVE
3. On ACTIVE, `_unload_local_model_after_stability()` unloads local model (already implemented)
4. Set `self._model_swapping = False`

**Emergency shortcut:** When callback fires with `'emergency'` (>2000 pageins/sec), skip Step 1 entirely. SIGSTOP the LLM process immediately, trigger GCP directly.

**PrimeRouter integration:** When `_model_swapping` is True:
- `route_request()` returns `RoutingTier.CLOUD_CLAUDE` (or queues if queue depth < 3)
- Circuit breaker for PRIME_LOCAL is NOT tripped (the swap is intentional, not a failure)
- Dashboard shows "Model swapping — temporary cloud fallback"

### 6. Downgrade Window Protection

**Problem:** During the 5-30s model swap window, no local inference is available. PrimeRouter must not circuit-break.

**Fix:** The `_model_swapping` flag in UnifiedModelServing is checked by PrimeRouter before it classifies a local failure:

```python
# In PrimeRouter routing logic:
if model_serving and model_serving._model_swapping:
    # Intentional swap, not a failure — use cloud temporarily
    return RoutingDecision(tier=RoutingTier.CLOUD_CLAUDE, reason=RoutingReason.CAPABILITY_REQUIRED)
```

This uses the existing hot-swap pattern (same as local-to-GCP transitions) extended to cover local model swaps.

---

## Files Modified

| File | Changes |
|---|---|
| `backend/core/gcp_vm_manager.py` | Fix `should_create_vm()` fail-open (line 4534), add budget gate to `ensure_static_vm_ready()` |
| `backend/core/memory_quantizer.py` | Memory reservation API, `vm_stat` pagein tracking, thrash detection, `register_thrash_callback()` |
| `backend/intelligence/unified_model_serving.py` | Thrash callback handler, one-tier downgrade cascade, `_model_swapping` flag, memory reservation on load |
| `backend/core/gcp_hybrid_prime_router.py` | Respect `_model_swapping` flag, expose `trigger_from_thrash()` for emergency bypass |

**No new files.**

---

## Expected Impact

| Metric | Before | After |
|---|---|---|
| `should_create_vm()` budget bypass | Fails open | Fails closed |
| `ensure_static_vm_ready()` budget check | None | Gated |
| Model loading RAM race | Unprotected 5-60s gap | Reserved in MemoryQuantizer accounting |
| Mmap thrash detection | None | `vm_stat` pageins/sec with 3 thresholds |
| Thrash response | None (infinite degradation) | Downgrade one tier → GCP offload cascade |
| Inference during model swap | Circuit breaker trips | Cloud fallback with `_model_swapping` flag |
