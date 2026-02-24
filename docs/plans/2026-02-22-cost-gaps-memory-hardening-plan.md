# Cost Gap Closure + LLM Memory Hardening — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close remaining cost leak vectors (fail-open wrappers, missing budget gates) and add defensive memory hardening (reservation system, mmap thrash detection, two-step cascade response).

**Architecture:** Part A fixes two cost leak paths in `gcp_vm_manager.py`. Part B adds memory reservation accounting to MemoryQuantizer, `vm_stat` pagein-based thrash detection, and a cascade response (downgrade model one tier → GCP offload) in UnifiedModelServing with PrimeRouter awareness.

**Tech Stack:** Python 3, asyncio, psutil, `vm_stat` (macOS), llama-cpp-python, existing MemoryQuantizer/UnifiedModelServing/GCPHybridPrimeRouter

**Design doc:** `docs/plans/2026-02-22-cost-gaps-memory-hardening-design.md`

---

## Part A: Cost Gap Closure

---

### Task 1: Fix `should_create_vm()` Fail-Open Wrapper

**Files:**
- Modify: `backend/core/gcp_vm_manager.py:4533-4534`

**Step 1: Change fail-open to fail-closed**

At line 4533-4534, replace:

```python
            except Exception as e:
                logger.warning(f"⚠️ Budget check failed (allowing VM): {e}")
```

With:

```python
            except Exception as e:
                logger.error(
                    f"🚫 [CostGuard] Budget check failed — blocking VM creation for safety: {e}"
                )
                return (False, f"Budget check error (blocking): {e}", 0.0)
```

**Step 2: Verify**

Run: `python3 -c "from backend.core.gcp_vm_manager import GCPVMManager; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add backend/core/gcp_vm_manager.py
git commit -m "$(cat <<'EOF'
fix(cost): close should_create_vm() fail-open wrapper

The except block at line 4534 caught can_create_vm() failures and allowed
VM creation anyway, completely negating the fail-closed fix in cost_tracker.
Now returns (False, reason, 0.0) on budget check failure.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Add Budget Gate to `ensure_static_vm_ready()`

**Files:**
- Modify: `backend/core/gcp_vm_manager.py:7320-7341` (after config validation, before VM lock)

**Step 1: Add budget gate after config validation**

After the config validation block (line 7341: `return False, None, f"CONFIG_INVALID: {validation_error}"`), and BEFORE the `async with self._vm_lock:` line (line 7344), add:

```python
        # v266.0: Budget gate — check cost before creating or starting VMs
        # For STOPPED VMs (fast restart), use a lenient check since starting
        # costs near-zero. For new CREATE operations, enforce full budget.
        if self.cost_tracker and hasattr(self.cost_tracker, 'can_create_vm'):
            try:
                allowed, reason, details = await self.cost_tracker.can_create_vm()
                if not allowed:
                    logger.warning(
                        f"🚫 [InvincibleNode] ensure_static_vm_ready blocked by budget: {reason}"
                    )
                    return (False, None, f"BUDGET_EXCEEDED: {reason}")
            except Exception as e:
                logger.error(
                    f"🚫 [InvincibleNode] Budget check failed — blocking for safety: {e}"
                )
                return (False, None, f"BUDGET_CHECK_ERROR: {e}")
```

**Step 2: Verify**

Run: `python3 -c "from backend.core.gcp_vm_manager import GCPVMManager; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add backend/core/gcp_vm_manager.py
git commit -m "$(cat <<'EOF'
fix(cost): add budget gate to ensure_static_vm_ready()

The Invincible Node creation path never called can_create_vm() — VMs were
created and started with zero budget awareness. This was likely responsible
for $30-45/mo of the $92 cost spike. Now checks budget before any VM
create/start operation. Fails closed on budget check errors.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Part B: LLM Memory Hardening

---

### Task 3: Add Memory Reservation System to MemoryQuantizer

**Files:**
- Modify: `backend/core/memory_quantizer.py:486-538` (get_current_metrics_async), `:985-987` (callbacks area)

**Step 1: Add reservation data structures and API**

Near the existing callback registration (line 985), add the reservation system:

```python
    # v266.0: Memory reservation system
    # Allows components to "reserve" RAM in accounting before actually allocating.
    # Reservations are factored into tier calculations so other components see
    # accurate available headroom.

    def reserve_memory(self, gb: float, component: str) -> str:
        """Reserve memory in accounting. Returns reservation ID.

        The reservation is factored into get_current_metrics_async() so all
        consumers see reduced available RAM. This prevents the stale-check
        race: component A checks RAM, component B allocates between check
        and A's actual allocation.
        """
        import uuid
        reservation_id = f"res_{component}_{uuid.uuid4().hex[:8]}"
        if not hasattr(self, '_memory_reservations'):
            self._memory_reservations: Dict[str, Tuple[float, str]] = {}
        self._memory_reservations[reservation_id] = (gb, component)
        logger.info(
            f"[MemReserve] Reserved {gb:.2f}GB for {component} (id={reservation_id})"
        )
        return reservation_id

    def release_reservation(self, reservation_id: str) -> None:
        """Release a memory reservation."""
        if not hasattr(self, '_memory_reservations'):
            return
        if reservation_id in self._memory_reservations:
            gb, component = self._memory_reservations.pop(reservation_id)
            logger.info(
                f"[MemReserve] Released {gb:.2f}GB from {component} (id={reservation_id})"
            )

    def get_total_reserved_gb(self) -> float:
        """Get total reserved memory in GB."""
        if not hasattr(self, '_memory_reservations'):
            return 0.0
        return sum(gb for gb, _ in self._memory_reservations.values())
```

**Step 2: Factor reservations into `get_current_metrics_async()`**

In `get_current_metrics_async()` (line 486), after the `macos_pressure_percent` calculation, add reservation adjustment:

```python
        # v266.0: Factor in memory reservations from other components
        reserved_gb = self.get_total_reserved_gb()
        if reserved_gb > 0:
            reserved_bytes = reserved_gb * 1024 * 1024 * 1024
            # Adjust pressure percent to include reservations
            effective_used = (wired + active + compressed) + reserved_bytes
            macos_pressure_percent = (effective_used / mem.total) * 100
            logger.debug(
                f"[MemReserve] Adjusted pressure: {macos_pressure_percent:.1f}% "
                f"(+{reserved_gb:.2f}GB reserved)"
            )
```

**Step 3: Initialize `_memory_reservations` in `__init__`**

Find the `__init__` method and add near the other dict initializations:

```python
        self._memory_reservations: Dict[str, Tuple[float, str]] = {}
```

**Step 4: Verify**

Run: `python3 -c "
import asyncio
from backend.core.memory_quantizer import MemoryQuantizer
q = MemoryQuantizer()
rid = q.reserve_memory(2.0, 'test')
print('Reserved:', q.get_total_reserved_gb(), 'GB')
q.release_reservation(rid)
print('After release:', q.get_total_reserved_gb(), 'GB')
print('OK')
"`
Expected: `Reserved: 2.0 GB`, `After release: 0.0 GB`, `OK`

**Step 5: Commit**

```bash
git add backend/core/memory_quantizer.py
git commit -m "$(cat <<'EOF'
feat(memory): add reservation system to MemoryQuantizer

Components can now reserve_memory(gb, component) before allocating,
which factors into tier calculations. This prevents the stale-RAM-check
race where component A checks available RAM, component B allocates in
the gap, and A's load exceeds actual headroom. Reservations are released
on successful load or failure.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Add `vm_stat` Pagein Thrash Detection to MemoryQuantizer

**Files:**
- Modify: `backend/core/memory_quantizer.py:684` (near async pressure method), `:919-956` (_monitor_loop), `:985` (callbacks)

**Step 1: Add thrash detection config constants**

Near the top of the file (after imports), add:

```python
# v266.0: Mmap thrash detection via vm_stat pageins
THRASH_PAGEIN_HEALTHY = int(os.getenv("THRASH_PAGEIN_HEALTHY", "100"))
THRASH_PAGEIN_WARNING = int(os.getenv("THRASH_PAGEIN_WARNING", "500"))
THRASH_PAGEIN_EMERGENCY = int(os.getenv("THRASH_PAGEIN_EMERGENCY", "2000"))
THRASH_SUSTAINED_SECONDS = int(os.getenv("THRASH_SUSTAINED_SECONDS", "10"))
```

**Step 2: Add state fields in `__init__`**

```python
        # v266.0: Thrash detection state
        self._last_pageins: Optional[int] = None
        self._last_pagein_time: float = 0.0
        self._pagein_rate: float = 0.0  # pageins/sec
        self._thrash_state: str = "healthy"  # healthy, thrashing, emergency
        self._thrash_warning_since: float = 0.0
        self._thrash_callbacks: List[Callable] = []
```

**Step 3: Add `_get_pagein_rate_async()` method**

Near `_get_memory_pressure_async()` (line 684), add:

```python
    async def _get_pagein_rate_async(self) -> float:
        """Get pageins/sec from vm_stat delta between two samples.

        Uses the 'Pageins' line from vm_stat. Returns pageins/sec since
        last call, or 0.0 on first call or error.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                'vm_stat',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)
            output = stdout.decode('utf-8', errors='replace')

            # Parse "Pageins:" line
            current_pageins = None
            for line in output.splitlines():
                if 'Pageins' in line or 'pageins' in line:
                    # Format: "Pageins:                          12345."
                    parts = line.split(':')
                    if len(parts) >= 2:
                        num_str = parts[1].strip().rstrip('.')
                        try:
                            current_pageins = int(num_str)
                        except ValueError:
                            pass
                    break

            if current_pageins is None:
                return 0.0

            now = time.time()

            if self._last_pageins is not None and self._last_pagein_time > 0:
                elapsed = now - self._last_pagein_time
                if elapsed > 0:
                    delta = current_pageins - self._last_pageins
                    rate = delta / elapsed
                    self._last_pageins = current_pageins
                    self._last_pagein_time = now
                    return max(0.0, rate)

            # First call — just store baseline
            self._last_pageins = current_pageins
            self._last_pagein_time = now
            return 0.0

        except asyncio.TimeoutError:
            return 0.0
        except FileNotFoundError:
            return 0.0
        except Exception:
            return 0.0
```

**Step 4: Add thrash callback registration**

Near `register_tier_change_callback()` (line 985):

```python
    def register_thrash_callback(self, callback: Callable) -> None:
        """Register callback for thrash state changes.

        Callback receives one argument: 'healthy', 'thrashing', or 'emergency'.
        Called when state transitions (not on every check).
        """
        self._thrash_callbacks.append(callback)
```

**Step 5: Add thrash checking to `_monitor_loop()`**

In `_monitor_loop()` (line 919), after the tier change check (line 933) and before the pattern learning (line 936), add:

```python
                # v266.0: Mmap thrash detection via vm_stat pageins
                self._pagein_rate = await self._get_pagein_rate_async()
                await self._check_thrash_state()
```

And add the `_check_thrash_state` method:

```python
    async def _check_thrash_state(self) -> None:
        """Check pagein rate and transition thrash state if needed."""
        rate = self._pagein_rate
        now = time.time()
        old_state = self._thrash_state

        if rate >= THRASH_PAGEIN_EMERGENCY:
            new_state = "emergency"
        elif rate >= THRASH_PAGEIN_WARNING:
            if self._thrash_warning_since == 0:
                self._thrash_warning_since = now
            elif now - self._thrash_warning_since >= THRASH_SUSTAINED_SECONDS:
                new_state = "thrashing"
            else:
                return  # Still accumulating sustained readings
        else:
            self._thrash_warning_since = 0.0
            new_state = "healthy"

        if new_state != old_state:
            self._thrash_state = new_state
            logger.warning(
                f"[ThrashDetect] State change: {old_state} -> {new_state} "
                f"(pageins/sec: {rate:.0f})"
            )
            for callback in self._thrash_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(new_state)
                    else:
                        callback(new_state)
                except Exception as e:
                    logger.debug(f"Thrash callback error: {e}")
```

**Step 6: Verify**

Run: `python3 -c "
import asyncio
from backend.core.memory_quantizer import MemoryQuantizer
q = MemoryQuantizer()
rate = asyncio.run(q._get_pagein_rate_async())
print(f'Pagein rate: {rate:.0f}/sec')
print(f'Thrash state: {q._thrash_state}')
print('OK')
"`
Expected: Pagein rate and thrash state printed, `OK`.

**Step 7: Commit**

```bash
git add backend/core/memory_quantizer.py
git commit -m "$(cat <<'EOF'
feat(memory): add vm_stat pagein thrash detection to MemoryQuantizer

Parse vm_stat pageins delta between samples to detect mmap thrashing.
Three thresholds: HEALTHY (<100/sec), THRASHING (>500/sec sustained 10s),
EMERGENCY (>2000/sec instant). Fires register_thrash_callback() on state
transitions. This is the direct, unambiguous signal for mmap page fault
storms — unlike inference latency which conflates with other delays.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Add Memory Reservation to UnifiedModelServing `load_model()`

**Files:**
- Modify: `backend/intelligence/unified_model_serving.py:608-837` (load_model method)

**Step 1: Add reservation before model loading**

In `load_model()`, after model selection succeeds (around line 670, after `_select_best_model()` returns) and BEFORE the `Llama()` constructor call (line 777), add memory reservation:

Find the section where `effective_model_size_gb` or `_effective_min` is calculated and before `self._model = Llama(...)`, add:

```python
            # v266.0: Reserve memory in MemoryQuantizer accounting
            # This prevents other startup phases from consuming headroom
            # between our RAM check and the actual Llama() allocation.
            _reservation_id = None
            try:
                from backend.core.memory_quantizer import get_memory_quantizer
                _mq = get_memory_quantizer()
                if _mq and hasattr(_mq, 'reserve_memory'):
                    _reserve_gb = (model_size_mb / 1024.0) * _mmap_factor + _headroom_gb
                    _reservation_id = _mq.reserve_memory(_reserve_gb, "unified_model_serving")
            except Exception as e:
                self.logger.debug(f"[v266.0] Memory reservation failed (non-fatal): {e}")
```

Note: `model_size_mb` is from `selected_entry["size_mb"]`, `_mmap_factor` and `_headroom_gb` are already in scope. Read the exact variable names from the code before implementing — they may be named differently.

**Step 2: Release reservation after load (success or failure)**

After the `self._model = Llama(...)` call succeeds (around line 784), add:

```python
            # v266.0: Release reservation — model itself now holds the RAM
            if _reservation_id:
                try:
                    _mq.release_reservation(_reservation_id)
                except Exception:
                    pass
```

In ALL error/exception paths that exit `load_model()` before the Llama call (or where Llama raises), add the same release:

```python
            # Release reservation on failure
            if _reservation_id:
                try:
                    _mq.release_reservation(_reservation_id)
                except Exception:
                    pass
```

**Step 3: Verify**

Run: `python3 -c "from backend.intelligence.unified_model_serving import UnifiedModelServing; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add backend/intelligence/unified_model_serving.py
git commit -m "$(cat <<'EOF'
feat(model): add memory reservation before Llama() allocation

Reserve expected RAM in MemoryQuantizer accounting after model selection
but before Llama() constructor. This prevents the stale-RAM-check race
where other startup phases consume headroom between our check and actual
allocation. Reservation released on successful load or failure.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Add Thrash Cascade Response to UnifiedModelServing

**Files:**
- Modify: `backend/intelligence/unified_model_serving.py:408` (__init__ area), `:2128-2208` (memory pressure monitor area)

**Step 1: Add thrash handling state in `__init__`**

Near the existing `_inference_executor` (line 408), add:

```python
        # v266.0: Mmap thrash cascade state
        self._model_swapping: bool = False
        self._current_model_entry: Optional[Dict[str, Any]] = None  # Current QUANT_CATALOG entry
        self._thrash_downgrade_attempted: bool = False
```

**Step 2: Track current model entry in `load_model()`**

In `load_model()`, after successful model selection, store the catalog entry:

```python
        self._current_model_entry = selected_entry  # v266.0: Track for thrash downgrade
```

**Step 3: Add thrash callback handler**

Near the memory pressure monitor method, add:

```python
    async def _handle_thrash_state_change(self, new_state: str) -> None:
        """Handle mmap thrash state changes from MemoryQuantizer.

        Two-step cascade:
        Step 1 (thrashing): Downgrade one tier in QUANT_CATALOG.
        Step 2 (still thrashing or emergency): Trigger GCP offload.
        """
        if new_state == "healthy":
            self._thrash_downgrade_attempted = False
            return

        if new_state == "emergency":
            # Skip downgrade, go straight to GCP
            self.logger.critical(
                "[ThrashCascade] EMERGENCY pagein rate — triggering GCP offload"
            )
            await self._trigger_gcp_offload_from_thrash()
            return

        if new_state == "thrashing":
            if self._thrash_downgrade_attempted:
                # Already tried downgrade, still thrashing — go to GCP
                self.logger.warning(
                    "[ThrashCascade] Still thrashing after downgrade — triggering GCP offload"
                )
                await self._trigger_gcp_offload_from_thrash()
                return

            # Step 1: Downgrade one tier
            self._thrash_downgrade_attempted = True
            await self._downgrade_model_one_tier()

    async def _downgrade_model_one_tier(self) -> None:
        """Unload current model, load next smaller from QUANT_CATALOG."""
        if not self._current_model_entry:
            self.logger.warning("[ThrashCascade] No current model entry — cannot downgrade")
            return

        current_rank = self._current_model_entry.get("quality_rank", 0)
        # Find next tier (higher quality_rank number = smaller model)
        smaller_entries = [
            e for e in self.QUANT_CATALOG
            if e["quality_rank"] > current_rank
        ]
        smaller_entries.sort(key=lambda e: e["quality_rank"])

        if not smaller_entries:
            self.logger.warning(
                "[ThrashCascade] Already on smallest model — cannot downgrade further"
            )
            await self._trigger_gcp_offload_from_thrash()
            return

        next_model = smaller_entries[0]
        self.logger.warning(
            f"[ThrashCascade] Downgrading: {self._current_model_entry['name']} "
            f"-> {next_model['name']}"
        )

        self._model_swapping = True
        try:
            # Unload current model
            await self.stop()

            # Load smaller model
            success = await self.load_model(model_name=next_model["filename"])
            if not success:
                self.logger.error("[ThrashCascade] Downgrade load failed — triggering GCP")
                await self._trigger_gcp_offload_from_thrash()
        except Exception as e:
            self.logger.error(f"[ThrashCascade] Downgrade error: {e}")
            await self._trigger_gcp_offload_from_thrash()
        finally:
            self._model_swapping = False

    async def _trigger_gcp_offload_from_thrash(self) -> None:
        """Signal GCP router to enter VM provisioning for thrash recovery."""
        self._model_swapping = True
        try:
            from backend.core.gcp_hybrid_prime_router import get_router
            router = get_router()
            if router and hasattr(router, '_transition_vm_lifecycle'):
                from backend.core.gcp_hybrid_prime_router import VMLifecycleState
                if router._vm_lifecycle_state == VMLifecycleState.IDLE:
                    router._transition_vm_lifecycle(
                        VMLifecycleState.TRIGGERING, "mmap_thrash_emergency"
                    )
                    router._transition_vm_lifecycle(
                        VMLifecycleState.PROVISIONING, "thrash_bypass"
                    )
                    await router._trigger_vm_provisioning(reason="mmap_thrash")
        except ImportError:
            self.logger.debug("[ThrashCascade] GCP router not available")
        except Exception as e:
            self.logger.error(f"[ThrashCascade] GCP offload trigger failed: {e}")
```

**Step 4: Register thrash callback on startup**

In the initialization area (near where the memory pressure monitor is started), or in a `start()` method if one exists, add:

```python
        # v266.0: Register for thrash detection
        try:
            from backend.core.memory_quantizer import get_memory_quantizer
            _mq = get_memory_quantizer()
            if _mq and hasattr(_mq, 'register_thrash_callback'):
                _mq.register_thrash_callback(self._handle_thrash_state_change)
                self.logger.info("[v266.0] Registered thrash detection callback")
        except Exception as e:
            self.logger.debug(f"[v266.0] Thrash callback registration: {e}")
```

**Step 5: Verify**

Run: `python3 -c "from backend.intelligence.unified_model_serving import UnifiedModelServing; print('OK')"`
Expected: `OK`

**Step 6: Commit**

```bash
git add backend/intelligence/unified_model_serving.py
git commit -m "$(cat <<'EOF'
feat(model): add two-step thrash cascade response

When MemoryQuantizer detects mmap thrashing via vm_stat pageins:
Step 1 (thrashing): Unload current model, load one tier smaller from
QUANT_CATALOG. Sets _model_swapping flag so PrimeRouter queues requests.
Step 2 (still thrashing or emergency): Trigger GCP VM lifecycle.
Emergency (>2000 pageins/sec): Skip downgrade, go straight to GCP.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Add `_model_swapping` Awareness to GCPHybridPrimeRouter

**Files:**
- Modify: `backend/core/gcp_hybrid_prime_router.py:2892` (route method area)

**Step 1: Add model-swapping check to `route()`**

In the `route()` method (line 2892), early in the method (after budget check, before local RAM check), add:

```python
        # v266.0: If local model is being swapped (thrash cascade), use cloud
        try:
            from backend.intelligence.unified_model_serving import get_model_serving
            model_serving = get_model_serving()
            if model_serving and getattr(model_serving, '_model_swapping', False):
                self.logger.info(
                    "[v266.0] Local model swapping (thrash cascade) — routing to cloud"
                )
                return RoutingDecision(
                    tier=RoutingTier.CLOUD_CLAUDE,
                    reason=RoutingReason.CAPABILITY_REQUIRED,
                    timeout_ms=CLOUD_API_TIMEOUT_MS,
                    metadata={"model_swapping": True},
                )
        except ImportError:
            pass
```

**Step 2: Verify**

Run: `python3 -c "from backend.core.gcp_hybrid_prime_router import GCPHybridPrimeRouter; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add backend/core/gcp_hybrid_prime_router.py
git commit -m "$(cat <<'EOF'
feat(router): respect _model_swapping flag during routing

When UnifiedModelServing is swapping models (thrash cascade downgrade),
route() returns CLOUD_CLAUDE instead of attempting local inference. This
prevents circuit breaker trips during intentional model swaps and provides
seamless cloud fallback during the 5-30s swap window.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Integration Verification

**Files:** None (verification only)

**Step 1: Verify all imports chain**

```bash
python3 -c "
from backend.core.gcp_vm_manager import GCPVMManager, VMAction, VMState
from backend.core.memory_quantizer import MemoryQuantizer
from backend.core.gcp_hybrid_prime_router import GCPHybridPrimeRouter, VMLifecycleState
from backend.intelligence.unified_model_serving import UnifiedModelServing
from backend.core.cost_tracker import CostTracker
print('All imports OK')
"
```

**Step 2: Verify reservation round-trip**

```bash
python3 -c "
import asyncio
from backend.core.memory_quantizer import MemoryQuantizer
q = MemoryQuantizer()
rid = q.reserve_memory(4.0, 'test_model')
metrics = asyncio.run(q.get_current_metrics_async())
print(f'With 4GB reservation: {metrics.system_memory_percent:.1f}%')
q.release_reservation(rid)
metrics2 = asyncio.run(q.get_current_metrics_async())
print(f'After release: {metrics2.system_memory_percent:.1f}%')
print(f'Delta: {metrics.system_memory_percent - metrics2.system_memory_percent:.1f}%')
print('OK')
"
```

**Step 3: Verify pagein rate works**

```bash
python3 -c "
import asyncio
from backend.core.memory_quantizer import MemoryQuantizer
q = MemoryQuantizer()
r1 = asyncio.run(q._get_pagein_rate_async())
import time; time.sleep(1)
r2 = asyncio.run(q._get_pagein_rate_async())
print(f'Pagein rates: {r1:.0f}/sec, {r2:.0f}/sec')
print(f'Thrash state: {q._thrash_state}')
print('OK')
"
```

**Step 4: Verify supervisor syntax**

```bash
python3 -c "import py_compile; py_compile.compile('unified_supervisor.py', doraise=True); print('Supervisor OK')"
```
