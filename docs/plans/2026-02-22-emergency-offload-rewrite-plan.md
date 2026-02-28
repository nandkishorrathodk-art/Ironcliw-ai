# Emergency Offload Path Rewrite — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace SIGSTOP-first emergency offload with clean model unloading as primary response, SIGSTOP as one-shot last resort only.

**Architecture:** Rewrite the emergency offload flow in GCPHybridPrimeRouter to use a 3-step escalation ladder: (1) clean model unload via COMPONENT_UNLOAD at CRITICAL tier, (2) post-unload verification + GCP attempt, (3) one-shot SIGSTOP at 95%+ as last resort. Remove the SIGSTOP→SIGCONT cycling logic that causes the death spiral.

**Tech Stack:** Python 3, asyncio, psutil, existing MemoryQuantizer/UnifiedModelServing COMPONENT_UNLOAD callbacks

**Design doc:** `docs/plans/2026-02-22-emergency-offload-rewrite-design.md`

---

### Task 1: Update Impact Predictions in MemoryQuantizer

**Files:**
- Modify: `backend/core/memory_quantizer.py:199-207`

**Step 1: Update `predict_optimization_impact()` estimates**

Replace lines 199-207:

```python
        impact_estimates = {
            OptimizationStrategy.CACHE_PRUNING: 50,
            OptimizationStrategy.LAZY_LOADING: 100,
            OptimizationStrategy.AGGRESSIVE_GC: 200,
            OptimizationStrategy.COMPONENT_UNLOAD: 300,
            OptimizationStrategy.BUFFER_REDUCTION: 150,
            OptimizationStrategy.EMERGENCY_CLEANUP: 500,
            OptimizationStrategy.PREDICTIVE_PREEMPT: 75
        }
```

With:

```python
        impact_estimates = {
            OptimizationStrategy.CACHE_PRUNING: 50,
            OptimizationStrategy.LAZY_LOADING: 100,
            OptimizationStrategy.AGGRESSIVE_GC: 200,
            # v266.0: COMPONENT_UNLOAD now actually unloads the LLM model (4-8GB)
            OptimizationStrategy.COMPONENT_UNLOAD: 6000,
            OptimizationStrategy.BUFFER_REDUCTION: 150,
            # v266.0: EMERGENCY_CLEANUP is gc.collect + flush — realistic estimate
            OptimizationStrategy.EMERGENCY_CLEANUP: 50,
            OptimizationStrategy.PREDICTIVE_PREEMPT: 75
        }
```

**Step 2: Verify**

Run: `python3 -c "
from backend.core.memory_quantizer import MemoryQuantizer, OptimizationStrategy
q = MemoryQuantizer()
print(f'COMPONENT_UNLOAD: {q.predict_optimization_impact(OptimizationStrategy.COMPONENT_UNLOAD)}MB')
print(f'EMERGENCY_CLEANUP: {q.predict_optimization_impact(OptimizationStrategy.EMERGENCY_CLEANUP)}MB')
print('OK')
"`
Expected: `COMPONENT_UNLOAD: 6000MB`, `EMERGENCY_CLEANUP: 50MB`, `OK`

**Step 3: Commit**

```bash
git add backend/core/memory_quantizer.py
git commit -m "$(cat <<'EOF'
fix(memory): update impact predictions for COMPONENT_UNLOAD and EMERGENCY_CLEANUP

COMPONENT_UNLOAD now actually unloads the LLM model (4-8GB freed) so
its impact prediction moves from 300MB to 6000MB. EMERGENCY_CLEANUP
only does gc.collect + flush (~50MB), so its inflated 500MB prediction
drops to 50MB. Accurate predictions help the tier strategy engine
make better optimization decisions.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Add Emergency Offload Configuration Constants

**Files:**
- Modify: `backend/core/gcp_hybrid_prime_router.py:168-178`

**Step 1: Replace and extend the emergency offload configuration block**

Replace lines 168-178:

```python
# Emergency offload configuration
EMERGENCY_OFFLOAD_RAM_PERCENT = float(os.getenv("EMERGENCY_OFFLOAD_RAM_PERCENT", "80.0"))  # SIGSTOP at 80%
EMERGENCY_OFFLOAD_TIMEOUT_SEC = float(os.getenv("EMERGENCY_OFFLOAD_TIMEOUT_SEC", "60.0"))  # Max time processes paused

# v192.0: Emergency offload anti-cycle protection
# Cooldown after releasing offload - prevents immediate re-trigger
EMERGENCY_OFFLOAD_COOLDOWN_SEC = float(os.getenv("EMERGENCY_OFFLOAD_COOLDOWN_SEC", "120.0"))  # 2 min cooldown
# Hysteresis threshold - RAM must drop this much below trigger before re-enabling
EMERGENCY_OFFLOAD_HYSTERESIS = float(os.getenv("EMERGENCY_OFFLOAD_HYSTERESIS", "10.0"))  # 10% below threshold
# Max consecutive offloads before forcing termination instead of pause
EMERGENCY_OFFLOAD_MAX_CYCLES = int(os.getenv("EMERGENCY_OFFLOAD_MAX_CYCLES", "3"))  # After 3 cycles, terminate
```

With:

```python
# v266.1: Emergency offload configuration — clean unload primary, SIGSTOP last resort
# Step 1 threshold: clean model unload via COMPONENT_UNLOAD
EMERGENCY_UNLOAD_RAM_PERCENT = float(os.getenv("EMERGENCY_UNLOAD_RAM_PERCENT", "85.0"))  # Clean unload at 85%
# Post-unload verification: wait this long to check if memory dropped
EMERGENCY_UNLOAD_VERIFY_DELAY_SEC = float(os.getenv("EMERGENCY_UNLOAD_VERIFY_DELAY_SEC", "12.0"))
# Minimum RAM drop (GB) to consider unload successful
EMERGENCY_UNLOAD_MIN_DROP_GB = float(os.getenv("EMERGENCY_UNLOAD_MIN_DROP_GB", "1.0"))
# Step 3 threshold: SIGSTOP as last resort (one-shot, no cycling)
EMERGENCY_SIGSTOP_RAM_PERCENT = float(os.getenv("EMERGENCY_SIGSTOP_RAM_PERCENT", "95.0"))
EMERGENCY_SIGSTOP_TIMEOUT_SEC = float(os.getenv("EMERGENCY_SIGSTOP_TIMEOUT_SEC", "60.0"))
# Cooldown applies to SIGSTOP only (clean unload has no cycling problem)
EMERGENCY_SIGSTOP_COOLDOWN_SEC = float(os.getenv("EMERGENCY_SIGSTOP_COOLDOWN_SEC", "120.0"))
# Hysteresis: RAM must drop this much below unload trigger before re-enabling
EMERGENCY_OFFLOAD_HYSTERESIS = float(os.getenv("EMERGENCY_OFFLOAD_HYSTERESIS", "10.0"))
```

NOTE: `EMERGENCY_OFFLOAD_MAX_CYCLES` is deliberately removed — no more cycling.

**Step 2: Verify**

Run: `python3 -c "from backend.core.gcp_hybrid_prime_router import EMERGENCY_UNLOAD_RAM_PERCENT, EMERGENCY_SIGSTOP_RAM_PERCENT; print(f'Unload: {EMERGENCY_UNLOAD_RAM_PERCENT}%, SIGSTOP: {EMERGENCY_SIGSTOP_RAM_PERCENT}%'); print('OK')"`
Expected: `Unload: 85.0%, SIGSTOP: 95.0%`, `OK`

**Step 3: Commit**

```bash
git add backend/core/gcp_hybrid_prime_router.py
git commit -m "$(cat <<'EOF'
refactor(router): replace emergency offload config with 3-step escalation

Replaces the SIGSTOP-at-80% config with clean-unload-at-85% +
SIGSTOP-at-95% thresholds. Removes EMERGENCY_OFFLOAD_MAX_CYCLES
(no more cycling). Adds post-unload verification delay and minimum
drop thresholds. SIGSTOP cooldown applies only to SIGSTOP (rare).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Update State Variables for New Offload Model

**Files:**
- Modify: `backend/core/gcp_hybrid_prime_router.py:761-774`

**Step 1: Replace emergency offload state variables**

Replace lines 761-774:

```python
        # Emergency offload state
        self._emergency_offload_active: bool = False
        self._emergency_offload_started_at: float = 0.0
        self._paused_processes: Dict[int, str] = {}  # pid -> process_name
        self._offload_lock: Optional[asyncio.Lock] = None  # Lazy init

        # v192.0: Anti-cycle protection for emergency offload
        self._emergency_offload_released_at: float = 0.0  # When last offload ended
        self._emergency_offload_cycle_count: int = 0  # Consecutive offload cycles
        self._emergency_offload_hysteresis_armed: bool = False  # True = waiting for RAM to drop

        # Process tracking for emergency offload
        self._ml_loader_ref = None  # Reference to ProcessIsolatedMLLoader
        self._local_llm_pids: Set[int] = set()  # PIDs of local LLM processes to pause
```

With:

```python
        # v266.1: Emergency offload state — 3-step escalation ladder
        self._emergency_offload_active: bool = False
        self._emergency_offload_started_at: float = 0.0
        self._offload_lock: Optional[asyncio.Lock] = None  # Lazy init
        self._clean_unload_fired: bool = False  # Step 1 completed
        self._clean_unload_verified: bool = False  # Post-unload verification passed

        # SIGSTOP state (last resort only)
        self._sigstop_active: bool = False
        self._paused_processes: Dict[int, str] = {}  # pid -> process_name
        self._sigstop_released_at: float = 0.0  # Cooldown tracking for SIGSTOP only

        # Hysteresis (applies to all emergency actions)
        self._emergency_offload_hysteresis_armed: bool = False

        # Process tracking for emergency offload
        self._ml_loader_ref = None  # Reference to ProcessIsolatedMLLoader
        self._local_llm_pids: Set[int] = set()  # PIDs of local LLM processes to pause
```

NOTE: `_emergency_offload_cycle_count` is deliberately removed.

**Step 2: Verify**

Run: `python3 -c "from backend.core.gcp_hybrid_prime_router import GCPHybridPrimeRouter; print('OK')"`

This may fail if the constructor requires arguments. If so, just verify syntax:
Run: `python3 -c "import py_compile; py_compile.compile('backend/core/gcp_hybrid_prime_router.py', doraise=True); print('OK')"`

**Step 3: Commit**

```bash
git add backend/core/gcp_hybrid_prime_router.py
git commit -m "$(cat <<'EOF'
refactor(router): update state variables for 3-step escalation model

Replaces cycle-based state (_emergency_offload_cycle_count) with
step-tracking state (_clean_unload_fired, _clean_unload_verified,
_sigstop_active). SIGSTOP cooldown tracked separately since it
now fires rarely. Hysteresis stays for all emergency actions.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Rewrite Emergency Offload Trigger Logic

**Files:**
- Modify: `backend/core/gcp_hybrid_prime_router.py:1147-1228` (the trigger block in the monitoring loop)

**Step 1: Replace the trigger block**

Find the block starting at line 1147:

```python
                # v93.0: Emergency offload check - highest priority
```

And ending at line 1228:

```python
                    continue
```

Replace the ENTIRE block (lines 1147-1228) with:

```python
                # v266.1: Emergency offload — 3-step escalation ladder
                # Step 1 (85%): Clean model unload via COMPONENT_UNLOAD
                # Step 2: Post-unload verification + GCP attempt
                # Step 3 (95%): SIGSTOP as one-shot last resort
                if not self._emergency_offload_active:
                    # Check hysteresis: must wait for RAM to drop before re-trigger
                    if self._emergency_offload_hysteresis_armed:
                        hysteresis_threshold = EMERGENCY_UNLOAD_RAM_PERCENT - EMERGENCY_OFFLOAD_HYSTERESIS
                        if used_percent < hysteresis_threshold:
                            self._emergency_offload_hysteresis_armed = False
                            self._clean_unload_fired = False
                            self._clean_unload_verified = False
                            self.logger.info(
                                f"[v266.1] RAM dropped to {used_percent:.1f}% — "
                                f"hysteresis disarmed, escalation reset"
                            )
                        else:
                            # Still above hysteresis threshold, skip
                            pass

                    # Step 1: Clean model unload at CRITICAL tier (85%+)
                    elif used_percent >= EMERGENCY_UNLOAD_RAM_PERCENT and not self._clean_unload_fired:
                        self.logger.warning(
                            f"[v266.1] RAM at {used_percent:.1f}% (>={EMERGENCY_UNLOAD_RAM_PERCENT}%) "
                            f"— firing COMPONENT_UNLOAD for clean model unload"
                        )
                        self._emergency_offload_active = True
                        self._emergency_offload_started_at = time.time()
                        self._clean_unload_fired = True
                        self._clean_unload_verified = False

                        # Fire COMPONENT_UNLOAD via MemoryQuantizer
                        try:
                            from backend.core.memory_quantizer import (
                                get_memory_quantizer, OptimizationStrategy,
                            )
                            _mq = await get_memory_quantizer()
                            if _mq:
                                await _mq._apply_strategy(OptimizationStrategy.COMPONENT_UNLOAD)
                                self.logger.info("[v266.1] COMPONENT_UNLOAD fired — waiting for verification")
                            else:
                                self.logger.warning("[v266.1] MemoryQuantizer unavailable")
                        except Exception as e:
                            self.logger.error(f"[v266.1] COMPONENT_UNLOAD failed: {e}")

                        # Signal to other repos
                        await self._signal_memory_pressure_to_repos(
                            status="offload_active",
                            action="unload",
                            used_percent=used_percent,
                            rate_mb_sec=memory_rate_mb_sec,
                        )
                        continue

                elif self._emergency_offload_active:
                    # Active offload — check escalation ladder
                    await self._check_offload_escalation(used_percent, memory_rate_mb_sec)
                    continue
```

**Step 2: Verify**

Run: `python3 -c "import py_compile; py_compile.compile('backend/core/gcp_hybrid_prime_router.py', doraise=True); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add backend/core/gcp_hybrid_prime_router.py
git commit -m "$(cat <<'EOF'
feat(router): rewrite emergency trigger for 3-step escalation

Replaces SIGSTOP-at-80% trigger with clean-unload-at-85% primary
response. Fires COMPONENT_UNLOAD via MemoryQuantizer to cleanly
unload the LLM model (4-8GB freed). Delegates active-offload
monitoring to _check_offload_escalation(). Removes cycling logic.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Implement `_check_offload_escalation()` — Verification + GCP + SIGSTOP

**Files:**
- Modify: `backend/core/gcp_hybrid_prime_router.py` (replace `_check_emergency_offload_release` at line 1604)

**Step 1: Replace `_check_emergency_offload_release()` with `_check_offload_escalation()`**

Replace the entire method `_check_emergency_offload_release` (lines 1604-1700) with:

```python
    async def _check_offload_escalation(self, current_used_percent: float, rate_mb_sec: float) -> None:
        """v266.1: Check escalation ladder during active emergency offload.

        Step 1 complete (clean unload fired). Now:
        - Verify unload worked (wait, re-read memory)
        - If RAM still high, attempt GCP
        - If GCP fails and RAM >= 95%, one-shot SIGSTOP (no cycling)
        """
        elapsed = time.time() - self._emergency_offload_started_at

        # --- Post-unload verification gate ---
        if self._clean_unload_fired and not self._clean_unload_verified:
            if elapsed < EMERGENCY_UNLOAD_VERIFY_DELAY_SEC:
                # Still waiting for verification window
                return

            # Verification window elapsed — read memory
            try:
                import psutil
                mem = psutil.virtual_memory()
                post_unload_percent = mem.percent
                available_gb = mem.available / (1024 ** 3)
            except Exception:
                post_unload_percent = current_used_percent
                available_gb = 0.0

            ram_dropped_gb = (current_used_percent - post_unload_percent) * psutil.virtual_memory().total / (100 * 1024 ** 3) if current_used_percent > post_unload_percent else 0.0

            if ram_dropped_gb >= EMERGENCY_UNLOAD_MIN_DROP_GB or post_unload_percent < EMERGENCY_UNLOAD_RAM_PERCENT:
                # Unload worked — cancel escalation
                self.logger.info(
                    f"[v266.1] Post-unload verification PASSED: "
                    f"RAM {post_unload_percent:.1f}% (dropped {ram_dropped_gb:.1f}GB). "
                    f"Escalation stopped."
                )
                self._clean_unload_verified = True
                self._emergency_offload_active = False
                self._emergency_offload_hysteresis_armed = True
                await self._signal_memory_pressure_to_repos(
                    status="normal", action=None, used_percent=post_unload_percent,
                )
                return

            # Unload didn't drop enough — proceed to GCP
            self._clean_unload_verified = True  # Mark verified (failed, but checked)
            self.logger.warning(
                f"[v266.1] Post-unload verification FAILED: "
                f"RAM still {post_unload_percent:.1f}% (dropped only {ram_dropped_gb:.1f}GB). "
                f"Attempting GCP offload."
            )

            # Step 2: GCP attempt
            if not self._gcp_permanently_unavailable:
                can_attempt, cooldown_reason = self._recovery_cascade.can_attempt_gcp()
                if can_attempt:
                    self.logger.info("[v266.1] Provisioning GCP VM for workload transfer...")
                    success = await self._trigger_vm_provisioning(
                        reason=f"post_unload_escalation_ram_{post_unload_percent:.0f}pct"
                    )
                    if success:
                        self.logger.info("[v266.1] GCP VM provisioned — escalation resolved")
                        self._recovery_cascade.record_success(RoutingTier.GCP_VM)
                        self._emergency_offload_active = False
                        self._emergency_offload_hysteresis_armed = True
                        return
                    else:
                        self.logger.warning("[v266.1] GCP provisioning failed")
                else:
                    self.logger.warning(f"[v266.1] GCP in cooldown: {cooldown_reason}")

            return  # Wait for next poll cycle to check SIGSTOP threshold

        # --- SIGSTOP last resort gate ---
        if current_used_percent >= EMERGENCY_SIGSTOP_RAM_PERCENT and not self._sigstop_active:
            # Check SIGSTOP cooldown
            time_since_sigstop = time.time() - self._sigstop_released_at
            if self._sigstop_released_at > 0 and time_since_sigstop < EMERGENCY_SIGSTOP_COOLDOWN_SEC:
                self.logger.warning(
                    f"[v266.1] RAM at {current_used_percent:.1f}% but SIGSTOP in cooldown "
                    f"({EMERGENCY_SIGSTOP_COOLDOWN_SEC - time_since_sigstop:.0f}s remaining)"
                )
                return

            self.logger.critical(
                f"[v266.1] LAST RESORT: RAM at {current_used_percent:.1f}% "
                f"(>={EMERGENCY_SIGSTOP_RAM_PERCENT}%) after unload + GCP failed. "
                f"One-shot SIGSTOP on Ironcliw-owned PIDs."
            )
            self._sigstop_active = True
            paused_count = await self._pause_local_llm_processes()
            self.logger.critical(f"[v266.1] SIGSTOP sent to {paused_count} process(es)")
            return

        # --- SIGSTOP timeout: terminate if no recovery ---
        if self._sigstop_active:
            sigstop_elapsed = time.time() - self._emergency_offload_started_at
            if sigstop_elapsed >= EMERGENCY_SIGSTOP_TIMEOUT_SEC:
                self.logger.critical(
                    f"[v266.1] SIGSTOP timeout ({sigstop_elapsed:.0f}s) — "
                    f"terminating paused processes (no cycling)"
                )
                await self._terminate_paused_processes()
                self._sigstop_active = False
                self._emergency_offload_active = False
                self._sigstop_released_at = time.time()
                self._emergency_offload_hysteresis_armed = True
                return

            # Check if GCP became available while SIGSTOP'd
            if self._gcp_controller and hasattr(self._gcp_controller, 'is_vm_available'):
                if self._gcp_controller.is_vm_available():
                    self.logger.info("[v266.1] GCP VM ready during SIGSTOP — terminating local processes")
                    await self._terminate_paused_processes()
                    self._sigstop_active = False
                    self._emergency_offload_active = False
                    return

        # --- Global timeout: force-end offload ---
        max_offload_duration = EMERGENCY_SIGSTOP_TIMEOUT_SEC * 3  # 180s absolute max
        if elapsed >= max_offload_duration:
            self.logger.warning(
                f"[v266.1] Global offload timeout ({elapsed:.0f}s) — force-ending"
            )
            if self._sigstop_active:
                await self._terminate_paused_processes()
            self._sigstop_active = False
            self._emergency_offload_active = False
            self._emergency_offload_hysteresis_armed = True
            return
```

**Step 2: Remove old `_release_emergency_offload()` method**

Delete the method at lines ~1702-1742 (`_release_emergency_offload`). It handled SIGCONT cycling which no longer exists. The SIGSTOP path now terminates directly (no SIGCONT → re-trigger).

IMPORTANT: Check if `_release_emergency_offload` is called from anywhere else in the file first. If it is, leave it but mark it deprecated. Use grep:

Run: `grep -n '_release_emergency_offload' backend/core/gcp_hybrid_prime_router.py`

If only called from the old `_check_emergency_offload_release` (which we just replaced), delete it safely. If called elsewhere, leave it with a deprecation note.

**Step 3: Verify**

Run: `python3 -c "import py_compile; py_compile.compile('backend/core/gcp_hybrid_prime_router.py', doraise=True); print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add backend/core/gcp_hybrid_prime_router.py
git commit -m "$(cat <<'EOF'
feat(router): implement 3-step escalation with verification gate

Replaces _check_emergency_offload_release (SIGCONT cycling) with
_check_offload_escalation (3-step ladder):
1. Post-unload verification: wait 12s, check if RAM dropped >=1GB
2. GCP attempt if verification fails
3. One-shot SIGSTOP at 95%+ as last resort, no cycling
Eliminates the death spiral root cause (SIGSTOP→SIGCONT→repeat).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Scope SIGSTOP to Ironcliw-Owned PIDs First

**Files:**
- Modify: `backend/core/gcp_hybrid_prime_router.py:1510-1574` (`_pause_local_llm_processes`)

**Step 1: Rewrite `_pause_local_llm_processes()` to prioritize tracked PIDs**

Replace the entire method (lines 1510-1574) with:

```python
    async def _pause_local_llm_processes(self) -> int:
        """v266.1: Pause local LLM processes via SIGSTOP (last resort only).

        Priority order:
        1. Ironcliw-owned tracked PIDs (_local_llm_pids) — always preferred
        2. ProcessIsolatedMLLoader PIDs — if available
        3. Pattern-based scan — fallback only when no tracked PIDs found

        Returns:
            Number of processes paused
        """
        paused_count = 0

        try:
            import psutil

            # Priority 1: Use tracked Ironcliw-owned PIDs
            if self._local_llm_pids:
                self.logger.info(
                    f"[v266.1] SIGSTOP targeting {len(self._local_llm_pids)} tracked Ironcliw PID(s)"
                )
                for pid in list(self._local_llm_pids):
                    if pid not in self._paused_processes and self._pause_process(pid, "tracked_llm"):
                        paused_count += 1
                if paused_count > 0:
                    return paused_count

            # Priority 2: ProcessIsolatedMLLoader PIDs
            if self._ml_loader_ref is None:
                try:
                    from backend.core.process_isolated_ml_loader import get_ml_loader
                    self._ml_loader_ref = await get_ml_loader()
                except Exception:
                    pass

            if self._ml_loader_ref and hasattr(self._ml_loader_ref, '_active_processes'):
                for pid in list(self._ml_loader_ref._active_processes.keys()):
                    if pid not in self._paused_processes and self._pause_process(pid, "ml_loader"):
                        paused_count += 1
                if paused_count > 0:
                    return paused_count

            # Priority 3: Pattern-based scan (fallback only)
            self.logger.warning(
                "[v266.1] No tracked PIDs — falling back to pattern-based process scan"
            )
            llm_patterns = [
                "ollama", "llama", "llama.cpp", "llamacpp",
                "text-generation", "vllm", "transformers",
                "jarvis-prime", "jarvis_prime",
            ]

            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    proc_info = proc.info
                    pid = proc_info['pid']

                    if pid in self._paused_processes:
                        continue

                    name = (proc_info.get('name') or '').lower()
                    cmdline = ' '.join(proc_info.get('cmdline') or []).lower()

                    is_llm_process = any(
                        pattern in name or pattern in cmdline
                        for pattern in llm_patterns
                    )

                    if is_llm_process:
                        self.logger.info(
                            f"[v266.1] Pattern match: PID {pid} ({name}) — sending SIGSTOP"
                        )
                        if self._pause_process(pid, name):
                            paused_count += 1

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

        except Exception as e:
            self.logger.error(f"[v266.1] Error pausing LLM processes: {e}")

        return paused_count
```

**Step 2: Verify**

Run: `python3 -c "import py_compile; py_compile.compile('backend/core/gcp_hybrid_prime_router.py', doraise=True); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add backend/core/gcp_hybrid_prime_router.py
git commit -m "$(cat <<'EOF'
feat(router): scope SIGSTOP to Ironcliw-owned PIDs first

Prioritizes tracked PIDs (_local_llm_pids) over pattern-based
scanning. Pattern scan is now fallback only when no tracked PIDs
exist. Reduces collateral risk of SIGSTOP hitting unrelated
processes (ollama instances, other ML workloads).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Remove Hysteresis Reset from Old Location

**Files:**
- Modify: `backend/core/gcp_hybrid_prime_router.py:1130-1145` (old hysteresis reset block)

**Step 1: Find and remove the old hysteresis reset block**

The old block at lines ~1130-1145 resets hysteresis and cycle count. The cycle count is gone, and hysteresis reset is now in the trigger block (Task 4). Find this block:

```python
                # v192.0: Reset hysteresis and cycle count when RAM drops below safe threshold
                hysteresis_threshold = EMERGENCY_OFFLOAD_RAM_PERCENT - EMERGENCY_OFFLOAD_HYSTERESIS
                if self._emergency_offload_hysteresis_armed and used_percent < hysteresis_threshold:
```

Remove it entirely (the new trigger block in Task 4 handles hysteresis reset).

IMPORTANT: Read the area around these lines first to make sure you're removing the right block and not breaking surrounding code. The block should be 6-8 lines ending with the cycle count reset.

**Step 2: Also remove any remaining references to `_emergency_offload_cycle_count` and `EMERGENCY_OFFLOAD_MAX_CYCLES`**

Search for both strings and remove/replace:
- `self._emergency_offload_cycle_count` — should be gone after Tasks 3-4
- `EMERGENCY_OFFLOAD_MAX_CYCLES` — should be gone after Task 2
- `EMERGENCY_OFFLOAD_RAM_PERCENT` — replace with `EMERGENCY_UNLOAD_RAM_PERCENT` where used
- `EMERGENCY_OFFLOAD_TIMEOUT_SEC` — replace with `EMERGENCY_SIGSTOP_TIMEOUT_SEC` where used
- `EMERGENCY_OFFLOAD_COOLDOWN_SEC` — replace with `EMERGENCY_SIGSTOP_COOLDOWN_SEC` where used

Run grep to find all occurrences:
```bash
grep -n "EMERGENCY_OFFLOAD_RAM_PERCENT\|EMERGENCY_OFFLOAD_MAX_CYCLES\|_emergency_offload_cycle_count\|EMERGENCY_OFFLOAD_TIMEOUT_SEC\|EMERGENCY_OFFLOAD_COOLDOWN_SEC" backend/core/gcp_hybrid_prime_router.py
```

Fix any remaining references.

**Step 3: Verify**

Run: `python3 -c "import py_compile; py_compile.compile('backend/core/gcp_hybrid_prime_router.py', doraise=True); print('OK')"`
Expected: `OK`

Also verify no old constants remain:
Run: `grep -c "EMERGENCY_OFFLOAD_RAM_PERCENT\|EMERGENCY_OFFLOAD_MAX_CYCLES\|_emergency_offload_cycle_count" backend/core/gcp_hybrid_prime_router.py`
Expected: `0`

**Step 4: Commit**

```bash
git add backend/core/gcp_hybrid_prime_router.py
git commit -m "$(cat <<'EOF'
refactor(router): remove legacy cycling logic and old constants

Removes _emergency_offload_cycle_count, EMERGENCY_OFFLOAD_MAX_CYCLES,
and the old hysteresis reset block. Renames EMERGENCY_OFFLOAD_RAM_PERCENT
to EMERGENCY_UNLOAD_RAM_PERCENT. The cycling logic was the root cause
of the SIGSTOP death spiral — now eliminated.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Integration Verification

**Files:** None (verification only)

**Step 1: Verify all imports**

```bash
python3 -c "
from backend.core.memory_quantizer import MemoryQuantizer, OptimizationStrategy
from backend.core.gcp_hybrid_prime_router import (
    EMERGENCY_UNLOAD_RAM_PERCENT,
    EMERGENCY_SIGSTOP_RAM_PERCENT,
    EMERGENCY_UNLOAD_VERIFY_DELAY_SEC,
)
print(f'Unload threshold: {EMERGENCY_UNLOAD_RAM_PERCENT}%')
print(f'SIGSTOP threshold: {EMERGENCY_SIGSTOP_RAM_PERCENT}%')
print(f'Verify delay: {EMERGENCY_UNLOAD_VERIFY_DELAY_SEC}s')
print('All imports OK')
"
```

**Step 2: Verify impact predictions**

```bash
python3 -c "
from backend.core.memory_quantizer import MemoryQuantizer, OptimizationStrategy
q = MemoryQuantizer()
assert q.predict_optimization_impact(OptimizationStrategy.COMPONENT_UNLOAD) == 6000
assert q.predict_optimization_impact(OptimizationStrategy.EMERGENCY_CLEANUP) == 50
print('Impact predictions correct')
"
```

**Step 3: Verify no old constants remain**

```bash
grep -c "EMERGENCY_OFFLOAD_RAM_PERCENT\|EMERGENCY_OFFLOAD_MAX_CYCLES\|_emergency_offload_cycle_count" backend/core/gcp_hybrid_prime_router.py
```
Expected: `0`

**Step 4: Verify COMPONENT_UNLOAD still fires callbacks**

```bash
python3 -c "
import asyncio
from backend.core.memory_quantizer import MemoryQuantizer, OptimizationStrategy
q = MemoryQuantizer()
_fired = []
q.register_unload_callback(lambda tier: _fired.append(tier))
asyncio.run(q._apply_strategy(OptimizationStrategy.COMPONENT_UNLOAD))
assert len(_fired) == 1, f'Expected 1 callback, got {len(_fired)}'
print('Callback round-trip OK')
"
```

**Step 5: Verify syntax of all modified files**

```bash
python3 -c "
import py_compile
py_compile.compile('backend/core/memory_quantizer.py', doraise=True)
py_compile.compile('backend/core/gcp_hybrid_prime_router.py', doraise=True)
print('All syntax OK')
"
```
