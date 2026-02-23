# Cascading Failure Hardening — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add post-crisis model auto-recovery (hybrid lazy+background warm-up) and startup memory phase gates (monotonic mode escalation, per-subprocess admission, ENOMEM handling).

**Architecture:** Two independent fixes. Fix A adds a recovery phase to the GCPHybridPrimeRouter monitoring loop that detects memory stabilization after emergency offload and reloads the LLM model in background while CLAUDE handles requests. Fix B adds memory gates at Phase 2/3 boundaries in the supervisor, plus a `can_spawn_heavy_process()` admission check before each subprocess creation, with `errno.ENOMEM` handling.

**Tech Stack:** Python 3, asyncio, psutil, existing MemoryQuantizer/UnifiedModelServing/GCPHybridPrimeRouter

**Design doc:** `docs/plans/2026-02-22-cascading-failure-hardening-design.md`

---

### Task 1: Add Recovery Configuration Constants

**Files:**
- Modify: `backend/core/gcp_hybrid_prime_router.py:181` (after existing emergency config block)

**Step 1: Add recovery constants after the existing emergency offload config**

After line 181 (`EMERGENCY_OFFLOAD_HYSTERESIS = ...`), add:

```python
# v266.2: Post-crisis model recovery — hybrid lazy + background warm-up
# After emergency offload, reload local model when memory stabilizes
RECOVERY_STABILITY_THRESHOLD_PERCENT = float(os.getenv("EMERGENCY_RECOVERY_THRESHOLD_PERCENT", "75.0"))
RECOVERY_STABILITY_DURATION_SEC = float(os.getenv("EMERGENCY_RECOVERY_STABILITY_SEC", "30.0"))
RECOVERY_MAX_ATTEMPTS = int(os.getenv("EMERGENCY_RECOVERY_MAX_ATTEMPTS", "3"))
```

**Step 2: Verify**

Run: `python3 -c "from backend.core.gcp_hybrid_prime_router import RECOVERY_STABILITY_THRESHOLD_PERCENT, RECOVERY_STABILITY_DURATION_SEC, RECOVERY_MAX_ATTEMPTS; print(f'{RECOVERY_STABILITY_THRESHOLD_PERCENT}% / {RECOVERY_STABILITY_DURATION_SEC}s / {RECOVERY_MAX_ATTEMPTS} attempts'); print('OK')"`

Expected: `75.0% / 30.0s / 3 attempts`, `OK`

**Step 3: Commit**

```bash
git add backend/core/gcp_hybrid_prime_router.py
git commit -m "$(cat <<'EOF'
feat(router): add post-crisis recovery configuration constants

Three new env-var-configurable constants for the hybrid lazy+background
model recovery after emergency offload resolves.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Add Recovery State Variables

**Files:**
- Modify: `backend/core/gcp_hybrid_prime_router.py:778` (after hysteresis state, before process tracking)

**Step 1: Add recovery state variables**

After line 778 (`self._emergency_offload_hysteresis_armed: bool = False`), add:

```python
        # v266.2: Post-crisis recovery state
        self._model_needs_recovery: bool = False
        self._recovery_stable_since: float = 0.0  # When RAM first dropped below recovery threshold
        self._recovery_attempts: int = 0
        self._recovery_in_progress: bool = False  # Prevents concurrent reload attempts
```

**Step 2: Verify**

Run: `python3 -c "import py_compile; py_compile.compile('backend/core/gcp_hybrid_prime_router.py', doraise=True); print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add backend/core/gcp_hybrid_prime_router.py
git commit -m "$(cat <<'EOF'
feat(router): add post-crisis recovery state variables

Track recovery need, stability window, attempt count, and reentrancy
guard for the hybrid lazy+background model recovery.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Add `reset_local_circuit_breaker()` to UnifiedModelServing

**Files:**
- Modify: `backend/intelligence/unified_model_serving.py` (near the `_unload_local_model` method, around line 2285)

**Step 1: Read the file to find exact insertion point**

Read `backend/intelligence/unified_model_serving.py` around line 2285 (end of `_unload_local_model`). Find the exact line after the method ends.

**Step 2: Add `reset_local_circuit_breaker()` method**

After `_unload_local_model()` ends, add:

```python
    def reset_local_circuit_breaker(self) -> None:
        """v266.2: Reset PRIME_LOCAL circuit breaker after verified model reload.

        Called by GCPHybridPrimeRouter when post-crisis recovery successfully
        reloads the local model. Directly closes the breaker so requests
        immediately route to LOCAL instead of waiting for HALF_OPEN timeout.
        """
        _provider = ModelProvider.PRIME_LOCAL.value
        cb = self._circuit_breaker._get_cb(_provider)
        if cb is not None:
            with self._circuit_breaker._lock:
                from backend.kernel.circuit_breaker import CircuitBreakerState as _CanonicalCBState
                cb._state = _CanonicalCBState.CLOSED
                cb._failure_count = 0
                cb._success_count = 0
            self._circuit_breaker._sync_state_from_canonical(_provider, cb)
        else:
            with self._circuit_breaker._lock:
                state = self._circuit_breaker._states.get(_provider)
                if state:
                    state.state = CircuitState.CLOSED
                    state.failure_count = 0
        self.logger.info(
            f"[v266.2] PRIME_LOCAL circuit breaker reset to CLOSED (post-crisis recovery)"
        )
```

Note: Check the actual import for `_CanonicalCBState`. The `_unload_local_model` code uses `CIRCUIT_BREAKER_FAILURE_THRESHOLD` which is a module-level constant. The canonical breaker state enum may be imported differently. Read the top of the file and the `CircuitBreaker._get_cb()` method to find the correct import. If the canonical breaker isn't available, the fallback path (setting `state.state = CircuitState.CLOSED`) handles it.

**Step 3: Verify**

Run: `python3 -c "from backend.intelligence.unified_model_serving import UnifiedModelServing; print(hasattr(UnifiedModelServing, 'reset_local_circuit_breaker')); print('OK')"`

Expected: `True`, `OK`

**Step 4: Commit**

```bash
git add backend/intelligence/unified_model_serving.py
git commit -m "$(cat <<'EOF'
feat(model): add reset_local_circuit_breaker() for post-crisis recovery

After emergency offload kills the model and recovery reloads it, this
method force-closes the PRIME_LOCAL circuit breaker so requests route
locally immediately instead of waiting for the HALF_OPEN timeout.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Implement `_check_model_recovery()` Method

**Files:**
- Modify: `backend/core/gcp_hybrid_prime_router.py` (new method, near `_check_offload_escalation`)

**Step 1: Read the file to find insertion point**

Read `backend/core/gcp_hybrid_prime_router.py` around the end of `_check_offload_escalation()` (line ~1618). The new method goes after it.

**Step 2: Add the recovery method**

After `_check_offload_escalation()` ends, add:

```python
    async def _check_model_recovery(self, current_used_percent: float) -> None:
        """v266.2: Check if conditions are right for post-crisis model reload.

        Called every monitoring poll when _model_needs_recovery is True.
        Implements hybrid lazy+background warm-up:
        - Waits for RAM to stabilize below RECOVERY_STABILITY_THRESHOLD_PERCENT
        - After RECOVERY_STABILITY_DURATION_SEC sustained, starts background reload
        - CLAUDE handles requests during reload (circuit breaker OPEN for LOCAL)
        - On success: resets circuit breaker, clears recovery flag
        - On failure: resets stability window, increments attempt counter
        """
        if self._recovery_in_progress:
            return  # Reload already running in background

        if self._recovery_attempts >= RECOVERY_MAX_ATTEMPTS:
            self.logger.info(
                f"[v266.2] Recovery exhausted ({RECOVERY_MAX_ATTEMPTS} attempts) "
                f"— staying on CLAUDE until next crisis cycle"
            )
            self._model_needs_recovery = False
            return

        # Check stability: RAM must be below threshold
        if current_used_percent < RECOVERY_STABILITY_THRESHOLD_PERCENT:
            if self._recovery_stable_since == 0.0:
                self._recovery_stable_since = time.time()
                self.logger.info(
                    f"[v266.2] RAM at {current_used_percent:.1f}% "
                    f"(< {RECOVERY_STABILITY_THRESHOLD_PERCENT}%) — "
                    f"stability window started"
                )
                return

            elapsed = time.time() - self._recovery_stable_since
            if elapsed < RECOVERY_STABILITY_DURATION_SEC:
                return  # Still waiting for stability

            # Stability window met — attempt background reload
            self.logger.warning(
                f"[v266.2] RAM stable at {current_used_percent:.1f}% "
                f"for {elapsed:.0f}s — starting background model reload "
                f"(attempt {self._recovery_attempts + 1}/{RECOVERY_MAX_ATTEMPTS})"
            )
            self._recovery_in_progress = True
            try:
                from backend.intelligence.unified_model_serving import get_model_serving
                model_serving = await get_model_serving()
                success = await model_serving.load_model()
                if success:
                    model_serving.reset_local_circuit_breaker()
                    self._model_needs_recovery = False
                    self._recovery_attempts = 0
                    self._recovery_stable_since = 0.0
                    self.logger.warning(
                        "[v266.2] Post-crisis recovery COMPLETE — "
                        "local model reloaded, PRIME_LOCAL circuit breaker reset"
                    )
                    await self._signal_memory_pressure_to_repos(
                        status="recovered", action=None,
                        used_percent=current_used_percent,
                    )
                else:
                    self._recovery_attempts += 1
                    self._recovery_stable_since = 0.0
                    self.logger.warning(
                        f"[v266.2] Model reload failed (attempt {self._recovery_attempts}/"
                        f"{RECOVERY_MAX_ATTEMPTS}) — waiting for next stability window"
                    )
            except Exception as e:
                self._recovery_attempts += 1
                self._recovery_stable_since = 0.0
                self.logger.error(f"[v266.2] Recovery error: {e}")
            finally:
                self._recovery_in_progress = False
        else:
            # RAM climbed back up — reset stability window
            if self._recovery_stable_since > 0.0:
                self.logger.debug(
                    f"[v266.2] RAM at {current_used_percent:.1f}% — "
                    f"stability window reset"
                )
                self._recovery_stable_since = 0.0
```

**Step 3: Verify**

Run: `python3 -c "import py_compile; py_compile.compile('backend/core/gcp_hybrid_prime_router.py', doraise=True); print('OK')"`

Expected: `OK`

**Step 4: Commit**

```bash
git add backend/core/gcp_hybrid_prime_router.py
git commit -m "$(cat <<'EOF'
feat(router): implement _check_model_recovery() for post-crisis reload

Background model reload when RAM stabilizes below 75% for 30s after
emergency offload. Hybrid lazy+background: CLAUDE handles requests
while local model reloads. Resets circuit breaker on success.
Max 3 attempts per crisis cycle.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Wire Recovery into the Monitoring Loop

**Files:**
- Modify: `backend/core/gcp_hybrid_prime_router.py:1149-1161` (hysteresis disarm block)
- Modify: `backend/core/gcp_hybrid_prime_router.py:1201` (after escalation check, before VM state machine)

**Step 1: Read lines 1149-1205 to see current state**

**Step 2: Set recovery flag when hysteresis disarms**

In the hysteresis disarm block (line 1153-1160), after the existing reset logic, add the recovery trigger:

Replace lines 1153-1160:
```python
                        if used_percent < hysteresis_threshold:
                            self._emergency_offload_hysteresis_armed = False
                            self._clean_unload_fired = False
                            self._clean_unload_verified = False
                            self.logger.info(
                                f"[v266.1] RAM dropped to {used_percent:.1f}% — "
                                f"hysteresis disarmed, escalation reset"
                            )
```

With:
```python
                        if used_percent < hysteresis_threshold:
                            self._emergency_offload_hysteresis_armed = False
                            self._clean_unload_fired = False
                            self._clean_unload_verified = False
                            # v266.2: Trigger post-crisis recovery if model was unloaded
                            if not self._model_needs_recovery:
                                self._model_needs_recovery = True
                                self._recovery_stable_since = 0.0
                                self._recovery_attempts = 0
                            self.logger.info(
                                f"[v266.2] RAM dropped to {used_percent:.1f}% — "
                                f"hysteresis disarmed, escalation reset, "
                                f"recovery armed"
                            )
```

**Step 3: Add recovery check in the monitoring loop**

After the offload escalation check (line 1201, `continue`), and before the VM state machine block (line 1203), add the recovery monitoring:

After line 1201 (`continue`), add a new block:

```python
                # v266.2: Post-crisis model recovery check
                if self._model_needs_recovery and not self._emergency_offload_active:
                    await self._check_model_recovery(used_percent)
```

IMPORTANT: This must go AFTER the `elif self._emergency_offload_active:` block ends (line 1201) and BEFORE the VM state machine block (line 1203). The indentation must match the `if not self._emergency_offload_active:` block — this is at the same level as both `if` and `elif` blocks, NOT inside either of them.

**Step 4: Verify**

Run: `python3 -c "import py_compile; py_compile.compile('backend/core/gcp_hybrid_prime_router.py', doraise=True); print('OK')"`

Expected: `OK`

**Step 5: Commit**

```bash
git add backend/core/gcp_hybrid_prime_router.py
git commit -m "$(cat <<'EOF'
feat(router): wire recovery into monitoring loop

When hysteresis disarms after emergency offload, set recovery flag.
Each monitoring poll checks _check_model_recovery() which waits for
RAM stability then reloads the model in background.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Add Monotonic Mode Constraint to `_reevaluate_startup_mode`

**Files:**
- Modify: `unified_supervisor.py:63803-63804` (the mode change condition in `_reevaluate_startup_mode`)

**Step 1: Read lines 63766-63823 to see current state**

**Step 2: Add monotonic constraint**

Replace lines 63803-63804:
```python
                # Only change if significantly different (at least 1 severity level)
                if abs(_ideal_sev - _current_sev) >= 1 and _ideal != _current:
```

With:
```python
                # Only change if significantly different (at least 1 severity level)
                # v266.2: During startup, mode can only degrade (never recover upward).
                # Upward recovery only after startup completes.
                _startup_complete = os.environ.get("JARVIS_STARTUP_COMPLETE", "") == "true"
                _is_degradation = _ideal_sev > _current_sev
                _can_change = _is_degradation or _startup_complete
                if abs(_ideal_sev - _current_sev) >= 1 and _ideal != _current and _can_change:
```

**Step 3: Verify**

Run: `python3 -c "import py_compile; py_compile.compile('unified_supervisor.py', doraise=True); print('OK')"`

Expected: `OK`

**Step 4: Commit**

```bash
git add unified_supervisor.py
git commit -m "$(cat <<'EOF'
feat(startup): enforce monotonic mode degradation during startup

Mode can only degrade (local_full → cloud_only) during startup.
Upward recovery blocked until JARVIS_STARTUP_COMPLETE=true.
Prevents oscillation when RAM fluctuates between phases.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Add `can_spawn_heavy_process()` Admission Gate

**Files:**
- Modify: `unified_supervisor.py` (near `_reevaluate_startup_mode`, around line 63823)

**Step 1: Read lines 63820-63830 to find insertion point**

**Step 2: Add admission gate function**

After `_reevaluate_startup_mode()` ends, add:

```python
        async def can_spawn_heavy_process(estimated_mb: int, label: str) -> bool:
            """v266.2: Admission gate before spawning heavy subprocesses.

            Checks if spawning a process that needs ~estimated_mb would push
            the system into CRITICAL/EMERGENCY memory tier.

            Returns True if safe to spawn, False if memory too tight.
            """
            try:
                import psutil
                _mem = psutil.virtual_memory()
                _avail_mb = _mem.available / (1024 ** 2)
                # Reserve 500MB safety margin on top of estimated need
                _needed_mb = estimated_mb + 500
                if _avail_mb < _needed_mb:
                    self.logger.warning(
                        f"[v266.2] Admission gate BLOCKED {label}: "
                        f"needs ~{_needed_mb}MB but only {_avail_mb:.0f}MB available"
                    )
                    return False

                # Also check MemoryQuantizer tier if available
                try:
                    from backend.core.memory_quantizer import get_memory_quantizer, MemoryTier
                    _mq = await asyncio.wait_for(get_memory_quantizer(), timeout=2.0)
                    if _mq:
                        _metrics = _mq.get_current_metrics()
                        if _metrics.tier in (MemoryTier.CRITICAL, MemoryTier.EMERGENCY):
                            self.logger.warning(
                                f"[v266.2] Admission gate BLOCKED {label}: "
                                f"memory tier is {_metrics.tier.value}"
                            )
                            return False
                except Exception:
                    pass  # MemoryQuantizer may not be ready yet

                self.logger.debug(
                    f"[v266.2] Admission gate OK for {label}: "
                    f"{_avail_mb:.0f}MB available, needs ~{_needed_mb}MB"
                )
                return True
            except Exception as e:
                self.logger.debug(f"[v266.2] Admission gate error for {label}: {e}")
                return True  # Fail-open: don't block on gate errors

```

**Step 3: Verify**

Run: `python3 -c "import py_compile; py_compile.compile('unified_supervisor.py', doraise=True); print('OK')"`

Expected: `OK`

**Step 4: Commit**

```bash
git add unified_supervisor.py
git commit -m "$(cat <<'EOF'
feat(startup): add can_spawn_heavy_process() admission gate

Lightweight pre-spawn check: verifies available RAM can handle the
estimated subprocess cost + 500MB safety margin. Also checks
MemoryQuantizer tier. Fail-open on gate errors.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Add Phase 2 and Phase 3 Memory Gates

**Files:**
- Modify: `unified_supervisor.py:67146` (top of `_phase_resources`, after timeout retrieval)
- Modify: `unified_supervisor.py:68138` (top of `_phase_backend`, after state update)

**Step 1: Read lines 67144-67160 and 68137-68160 to find insertion points**

**Step 2: Add Phase 2 gate**

After the `resource_timeout` line in `_phase_resources()` (line ~67146), add:

```python
            # v266.2: Memory gate — re-evaluate mode at phase boundary
            await _reevaluate_startup_mode("phase_2_resources")
            _mode = os.environ.get("JARVIS_STARTUP_MEMORY_MODE", "local_full")
            if _mode in ("cloud_only", "minimal"):
                self.logger.warning(
                    f"[v266.2] Phase 2: mode={_mode} — deferring heavy local resources"
                )
                os.environ["JARVIS_CAPABILITY_DOCKER"] = "deferred"
                os.environ["JARVIS_CAPABILITY_LOCAL_STORAGE"] = "deferred"
```

**Step 3: Add Phase 3 gate**

After the state update line in `_phase_backend()` (line ~68138, after `self._state = KernelState.STARTING_BACKEND`), add:

```python
            # v266.2: Memory gate — re-evaluate mode at phase boundary
            await _reevaluate_startup_mode("phase_3_backend")
            _mode = os.environ.get("JARVIS_STARTUP_MEMORY_MODE", "local_full")
            if _mode in ("cloud_first", "cloud_only", "minimal"):
                import psutil as _ps3
                _avail_gb = _ps3.virtual_memory().available / (1024**3)
                if _avail_gb < 2.0:
                    self.logger.warning(
                        f"[v266.2] Phase 3: mode={_mode}, available={_avail_gb:.1f}GB "
                        f"— setting JARVIS_BACKEND_MINIMAL=true"
                    )
                    os.environ["JARVIS_BACKEND_MINIMAL"] = "true"
```

**Step 4: Verify**

Run: `python3 -c "import py_compile; py_compile.compile('unified_supervisor.py', doraise=True); print('OK')"`

Expected: `OK`

**Step 5: Commit**

```bash
git add unified_supervisor.py
git commit -m "$(cat <<'EOF'
feat(startup): add Phase 2/3 memory gates with capability flags

Phase 2: re-evaluates mode, defers Docker/local-storage on cloud_only.
Phase 3: re-evaluates mode, sets JARVIS_BACKEND_MINIMAL=true when
cloud_first + available RAM < 2GB. Downstream components branch on
capability flags rather than assuming resource availability.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Add Subprocess ENOMEM Handler

**Files:**
- Modify: `unified_supervisor.py:68437` (the `create_subprocess_exec` call in `_start_backend_subprocess`)

**Step 1: Read lines 68430-68450 to see current subprocess creation**

**Step 2: Wrap with ENOMEM handling**

Replace the subprocess creation block. Find the exact `self._backend_process = await asyncio.create_subprocess_exec(` line and its surrounding try block.

Before the `create_subprocess_exec` call, add the admission gate check:

```python
                # v266.2: Pre-spawn admission gate
                if not await can_spawn_heavy_process(500, "backend_subprocess"):
                    _mode = os.environ.get("JARVIS_STARTUP_MEMORY_MODE", "local_full")
                    self.logger.warning(
                        f"[v266.2] Backend subprocess blocked by admission gate "
                        f"(mode={_mode}) — escalating mode"
                    )
                    await _reevaluate_startup_mode("backend_subprocess_blocked")
                    os.environ["JARVIS_BACKEND_MINIMAL"] = "true"
```

Then wrap the `create_subprocess_exec` call with ENOMEM handling. After the existing try block that contains the call, add an `except OSError` clause BEFORE the generic `except Exception`:

```python
            except OSError as e:
                import errno
                if e.errno == errno.ENOMEM:
                    self.logger.error(
                        f"[v266.2] Backend subprocess OOM (Cannot allocate memory). "
                        f"Escalating startup mode."
                    )
                    await _reevaluate_startup_mode("subprocess_enomem")
                    os.environ["JARVIS_BACKEND_MINIMAL"] = "true"
                    return False
                raise  # Re-raise non-ENOMEM OSErrors
            except MemoryError:
                self.logger.error(
                    "[v266.2] Backend subprocess MemoryError. Escalating startup mode."
                )
                await _reevaluate_startup_mode("subprocess_memory_error")
                return False
```

IMPORTANT: Read the actual try/except structure around the subprocess exec call to find the correct insertion point. The `except OSError` must be BEFORE `except Exception` in the handler chain.

**Step 3: Verify**

Run: `python3 -c "import py_compile; py_compile.compile('unified_supervisor.py', doraise=True); print('OK')"`

Expected: `OK`

**Step 4: Commit**

```bash
git add unified_supervisor.py
git commit -m "$(cat <<'EOF'
feat(startup): add subprocess ENOMEM handler and admission gate

Pre-spawn admission gate checks available RAM before backend subprocess.
Catches OSError with errno.ENOMEM and MemoryError, escalates startup
mode, and sets JARVIS_BACKEND_MINIMAL=true. Uses errno detection,
not string matching.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Integration Verification

**Files:** None (verification only)

**Step 1: Verify all imports**

```bash
python3 -c "
from backend.core.gcp_hybrid_prime_router import (
    RECOVERY_STABILITY_THRESHOLD_PERCENT,
    RECOVERY_STABILITY_DURATION_SEC,
    RECOVERY_MAX_ATTEMPTS,
)
print(f'Recovery: {RECOVERY_STABILITY_THRESHOLD_PERCENT}% / {RECOVERY_STABILITY_DURATION_SEC}s / {RECOVERY_MAX_ATTEMPTS} attempts')
from backend.intelligence.unified_model_serving import UnifiedModelServing
print(f'reset_local_circuit_breaker exists: {hasattr(UnifiedModelServing, \"reset_local_circuit_breaker\")}')
print('All imports OK')
"
```

Expected: `Recovery: 75.0% / 30.0s / 3 attempts`, `reset_local_circuit_breaker exists: True`, `All imports OK`

**Step 2: Verify syntax of all modified files**

```bash
python3 -c "
import py_compile
py_compile.compile('backend/core/gcp_hybrid_prime_router.py', doraise=True)
py_compile.compile('backend/intelligence/unified_model_serving.py', doraise=True)
py_compile.compile('unified_supervisor.py', doraise=True)
print('All 3 files syntax OK')
"
```

Expected: `All 3 files syntax OK`

**Step 3: Verify monotonic mode constraint**

```bash
python3 -c "
import os
os.environ['JARVIS_STARTUP_COMPLETE'] = ''
# During startup, should not allow recovery upward
print('Monotonic constraint env var test: JARVIS_STARTUP_COMPLETE =', repr(os.environ.get('JARVIS_STARTUP_COMPLETE', '')))
print('Expected: empty string (blocks upward recovery during startup)')
os.environ['JARVIS_STARTUP_COMPLETE'] = 'true'
print('After startup: JARVIS_STARTUP_COMPLETE =', repr(os.environ.get('JARVIS_STARTUP_COMPLETE')))
print('Expected: true (allows upward recovery)')
print('OK')
"
```

Expected: Prints both states, `OK`

**Step 4: Verify no old recovery references conflict**

```bash
python3 -c "
# Verify no naming conflicts with existing code
from backend.core.gcp_hybrid_prime_router import GCPHybridPrimeRouter
print('GCPHybridPrimeRouter importable: True')
print('OK')
"
```

Expected: `GCPHybridPrimeRouter importable: True`, `OK`
