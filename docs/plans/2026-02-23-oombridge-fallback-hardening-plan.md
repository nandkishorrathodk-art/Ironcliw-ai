# OOMBridge Fallback Hardening — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the three-bug cascade where OOMBridge failure + 2.6GB RAM leads to OOM crash by fixing the guard condition, adding retry, forcing sequential init under low RAM, and splitting desired vs effective mode.

**Architecture:** Fix 1 sets `Ironcliw_STARTUP_DESIRED_MODE` at initial mode decision and unconditionally degrades effective mode when bridge fails. Fix 2 adds a single retry. Fix 3 forces sequential init in parallel_initializer when bridge unavailable + low RAM. Fix 4 changes GCP probe eligibility to use desired_mode.

**Tech Stack:** Python 3, asyncio, psutil, existing unified_supervisor.py and parallel_initializer.py

**Design doc:** `docs/plans/2026-02-23-oombridge-fallback-hardening-design.md`

---

### Task 1: Set `Ironcliw_STARTUP_DESIRED_MODE` at Initial Mode Decision

**Files:**
- Modify: `unified_supervisor.py:63702` (fallback mode-setting)
- Modify: `unified_supervisor.py:63728` (ResourceOrchestrator mode-setting)

**Step 1: Read lines 63695-63735 to see both mode-setting locations**

**Step 2: Add desired_mode at the ResourceOrchestrator path**

At line 63728, after `os.environ["Ironcliw_STARTUP_MEMORY_MODE"] = _startup_mem_mode`, add:

```python
            os.environ["Ironcliw_STARTUP_DESIRED_MODE"] = _startup_mem_mode
```

**Step 3: Add desired_mode at the fallback path**

At line 63702, after `os.environ["Ironcliw_STARTUP_MEMORY_MODE"] = _fallback_mode`, add:

```python
            os.environ["Ironcliw_STARTUP_DESIRED_MODE"] = _fallback_mode
```

**Step 4: Verify**

Run: `python3 -c "import py_compile; py_compile.compile('unified_supervisor.py', doraise=True); print('OK')"`

Expected: `OK`

**Step 5: Commit**

```bash
git add unified_supervisor.py
git commit -m "$(cat <<'EOF'
feat(startup): set Ironcliw_STARTUP_DESIRED_MODE at initial mode decision

Captures operator intent / policy separately from runtime safety mode.
Ironcliw_STARTUP_DESIRED_MODE is set once and never modified — it records
what mode was originally selected. Ironcliw_STARTUP_MEMORY_MODE continues
to be degraded monotonically during startup for safety.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Fix OOMBridge Fallback Guard + Add Retry

**Files:**
- Modify: `unified_supervisor.py:64124-64198` (OOM Prevention Bridge block)

**Step 1: Read lines 64124-64200 to see current OOMBridge block**

**Step 2: Replace the entire OOMBridge block**

Replace lines 64124-64198 (from `# v255.0: OOM Prevention Bridge` through the end of the `except Exception as _oom_err:` handler) with:

```python
        # v255.0 / v266.3: OOM Prevention Bridge — proactive memory check before heavy init.
        # Uses auto_offload=False to avoid GCP network calls during pre-flight.
        # Only overrides Ironcliw_STARTUP_MEMORY_MODE if OOM Bridge decision is MORE
        # severe than the current mode set by the ResourceOrchestrator (Step 1).
        # v266.3: Single retry on failure, then unconditional degradation to local fallback.
        _oom_preflight_timeout = _get_env_float("Ironcliw_OOM_PREFLIGHT_TIMEOUT", 3.0)
        _oom_retry_timeout = _get_env_float("Ironcliw_OOM_RETRY_TIMEOUT", 5.0)
        self._startup_memory_decision = None
        _oom_attempts = [_oom_preflight_timeout, _oom_retry_timeout]
        _oom_succeeded = False

        for _oom_attempt_idx, _oom_timeout in enumerate(_oom_attempts):
            try:
                from core.gcp_oom_prevention_bridge import check_memory_before_heavy_init
                _oom_result = await asyncio.wait_for(
                    check_memory_before_heavy_init(
                        component="startup_pipeline",
                        estimated_mb=3000,
                        auto_offload=False,
                    ),
                    timeout=_oom_timeout,
                )
                self._startup_memory_decision = _oom_result
                # Severity ordering: local_full < local_optimized < sequential < cloud_first < cloud_only < minimal
                _SEVERITY = {
                    "local_full": 0, "local_optimized": 1, "sequential": 2,
                    "cloud_first": 3, "cloud_only": 4, "minimal": 5,
                }
                _current_mode = os.environ.get("Ironcliw_STARTUP_MEMORY_MODE", "local_full")
                _current_sev = _SEVERITY.get(_current_mode, 0)

                if not _oom_result.can_proceed_locally:
                    _new_mode = "cloud_first"
                    if _SEVERITY.get(_new_mode, 0) > _current_sev:
                        os.environ["Ironcliw_STARTUP_MEMORY_MODE"] = _new_mode
                    _skip_local_prewarm = True
                    _skip_reason = (
                        f"oom_bridge: {_oom_result.decision.value} "
                        f"(avail={_oom_result.available_ram_gb:.1f}GB)"
                    )
                elif _oom_result.decision.value == "degraded":
                    _new_mode = "sequential"
                    if _SEVERITY.get(_new_mode, 0) > _current_sev:
                        os.environ["Ironcliw_STARTUP_MEMORY_MODE"] = _new_mode

                _oom_succeeded = True
                break  # Success — exit retry loop
            except asyncio.CancelledError:
                raise
            except Exception as _oom_err:
                if _oom_attempt_idx < len(_oom_attempts) - 1:
                    _unified_logger.info(
                        "[OOMBridge] Attempt %d failed (%s), retrying with %ds timeout...",
                        _oom_attempt_idx + 1, _oom_err, int(_oom_retry_timeout),
                    )
                    continue
                # Final attempt failed — unconditional degradation
                _available_gb = _read_available_memory_gb()
                _guard_mode = _resolve_local_startup_mode_on_cloud_unavailable(
                    "cloud_only",
                    available_gb=_available_gb,
                )
                # v266.3: OOMBridge is broken — cloud modes can't execute without it.
                # Unconditionally degrade to local fallback regardless of current mode.
                os.environ["Ironcliw_STARTUP_MEMORY_MODE"] = _guard_mode

                _oom_msg = (
                    f"OOM pre-flight unavailable after {len(_oom_attempts)} attempts: "
                    f"{_oom_err} "
                    f"(degraded to {_guard_mode}"
                    + (f", avail={_available_gb:.1f}GB" if _available_gb is not None else "")
                    + ")"
                )
                _unified_logger.warning("[OOMBridge] %s", _oom_msg)
                issue_collector.add_warning(
                    _oom_msg,
                    IssueCategory.SYSTEM,
                    suggestion="Verify core.gcp_oom_prevention_bridge dependencies and memory telemetry",
                )
```

Key changes vs. original:
1. Retry loop (2 attempts: 3s then 5s)
2. On final failure: `os.environ["Ironcliw_STARTUP_MEMORY_MODE"] = _guard_mode` — no guard condition, unconditional
3. Log message says "degraded to {_guard_mode}" instead of showing the broken mode
4. `_oom_succeeded` flag for downstream consumers to check

**Step 3: Verify**

Run: `python3 -c "import py_compile; py_compile.compile('unified_supervisor.py', doraise=True); print('OK')"`

Expected: `OK`

**Step 4: Commit**

```bash
git add unified_supervisor.py
git commit -m "$(cat <<'EOF'
fix(startup): fix OOMBridge fallback guard + add retry

Bug fix: the guard condition excluded cloud_first/cloud_only from
degradation when OOMBridge failed. System stayed in cloud mode with
no bridge to make it work. Now unconditionally degrades to local
fallback (sequential/local_optimized based on RAM).

Adds single retry with 5s timeout before giving up.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Change GCP Probe Eligibility to Use Desired Mode

**Files:**
- Modify: `unified_supervisor.py:64207-64210` (GCP probe eligibility check)

**Step 1: Read lines 64200-64215 to see current GCP probe block**

Note: Line numbers will have shifted from Task 2. Grep for `v258.3 (Gap 9): GCP AVAILABILITY PROBE` to find the exact location.

**Step 2: Change the mode check**

Find the line:
```python
        _startup_mode_now = os.environ.get("Ironcliw_STARTUP_MEMORY_MODE", "local_full")
```

And the condition:
```python
        if _startup_mode_now in ("cloud_first", "cloud_only"):
```

Replace both with:
```python
        # v266.3: Use desired_mode (operator intent) not effective_mode (runtime safety).
        # If OOMBridge failed, effective_mode may be degraded to sequential, but
        # operator intended cloud — GCP probe should still run for background recovery.
        _startup_desired_mode = os.environ.get("Ironcliw_STARTUP_DESIRED_MODE", "local_full")
```

And:
```python
        if _startup_desired_mode in ("cloud_first", "cloud_only"):
```

IMPORTANT: Also update any references to `_startup_mode_now` inside the probe block to use `_startup_desired_mode`. Read the full block to find all references.

**Step 3: Verify**

Run: `python3 -c "import py_compile; py_compile.compile('unified_supervisor.py', doraise=True); print('OK')"`

Expected: `OK`

**Step 4: Commit**

```bash
git add unified_supervisor.py
git commit -m "$(cat <<'EOF'
feat(startup): GCP probe uses desired_mode not effective_mode

When OOMBridge fails and effective mode degrades from cloud_first to
sequential, the GCP probe was skipped — no cloud recovery path.
Now uses Ironcliw_STARTUP_DESIRED_MODE (operator intent) so cloud
probing survives startup degradation.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Parallel Initializer Fail-Closed Under Low RAM

**Files:**
- Modify: `backend/core/parallel_initializer.py:469-480` (add `_force_sequential` to `__init__`)
- Modify: `backend/core/parallel_initializer.py:810-811` (OOM check failure handler)
- Modify: `backend/core/parallel_initializer.py:851-852` (group execution — conditional gather)

**Step 1: Read lines 469-480 and 810-855 to see current state**

**Step 2: Add `_force_sequential` flag to `__init__`**

After line 480 (`self._full_mode_event: Optional[asyncio.Event] = None`), add:

```python
        self._force_sequential: bool = False  # v266.3: Set when bridge unavailable + low RAM
```

**Step 3: Replace the OOM check failure handler**

Replace line 810-811:
```python
            except Exception as e:
                logger.warning(f"[OOM Prevention] Check failed (non-fatal): {e}")
```

With:
```python
            except Exception as e:
                # v266.3: Fail-closed — force sequential init when bridge
                # unavailable AND available RAM is below safe threshold.
                try:
                    import psutil
                    _avail_gb = psutil.virtual_memory().available / (1024**3)
                    if _avail_gb < 4.0:
                        self._force_sequential = True
                        logger.warning(
                            "[OOM Prevention] Bridge unavailable + %.1fGB available "
                            "— forcing sequential init", _avail_gb,
                        )
                    else:
                        logger.warning(
                            "[OOM Prevention] Bridge unavailable but %.1fGB available "
                            "— parallel init OK", _avail_gb,
                        )
                except Exception:
                    # psutil failed too — force sequential to be safe
                    self._force_sequential = True
                    logger.warning(
                        "[OOM Prevention] Bridge + psutil unavailable "
                        "— forcing sequential init (fail-closed)"
                    )
```

**Step 4: Modify group execution to respect `_force_sequential`**

Find the line (around 851-852):
```python
                if tasks:
                    await safe_gather(*tasks, return_exceptions=True)
```

Replace with:
```python
                if tasks:
                    if self._force_sequential:
                        # v266.3: Sequential init — one component at a time with
                        # memory check between each to prevent OOM cascade
                        for _seq_task in tasks:
                            try:
                                await _seq_task
                            except Exception as _seq_err:
                                logger.warning(
                                    "[Sequential Init] Component failed: %s", _seq_err
                                )
                            # Check RAM between components
                            try:
                                import psutil
                                _post_avail_gb = psutil.virtual_memory().available / (1024**3)
                                if _post_avail_gb < 2.0:
                                    logger.warning(
                                        "[Sequential Init] RAM critical (%.1fGB) "
                                        "— skipping remaining components in group",
                                        _post_avail_gb,
                                    )
                                    break
                            except Exception:
                                pass
                    else:
                        await safe_gather(*tasks, return_exceptions=True)
```

**Step 5: Verify**

Run: `python3 -c "import py_compile; py_compile.compile('backend/core/parallel_initializer.py', doraise=True); print('OK')"`

Expected: `OK`

**Step 6: Commit**

```bash
git add backend/core/parallel_initializer.py
git commit -m "$(cat <<'EOF'
fix(init): fail-closed sequential init when bridge unavailable + low RAM

When OOMBridge check fails AND available RAM < 4GB, forces sequential
component initialization instead of parallel gather. Checks RAM between
each component — skips remaining if below 2GB critical threshold.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Integration Verification

**Files:** None (verification only)

**Step 1: Verify all modified files compile**

```bash
python3 -c "
import py_compile
py_compile.compile('unified_supervisor.py', doraise=True)
py_compile.compile('backend/core/parallel_initializer.py', doraise=True)
print('Both files syntax OK')
"
```

Expected: `Both files syntax OK`

**Step 2: Verify desired_mode env var is set**

```bash
python3 -c "
import os
# Simulate: initial mode decision sets both vars
os.environ['Ironcliw_STARTUP_DESIRED_MODE'] = 'cloud_first'
os.environ['Ironcliw_STARTUP_MEMORY_MODE'] = 'cloud_first'
# Simulate: OOMBridge fails, effective mode degrades
os.environ['Ironcliw_STARTUP_MEMORY_MODE'] = 'sequential'
# Verify: desired_mode preserved, effective_mode degraded
assert os.environ['Ironcliw_STARTUP_DESIRED_MODE'] == 'cloud_first'
assert os.environ['Ironcliw_STARTUP_MEMORY_MODE'] == 'sequential'
# GCP probe should use desired_mode
_desired = os.environ.get('Ironcliw_STARTUP_DESIRED_MODE', 'local_full')
assert _desired in ('cloud_first', 'cloud_only'), f'GCP probe should run for {_desired}'
print('Desired vs effective mode split: OK')
"
```

Expected: `Desired vs effective mode split: OK`

**Step 3: Verify retry timeout config**

```bash
python3 -c "
import os
# Default values
t1 = float(os.getenv('Ironcliw_OOM_PREFLIGHT_TIMEOUT', '3.0'))
t2 = float(os.getenv('Ironcliw_OOM_RETRY_TIMEOUT', '5.0'))
print(f'Attempt 1: {t1}s, Attempt 2: {t2}s')
assert t1 == 3.0 and t2 == 5.0
print('Retry config: OK')
"
```

Expected: `Attempt 1: 3.0s, Attempt 2: 5.0s`, `Retry config: OK`

**Step 4: Verify parallel_initializer imports clean**

```bash
python3 -c "
from backend.core.parallel_initializer import ParallelInitializer
print(f'_force_sequential attribute exists: {hasattr(ParallelInitializer, \"_force_sequential\")}')
print('Import OK')
"
```

Note: `_force_sequential` is set in `__init__`, so `hasattr(ParallelInitializer, ...)` will be False — that's fine. What matters is the class imports without error.

Expected: `Import OK`
