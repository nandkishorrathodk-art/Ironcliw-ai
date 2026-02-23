# GCP Pressure-Driven Lifecycle — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace eager GCP VM creation with demand-driven provisioning based on real memory pressure, stop VMs on session shutdown, and fix cost gating — cutting GCP costs from ~$92/mo to ~$20-30/mo.

**Architecture:** Enhance the existing `GCPHybridPrimeRouter` with a formal VM lifecycle state machine and hysteresis bands. Centralize the scattered InvincibleGuard logic. Extend `terminate_vm()` to support STOP actions. Remove eager pre-warm code from the supervisor. Fix MemoryQuantizer event loop blocking and cost tracker fail-open.

**Tech Stack:** Python 3, asyncio, psutil, google-cloud-compute API, existing MemoryQuantizer/GCPHybridPrimeRouter/GCPVMManager

**Design doc:** `docs/plans/2026-02-22-gcp-pressure-driven-lifecycle-design.md`

---

## Task 1: Centralize InvincibleGuard — Add VMAction Enum and `check_vm_protection()`

**Files:**
- Modify: `backend/core/gcp_vm_manager.py:231-241` (VMState enum area), `:6019-6059` (terminate_vm guard), `:6241-6255` (_force_delete_vm guard), `:6799-6815` (monitoring loop guard)

**Step 1: Add VMAction enum and STOPPED state near existing VMState enum**

At `backend/core/gcp_vm_manager.py`, after the `VMState` enum (line ~241), add:

```python
class VMAction(Enum):
    """Action types for VM lifecycle operations."""
    STOP = "stop"          # Halt VM, preserve disk/IP
    DELETE = "delete"       # Remove VM and ephemeral resources
    TERMINATE = "terminate" # Force-terminate (monitoring loop)


# Add STOPPED to VMState enum:
# Insert after STOPPING = "stopping" (line 236):
#   STOPPED = "stopped"
```

Also add `STOPPED = "stopped"` as a new value in the existing `VMState` enum at line ~236.

**Step 2: Add `check_vm_protection()` method on GCPVMManager**

Add this method near the `terminate_vm()` method (before line ~6019):

```python
def check_vm_protection(self, vm_name: str, action: VMAction, reason: str = "") -> Tuple[bool, str]:
    """
    Centralized VM protection check. Single source of truth for all guard logic.

    Returns:
        (is_protected, explanation) — True means the action is BLOCKED.
    """
    # Detection: is this an invincible VM?
    static_name = getattr(self.config, 'static_instance_name', 'jarvis-prime-node')
    is_invincible = vm_name.startswith(static_name)

    if not is_invincible and vm_name in self.managed_vms:
        vm_meta = self.managed_vms[vm_name].metadata or {}
        is_invincible = (
            vm_meta.get("vm_class") == "invincible"
            or vm_meta.get("labels", {}).get("vm-class") == "invincible"
        )

    if not is_invincible:
        return False, ""  # Not protected, allow any action

    # Invincible VM — check action type
    if action in (VMAction.DELETE, VMAction.TERMINATE):
        msg = (
            f"[InvincibleGuard] BLOCKED {action.value} of persistent VM '{vm_name}' "
            f"(reason: {reason}). Use GCP Console or gcloud CLI for manual override."
        )
        logger.warning(msg)
        return True, msg

    if action == VMAction.STOP:
        # STOP is allowed for session shutdown when session lifecycle is enabled
        session_lifecycle = _get_env_bool("JARVIS_GCP_SESSION_LIFECYCLE", True)
        is_session_reason = reason in ("session_shutdown", "supervisor_cleanup", "emergency_shutdown")

        if session_lifecycle and is_session_reason:
            logger.info(
                f"[InvincibleGuard] Allowing STOP of '{vm_name}' "
                f"(session lifecycle, reason: {reason})"
            )
            return False, ""  # Allow STOP

        msg = (
            f"[InvincibleGuard] BLOCKED STOP of persistent VM '{vm_name}' "
            f"(reason: {reason}, session_lifecycle={session_lifecycle})"
        )
        logger.warning(msg)
        return True, msg

    return True, f"[InvincibleGuard] Unknown action {action} for '{vm_name}'"
```

**Step 3: Replace scattered guard in `terminate_vm()` (lines 6040-6059)**

Replace the inline guard block with:

```python
# v153.0 → v266.0: Centralized protection check
is_protected, guard_msg = self.check_vm_protection(vm_name, action, reason)
if is_protected:
    return False
```

**Step 4: Replace scattered guard in `_force_delete_vm()` (lines 6249-6255)**

Replace the name-only check with:

```python
# v153.0 → v266.0: Centralized protection check (was name-only, now full check)
is_protected, guard_msg = self.check_vm_protection(vm_name, VMAction.DELETE, reason)
if is_protected:
    return False
```

**Step 5: Replace scattered guard in monitoring loop (lines 6799-6815)**

Replace the inline `_is_invincible` check with:

```python
# v153.0 → v266.0: Centralized protection check
_is_protected, _ = self.check_vm_protection(vm_name, VMAction.TERMINATE, "monitoring_loop")
if _is_protected:
    logger.debug(f"[InvincibleVM] Skipping cost-cutting checks for '{vm_name}'")
    continue
```

**Step 6: Verify no regressions**

Run: `python3 -c "from backend.core.gcp_vm_manager import VMAction, VMState; print('VMAction:', list(VMAction)); print('VMState:', list(VMState))"`
Expected: Both enums print with new values (STOP/DELETE/TERMINATE and STOPPED).

**Step 7: Commit**

```bash
git add backend/core/gcp_vm_manager.py
git commit -m "refactor(gcp): centralize InvincibleGuard into check_vm_protection()

Extracts scattered guard logic from 3 locations (terminate_vm, _force_delete_vm,
monitoring loop) into a single method. Adds VMAction enum (STOP/DELETE/TERMINATE)
and VMState.STOPPED. The guard is now action-type-aware: DELETE/TERMINATE always
blocked for invincible VMs, STOP allowed for session_shutdown when session
lifecycle is enabled."
```

---

## Task 2: Extend `terminate_vm()` with Action Parameter

**Files:**
- Modify: `backend/core/gcp_vm_manager.py:6019` (terminate_vm signature and body)

**Step 1: Update `terminate_vm()` signature**

Change line 6019 from:
```python
async def terminate_vm(self, vm_name: str, reason: str = "Manual termination") -> bool:
```
To:
```python
async def terminate_vm(self, vm_name: str, reason: str = "Manual termination",
                       action: VMAction = VMAction.DELETE) -> bool:
```

**Step 2: Add STOP action branch after the guard check**

After the centralized guard check (from Task 1), before the existing DELETE logic, add a branch:

```python
if action == VMAction.STOP:
    return await self._stop_vm_instance(vm_name, reason)
```

**Step 3: Add `_stop_vm_instance()` method**

Add near `terminate_vm()`:

```python
async def _stop_vm_instance(self, vm_name: str, reason: str) -> bool:
    """Stop a VM instance (preserve disk/IP). Used for session shutdown."""
    try:
        logger.info(f"[VMLifecycle] Stopping VM '{vm_name}' (reason: {reason})")

        # GCP API: stop (not delete)
        zone = self.config.zone
        project = self.config.project_id

        operation = self._instances_client.stop(
            project=project,
            zone=zone,
            instance=vm_name,
        )

        # Wait for operation with timeout
        await asyncio.wait_for(
            self._wait_for_operation(operation, project, zone),
            timeout=60.0
        )

        # Update managed_vms state (don't remove — VM still exists)
        if vm_name in self.managed_vms:
            self.managed_vms[vm_name].state = VMState.STOPPED
            self.managed_vms[vm_name].last_activity_time = time.time()

        # End cost tracking session for this VM
        if self._cost_tracker:
            try:
                await self._cost_tracker.record_vm_deleted(
                    vm_name=vm_name,
                    reason=f"stopped: {reason}",
                    runtime_seconds=self._get_vm_runtime(vm_name),
                )
            except Exception as cost_err:
                logger.debug(f"Cost tracker update on stop: {cost_err}")

        # Dashboard event
        self._emit_event("vm_stopped", {
            "vm_name": vm_name,
            "reason": reason,
            "action": "stop",
        })

        logger.info(f"[VMLifecycle] VM '{vm_name}' stopped successfully")
        return True

    except asyncio.TimeoutError:
        logger.warning(f"[VMLifecycle] Timeout stopping VM '{vm_name}' (60s)")
        return False
    except Exception as e:
        logger.error(f"[VMLifecycle] Failed to stop VM '{vm_name}': {e}")
        return False
```

**Step 4: Update monitoring loop to skip STOPPED VMs**

In the monitoring loop (around line 6581+), after the invincible guard check, add:

```python
# Skip VMs that are already STOPPED
if vm_name in self.managed_vms and self.managed_vms[vm_name].state == VMState.STOPPED:
    continue
```

**Step 5: Verify import chain works**

Run: `python3 -c "from backend.core.gcp_vm_manager import GCPVMManager; print('OK')"`
Expected: `OK` (no import errors)

**Step 6: Commit**

```bash
git add backend/core/gcp_vm_manager.py
git commit -m "feat(gcp): extend terminate_vm() with STOP action for session lifecycle

terminate_vm() now accepts action=VMAction.STOP which calls instances_client.stop()
instead of delete. VM state transitions to VMState.STOPPED in managed_vms dict
(not removed). Cost tracking session ends. Monitoring loop skips STOPPED VMs."
```

---

## Task 3: Fix MemoryQuantizer Event Loop Blocking

**Files:**
- Modify: `backend/core/memory_quantizer.py:592-650` (_get_memory_pressure), `:443` (get_current_metrics)

**Step 1: Convert `_get_memory_pressure()` from sync to async**

Change method at line 592 from:
```python
def _get_memory_pressure(self) -> MemoryPressure:
```
To:
```python
async def _get_memory_pressure(self) -> MemoryPressure:
```

Replace the `subprocess.run` call (around line 607) with:

```python
try:
    proc = await asyncio.create_subprocess_exec(
        'memory_pressure',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)
    output = stdout_bytes.decode('utf-8', errors='replace').lower()
except asyncio.TimeoutError:
    logger.debug("memory_pressure command timed out (2s)")
    return self._fallback_pressure_from_percent(pressure_percent)
except FileNotFoundError:
    logger.debug("memory_pressure command not found")
    return self._fallback_pressure_from_percent(pressure_percent)
except Exception as e:
    logger.debug(f"memory_pressure failed: {e}")
    return self._fallback_pressure_from_percent(pressure_percent)
```

Extract the existing fallback logic (the `macos_pressure_percent` threshold checks from the original `except` blocks) into a helper:

```python
def _fallback_pressure_from_percent(self, pressure_percent: float) -> MemoryPressure:
    """Fallback when memory_pressure command is unavailable."""
    if pressure_percent >= 90:
        return MemoryPressure.CRITICAL
    elif pressure_percent >= 85:
        return MemoryPressure.WARN
    return MemoryPressure.NORMAL
```

**Step 2: Add async variant of `get_current_metrics()`**

The existing `get_current_metrics()` (line 443) is sync and called from both sync and async contexts. Add an async counterpart that the `_monitor_loop()` and callbacks use:

```python
async def get_current_metrics_async(self) -> MemoryMetrics:
    """Async version — uses non-blocking memory_pressure call."""
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()

    # macOS true-used: (wired + active + compressed) / total
    wired = getattr(mem, 'wired', 0)
    active = getattr(mem, 'active', 0)
    # Compressed memory from vm_stat
    compressed = self._get_compressed_memory()
    macos_pressure_percent = ((wired + active + compressed) / mem.total) * 100

    kernel_pressure = await self._get_memory_pressure()  # now async
    tier = self._calculate_tier_macos(kernel_pressure, macos_pressure_percent, swap)

    return MemoryMetrics(
        system_memory_percent=macos_pressure_percent,
        tier=tier,
        pressure=kernel_pressure,
        # ... other fields matching existing get_current_metrics return
    )
```

**Step 3: Update `_monitor_loop()` to use async variant**

In `_monitor_loop()` (line ~811), replace:
```python
metrics = self.get_current_metrics()
```
With:
```python
metrics = await self.get_current_metrics_async()
```

The sync `get_current_metrics()` stays for backward compat (one-off sync callers in `backend/main.py`).

**Step 4: Verify**

Run: `python3 -c "import asyncio; from backend.core.memory_quantizer import MemoryQuantizer; q = MemoryQuantizer(); print(asyncio.run(q.get_current_metrics_async()))"`
Expected: MemoryMetrics object printed, no blocking.

**Step 5: Commit**

```bash
git add backend/core/memory_quantizer.py
git commit -m "fix(memory): make MemoryQuantizer non-blocking on event loop

Replace subprocess.run(['memory_pressure']) with asyncio.create_subprocess_exec
in _get_memory_pressure(). Add get_current_metrics_async() for async callers.
Same pattern as ECAPA v265.2 and ChromaDB v265.2 event loop fixes."
```

---

## Task 4: Fix Cost Tracker — Fail Closed

**Files:**
- Modify: `backend/core/cost_tracker.py:1955-1962` (can_create_vm exception handler)

**Step 1: Change fail-open to fail-closed**

At line 1955-1962, replace:

```python
except Exception as e:
    log_component_failure(...)
    # On error, allow VM creation (don't block due to tracking issues)
    return True, f"Budget check error (allowing): {e}", details
```

With:

```python
except Exception as e:
    log_component_failure(
        "CostTracker", "can_create_vm",
        str(e), severity="warning"
    )
    logger.error(
        f"[CostTracker] Budget check failed — blocking VM creation for safety: {e}"
    )
    details["error"] = str(e)
    details["fail_mode"] = "closed"
    return False, f"Budget check error (blocking for safety): {e}", details
```

**Step 2: Verify**

Run: `python3 -c "from backend.core.cost_tracker import CostTracker; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add backend/core/cost_tracker.py
git commit -m "fix(cost): change can_create_vm() from fail-open to fail-closed

Under pressure-driven VM creation, a budget tracking failure should block
creation rather than allow unlimited VM provisioning. Fail-closed is the
safe default for solo developer cost protection."
```

---

## Task 5: VM Lifecycle State Machine in GCPHybridPrimeRouter

**Files:**
- Modify: `backend/core/gcp_hybrid_prime_router.py:86` (imports), `:212-218` (enums area), `:702-753` (__init__ state fields), `:994-1209` (_memory_pressure_monitor), `:1085-1196` (threshold trigger blocks), `:2011-2174` (_trigger_vm_provisioning)

**Step 1: Add VMLifecycleState enum after RoutingTier**

After line 218, add:

```python
class VMLifecycleState(Enum):
    """State machine for pressure-driven GCP VM lifecycle."""
    IDLE = "idle"                # No GCP VM active, local-only
    TRIGGERING = "triggering"    # Pressure detected, accumulating readings
    PROVISIONING = "provisioning"  # VM creation/restart API call in-flight
    BOOTING = "booting"          # VM created, waiting for health check
    ACTIVE = "active"            # VM serving requests
    COOLING_DOWN = "cooling_down"  # Pressure dropped, grace timer running
    STOPPING = "stopping"        # Session shutdown, VM being STOPPED
```

**Step 2: Add hysteresis config constants**

Near the existing constants (lines ~149-209), add:

```python
GCP_RELEASE_RAM_PERCENT = float(os.getenv("GCP_RELEASE_RAM_PERCENT", "70.0"))
GCP_TRIGGER_READINGS_REQUIRED = int(os.getenv("GCP_TRIGGER_READINGS_REQUIRED", "3"))
GCP_TRIGGER_READINGS_WINDOW = int(os.getenv("GCP_TRIGGER_READINGS_WINDOW", "5"))
GCP_ACTIVE_STABILITY_CHECKS = int(os.getenv("GCP_ACTIVE_STABILITY_CHECKS", "3"))
GCP_COOLING_GRACE_SECONDS = float(os.getenv("GCP_COOLING_GRACE_SECONDS", "120.0"))
```

**Step 3: Replace boolean fields in `__init__` with state machine**

In `__init__` (around lines 702-753), replace the scattered booleans:

```python
# v266.0: VM Lifecycle State Machine (replaces scattered booleans)
self._vm_lifecycle_state: VMLifecycleState = VMLifecycleState.IDLE
self._vm_lifecycle_changed_at: float = 0.0
self._trigger_readings: Deque[bool] = deque(maxlen=GCP_TRIGGER_READINGS_WINDOW)
self._active_stability_count: int = 0
self._cooling_started_at: float = 0.0

# Keep these for backward compat (read-only properties derived from state)
# Remove the old boolean assignments for:
#   _vm_provisioning_in_progress (line 705) → derived from state == PROVISIONING
#   _emergency_offload_active (line 737) → derived from state == ACTIVE
#   _emergency_offload_released_at (line 743) → replaced by _cooling_started_at
#   _emergency_offload_cycle_count (line 744) → tracked by _trigger_readings
#   _emergency_offload_hysteresis_armed (line 745) → implicit in COOLING_DOWN state
```

Add state transition method:

```python
def _transition_vm_lifecycle(self, new_state: VMLifecycleState, reason: str = "") -> bool:
    """Transition the VM lifecycle state machine. Returns True if transition was valid."""
    old_state = self._vm_lifecycle_state

    # Define valid transitions
    valid_transitions = {
        VMLifecycleState.IDLE: {VMLifecycleState.TRIGGERING, VMLifecycleState.STOPPING},
        VMLifecycleState.TRIGGERING: {VMLifecycleState.PROVISIONING, VMLifecycleState.IDLE, VMLifecycleState.STOPPING},
        VMLifecycleState.PROVISIONING: {VMLifecycleState.BOOTING, VMLifecycleState.COOLING_DOWN, VMLifecycleState.STOPPING},
        VMLifecycleState.BOOTING: {VMLifecycleState.ACTIVE, VMLifecycleState.COOLING_DOWN, VMLifecycleState.STOPPING},
        VMLifecycleState.ACTIVE: {VMLifecycleState.COOLING_DOWN, VMLifecycleState.STOPPING},
        VMLifecycleState.COOLING_DOWN: {VMLifecycleState.IDLE, VMLifecycleState.TRIGGERING, VMLifecycleState.STOPPING},
        VMLifecycleState.STOPPING: {VMLifecycleState.IDLE},
    }

    # STOPPING is always reachable (session shutdown)
    if new_state == VMLifecycleState.STOPPING:
        pass  # Always allowed
    elif new_state not in valid_transitions.get(old_state, set()):
        self.logger.warning(
            f"[VMLifecycle] Invalid transition {old_state.value} -> {new_state.value} "
            f"(reason: {reason})"
        )
        return False

    self._vm_lifecycle_state = new_state
    self._vm_lifecycle_changed_at = time.time()
    self.logger.info(
        f"[VMLifecycle] {old_state.value} -> {new_state.value} (reason: {reason})"
    )
    return True
```

**Step 4: Add backward-compatible properties**

```python
@property
def _vm_provisioning_in_progress(self) -> bool:
    """Backward compat: True when VM is being provisioned or booting."""
    return self._vm_lifecycle_state in (
        VMLifecycleState.PROVISIONING, VMLifecycleState.BOOTING
    )

@property
def _emergency_offload_active(self) -> bool:
    """Backward compat: True when GCP VM is actively serving."""
    return self._vm_lifecycle_state == VMLifecycleState.ACTIVE
```

**Step 5: Rewrite `_memory_pressure_monitor()` threshold blocks**

Replace the 80% emergency trigger block (lines 1085-1163) and the 70% standard trigger block (lines 1182-1196) with state-machine-aware logic:

```python
# v266.0: State-machine-driven pressure response
current_state = self._vm_lifecycle_state

if current_state == VMLifecycleState.IDLE:
    # Check if pressure warrants triggering
    above_trigger = used_percent >= CRITICAL_RAM_PERCENT  # 85%
    self._trigger_readings.append(above_trigger)

    # Spike bypass: 100MB/sec rate still triggers instantly
    if spike_detected and not self._is_in_cooldown():
        self._transition_vm_lifecycle(VMLifecycleState.TRIGGERING,
                                       f"memory_spike_{memory_rate_mb_sec:.0f}mb_sec")
        # Immediate escalation to provisioning for spikes
        self._transition_vm_lifecycle(VMLifecycleState.PROVISIONING,
                                       "spike_bypass")
        success = await self._trigger_vm_provisioning(
            reason=f"memory_spike_{memory_rate_mb_sec:.0f}mb_sec"
        )
        if not success:
            self._transition_vm_lifecycle(VMLifecycleState.COOLING_DOWN,
                                           "provisioning_failed")
            self._cooling_started_at = time.time()
    elif above_trigger:
        self._transition_vm_lifecycle(VMLifecycleState.TRIGGERING,
                                       f"pressure_{used_percent:.1f}pct")

elif current_state == VMLifecycleState.TRIGGERING:
    above_trigger = used_percent >= CRITICAL_RAM_PERCENT
    self._trigger_readings.append(above_trigger)

    # Count readings above threshold in the window
    above_count = sum(1 for r in self._trigger_readings if r)

    if above_count >= GCP_TRIGGER_READINGS_REQUIRED:
        # Sustained pressure confirmed — provision
        self._transition_vm_lifecycle(VMLifecycleState.PROVISIONING,
                                       f"sustained_{above_count}/{len(self._trigger_readings)}")
        success = await self._trigger_vm_provisioning(reason="sustained_pressure")
        if not success:
            self._transition_vm_lifecycle(VMLifecycleState.COOLING_DOWN,
                                           "provisioning_failed")
            self._cooling_started_at = time.time()
    elif not above_trigger and above_count == 0:
        # Pressure gone completely — back to idle
        self._transition_vm_lifecycle(VMLifecycleState.IDLE, "pressure_cleared")
        self._trigger_readings.clear()

elif current_state == VMLifecycleState.ACTIVE:
    # Check release threshold (hysteresis: must drop below 70%)
    if used_percent < GCP_RELEASE_RAM_PERCENT:
        self._transition_vm_lifecycle(VMLifecycleState.COOLING_DOWN,
                                       f"pressure_released_{used_percent:.1f}pct")
        self._cooling_started_at = time.time()

elif current_state == VMLifecycleState.COOLING_DOWN:
    elapsed = time.time() - self._cooling_started_at
    if elapsed >= GCP_COOLING_GRACE_SECONDS:
        self._transition_vm_lifecycle(VMLifecycleState.IDLE, "cooling_complete")
        self._trigger_readings.clear()
    elif used_percent >= CRITICAL_RAM_PERCENT:
        # Pressure back — return to active (VM still running)
        self._transition_vm_lifecycle(VMLifecycleState.TRIGGERING,
                                       "pressure_returned")

# PROVISIONING and BOOTING states are managed by _trigger_vm_provisioning()
# and health check callbacks — not by the pressure monitor
```

**Step 6: Update `_trigger_vm_provisioning()` to use state transitions**

At line 2024-2028, replace the boolean guard:
```python
# Before:
if self._vm_provisioning_in_progress:
    return False
self._vm_provisioning_in_progress = True
```

With:
```python
# v266.0: State machine handles guard
if self._vm_lifecycle_state in (VMLifecycleState.PROVISIONING, VMLifecycleState.BOOTING, VMLifecycleState.ACTIVE):
    self.logger.debug(f"VM lifecycle in {self._vm_lifecycle_state.value}, skipping provision")
    return False
```

And in the `finally` block (line ~2174), replace:
```python
self._vm_provisioning_in_progress = False
```
With:
```python
# State transition handled by success/failure paths above, not blindly here
pass
```

Add on success (after controller.create_vm succeeds):
```python
self._transition_vm_lifecycle(VMLifecycleState.BOOTING, "vm_created")
```

Add when health check passes (in the health check callback):
```python
self._transition_vm_lifecycle(VMLifecycleState.ACTIVE, "health_check_passed")
```

**Step 7: Update `stop()` to handle state machine shutdown**

In `stop()` (line 2233), add at the beginning:

```python
# Transition to STOPPING state
if self._vm_lifecycle_state != VMLifecycleState.IDLE:
    self._transition_vm_lifecycle(VMLifecycleState.STOPPING, "router_shutdown")
```

**Step 8: Verify**

Run: `python3 -c "from backend.core.gcp_hybrid_prime_router import VMLifecycleState; print(list(VMLifecycleState))"`
Expected: All 7 states printed.

**Step 9: Commit**

```bash
git add backend/core/gcp_hybrid_prime_router.py
git commit -m "feat(gcp): add VM lifecycle state machine to GCPHybridPrimeRouter

Replace scattered boolean fields with formal VMLifecycleState enum and
state machine. States: IDLE -> TRIGGERING -> PROVISIONING -> BOOTING ->
ACTIVE -> COOLING_DOWN -> STOPPING. Hysteresis bands: trigger at 85%,
release at 70%. Sustained pressure requires 3/5 consecutive readings.
100MB/sec spike still bypasses N-of-M check for instant trigger."
```

---

## Task 6: Wire MemoryQuantizer Callback into Router

**Files:**
- Modify: `backend/core/gcp_hybrid_prime_router.py:938-992` (start method)

**Step 1: Add MemoryQuantizer callback registration in `start()`**

In the `start()` method (around line 962-991), after the GCP check and before creating `_memory_pressure_task`, add:

```python
# v266.0: Register for authoritative macOS memory tier changes
try:
    from backend.core.memory_quantizer import get_memory_quantizer, MemoryTier
    mq = await get_memory_quantizer()

    async def _on_tier_change(old_tier: MemoryTier, new_tier: MemoryTier):
        """MemoryQuantizer authoritative tier change callback."""
        tier_severity = {
            MemoryTier.ABUNDANT: 0, MemoryTier.OPTIMAL: 1,
            MemoryTier.ELEVATED: 2, MemoryTier.CONSTRAINED: 3,
            MemoryTier.CRITICAL: 4, MemoryTier.EMERGENCY: 5,
        }
        new_sev = tier_severity.get(new_tier, 0)

        if new_sev >= 4 and self._vm_lifecycle_state == VMLifecycleState.IDLE:
            # CRITICAL or EMERGENCY from macOS-aware metric — this is authoritative
            self.logger.info(
                f"[VMLifecycle] MemoryQuantizer tier change: {old_tier.value} -> {new_tier.value}"
            )
            # Count as an above-threshold reading for the trigger window
            self._trigger_readings.append(True)
            above_count = sum(1 for r in self._trigger_readings if r)
            if above_count >= GCP_TRIGGER_READINGS_REQUIRED:
                self._transition_vm_lifecycle(
                    VMLifecycleState.TRIGGERING,
                    f"mq_tier_{new_tier.value}"
                )

    mq.register_tier_change_callback(_on_tier_change)
    self.logger.info("[VMLifecycle] Registered MemoryQuantizer tier callback")
except ImportError:
    self.logger.debug("MemoryQuantizer not available — using psutil-only polling")
except Exception as e:
    self.logger.debug(f"MemoryQuantizer callback registration failed: {e}")
```

**Step 2: Verify**

Run: `python3 -c "from backend.core.gcp_hybrid_prime_router import GCPHybridPrimeRouter; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add backend/core/gcp_hybrid_prime_router.py
git commit -m "feat(gcp): wire MemoryQuantizer tier callbacks into router

Register for macOS-aware tier change events from MemoryQuantizer. When tier
reaches CRITICAL/EMERGENCY, counts as an above-threshold reading for the
3-of-5 sustained pressure window. MemoryQuantizer uses (wired + active +
compressed) / total which is the correct metric for M1 unified memory."
```

---

## Task 7: Local Model Unload After GCP Takeover

**Files:**
- Modify: `backend/core/gcp_hybrid_prime_router.py` (add unload logic after ACTIVE transition)

**Step 1: Add model unload method**

Add to `GCPHybridPrimeRouter`:

```python
async def _unload_local_model_after_stability(self) -> None:
    """After GCP VM proves stable, unload local model to reclaim RAM."""
    try:
        self._active_stability_count = 0

        for _ in range(GCP_ACTIVE_STABILITY_CHECKS):
            await asyncio.sleep(10.0)

            # Check VM is still ACTIVE and healthy
            if self._vm_lifecycle_state != VMLifecycleState.ACTIVE:
                self.logger.info("[VMLifecycle] Left ACTIVE state, aborting model unload")
                return

            if self._gcp_controller and hasattr(self._gcp_controller, 'is_vm_available'):
                if not self._gcp_controller.is_vm_available():
                    self.logger.info("[VMLifecycle] GCP VM unhealthy, aborting model unload")
                    return

            self._active_stability_count += 1

        # GCP stable for 30s — unload local model
        self.logger.info(
            f"[VMLifecycle] GCP VM stable for {GCP_ACTIVE_STABILITY_CHECKS * 10}s, "
            f"unloading local model to reclaim RAM"
        )

        try:
            from backend.intelligence.unified_model_serving import get_model_serving
            model_serving = get_model_serving()
            if model_serving and hasattr(model_serving, 'stop'):
                await model_serving.stop()
                self.logger.info("[VMLifecycle] Local model unloaded — RAM reclaimed")
                os.environ["JARVIS_GCP_OFFLOAD_ACTIVE"] = "true"
        except ImportError:
            self.logger.debug("UnifiedModelServing not available for unload")
        except Exception as e:
            self.logger.warning(f"[VMLifecycle] Local model unload failed: {e}")

    except asyncio.CancelledError:
        pass
    except Exception as e:
        self.logger.debug(f"[VMLifecycle] Model unload task error: {e}")
```

**Step 2: Launch unload task when transitioning to ACTIVE**

In `_transition_vm_lifecycle()`, add after the state assignment:

```python
if new_state == VMLifecycleState.ACTIVE:
    # Start background task to unload local model after stability confirmed
    if not hasattr(self, '_model_unload_task') or self._model_unload_task is None or self._model_unload_task.done():
        self._model_unload_task = asyncio.create_task(
            self._unload_local_model_after_stability(),
            name="gcp_model_unload_after_stability"
        )
```

**Step 3: Cancel unload task on state transitions away from ACTIVE**

In `_transition_vm_lifecycle()`, add:

```python
if old_state == VMLifecycleState.ACTIVE and new_state != VMLifecycleState.ACTIVE:
    # Cancel pending model unload if we leave ACTIVE
    if hasattr(self, '_model_unload_task') and self._model_unload_task and not self._model_unload_task.done():
        self._model_unload_task.cancel()
    # Clear offload flag
    os.environ.pop("JARVIS_GCP_OFFLOAD_ACTIVE", None)
```

**Step 4: Commit**

```bash
git add backend/core/gcp_hybrid_prime_router.py
git commit -m "feat(gcp): unload local model after GCP VM proves stable

After VM lifecycle reaches ACTIVE and 3 consecutive health checks pass (30s),
call UnifiedModelServing.stop() to unload the local GGUF model and reclaim
4-8GB RAM. This closes the feedback loop: pressure triggers GCP, then model
unload actually frees the memory. If GCP becomes unhealthy, load_model()
re-loads locally via existing hot-swap."
```

---

## Task 8: Delete Eager Pre-Warm from Supervisor

**Files:**
- Modify: `unified_supervisor.py:63841-63904` (early_spot_vm_warm), `:63906-64495` (early_invincible_node_prewarm)

**Step 1: Remove `early_spot_vm_warm` block (lines 63841-63904)**

Delete the entire block from the comment `# v258.3 (Gap 10): PROACTIVE SPOT VM WARM` through the task creation and log statement. Replace with a comment:

```python
# v266.0: Proactive Spot VM warm REMOVED — VM creation is now pressure-driven
# via GCPHybridPrimeRouter state machine. See design doc:
# docs/plans/2026-02-22-gcp-pressure-driven-lifecycle-design.md
```

**Step 2: Remove `early_invincible_node_prewarm` block (lines 63906-64495)**

This is a larger block that includes the `_early_wake_invincible_node` inner function, the golden image decision tree, and the local prime prewarm launch.

**Important:** The local Prime prewarm (`JARVIS_EARLY_PRIME_PREWARM`) logic around lines 64040-64300 should be KEPT — it's for local model prewarming, not GCP. Only delete:
- The `_early_wake_invincible_node` async def and its task creation (lines ~64315-64495)
- The GCP-specific env var checks that gate it (the `_early_gcp_golden` and `_early_gcp_on` conditions)

Replace the deleted invincible node section with:

```python
# v266.0: Invincible Node eager prewarm REMOVED — VM creation is now
# pressure-driven via GCPHybridPrimeRouter. The router provisions VMs
# when memory pressure exceeds 85% sustained (3/5 readings).
```

**Step 3: Clean up any task references**

Search for `_early_invincible_task` and `_early_spot_vm_task` in the rest of the file (they may be awaited or checked elsewhere). Remove or null-check those references.

**Step 4: Verify supervisor still starts**

Run: `python3 unified_supervisor.py --dry-run`
Expected: Config summary prints and exits with code 0. No errors about missing early tasks.

**Step 5: Commit**

```bash
git add unified_supervisor.py
git commit -m "refactor(startup): remove eager GCP VM pre-warm from supervisor

Delete early_invincible_node_prewarm and early_spot_vm_warm task blocks.
VM creation is now demand-driven by GCPHybridPrimeRouter's pressure-driven
state machine. GCPInstanceManager still syncs existing VMs in Phase 2.
Local Prime prewarm (JARVIS_EARLY_PRIME_PREWARM) is preserved."
```

---

## Task 9: Update Supervisor Shutdown to STOP VMs

**Files:**
- Modify: `unified_supervisor.py:78213-78234` (cleanup GCP section), `:61964-61979` (emergency shutdown GCP section)

**Step 1: Update normal `cleanup()` GCP section**

Replace the `shutdown_orchestrator()` call at lines 78219-78234 with:

```python
# =====================================================================
# v266.0: GCP VM SESSION LIFECYCLE — STOP VMs on shutdown
# =====================================================================
try:
    from backend.core.gcp_vm_manager import get_gcp_vm_manager_safe, VMAction
    vm_manager = get_gcp_vm_manager_safe()
    if vm_manager:
        # Stop all tracked VMs (STOP, not DELETE — preserves disk/IP)
        for vm_name, vm_info in list(vm_manager.managed_vms.items()):
            if vm_info.state.value in ("running", "staging", "provisioning"):
                try:
                    await asyncio.wait_for(
                        vm_manager.terminate_vm(
                            vm_name,
                            reason="session_shutdown",
                            action=VMAction.STOP,
                        ),
                        timeout=15.0,
                    )
                except asyncio.TimeoutError:
                    self.logger.warning(f"[Kernel] Timeout stopping VM '{vm_name}'")
                except Exception as e:
                    self.logger.debug(f"[Kernel] VM stop error for '{vm_name}': {e}")

    # Still call shutdown_orchestrator for other cross-repo cleanup
    if CROSS_REPO_ORCHESTRATOR_AVAILABLE:
        from backend.supervisor.cross_repo_startup_orchestrator import (
            shutdown_orchestrator,
        )
        await asyncio.wait_for(shutdown_orchestrator(), timeout=10.0)
except ImportError:
    pass
except Exception as e:
    self.logger.debug(f"[Kernel] GCP cleanup: {e}")
```

**Step 2: Update `_emergency_shutdown()` GCP section**

At lines 61964-61979, add the same VM STOP logic before the `shutdown_orchestrator()` call:

```python
# v266.0: Emergency VM STOP
try:
    from backend.core.gcp_vm_manager import get_gcp_vm_manager_safe, VMAction
    vm_manager = get_gcp_vm_manager_safe()
    if vm_manager:
        for vm_name, vm_info in list(vm_manager.managed_vms.items()):
            if vm_info.state.value in ("running", "staging"):
                try:
                    await asyncio.wait_for(
                        vm_manager.terminate_vm(
                            vm_name, reason="emergency_shutdown", action=VMAction.STOP
                        ),
                        timeout=10.0,
                    )
                except Exception:
                    pass
except Exception:
    pass
```

**Step 3: Verify**

Run: `python3 unified_supervisor.py --dry-run`
Expected: Clean exit.

**Step 4: Commit**

```bash
git add unified_supervisor.py
git commit -m "feat(shutdown): STOP all GCP VMs on supervisor shutdown

Both normal cleanup() and _emergency_shutdown() now call terminate_vm()
with action=VMAction.STOP for all tracked running VMs. This ensures VMs
are STOPPED (not deleted) when JARVIS exits, preserving disk and IP for
fast 30s restart while eliminating 24/7 compute charges."
```

---

## Task 10: Update .env.gcp Defaults

**Files:**
- Modify: `.env.gcp:90` (JARVIS_INVINCIBLE_NODE_ENABLED)

**Step 1: Change default**

At line 90, change:
```
JARVIS_INVINCIBLE_NODE_ENABLED=true
```
To:
```
JARVIS_INVINCIBLE_NODE_ENABLED=false
```

Add a comment explaining the change:

```
# v266.0: Invincible Node no longer eagerly created at startup.
# VM provisioning is now pressure-driven by GCPHybridPrimeRouter
# when memory exceeds 85% sustained. Set to true to re-enable
# eager prewarm (not recommended for dev/testing).
JARVIS_INVINCIBLE_NODE_ENABLED=false
```

**Step 2: Commit**

```bash
git add .env.gcp
git commit -m "config: disable eager Invincible Node in .env.gcp

Set JARVIS_INVINCIBLE_NODE_ENABLED=false. VM provisioning is now
pressure-driven. This eliminates the primary cost driver ($21/mo
from 24/7 VM) during development and testing."
```

---

## Task 11: Integration Verification

**Step 1: Full startup test (local only)**

```bash
python3 unified_supervisor.py --dry-run
```
Expected: Config summary shows `gcp_enabled: True` but no VM creation tasks listed.

**Step 2: Import chain verification**

```bash
python3 -c "
from backend.core.gcp_vm_manager import VMAction, VMState, GCPVMManager
from backend.core.gcp_hybrid_prime_router import VMLifecycleState, GCPHybridPrimeRouter
from backend.core.memory_quantizer import MemoryQuantizer
from backend.core.cost_tracker import CostTracker
print('All imports OK')
print('VMAction:', [a.value for a in VMAction])
print('VMState:', [s.value for s in VMState])
print('VMLifecycleState:', [s.value for s in VMLifecycleState])
"
```
Expected: All imports succeed, enums print correctly.

**Step 3: Memory quantizer async test**

```bash
python3 -c "
import asyncio
from backend.core.memory_quantizer import MemoryQuantizer
async def test():
    q = MemoryQuantizer()
    m = await q.get_current_metrics_async()
    print(f'Tier: {m.tier}, Pressure: {m.system_memory_percent:.1f}%')
asyncio.run(test())
"
```
Expected: Tier and pressure printed, no blocking.

**Step 4: Stop any running GCP VMs now**

```bash
gcloud compute instances list --project=jarvis-473803 --filter="name~jarvis" --format="table(name,status,zone)"
```

If any are RUNNING, stop them:
```bash
gcloud compute instances stop jarvis-prime-node --zone=us-central1-a --project=jarvis-473803
```

**Step 5: Final commit (if any fixups needed)**

```bash
git add -A
git commit -m "fix: integration fixups for pressure-driven GCP lifecycle"
```

---

## Summary of Commits

| Task | Commit Message | Files |
|------|---------------|-------|
| 1 | `refactor(gcp): centralize InvincibleGuard` | `gcp_vm_manager.py` |
| 2 | `feat(gcp): extend terminate_vm() with STOP action` | `gcp_vm_manager.py` |
| 3 | `fix(memory): make MemoryQuantizer non-blocking` | `memory_quantizer.py` |
| 4 | `fix(cost): can_create_vm() fail-closed` | `cost_tracker.py` |
| 5 | `feat(gcp): VM lifecycle state machine` | `gcp_hybrid_prime_router.py` |
| 6 | `feat(gcp): wire MemoryQuantizer callbacks` | `gcp_hybrid_prime_router.py` |
| 7 | `feat(gcp): unload local model after GCP stable` | `gcp_hybrid_prime_router.py` |
| 8 | `refactor(startup): remove eager pre-warm` | `unified_supervisor.py` |
| 9 | `feat(shutdown): STOP VMs on shutdown` | `unified_supervisor.py` |
| 10 | `config: disable eager Invincible Node` | `.env.gcp` |
| 11 | Integration verification | (fixups if needed) |

**Estimated cost reduction:** ~$50-92/mo → ~$20-30/mo (60-70% reduction)
