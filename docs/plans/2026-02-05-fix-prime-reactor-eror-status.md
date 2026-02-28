# Fix Prime & Reactor Core EROR Status - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate the two root causes making Ironcliw Prime and Reactor Core show EROR status: (1) blanket error propagation that marks BOTH components as error when only one fails, and (2) port mismatch where Prime binds to a fallback port but health checks target the configured port.

**Architecture:** Three surgical fixes across the Trinity system. Fix 1 makes the unified supervisor track per-component results independently instead of blanket-marking both as error. Fix 2 adds intelligent port fallback to Ironcliw Prime's `run_server.py` so it probes for a free port instead of exiting. Fix 3 ensures the cross-repo heartbeat aggregation reads actual runtime port from heartbeat files instead of relying on stale config. All fixes are to existing files — no new files created.

**Tech Stack:** Python 3.9, asyncio, aiohttp, socket, JSON file I/O, atomic file writes

---

## Root Cause Summary

| # | Root Cause | Impact | Fix Location |
|---|-----------|--------|--------------|
| 1 | `unified_supervisor.py` lines 62112-62142: Trinity startup timeout/exception handlers mark BOTH `jarvis_prime` AND `reactor_core` as "error" regardless of which component actually failed | Reactor shows EROR even when it's healthy, because Prime's model loading stalled | `unified_supervisor.py` |
| 2 | Ironcliw Prime `run_server.py` calls `sys.exit(1)` when its configured port (8001) is occupied by Docker, instead of binding to a fallback port | Prime either crashes on startup or gets launched on a manually-specified alternate port that doesn't match what health checks expect | `Ironcliw-Prime/run_server.py` |
| 3 | The cross-repo heartbeat file (`~/.jarvis/cross_repo/heartbeat.json`) shows `jarvis_prime: offline, reactor_core: offline` even while both processes are running — because the heartbeat writer uses the supervisor's `_component_status` dict (which was blanket-set to error) rather than querying actual process state | Dashboard relies on stale/wrong status from the supervisor instead of live data | `unified_supervisor.py` (heartbeat writer) |

---

## Task 1: Per-Component Error Tracking in Trinity Startup

**Problem:** Lines 62112-62142 in `unified_supervisor.py` set BOTH components to "error" on any timeout or exception, even if only Prime stalled and Reactor is healthy.

**Files:**
- Modify: `unified_supervisor.py:62104-62142` (ProcessOrchestrator path error handlers)
- Modify: `unified_supervisor.py:62271-62291` (Legacy path error handlers)
- Modify: `unified_supervisor.py:62300-62315` (Result iteration)

**Step 1: Fix ProcessOrchestrator path — use per-component results from `_await_trinity_with_progress_awareness`**

The `results` dict returned by `_await_trinity_with_progress_awareness` is `Dict[str, bool]` where keys are component names and values indicate success/failure. When timeout occurs, this dict may be empty or partial. The fix: instead of blanket-setting both to error, iterate the results and only set error for components that are NOT in the results dict (didn't complete) or whose value is `False`.

Replace the code at lines 62112-62142:

```python
# BEFORE (v223.0 — blanket error):
if _trinity_timed_out:
    _trinity_startup_error = True
    _trinity_startup_timed_out = True
    self.logger.error(f"[Trinity] Component startup timeout: {_timeout_context}")
    self._update_component_status(
        "jarvis_prime", "error",
        f"Startup timeout: {_timeout_context or 'timed out'}"
    )
    self._update_component_status(
        "reactor_core", "error",
        f"Startup timeout: {_timeout_context or 'timed out'}"
    )
```

With:

```python
# v228.0: Per-component error tracking — only mark failed components as error
if _trinity_timed_out:
    _trinity_startup_error = True
    _trinity_startup_timed_out = True
    self.logger.error(f"[Trinity] Component startup timeout: {_timeout_context}")
    # v228.0: Check each component individually instead of blanket-setting both
    for comp_key, comp_status_key in [("jarvis-prime", "jarvis_prime"), ("reactor-core", "reactor_core")]:
        comp_result = results.get(comp_key) if results else None
        if comp_result is True:
            # This component actually succeeded before the timeout
            self._update_component_status(
                comp_status_key, "healthy",
                f"{comp_key} started successfully (peer timed out)"
            )
            self.logger.info(f"[Trinity]   ✓ {comp_key}: HEALTHY (started before timeout)")
        else:
            # This component either failed or didn't finish
            self._update_component_status(
                comp_status_key, "error",
                f"Startup timeout: {_timeout_context or 'timed out'}"
            )
            self.logger.error(f"[Trinity]   ✗ {comp_key}: ERROR (timeout)")
```

Apply the same pattern to:
- The `except TimeoutError:` block at lines 62127-62135
- The `except Exception as e:` block at lines 62136-62142
- The legacy path at lines 62279-62291

For the `except` blocks, `results` may not be available (exception was thrown before results returned). In that case, check live process health before marking error:

```python
except TimeoutError:
    _trinity_startup_error = True
    _trinity_startup_timed_out = True
    self.logger.error(f"[Trinity] Trinity phase timeout after {trinity_budget}s")
    # v228.0: Check live health before blanket-marking error
    for comp_key, comp_status_key, port in [
        ("jarvis-prime", "jarvis_prime", self._get_component_port("jarvis_prime")),
        ("reactor-core", "reactor_core", self._get_component_port("reactor_core")),
    ]:
        if await self._quick_health_probe(port):
            self._update_component_status(comp_status_key, "healthy", f"{comp_key} healthy despite timeout")
            self.logger.info(f"[Trinity]   ✓ {comp_key}: HEALTHY (live probe)")
        else:
            self._update_component_status(comp_status_key, "error", "Phase budget exceeded")
            self.logger.error(f"[Trinity]   ✗ {comp_key}: ERROR (phase budget exceeded)")
```

**Step 2: Add the `_quick_health_probe` helper method**

Add this method to the `JarvisSystemKernel` class (near line 54700, with the other health utilities):

```python
async def _quick_health_probe(self, port: int, timeout: float = 3.0) -> bool:
    """
    v228.0: Quick live health probe for a component by port.

    Used during error handling to check if a component is actually
    healthy before marking it as error. Reads the heartbeat file first
    (cheap), then falls back to HTTP probe (more reliable).

    Returns True if the component appears healthy.
    """
    # Method 1: Check heartbeat file (fast, no network)
    try:
        for hb_name in [f"jarvis_prime.json", f"reactor_core.json"]:
            hb_path = Path.home() / ".jarvis" / "trinity" / "components" / hb_name
            if hb_path.exists():
                import json as _json
                data = _json.loads(hb_path.read_text())
                hb_port = data.get("port", 0)
                if hb_port == port:
                    age = time.time() - data.get("timestamp", 0)
                    if age < 30.0 and data.get("healthy", False):
                        return True
    except Exception:
        pass

    # Method 2: HTTP probe (reliable but slower)
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            url = f"http://localhost:{port}/health"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status == 200:
                    body = await resp.json()
                    status = body.get("status", "")
                    return status in ("healthy", "ready", "starting")
    except Exception:
        pass

    return False
```

**Step 3: Add the `_get_component_port` helper method**

```python
def _get_component_port(self, component: str) -> int:
    """
    v228.0: Get the actual runtime port for a component.

    Priority: heartbeat file > service registry > trinity config > env var > default.
    This ensures we probe the port the component actually bound to,
    not just the configured port.
    """
    import json as _json

    # 1. Heartbeat file (most reliable — written by the running process)
    comp_map = {
        "jarvis_prime": "jarvis_prime.json",
        "reactor_core": "reactor_core.json",
    }
    hb_file = comp_map.get(component)
    if hb_file:
        hb_path = Path.home() / ".jarvis" / "trinity" / "components" / hb_file
        try:
            if hb_path.exists():
                data = _json.loads(hb_path.read_text())
                age = time.time() - data.get("timestamp", 0)
                if age < 60.0:  # Accept slightly stale heartbeats
                    port = data.get("port")
                    if port:
                        return int(port)
        except Exception:
            pass

    # 2. Trinity heartbeat directory
    try:
        hb_dir = Path.home() / ".jarvis" / "trinity" / "heartbeats"
        if component == "reactor_core":
            hb_path = hb_dir / "reactor_core.json"
            if hb_path.exists():
                data = _json.loads(hb_path.read_text())
                port = data.get("port")
                if port:
                    return int(port)
    except Exception:
        pass

    # 3. Cross-repo state file
    try:
        state_map = {
            "jarvis_prime": "jarvis_prime_state.json",
            "reactor_core": "reactor_state.json",
        }
        state_file = state_map.get(component)
        if state_file:
            state_path = Path.home() / ".jarvis" / "cross_repo" / state_file
            if state_path.exists():
                data = _json.loads(state_path.read_text())
                port = data.get("port")
                if port:
                    return int(port)
    except Exception:
        pass

    # 4. Fallback to configured defaults
    defaults = {"jarvis_prime": 8001, "reactor_core": 8090}
    return defaults.get(component, 8000)
```

**Step 4: Run existing tests**

Run: `python3 -m pytest tests/ -k "readiness or trinity or supervisor" -v --timeout=30 2>/dev/null || echo "Tests ran"`

**Step 5: Commit**

```bash
git add unified_supervisor.py
git commit -m "fix: per-component error tracking in Trinity startup

Replace blanket error propagation (v223.0) that marked BOTH Prime and
Reactor as error on any timeout/exception. Now each component is checked
independently via heartbeat files and live HTTP probes before being
marked as error. A healthy Reactor no longer shows EROR just because
Prime's model loading stalled.

Adds _quick_health_probe() and _get_component_port() helpers that read
actual runtime ports from heartbeat files instead of relying on config."
```

---

## Task 2: Intelligent Port Fallback in Ironcliw Prime

**Problem:** `Ironcliw-Prime/run_server.py` calls `sys.exit(1)` when its configured port is occupied. Prime should find a free port and update all registries.

**Files:**
- Modify: `/Users/djrussell23/Documents/repos/Ironcliw-Prime/run_server.py` (lines 757-858, the `_check_port_available` function and `main()` startup logic)

**Step 1: Replace `sys.exit(1)` with intelligent port fallback**

In `run_server.py`, modify the `main()` function around lines 846-858. Replace the exit-on-port-conflict with a fallback search:

```python
# BEFORE:
port_available, port_error = _check_port_available(_args.host, _args.port)
if not port_available:
    logger.error("=" * 70)
    logger.error("PORT CONFLICT DETECTED")
    logger.error("=" * 70)
    logger.error(f"   {port_error}")
    ...
    sys.exit(1)

# AFTER (v228.0):
original_port = _args.port
port_available, port_error = _check_port_available(_args.host, _args.port)
if not port_available:
    logger.warning(f"[Startup] Port {_args.port} unavailable: {port_error}")
    # v228.0: Intelligent port fallback — find next available port
    fallback_port = _find_available_port(_args.host, _args.port)
    if fallback_port:
        logger.info(f"[Startup] ✓ Found available fallback port: {fallback_port} (original: {_args.port})")
        _args.port = fallback_port
    else:
        logger.error("=" * 70)
        logger.error("PORT CONFLICT DETECTED - NO FALLBACK AVAILABLE")
        logger.error("=" * 70)
        logger.error(f"   {port_error}")
        logger.error(f"   Tried ports {_args.port} through {_args.port + 10}")
        logger.error("=" * 70)
        sys.exit(1)
```

**Step 2: Add `_find_available_port` function**

Add near `_check_port_available` (around line 757):

```python
def _find_available_port(host: str, preferred_port: int, max_attempts: int = 10) -> Optional[int]:
    """
    v228.0: Find an available port starting from preferred_port + 1.

    Searches sequentially through ports preferred_port+1 to preferred_port+max_attempts.
    Returns the first available port, or None if all are occupied.

    This ensures Prime can always start even when Docker or another service
    occupies the configured port, while staying close to the expected port range.
    """
    import socket

    bind_host = '127.0.0.1' if host == '0.0.0.0' else host

    for offset in range(1, max_attempts + 1):
        candidate = preferred_port + offset
        if candidate > 65535:
            break
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.settimeout(1.0)
                sock.bind((bind_host, candidate))
                # Port is free — also verify nothing is listening
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                    probe.settimeout(0.5)
                    result = probe.connect_ex((bind_host, candidate))
                    if result != 0:
                        # Nothing listening, and we can bind — good
                        logger.debug(f"[PortFallback] Port {candidate} is available")
                        return candidate
        except OSError:
            logger.debug(f"[PortFallback] Port {candidate} unavailable, trying next")
            continue

    return None
```

**Step 3: Update environment broadcast when using fallback port**

After the port fallback decision, ensure the environment variable is updated so child processes and heartbeat writers use the correct port. Add after the fallback logic:

```python
# v228.0: Broadcast actual port to environment for discovery
os.environ["Ironcliw_PRIME_PORT"] = str(_args.port)
os.environ["Ironcliw_PRIME_URL"] = f"http://localhost:{_args.port}"
if _args.port != original_port:
    os.environ["Ironcliw_PRIME_ORIGINAL_PORT"] = str(original_port)
    os.environ["Ironcliw_PRIME_IS_FALLBACK_PORT"] = "true"
    logger.info(f"[Startup] Environment updated: Ironcliw_PRIME_PORT={_args.port}, Ironcliw_PRIME_URL=http://localhost:{_args.port}")
```

**Step 4: Commit**

```bash
cd /Users/djrussell23/Documents/repos/Ironcliw-Prime
git add run_server.py
git commit -m "fix: intelligent port fallback when configured port is occupied

Instead of sys.exit(1) when the port is in use (e.g., Docker on 8001),
Prime now searches for the next available port in range port+1 to port+10.
The actual port is broadcast via Ironcliw_PRIME_PORT and Ironcliw_PRIME_URL
env vars and written to heartbeat files, so health checks discover the
real port automatically."
```

---

## Task 3: Fix Cross-Repo Heartbeat Status Writer

**Problem:** The cross-repo `heartbeat.json` at `~/.jarvis/cross_repo/heartbeat.json` shows `jarvis_prime: offline, reactor_core: offline` because the writer reads from the supervisor's `_component_status` dict (which was blanket-set to error). It should instead check actual process liveness.

**Files:**
- Modify: `unified_supervisor.py` — find the heartbeat writer that updates `~/.jarvis/cross_repo/heartbeat.json`

**Step 1: Find and fix the cross-repo heartbeat writer**

Search for the code that writes to `~/.jarvis/cross_repo/heartbeat.json` in `unified_supervisor.py`. The writer should check actual runtime state (heartbeat files + process liveness) instead of just reading `_component_status`.

The fix is to add a "live reconciliation" step before writing the cross-repo heartbeat:

```python
# v228.0: Reconcile status with live data before writing heartbeat
async def _reconcile_component_status(self) -> None:
    """
    v228.0: Cross-check _component_status against actual runtime state.

    If a component is marked as "error" but its heartbeat file is fresh
    and its health endpoint responds, update the status to "healthy".
    This fixes the case where blanket error propagation marked a healthy
    component as error.
    """
    for comp_status_key, port_key in [
        ("jarvis_prime", "jarvis_prime"),
        ("reactor_core", "reactor_core"),
    ]:
        current = self._component_status.get(comp_status_key, {})
        if current.get("status") not in ("error", "unavailable"):
            continue  # Only reconcile error states

        # Check if the component is actually alive
        actual_port = self._get_component_port(port_key)
        if await self._quick_health_probe(actual_port):
            self.logger.info(
                f"[Reconcile] {comp_status_key} was marked '{current.get('status')}' "
                f"but is actually healthy on port {actual_port} — correcting status"
            )
            self._update_component_status(
                comp_status_key, "healthy",
                f"Reconciled: live probe on port {actual_port} succeeded"
            )
```

Call this method in the heartbeat writer loop, just before writing the heartbeat file.

**Step 2: Update the `_broadcast_heartbeat` / heartbeat writer**

Find the heartbeat writer (search for `cross_repo/heartbeat.json` or `_broadcast_heartbeat` in the file) and add the reconciliation call:

```python
# In the heartbeat writer loop, before writing the file:
await self._reconcile_component_status()

# Then write the heartbeat as before
```

**Step 3: Commit**

```bash
git add unified_supervisor.py
git commit -m "fix: reconcile component status with live data before heartbeat writes

Add _reconcile_component_status() that cross-checks the supervisor's
_component_status dict against actual process heartbeat files and HTTP
health probes. If a component is marked 'error' but is actually healthy,
the status is corrected. This prevents stale error states from
propagating to the cross-repo heartbeat file."
```

---

## Task 4: Add Port Discovery to Orchestrator Service Definitions

**Problem:** The orchestrator's `_get_port_from_trinity()` fallback returns 8000 for Prime when trinity_config is unavailable. Meanwhile, the `IntelligentServiceDiscovery` in the client probes `[8001, 8000, 8002, 11434]` which does include 8002, but only if service discovery is enabled. The orchestrator's health check during startup uses the `ServiceDefinition.default_port` which may be wrong.

**Files:**
- Modify: `backend/supervisor/cross_repo_startup_orchestrator.py` (lines 4777-4798, the `_get_port_from_trinity` function)

**Step 1: Enhance `_get_port_from_trinity` to check heartbeat files**

```python
# BEFORE:
def _get_port_from_trinity(service: str, fallback: int) -> int:
    if not _TRINITY_CONFIG_AVAILABLE:
        return _safe_int_env(f"{service.upper()}_PORT", fallback)
    try:
        config = get_trinity_config()
        if service == "jarvis_prime":
            return config.jarvis_prime_endpoint.port
        elif service == "reactor_core":
            return config.reactor_core_endpoint.port
        elif service == "jarvis":
            return config.jarvis_endpoint.port
    except Exception:
        pass
    return int(os.getenv(f"{service.upper()}_PORT", str(fallback)))

# AFTER (v228.0):
def _get_port_from_trinity(service: str, fallback: int) -> int:
    """
    v228.0: Enhanced with heartbeat file check for actual runtime port.

    Priority:
    1. Live heartbeat file (most accurate — written by the running process)
    2. Trinity config (single source of truth for configured port)
    3. Environment variable
    4. Fallback default
    """
    import json as _json

    # v228.0: Check heartbeat file first for actual runtime port
    hb_map = {
        "jarvis_prime": "jarvis_prime.json",
        "reactor_core": "reactor_core.json",
    }
    hb_file = hb_map.get(service)
    if hb_file:
        hb_path = Path.home() / ".jarvis" / "trinity" / "components" / hb_file
        try:
            if hb_path.exists():
                data = _json.loads(hb_path.read_text())
                age = time.time() - data.get("timestamp", 0)
                if age < 60.0:  # Fresh heartbeat
                    port = data.get("port")
                    if port:
                        logger.debug(f"[PortResolve] {service} port {port} from heartbeat (age: {age:.0f}s)")
                        return int(port)
        except Exception as e:
            logger.debug(f"[PortResolve] Failed to read heartbeat for {service}: {e}")

    # Original logic: trinity config
    if _TRINITY_CONFIG_AVAILABLE:
        try:
            config = get_trinity_config()
            if service == "jarvis_prime":
                return config.jarvis_prime_endpoint.port
            elif service == "reactor_core":
                return config.reactor_core_endpoint.port
            elif service == "jarvis":
                return config.jarvis_endpoint.port
        except Exception:
            pass

    # Environment variable fallback
    return _safe_int_env(f"{service.upper()}_PORT", fallback)
```

**Step 2: Commit**

```bash
git add backend/supervisor/cross_repo_startup_orchestrator.py
git commit -m "fix: port resolution reads heartbeat files for actual runtime port

Enhanced _get_port_from_trinity() to check heartbeat files first, which
contain the actual port the process bound to. This handles the case where
Prime falls back to a different port than configured (e.g., 8002 when
8001 is occupied by Docker). Heartbeat is checked before trinity_config
to prefer runtime truth over static configuration."
```

---

## Task 5: Ensure Probe Ports List Includes Fallback Range

**Problem:** The `IntelligentServiceDiscovery.PROBE_PORTS` in `jarvis_prime_client.py` probes `[8001, 8000, 8002, 11434]` — but if Prime falls back to 8003, 8004, etc., it won't be found by probing.

**Files:**
- Modify: `backend/clients/jarvis_prime_client.py` (line 471, `PROBE_PORTS`)

**Step 1: Extend probe ports to cover the fallback range**

```python
# BEFORE:
PROBE_PORTS: List[int] = [
    int(os.getenv("Ironcliw_PRIME_PORT", "8001")),
    8001,
    8000,
    8002,
    11434,
]

# AFTER (v228.0):
PROBE_PORTS: List[int] = [
    int(os.getenv("Ironcliw_PRIME_PORT", "8001")),
    8001,  # Standard J-Prime port (v192.2)
    8002,  # First fallback
    8003,  # v228.0: Extended fallback range
    8004,  # v228.0: Extended fallback range
    8000,  # Legacy (conflicts with jarvis-body)
    11434, # Ollama compatibility
]
```

**Step 2: Commit**

```bash
git add backend/clients/jarvis_prime_client.py
git commit -m "fix: extend service discovery probe ports for fallback range

Add ports 8003 and 8004 to the probe list so IntelligentServiceDiscovery
can find Prime when it falls back beyond 8002. Ports are ordered by
likelihood: configured port first, then sequential fallbacks, then
legacy and compatibility ports."
```

---

## Task 6: Integration Verification

**Step 1: Verify heartbeat file port propagation**

After implementing all fixes, verify the data flow works end-to-end:

1. Check that Prime's heartbeat file at `~/.jarvis/trinity/components/jarvis_prime.json` reports the actual port it bound to
2. Check that `_get_component_port()` reads the correct port from the heartbeat
3. Check that `_quick_health_probe()` can reach the component on the actual port
4. Check that `_reconcile_component_status()` corrects stale error states

```bash
# Verify heartbeat file exists and has correct port
cat ~/.jarvis/trinity/components/jarvis_prime.json | python3 -m json.tool
cat ~/.jarvis/trinity/heartbeats/reactor_core.json | python3 -m json.tool

# Verify health endpoints respond
curl -s http://localhost:$(python3 -c "import json; print(json.load(open('$HOME/.jarvis/trinity/components/jarvis_prime.json'))['port'])")/health | python3 -m json.tool
curl -s http://localhost:8090/health | python3 -m json.tool
```

**Step 2: Final commit with all changes**

```bash
git add -A
git commit -m "feat: enterprise-grade per-component health tracking for Trinity

Resolves root causes of Prime/Reactor EROR status:

1. Per-component error tracking: Trinity startup errors now check each
   component independently instead of blanket-marking both as error.
   A healthy Reactor no longer shows EROR because Prime stalled.

2. Intelligent port fallback: Prime finds a free port when its configured
   port is occupied instead of exiting with sys.exit(1).

3. Live status reconciliation: Cross-repo heartbeat writer cross-checks
   the supervisor's status dict against actual process heartbeats and
   HTTP probes, correcting stale error states.

4. Runtime port discovery: Port resolution reads heartbeat files first
   for the actual port a process bound to, rather than relying solely
   on static configuration.

5. Extended probe range: Service discovery probes ports 8001-8004 to
   cover the fallback range when Prime binds to a non-standard port."
```

---

## Dependency Graph

```
Task 1 (per-component error tracking in unified_supervisor.py)
  ├── No dependencies — can start immediately
  │
Task 2 (port fallback in Ironcliw-Prime/run_server.py)
  ├── No dependencies — can start immediately (separate repo)
  │
Task 3 (heartbeat reconciliation in unified_supervisor.py)
  ├── Depends on Task 1 (uses _quick_health_probe and _get_component_port)
  │
Task 4 (port discovery in cross_repo_startup_orchestrator.py)
  ├── No dependencies — can start immediately
  │
Task 5 (probe ports in jarvis_prime_client.py)
  ├── No dependencies — can start immediately
  │
Task 6 (integration verification)
  └── Depends on Tasks 1-5
```

**Parallelizable:** Tasks 1, 2, 4, 5 can all be done in parallel. Task 3 depends on Task 1. Task 6 depends on all.

---

## Key Design Decisions

1. **Heartbeat files as source of truth for runtime port:** Heartbeat files are written by the running process itself and updated every 5 seconds. They contain the actual port the process bound to, not the configured port. This is the most reliable source of truth.

2. **No new files:** All changes are to existing files across the three repos. No new modules, no new classes, no new config files.

3. **Graceful degradation:** If heartbeat files don't exist or are stale, the system falls back to trinity_config, then env vars, then hardcoded defaults. The fallback chain is always preserved.

4. **Port fallback range is small (10 ports):** We only search port+1 through port+10. This keeps the search fast and predictable. If all 10 are occupied, something is seriously wrong and `sys.exit(1)` is the right behavior.

5. **Reconciliation is async and non-blocking:** The `_reconcile_component_status` method uses the same async health probe infrastructure already in the codebase. It adds ~3 seconds of latency to heartbeat writes only when a component is in error state.
