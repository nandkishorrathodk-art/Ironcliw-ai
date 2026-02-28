# Enterprise-Grade System Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Harden the Ironcliw unified supervisor into an enterprise-grade control plane with deterministic lifecycle management, startup reliability with autonomous recovery, and comprehensive operational visibility through CLI dashboards.

**Architecture:** The implementation adds missing CLI monitor commands (`--monitor-prime`, `--monitor-reactor`, `--monitor-trinity`), extends IPC status responses to include Trinity/Invincible data, adds a `--check-only` dry-run mode, and fixes the dashboard `update_component` signature. All changes are made in `unified_supervisor.py` with zero new files, using existing config/env patterns.

**Tech Stack:** Python 3.9+, asyncio, Unix domain sockets (IPC), ANSI terminal formatting, aiohttp (HTTP health checks)

---

## Task Overview

| # | Task | Files | Description |
|---|------|-------|-------------|
| 1 | Extend IPC status with Trinity data | unified_supervisor.py | Add invincible_node status to IPC response |
| 2 | Add --monitor-prime CLI | unified_supervisor.py | Prime-specific dashboard with health checks |
| 3 | Add --monitor-reactor CLI | unified_supervisor.py | Reactor-specific dashboard with health checks |
| 4 | Add --monitor-trinity CLI | unified_supervisor.py | Unified Trinity dashboard (Prime + Reactor + Invincible) |
| 5 | Fallback health when kernel down | unified_supervisor.py | Direct HTTP health checks when IPC unavailable |
| 6 | Verify --check-only completeness | unified_supervisor.py | Ensure check-only validates Trinity repos |

---

## Task 1: Extend IPC Status with Invincible Node Data

**Files:**
- Modify: `unified_supervisor.py:59653-59702` (_ipc_status method)

**Step 1: Read the current _ipc_status method**

The current method already includes `trinity` status but is missing `invincible_node` top-level field.

**Step 2: Add invincible_node to IPC status response**

In `_ipc_status()`, add after line 59700 (after agi_os block):

```python
        # v201.1: Invincible Node status for CLI dashboards
        status["invincible_node"] = {
            "enabled": self.config.invincible_node_enabled,
            "instance_name": self.config.invincible_node_instance_name,
            "port": self.config.invincible_node_port,
            "static_ip_name": self.config.invincible_node_static_ip_name,
            "status": getattr(self, 'invincible_node_status', {}),
        }
```

**Step 3: Verify the change**

Run: `python unified_supervisor.py --status` (with kernel running)
Expected: Response includes `invincible_node` key with configuration

**Step 4: Commit**

```bash
git add unified_supervisor.py
git commit -m "$(cat <<'EOF'
feat(ipc): Add invincible_node to IPC status response

v201.1: Include Invincible Node configuration and status in IPC
status response for CLI dashboard consumption.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add --monitor-prime CLI Command

**Files:**
- Modify: `unified_supervisor.py` (argparse section ~61974, handlers ~62440)

**Step 1: Add argparse argument for --monitor-prime**

After the `--reactor-path` argument (around line 61989), add:

```python
    trinity.add_argument(
        "--monitor-prime",
        action="store_true",
        help="Display J-Prime component status dashboard",
    )
```

**Step 2: Create handle_monitor_prime function**

Add after `handle_cloud_monitor_logs()` (around line 62700):

```python
async def handle_monitor_prime() -> int:
    """
    Handle --monitor-prime command: Display J-Prime status dashboard.

    v201.1: Shows J-Prime status whether kernel is running or not.
    When kernel is running, uses IPC. When down, does direct HTTP check.
    """
    # ANSI colors
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    # Box drawing
    def box_line(text: str, width: int = 70) -> str:
        padded = f" {text}".ljust(width - 2)
        return f"\u2551{padded}\u2551"

    def header(width: int = 70) -> str:
        return f"\u2554{'═' * (width - 2)}\u2557"

    def footer(width: int = 70) -> str:
        return f"\u255a{'═' * (width - 2)}\u255d"

    def separator(width: int = 70) -> str:
        return f"\u2560{'═' * (width - 2)}\u2563"

    print()
    print(f"{BOLD}{BLUE}" + header() + RESET)
    print(f"{BOLD}{BLUE}" + box_line("🧠  J-PRIME STATUS MONITOR") + RESET)
    print(f"{BOLD}{BLUE}" + separator() + RESET)

    # Get port from environment (same source as TrinityIntegrator)
    prime_port = int(os.getenv("TRINITY_JPRIME_PORT", "8000"))
    prime_host = os.getenv("TRINITY_JPRIME_HOST", "localhost")

    # Try IPC first (kernel running)
    socket_path = Path.home() / ".jarvis" / "locks" / "kernel.sock"
    trinity_status = None
    kernel_running = False

    if socket_path.exists():
        try:
            reader, writer = await asyncio.open_unix_connection(str(socket_path))
            request = json.dumps({"command": "status"}) + "\n"
            writer.write(request.encode())
            await writer.drain()
            response_data = await asyncio.wait_for(reader.readline(), timeout=5.0)
            response = json.loads(response_data.decode())
            writer.close()
            await writer.wait_closed()

            if response.get("success"):
                result = response.get("result", {})
                trinity_status = result.get("trinity", {})
                kernel_running = True
        except Exception:
            pass

    # Extract Prime status from IPC or do direct check
    prime_data = None
    if trinity_status:
        prime_data = trinity_status.get("components", {}).get("jarvis-prime", {})

    # Display kernel status
    if kernel_running:
        print(box_line(f"Kernel:       {GREEN}Running{RESET}"))
    else:
        print(box_line(f"Kernel:       {YELLOW}Not running{RESET} (direct health check)"))

    print(separator())

    # Display Prime configuration
    print(box_line(f"Host:         {prime_host}"))
    print(box_line(f"Port:         {prime_port}"))

    if prime_data:
        # Use IPC data
        configured = prime_data.get("configured", False)
        state = prime_data.get("state", "unknown")
        running = prime_data.get("running", False)
        healthy = prime_data.get("healthy", False)
        pid = prime_data.get("pid")
        repo_path = prime_data.get("repo_path")
        restart_count = prime_data.get("restart_count", 0)

        print(box_line(f"Configured:   {GREEN}Yes{RESET}" if configured else f"Configured:   {RED}No{RESET}"))
        print(box_line(f"State:        {state}"))
        print(box_line(f"Running:      {GREEN}Yes{RESET}" if running else f"Running:      {RED}No{RESET}"))
        print(box_line(f"Healthy:      {GREEN}Yes{RESET}" if healthy else f"Healthy:      {RED}No{RESET}"))
        if pid:
            print(box_line(f"PID:          {pid}"))
        if repo_path:
            print(box_line(f"Repo:         {DIM}{repo_path}{RESET}"))
        if restart_count > 0:
            print(box_line(f"Restarts:     {YELLOW}{restart_count}{RESET}"))

    else:
        # Direct HTTP health check
        print(separator())
        print(box_line(f"{CYAN}Direct Health Check{RESET}"))

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                url = f"http://{prime_host}:{prime_port}/health"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        health = await resp.json()
                        print(box_line(f"Reachable:    {GREEN}Yes{RESET}"))
                        status = health.get("status", "unknown")
                        if status == "healthy":
                            print(box_line(f"Status:       {GREEN}{status}{RESET}"))
                        else:
                            print(box_line(f"Status:       {YELLOW}{status}{RESET}"))
                        if health.get("model_loaded"):
                            print(box_line(f"Model:        {GREEN}Loaded{RESET}"))
                        if health.get("active_model"):
                            print(box_line(f"Model:        {health['active_model']}"))
                    else:
                        print(box_line(f"Reachable:    {YELLOW}Yes (HTTP {resp.status}){RESET}"))
        except aiohttp.ClientConnectorError:
            print(box_line(f"Reachable:    {RED}No (connection refused){RESET}"))
        except asyncio.TimeoutError:
            print(box_line(f"Reachable:    {RED}No (timeout){RESET}"))
        except ImportError:
            # Fallback without aiohttp
            import urllib.request
            import urllib.error
            try:
                url = f"http://{prime_host}:{prime_port}/health"
                req = urllib.request.Request(url, method='GET')
                with urllib.request.urlopen(req, timeout=5) as resp:
                    print(box_line(f"Reachable:    {GREEN}Yes{RESET}"))
            except urllib.error.URLError:
                print(box_line(f"Reachable:    {RED}No{RESET}"))
            except Exception:
                print(box_line(f"Reachable:    {RED}Unknown{RESET}"))
        except Exception as e:
            print(box_line(f"Error:        {RED}{e}{RESET}"))

    print(footer())

    # Quick actions
    print()
    print(f"{BOLD}Quick Actions:{RESET}")
    print(f"  • Full status:  python unified_supervisor.py --status")
    print(f"  • Start kernel: python unified_supervisor.py")
    print(f"  • Health check: curl http://{prime_host}:{prime_port}/health")
    print()

    return 0
```

**Step 3: Wire up the handler in main()**

In the `main()` function (around line 62950-62970), add after the `--monitor-logs` check:

```python
    # Handle --monitor-prime
    if args.monitor_prime:
        return await handle_monitor_prime()
```

**Step 4: Test the command**

Run: `python unified_supervisor.py --monitor-prime`
Expected: Boxed dashboard showing Prime status

**Step 5: Commit**

```bash
git add unified_supervisor.py
git commit -m "$(cat <<'EOF'
feat(cli): Add --monitor-prime dashboard command

v201.1: New CLI command to display J-Prime component status.
Works with or without kernel running (falls back to direct HTTP).

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add --monitor-reactor CLI Command

**Files:**
- Modify: `unified_supervisor.py` (argparse, handlers)

**Step 1: Add argparse argument**

After `--monitor-prime`:

```python
    trinity.add_argument(
        "--monitor-reactor",
        action="store_true",
        help="Display Reactor-Core component status dashboard",
    )
```

**Step 2: Create handle_monitor_reactor function**

Similar pattern to Prime, but for Reactor port (default 8090):

```python
async def handle_monitor_reactor() -> int:
    """
    Handle --monitor-reactor command: Display Reactor-Core status dashboard.

    v201.1: Shows Reactor status whether kernel is running or not.
    """
    # ANSI colors
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    def box_line(text: str, width: int = 70) -> str:
        padded = f" {text}".ljust(width - 2)
        return f"\u2551{padded}\u2551"

    def header(width: int = 70) -> str:
        return f"\u2554{'═' * (width - 2)}\u2557"

    def footer(width: int = 70) -> str:
        return f"\u255a{'═' * (width - 2)}\u255d"

    def separator(width: int = 70) -> str:
        return f"\u2560{'═' * (width - 2)}\u2563"

    print()
    print(f"{BOLD}{BLUE}" + header() + RESET)
    print(f"{BOLD}{BLUE}" + box_line("⚡  REACTOR-CORE STATUS MONITOR") + RESET)
    print(f"{BOLD}{BLUE}" + separator() + RESET)

    # Get port from environment
    reactor_port = int(os.getenv("TRINITY_REACTOR_PORT", "8090"))
    reactor_host = os.getenv("TRINITY_REACTOR_HOST", "localhost")

    # Try IPC first
    socket_path = Path.home() / ".jarvis" / "locks" / "kernel.sock"
    trinity_status = None
    kernel_running = False

    if socket_path.exists():
        try:
            reader, writer = await asyncio.open_unix_connection(str(socket_path))
            request = json.dumps({"command": "status"}) + "\n"
            writer.write(request.encode())
            await writer.drain()
            response_data = await asyncio.wait_for(reader.readline(), timeout=5.0)
            response = json.loads(response_data.decode())
            writer.close()
            await writer.wait_closed()

            if response.get("success"):
                result = response.get("result", {})
                trinity_status = result.get("trinity", {})
                kernel_running = True
        except Exception:
            pass

    reactor_data = None
    if trinity_status:
        reactor_data = trinity_status.get("components", {}).get("reactor-core", {})

    if kernel_running:
        print(box_line(f"Kernel:       {GREEN}Running{RESET}"))
    else:
        print(box_line(f"Kernel:       {YELLOW}Not running{RESET} (direct health check)"))

    print(separator())
    print(box_line(f"Host:         {reactor_host}"))
    print(box_line(f"Port:         {reactor_port}"))

    if reactor_data:
        configured = reactor_data.get("configured", False)
        state = reactor_data.get("state", "unknown")
        running = reactor_data.get("running", False)
        healthy = reactor_data.get("healthy", False)
        pid = reactor_data.get("pid")
        repo_path = reactor_data.get("repo_path")
        restart_count = reactor_data.get("restart_count", 0)

        print(box_line(f"Configured:   {GREEN}Yes{RESET}" if configured else f"Configured:   {RED}No{RESET}"))
        print(box_line(f"State:        {state}"))
        print(box_line(f"Running:      {GREEN}Yes{RESET}" if running else f"Running:      {RED}No{RESET}"))
        print(box_line(f"Healthy:      {GREEN}Yes{RESET}" if healthy else f"Healthy:      {RED}No{RESET}"))
        if pid:
            print(box_line(f"PID:          {pid}"))
        if repo_path:
            print(box_line(f"Repo:         {DIM}{repo_path}{RESET}"))
        if restart_count > 0:
            print(box_line(f"Restarts:     {YELLOW}{restart_count}{RESET}"))
    else:
        print(separator())
        print(box_line(f"{CYAN}Direct Health Check{RESET}"))

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                url = f"http://{reactor_host}:{reactor_port}/health"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        health = await resp.json()
                        print(box_line(f"Reachable:    {GREEN}Yes{RESET}"))
                        status = health.get("status", "unknown")
                        if status == "healthy":
                            print(box_line(f"Status:       {GREEN}{status}{RESET}"))
                        else:
                            print(box_line(f"Status:       {YELLOW}{status}{RESET}"))
                    else:
                        print(box_line(f"Reachable:    {YELLOW}Yes (HTTP {resp.status}){RESET}"))
        except Exception:
            print(box_line(f"Reachable:    {RED}No{RESET}"))

    print(footer())

    print()
    print(f"{BOLD}Quick Actions:{RESET}")
    print(f"  • Full status:  python unified_supervisor.py --status")
    print(f"  • Health check: curl http://{reactor_host}:{reactor_port}/health")
    print()

    return 0
```

**Step 3: Wire up in main()**

```python
    if args.monitor_reactor:
        return await handle_monitor_reactor()
```

**Step 4: Commit**

```bash
git add unified_supervisor.py
git commit -m "$(cat <<'EOF'
feat(cli): Add --monitor-reactor dashboard command

v201.1: New CLI command to display Reactor-Core component status.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Add --monitor-trinity Unified Dashboard

**Files:**
- Modify: `unified_supervisor.py`

**Step 1: Add argparse argument**

```python
    trinity.add_argument(
        "--monitor-trinity",
        action="store_true",
        help="Display unified Trinity status dashboard (Prime + Reactor + Invincible)",
    )
```

**Step 2: Create handle_monitor_trinity function**

This is the comprehensive dashboard combining all three:

```python
async def handle_monitor_trinity() -> int:
    """
    Handle --monitor-trinity command: Unified Trinity dashboard.

    v201.1: Shows Prime, Reactor, and Invincible Node status in one view.
    """
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    def box_line(text: str, width: int = 70) -> str:
        padded = f" {text}".ljust(width - 2)
        return f"\u2551{padded}\u2551"

    def header(width: int = 70) -> str:
        return f"\u2554{'═' * (width - 2)}\u2557"

    def footer(width: int = 70) -> str:
        return f"\u255a{'═' * (width - 2)}\u255d"

    def separator(width: int = 70) -> str:
        return f"\u2560{'═' * (width - 2)}\u2563"

    def section_header(title: str, width: int = 70) -> str:
        return f"\u2560{'─' * 2} {title} {'─' * (width - len(title) - 6)}\u2563"

    print()
    print(f"{BOLD}{BLUE}" + header() + RESET)
    print(f"{BOLD}{BLUE}" + box_line("🔺  TRINITY UNIFIED STATUS MONITOR") + RESET)
    print(f"{BOLD}{BLUE}" + separator() + RESET)

    # Try IPC
    socket_path = Path.home() / ".jarvis" / "locks" / "kernel.sock"
    ipc_result = None
    kernel_running = False

    if socket_path.exists():
        try:
            reader, writer = await asyncio.open_unix_connection(str(socket_path))
            request = json.dumps({"command": "status"}) + "\n"
            writer.write(request.encode())
            await writer.drain()
            response_data = await asyncio.wait_for(reader.readline(), timeout=5.0)
            response = json.loads(response_data.decode())
            writer.close()
            await writer.wait_closed()

            if response.get("success"):
                ipc_result = response.get("result", {})
                kernel_running = True
        except Exception:
            pass

    # Kernel status
    if kernel_running:
        state = ipc_result.get("state", "unknown")
        uptime = ipc_result.get("uptime_seconds", 0)
        uptime_str = f"{int(uptime // 60)}m {int(uptime % 60)}s"
        print(box_line(f"Kernel:       {GREEN}{state}{RESET} (uptime: {uptime_str})"))
    else:
        print(box_line(f"Kernel:       {YELLOW}Not running{RESET}"))

    trinity_status = ipc_result.get("trinity", {}) if ipc_result else {}
    invincible_status = ipc_result.get("invincible_node", {}) if ipc_result else {}

    # Prime section
    print(section_header("J-Prime"))
    prime_data = trinity_status.get("components", {}).get("jarvis-prime", {})
    prime_port = int(os.getenv("TRINITY_JPRIME_PORT", "8000"))

    if prime_data:
        running = prime_data.get("running", False)
        healthy = prime_data.get("healthy", False)
        state = prime_data.get("state", "unknown")
        status_icon = f"{GREEN}●{RESET}" if healthy else (f"{YELLOW}●{RESET}" if running else f"{RED}●{RESET}")
        print(box_line(f"{status_icon} State: {state}  |  Port: {prime_port}  |  PID: {prime_data.get('pid', '-')}"))
    else:
        print(box_line(f"{DIM}Not configured or kernel not running{RESET}"))

    # Reactor section
    print(section_header("Reactor-Core"))
    reactor_data = trinity_status.get("components", {}).get("reactor-core", {})
    reactor_port = int(os.getenv("TRINITY_REACTOR_PORT", "8090"))

    if reactor_data:
        running = reactor_data.get("running", False)
        healthy = reactor_data.get("healthy", False)
        state = reactor_data.get("state", "unknown")
        status_icon = f"{GREEN}●{RESET}" if healthy else (f"{YELLOW}●{RESET}" if running else f"{RED}●{RESET}")
        print(box_line(f"{status_icon} State: {state}  |  Port: {reactor_port}  |  PID: {reactor_data.get('pid', '-')}"))
    else:
        print(box_line(f"{DIM}Not configured or kernel not running{RESET}"))

    # Invincible Node section
    print(section_header("Invincible Node"))
    config = SystemKernelConfig()

    if config.invincible_node_enabled:
        inv_enabled = invincible_status.get("enabled", False)
        inv_status_data = invincible_status.get("status", {})
        if inv_status_data:
            gcp_status = inv_status_data.get("gcp_status", "UNKNOWN")
            static_ip = inv_status_data.get("static_ip", "N/A")
            health = inv_status_data.get("health", {})
            reachable = health.get("reachable", False)
            ready = health.get("ready_for_inference", False)

            if gcp_status == "RUNNING" and ready:
                status_icon = f"{GREEN}●{RESET}"
            elif gcp_status == "RUNNING":
                status_icon = f"{YELLOW}●{RESET}"
            else:
                status_icon = f"{RED}●{RESET}"

            print(box_line(f"{status_icon} GCP: {gcp_status}  |  IP: {static_ip}  |  Inference: {'Ready' if ready else 'Not ready'}"))
        else:
            print(box_line(f"{YELLOW}●{RESET} Enabled but no status data (run --monitor for full check)"))
    else:
        print(box_line(f"{DIM}Disabled{RESET}"))

    # Overall health summary
    print(separator())
    all_healthy = True
    if prime_data and not prime_data.get("healthy"):
        all_healthy = False
    if reactor_data and not reactor_data.get("healthy"):
        all_healthy = False

    if all_healthy and kernel_running:
        print(box_line(f"{GREEN}✓ Trinity System: All components healthy{RESET}"))
    elif kernel_running:
        print(box_line(f"{YELLOW}⚠ Trinity System: Some components degraded{RESET}"))
    else:
        print(box_line(f"{DIM}Cannot determine health - kernel not running{RESET}"))

    print(footer())

    print()
    print(f"{BOLD}Component Dashboards:{RESET}")
    print(f"  • J-Prime:     python unified_supervisor.py --monitor-prime")
    print(f"  • Reactor:     python unified_supervisor.py --monitor-reactor")
    print(f"  • Invincible:  python unified_supervisor.py --monitor")
    print()

    return 0
```

**Step 3: Wire up in main()**

```python
    if args.monitor_trinity:
        return await handle_monitor_trinity()
```

**Step 4: Commit**

```bash
git add unified_supervisor.py
git commit -m "$(cat <<'EOF'
feat(cli): Add --monitor-trinity unified dashboard

v201.1: Unified dashboard showing Prime, Reactor, and Invincible Node
status in a single view with health summary.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Add Fallback Health Check Helper

**Files:**
- Modify: `unified_supervisor.py`

**Step 1: Create shared health check utility**

Add a reusable function for direct HTTP health checks (around line 62200, before the monitor handlers):

```python
async def _direct_health_check(host: str, port: int, timeout: float = 5.0) -> Dict[str, Any]:
    """
    v201.1: Perform direct HTTP health check (used when kernel not running).

    Returns:
        Dict with 'reachable', 'status', 'data' keys
    """
    result = {"reachable": False, "status": "unknown", "data": {}}

    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            url = f"http://{host}:{port}/health"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                result["reachable"] = True
                result["status"] = "healthy" if resp.status == 200 else f"http_{resp.status}"
                if resp.status == 200:
                    try:
                        result["data"] = await resp.json()
                    except Exception:
                        result["data"] = {"raw": await resp.text()}
    except ImportError:
        # Fallback to urllib
        import urllib.request
        import urllib.error
        try:
            url = f"http://{host}:{port}/health"
            req = urllib.request.Request(url, method='GET')
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result["reachable"] = True
                result["status"] = "healthy" if resp.status == 200 else f"http_{resp.status}"
        except urllib.error.URLError:
            result["status"] = "unreachable"
        except Exception:
            result["status"] = "error"
    except Exception as e:
        result["status"] = f"error: {e}"

    return result
```

**Step 2: Refactor monitor handlers to use it**

Update `handle_monitor_prime` and `handle_monitor_reactor` to use `_direct_health_check()` for cleaner code.

**Step 3: Commit**

```bash
git add unified_supervisor.py
git commit -m "$(cat <<'EOF'
refactor(cli): Add shared _direct_health_check utility

v201.1: Reusable async health check for CLI dashboards when
kernel is not running.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Verify --check-only Validates Trinity

**Files:**
- Modify: `unified_supervisor.py:62206-62431` (handle_check_only)

**Step 1: Review current check-only implementation**

The current implementation checks Docker, GCP, ports. Verify Trinity repo paths are checked.

**Step 2: Add Trinity repo validation if missing**

In `handle_check_only()`, add after the port checks (around line 62410):

```python
    # Trinity repo validation
    print(f"{BOLD}{BLUE}║{RESET}")
    print(f"{BOLD}{BLUE}║{RESET}  {CYAN}Trinity Repositories{RESET}")

    prime_path = config.jarvis_prime_path or os.environ.get("Ironcliw_PRIME_PATH", "")
    reactor_path = config.reactor_core_path or os.environ.get("REACTOR_CORE_PATH", "")

    if prime_path:
        prime_exists = Path(prime_path).exists()
        print(f"{BOLD}{BLUE}║{RESET}    {check_mark(prime_exists)} J-Prime: {prime_path}")
        if not prime_exists:
            warnings.append("J-Prime path does not exist")
    else:
        print(f"{BOLD}{BLUE}║{RESET}    {warn_mark()} J-Prime: Not configured")
        warnings.append("J-Prime path not set (Ironcliw_PRIME_PATH)")

    if reactor_path:
        reactor_exists = Path(reactor_path).exists()
        print(f"{BOLD}{BLUE}║{RESET}    {check_mark(reactor_exists)} Reactor: {reactor_path}")
        if not reactor_exists:
            warnings.append("Reactor path does not exist")
    else:
        print(f"{BOLD}{BLUE}║{RESET}    {warn_mark()} Reactor: Not configured")
        warnings.append("Reactor path not set (REACTOR_CORE_PATH)")
```

**Step 3: Test**

Run: `python unified_supervisor.py --check-only`
Expected: Trinity repo paths shown in output

**Step 4: Commit**

```bash
git add unified_supervisor.py
git commit -m "$(cat <<'EOF'
feat(cli): Add Trinity repo validation to --check-only

v201.1: Pre-flight check now validates J-Prime and Reactor-Core
repository paths exist.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Verification Checklist

After all tasks complete, verify:

1. [ ] `python unified_supervisor.py --status` includes `trinity` and `invincible_node`
2. [ ] `python unified_supervisor.py --monitor-prime` shows Prime dashboard
3. [ ] `python unified_supervisor.py --monitor-reactor` shows Reactor dashboard
4. [ ] `python unified_supervisor.py --monitor-trinity` shows unified dashboard
5. [ ] `python unified_supervisor.py --monitor` shows Invincible Node (already exists)
6. [ ] `python unified_supervisor.py --check-only` validates Trinity repos
7. [ ] All dashboards work when kernel is NOT running (direct HTTP fallback)

---

## Summary

This plan implements the P1 enterprise hardening requirements:

| Requirement | Task | Status |
|-------------|------|--------|
| Unified control plane | All tasks | One file, deterministic |
| IPC includes Trinity/Invincible | Task 1 | Extended status response |
| CLI dashboards (Prime/Reactor/Trinity) | Tasks 2-4 | Boxed dashboards |
| Works without kernel | Task 5 | Direct HTTP fallback |
| --check-only validates Trinity | Task 6 | Pre-flight check |

All changes in `unified_supervisor.py`, zero new files, using existing config/env patterns.
