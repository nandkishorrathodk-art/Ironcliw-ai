# Unified Monolith Verification Checklist (v111.1)

## Overview

This checklist verifies the Unified Monolith Refactor is working correctly.
The goal: `python3 run_supervisor.py` starts everything in a single process.

## Pre-Flight Checks

### 1. Code Integrity
- [ ] `run_supervisor.py` syntax valid
- [ ] `backend/core/async_system_manager.py` syntax valid
- [ ] `backend/supervisor/cross_repo_startup_orchestrator.py` syntax valid
- [ ] Integration tests pass (target: 28+ passed)

### 2. Environment
- [ ] `Ironcliw_IN_PROCESS_MODE` defaults to `true`
- [ ] No conflicting supervisor locks exist
- [ ] Service registry directory exists (`~/.jarvis/registry/`)

## Startup Verification

### 3. Single Process Verification
```bash
# Start the system
python3 run_supervisor.py &

# Wait for startup
sleep 10

# Verify single process (should show ONLY ONE Python process for Ironcliw)
ps aux | grep -E "run_supervisor|uvicorn" | grep -v grep

# Expected: ONE process with both run_supervisor and uvicorn in same PID
```

### 4. In-Process Backend Verification
```bash
# Check that backend is running on expected port
curl -s http://localhost:8010/health | jq .

# Expected: {"status": "healthy", ...}
```

### 5. Service Registry Verification
```bash
# Check jarvis-body is registered
cat ~/.jarvis/registry/services.json | jq '.["jarvis-body"]'

# Expected:
# {
#   "service_name": "jarvis-body",
#   "mode": "in-process",
#   "unified_monolith": true,
#   ...
# }
```

### 6. Cross-Repo Discovery
```bash
# Verify jarvis-body is discoverable (metadata should show in-process mode)
cat ~/.jarvis/registry/services.json | jq '.["jarvis-body"].metadata'

# Expected: {"mode": "in-process", "unified_monolith": true, ...}
```

## Shutdown Verification

### 7. Graceful Shutdown (Ctrl+C)
```bash
# Send SIGINT (Ctrl+C equivalent)
kill -INT $(pgrep -f run_supervisor)

# Watch for graceful shutdown messages:
# - "[v111.0] Stopping backend..."
# - "[v111.1] jarvis-body deregistered from service registry"
# - "Backend stopped gracefully"
```

### 8. Service Deregistration
```bash
# After shutdown, verify jarvis-body is removed from registry
cat ~/.jarvis/registry/services.json | jq 'has("jarvis-body")'

# Expected: false (service should be deregistered)
```

### 9. No Orphan Processes
```bash
# Verify no zombie processes
ps aux | grep -E "python.*jarvis|uvicorn" | grep -v grep

# Expected: No output (all processes cleaned up)
```

## Signal Handling Verification

### 10. Escalating Shutdown
```bash
# First Ctrl+C: Graceful shutdown (10s timeout)
# Second Ctrl+C: Fast shutdown (3s timeout)
# Third Ctrl+C: Immediate exit (os._exit)

# Test by sending multiple SIGINTs in quick succession
```

## Performance Verification

### 11. Startup Time
```bash
# Time from start to "Backend started in-process"
# Target: < 30 seconds for backend startup
```

### 12. Memory Usage
```bash
# Single process should use less memory than subprocess model
ps aux | grep run_supervisor | awk '{print $6 " KB"}'

# Compare with historical subprocess model (should be lower)
```

## Success Criteria

| Criterion | Expected | Actual |
|-----------|----------|--------|
| Single Python process | Yes | |
| Backend health endpoint | 200 OK | |
| jarvis-body in registry | Yes | |
| In-process mode metadata | Yes | |
| Graceful shutdown | Clean | |
| No orphan processes | Zero | |
| Startup time | < 30s | |

## Troubleshooting

### Backend Won't Start
1. Check port 8010 is free: `lsof -i :8010`
2. Check for import errors in logs
3. Verify `backend/main.py` is import-safe

### Service Registry Issues
1. Check directory exists: `ls -la ~/.jarvis/registry/`
2. Check lock file: `ls -la ~/.jarvis/registry/services.json.lock`
3. Remove stale lock: `rm ~/.jarvis/registry/services.json.lock`

### Shutdown Hangs
1. Check for blocking operations
2. Verify signal handlers installed
3. Force kill: `kill -9 $(pgrep -f run_supervisor)`

---

## Verification Script

Run this automated verification:

```bash
#!/bin/bash
# unified_monolith_verify.sh

echo "=== Unified Monolith Verification ==="

# 1. Syntax check
echo -n "1. Syntax check... "
python3 -m py_compile run_supervisor.py && echo "OK" || echo "FAIL"

# 2. Integration tests
echo -n "2. Integration tests... "
python3 -m pytest tests/integration/test_unified_monolith.py -q 2>/dev/null | tail -1

# 3. Environment
echo -n "3. In-process mode default... "
python3 -c "import os; print('OK' if os.getenv('Ironcliw_IN_PROCESS_MODE', 'true').lower() == 'true' else 'FAIL')"

# 4. Registry directory
echo -n "4. Registry directory... "
[[ -d ~/.jarvis/registry ]] && echo "OK" || echo "MISSING (will be created)"

echo ""
echo "=== Pre-flight complete ==="
echo "Run 'python3 run_supervisor.py' to start the system"
```

---

*Last Updated: v111.1 - Cross-Repo Integration*
