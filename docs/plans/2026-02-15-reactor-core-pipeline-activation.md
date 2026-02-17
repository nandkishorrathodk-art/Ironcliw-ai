# Reactor Core Pipeline Activation — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Activate the end-to-end training pipeline: telemetry -> ingestion -> training -> deployment gate -> deploy -> probation -> feedback, across JARVIS-AI-Agent, reactor-core, and jarvis-prime.

**Architecture:** Supervisor-driven activation. The unified_supervisor coordinates the loop via ReactorCoreClient. Reactor-Core owns job execution (survives supervisor restarts). Six specific wiring disconnects are fixed, then the pipeline is hardened with circuit breakers, quality-weighted triggers, deployment gates, post-deployment probation, resource awareness, and full observability via TrinityEventBus correlation IDs.

**Tech Stack:** Python 3.9+, pytest + pytest-asyncio, aiohttp, FastAPI (reactor-core), llama-cpp-python (deployment gate), existing CircuitBreaker/TrinityEventBus/MemoryQuantizer/Gatekeeper/DeadManSwitch patterns.

**Repos:**
- JARVIS: `/Users/djrussell23/Documents/repos/JARVIS-AI-Agent`
- Reactor: `/Users/djrussell23/Documents/repos/reactor-core`
- Prime: `/Users/djrussell23/Documents/repos/jarvis-prime`

---

## Phase 1: Verify the Wiring (Manual Validation)

### Task 1: Verify Reactor-Core API is Listening

**Purpose:** Confirm Reactor-Core's FastAPI server accepts requests on port 8090 before changing any code.

**Step 1: Check if Reactor-Core is running**

Run:
```bash
curl -s http://localhost:8090/health | python3 -m json.tool
```
Expected: JSON with `status`, `phase`, `training_ready`, `trinity_connected` fields.

If connection refused: Reactor-Core is not running. Start it:
```bash
cd /Users/djrussell23/Documents/repos/reactor-core && python3 run_reactor.py --port 8090 &
```
Wait 30s, retry curl.

**Step 2: Verify the experience streaming endpoint exists**

Run:
```bash
curl -s -X POST http://localhost:8090/api/v1/experiences/stream \
  -H "Content-Type: application/json" \
  -d '{"experience": {"event_type": "INTERACTION", "user_input": "test query", "assistant_output": "test response", "source": "JARVIS_BODY", "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"}, "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'", "source": "jarvis_agent"}' | python3 -m json.tool
```
Expected: `{"accepted": true, "count": 1}`

**Step 3: Verify the training trigger endpoint exists**

Run:
```bash
curl -s -X POST http://localhost:8090/api/v1/train \
  -H "Content-Type: application/json" \
  -d '{"experience_count": 0, "priority": "normal", "sources": ["jarvis_experience"], "metadata": {}, "triggered_by": "manual_test"}' | python3 -m json.tool
```
Expected: JSON with `job_id` field, or 409 if a job is already running.

**Step 4: Verify the wrong endpoints return 404**

Run:
```bash
curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8090/api/experiences/stream -d '{}'
curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8090/api/training/trigger -d '{}'
```
Expected: `404` for both (or `422`). These are the WRONG paths that ReactorCoreClient currently uses.

**Step 5: Document results**

Record which endpoints responded, which failed, and any unexpected errors. This informs Task 2.

---

## Phase 2: Fix the 6 Core Disconnects

### Task 2: Fix Endpoint Path Mismatches (Disconnect #6)

**Files:**
- Modify: `backend/clients/reactor_core_client.py` (lines 714, 579, 732)
- Test: `tests/unit/backend/clients/test_reactor_core_client.py` (create)

**Step 1: Create test file with path validation tests**

Create: `tests/unit/backend/clients/test_reactor_core_client.py`

```python
"""Tests for ReactorCoreClient endpoint paths and core functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

# We test that the client calls _request with correct paths


@pytest.fixture
def mock_session():
    """Mock aiohttp session."""
    session = AsyncMock()
    response = AsyncMock()
    response.status = 200
    response.json = AsyncMock(return_value={"accepted": True, "count": 1})
    session.get = AsyncMock(return_value=response)
    session.post = AsyncMock(return_value=response)
    # Make context manager work
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=False)
    return session


@pytest.fixture
def client(mock_session):
    """Create a ReactorCoreClient with mocked session."""
    from backend.clients.reactor_core_client import ReactorCoreClient, ReactorCoreConfig
    config = ReactorCoreConfig()
    c = ReactorCoreClient.__new__(ReactorCoreClient)
    c.config = config
    c._session = mock_session
    c._is_online = True
    c._requests_sent = 0
    c._requests_failed = 0
    c._last_trigger_time = None
    c._training_triggers = 0
    c._callbacks = {}
    c._bridge_dir = None
    return c


class TestEndpointPaths:
    """Verify all endpoint paths match Reactor-Core's server.py routes."""

    async def test_stream_experience_uses_v1_path(self, client):
        """stream_experience must POST to /api/v1/experiences/stream."""
        with patch.object(client, '_request', new_callable=AsyncMock) as mock_req:
            mock_req.return_value = {"accepted": True, "count": 1}
            await client.stream_experience({"event_type": "INTERACTION"})
            mock_req.assert_called_once()
            call_args = mock_req.call_args
            assert call_args[0][0] == "POST"
            assert call_args[0][1] == "/api/v1/experiences/stream"

    async def test_trigger_training_uses_v1_train_path(self, client):
        """trigger_training must POST to /api/v1/train."""
        with patch.object(client, '_request', new_callable=AsyncMock) as mock_req:
            mock_req.return_value = {"job_id": "test-123", "status": "queued"}
            with patch.object(client, '_emit_event', new_callable=AsyncMock):
                with patch.object(client, '_write_bridge_event', new_callable=AsyncMock):
                    await client.trigger_training(experience_count=100, force=True)
            mock_req.assert_called_once()
            call_args = mock_req.call_args
            assert call_args[0][0] == "POST"
            assert call_args[0][1] == "/api/v1/train"

    async def test_get_experience_count_uses_v1_path(self, client):
        """get_experience_count must GET /api/v1/experiences/count."""
        with patch.object(client, '_request', new_callable=AsyncMock) as mock_req:
            mock_req.return_value = {"count": 42}
            result = await client.get_experience_count()
            assert result == 42
            mock_req.assert_called_once()
            call_args = mock_req.call_args
            assert call_args[0][0] == "GET"
            assert call_args[0][1] == "/api/v1/experiences/count"
```

**Step 2: Run tests to verify they fail**

Run:
```bash
cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent && python3 -m pytest tests/unit/backend/clients/test_reactor_core_client.py -v -x 2>&1 | head -60
```
Expected: FAIL — paths don't match (`/api/experiences/stream` != `/api/v1/experiences/stream`).

**Step 3: Fix the endpoint paths**

In `backend/clients/reactor_core_client.py`:

**Fix 1 — `stream_experience()` (~line 714):**
Change:
```python
            data = await self._request("POST", "/api/experiences/stream", json=payload)
```
To:
```python
            data = await self._request("POST", "/api/v1/experiences/stream", json=payload)
```

**Fix 2 — `trigger_training()` (~line 579):**
Change:
```python
            data = await self._request("POST", "/api/training/trigger", json=payload)
```
To:
```python
            data = await self._request("POST", "/api/v1/train", json=payload)
```

**Fix 3 — `get_experience_count()` (~line 732):**
Change:
```python
            data = await self._request("GET", "/api/experiences/count")
```
To:
```python
            data = await self._request("GET", "/api/v1/experiences/count")
```

**Step 4: Audit all remaining `_request()` calls**

Search the file for all `_request(` calls. For each one, verify the path matches an endpoint registered in `reactor-core/reactor_core/api/server.py`. Fix any mismatches found.

Paths to verify:
- `cancel_training()`: should be `/api/v1/jobs/{job_id}/cancel`
- `get_training_job()`: should be `/api/v1/jobs/{job_id}`
- `get_training_history()`: should be `/api/v1/jobs`
- Health check: `/health` (this one is correct)

**Step 5: Run tests to verify they pass**

Run:
```bash
cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent && python3 -m pytest tests/unit/backend/clients/test_reactor_core_client.py -v -x 2>&1 | head -60
```
Expected: All 3 tests PASS.

**Step 6: Commit**

```bash
cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent && git add backend/clients/reactor_core_client.py tests/unit/backend/clients/test_reactor_core_client.py && git commit -m "fix: correct endpoint path mismatches in ReactorCoreClient

ReactorCoreClient was sending to /api/experiences/stream (404) instead of
/api/v1/experiences/stream, and /api/training/trigger instead of /api/v1/train.
These path mismatches meant experiences never reached Reactor-Core and training
was never triggered despite the infrastructure being fully built.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Enrich Health Monitor with Training Readiness (Disconnect #2)

**Files:**
- Modify: `backend/clients/reactor_core_client.py` (health_check method ~lines 344-468, class __init__)
- Test: `tests/unit/backend/clients/test_reactor_core_client.py` (append)

**Step 1: Add tests for training readiness parsing**

Append to `tests/unit/backend/clients/test_reactor_core_client.py`:

```python
class TestHealthCheckEnrichment:
    """Verify health check parses training readiness from response body."""

    async def test_health_check_parses_training_ready(self, client, mock_session):
        """Health check should store training_ready from response JSON."""
        response = AsyncMock()
        response.status = 200
        response.json = AsyncMock(return_value={
            "status": "healthy",
            "phase": "ready",
            "training_ready": True,
            "trinity_connected": True,
        })
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = AsyncMock(return_value=response)

        # Initialize missing attributes for health_check
        client._last_health_check = None
        client._consecutive_failures = 0
        client._consecutive_successes = 0
        client._last_failure_reason = None
        client._current_timeout = 10.0
        client._health_check_interval_multiplier = 1.0
        client._training_ready = False
        client._reactor_phase = "unknown"
        client._trinity_connected = False

        result = await client.health_check()
        assert result is True
        assert client._training_ready is True
        assert client._reactor_phase == "ready"
        assert client._trinity_connected is True

    async def test_health_check_stores_phase_on_not_ready(self, client, mock_session):
        """Health check should store phase even when training not ready."""
        response = AsyncMock()
        response.status = 200
        response.json = AsyncMock(return_value={
            "status": "starting",
            "phase": "initializing",
            "training_ready": False,
            "trinity_connected": False,
        })
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = AsyncMock(return_value=response)

        client._last_health_check = None
        client._consecutive_failures = 0
        client._consecutive_successes = 0
        client._last_failure_reason = None
        client._current_timeout = 10.0
        client._health_check_interval_multiplier = 1.0
        client._training_ready = False
        client._reactor_phase = "unknown"
        client._trinity_connected = False

        result = await client.health_check()
        assert result is True
        assert client._training_ready is False
        assert client._reactor_phase == "initializing"

    async def test_is_training_ready_property(self, client):
        """is_training_ready should combine is_online and _training_ready."""
        client._training_ready = True
        assert client.is_training_ready is True
        client._is_online = False
        assert client.is_training_ready is False
```

**Step 2: Run tests to verify they fail**

Run:
```bash
cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent && python3 -m pytest tests/unit/backend/clients/test_reactor_core_client.py::TestHealthCheckEnrichment -v -x 2>&1 | head -40
```
Expected: FAIL — `_training_ready` attribute doesn't exist, `is_training_ready` property doesn't exist.

**Step 3: Add training readiness state to `__init__` and property**

In `backend/clients/reactor_core_client.py`, find the `__init__` method of `ReactorCoreClient`. Add these instance variables alongside the existing ones:

```python
        # Training readiness state (v2.1: enriched health monitoring)
        self._training_ready: bool = False
        self._reactor_phase: str = "unknown"
        self._trinity_connected: bool = False
```

Add a property after `is_online`:

```python
    @property
    def is_training_ready(self) -> bool:
        """Whether Reactor-Core is online AND training subsystem is ready."""
        return self._is_online and self._training_ready
```

**Step 4: Modify `health_check()` to parse response JSON**

In the `health_check()` method, after `if response.status == 200:` (around line 372), add JSON parsing before the existing success-tracking code:

```python
                if response.status == 200:
                    # v2.1: Parse training readiness from response body
                    try:
                        health_data = await response.json()
                        prev_phase = self._reactor_phase
                        self._training_ready = health_data.get("training_ready", False)
                        self._reactor_phase = health_data.get("phase", "unknown")
                        self._trinity_connected = health_data.get("trinity_connected", False)
                        # Log phase transitions
                        if prev_phase != self._reactor_phase:
                            logger.info(
                                f"[ReactorClient] Reactor-Core phase: {prev_phase} -> {self._reactor_phase} "
                                f"(training_ready={self._training_ready})"
                            )
                    except Exception:
                        pass  # Don't fail health check if JSON parsing fails

                    self._last_health_check = datetime.now()
                    # ... rest of existing success code unchanged
```

**Step 5: Run tests to verify they pass**

Run:
```bash
cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent && python3 -m pytest tests/unit/backend/clients/test_reactor_core_client.py::TestHealthCheckEnrichment -v -x 2>&1 | head -40
```
Expected: All 3 tests PASS.

**Step 6: Commit**

```bash
cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent && git add backend/clients/reactor_core_client.py tests/unit/backend/clients/test_reactor_core_client.py && git commit -m "feat: enrich ReactorCoreClient health check with training readiness

Parse training_ready, phase, and trinity_connected from Reactor-Core's
/health response JSON. Expose is_training_ready property. Log phase
transitions. Previously only checked HTTP 200 status, ignoring all
semantic health information.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Add Experience Count Checking + Auto-Trigger to Health Monitor (Disconnects #3, #1)

**Files:**
- Modify: `backend/clients/reactor_core_client.py` (`_health_monitor_loop` ~line 1005)
- Modify: `unified_supervisor.py` (~lines 22773-22788, DataFlywheelManager)
- Test: `tests/unit/backend/clients/test_reactor_core_client.py` (append)

**Step 1: Add test for auto-trigger in health monitor**

Append to test file:

```python
class TestHealthMonitorAutoTrigger:
    """Verify health monitor checks experience count and triggers training."""

    async def test_monitor_triggers_training_when_threshold_met(self, client):
        """Health monitor should trigger training when experience count >= threshold."""
        client._training_ready = True
        client.config.auto_trigger_enabled = True
        client.config.experience_threshold = 100

        with patch.object(client, 'health_check', new_callable=AsyncMock, return_value=True):
            with patch.object(client, 'get_experience_count', new_callable=AsyncMock, return_value=150):
                with patch.object(client, 'trigger_training', new_callable=AsyncMock) as mock_trigger:
                    mock_trigger.return_value = MagicMock(job_id="test-job")
                    await client._check_and_auto_trigger()
                    mock_trigger.assert_called_once()

    async def test_monitor_skips_trigger_when_not_training_ready(self, client):
        """Health monitor should NOT trigger when training_ready is False."""
        client._training_ready = False
        client.config.auto_trigger_enabled = True

        with patch.object(client, 'get_experience_count', new_callable=AsyncMock) as mock_count:
            await client._check_and_auto_trigger()
            mock_count.assert_not_called()

    async def test_monitor_skips_trigger_when_below_threshold(self, client):
        """Health monitor should NOT trigger when experience count < threshold."""
        client._training_ready = True
        client.config.auto_trigger_enabled = True
        client.config.experience_threshold = 100

        with patch.object(client, 'get_experience_count', new_callable=AsyncMock, return_value=50):
            with patch.object(client, 'trigger_training', new_callable=AsyncMock) as mock_trigger:
                await client._check_and_auto_trigger()
                mock_trigger.assert_not_called()

    async def test_monitor_skips_trigger_when_job_active(self, client):
        """Health monitor should NOT trigger when a training job is already active."""
        client._training_ready = True
        client.config.auto_trigger_enabled = True
        client._active_job_id = "existing-job"

        with patch.object(client, 'get_experience_count', new_callable=AsyncMock) as mock_count:
            await client._check_and_auto_trigger()
            mock_count.assert_not_called()
```

**Step 2: Run tests to verify they fail**

Run:
```bash
cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent && python3 -m pytest tests/unit/backend/clients/test_reactor_core_client.py::TestHealthMonitorAutoTrigger -v -x 2>&1 | head -40
```
Expected: FAIL — `_check_and_auto_trigger` doesn't exist.

**Step 3: Implement `_check_and_auto_trigger` method**

Add to `ReactorCoreClient` class, after the `_health_monitor_loop` method:

```python
    async def _check_and_auto_trigger(self) -> None:
        """
        v2.1: Check experience count and auto-trigger training if threshold met.

        Called from the health monitor loop after each successful health check.
        Guards:
        - Auto-trigger must be enabled
        - Reactor-Core must be training-ready
        - No active training job
        - Experience count must meet threshold
        - Minimum interval must have elapsed (enforced by trigger_training)
        """
        if not self.config.auto_trigger_enabled:
            return
        if not self._training_ready:
            return
        if getattr(self, '_active_job_id', None):
            return

        try:
            count = await self.get_experience_count()
            if count >= self.config.experience_threshold:
                logger.info(
                    f"[ReactorClient] Auto-trigger: {count} experiences "
                    f">= threshold {self.config.experience_threshold}"
                )
                job = await self.trigger_training(
                    experience_count=count,
                    priority=TrainingPriority.NORMAL,
                )
                if job:
                    self._active_job_id = job.job_id
                    logger.info(f"[ReactorClient] Auto-triggered job: {job.job_id}")
        except Exception as e:
            logger.warning(f"[ReactorClient] Auto-trigger check error: {e}")
```

Initialize `_active_job_id = None` in `__init__`.

**Step 4: Wire `_check_and_auto_trigger` into `_health_monitor_loop`**

In `_health_monitor_loop`, after `healthy = await self.health_check()`, add:

```python
                if healthy:
                    # v2.1: Check experience count and auto-trigger training
                    await self._check_and_auto_trigger()
```

**Step 5: Run tests to verify they pass**

Run:
```bash
cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent && python3 -m pytest tests/unit/backend/clients/test_reactor_core_client.py::TestHealthMonitorAutoTrigger -v -x 2>&1 | head -40
```
Expected: All 4 tests PASS.

**Step 6: Replace DataFlywheelManager raw HTTP in unified_supervisor.py**

Search `unified_supervisor.py` for the DataFlywheelManager training trigger code (~lines 22773-22788). It currently does a raw HTTP POST bypassing ReactorCoreClient. Replace the raw HTTP block with:

```python
                    # v2.1: Use ReactorCoreClient instead of raw HTTP
                    from backend.clients.reactor_core_client import check_and_trigger_training, TrainingPriority
                    job = await check_and_trigger_training(
                        experience_count=self._stats["total_queued"],
                        priority=TrainingPriority.NORMAL,
                    )
                    if job:
                        logger.info(f"[DataFlywheel] Training triggered via ReactorCoreClient: {job.job_id}")
```

**Step 7: Commit**

```bash
cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent && git add backend/clients/reactor_core_client.py unified_supervisor.py tests/unit/backend/clients/test_reactor_core_client.py && git commit -m "feat: add auto-trigger to health monitor, wire DataFlywheelManager through client

Health monitor now checks experience count after each successful health
check and auto-triggers training when threshold is met. DataFlywheelManager
now routes through ReactorCoreClient instead of raw HTTP POST, respecting
configurable thresholds, min intervals, and priority levels.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 5: Add Training Job Polling (Disconnect #4)

**Files:**
- Modify: `backend/clients/reactor_core_client.py` (`_health_monitor_loop`, new `_poll_active_job`)
- Test: `tests/unit/backend/clients/test_reactor_core_client.py` (append)

**Step 1: Add tests for job polling**

Append to test file:

```python
class TestJobPolling:
    """Verify health monitor polls active training jobs."""

    async def test_poll_detects_completed_job(self, client):
        """Poll should clear active_job_id when job completes."""
        client._active_job_id = "test-job-123"

        with patch.object(client, 'get_training_job', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"job_id": "test-job-123", "status": "completed", "metrics": {"loss": 0.5}}
            with patch.object(client, '_emit_event', new_callable=AsyncMock):
                with patch.object(client, '_write_bridge_event', new_callable=AsyncMock):
                    await client._poll_active_job()
            assert client._active_job_id is None

    async def test_poll_detects_failed_job(self, client):
        """Poll should clear active_job_id and log error when job fails."""
        client._active_job_id = "test-job-456"

        with patch.object(client, 'get_training_job', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"job_id": "test-job-456", "status": "failed", "error": "OOM"}
            with patch.object(client, '_emit_event', new_callable=AsyncMock):
                with patch.object(client, '_write_bridge_event', new_callable=AsyncMock):
                    await client._poll_active_job()
            assert client._active_job_id is None

    async def test_poll_keeps_running_job(self, client):
        """Poll should keep active_job_id for running jobs."""
        client._active_job_id = "test-job-789"

        with patch.object(client, 'get_training_job', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"job_id": "test-job-789", "status": "running", "stage": "training"}
            await client._poll_active_job()
            assert client._active_job_id == "test-job-789"

    async def test_poll_noop_when_no_active_job(self, client):
        """Poll should do nothing when no active job."""
        client._active_job_id = None
        await client._poll_active_job()  # Should not raise
```

**Step 2: Run tests to verify they fail**

Run:
```bash
cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent && python3 -m pytest tests/unit/backend/clients/test_reactor_core_client.py::TestJobPolling -v -x 2>&1 | head -40
```
Expected: FAIL — `_poll_active_job` doesn't exist.

**Step 3: Implement `_poll_active_job` method**

Add to `ReactorCoreClient`:

```python
    async def _poll_active_job(self) -> None:
        """
        v2.1: Poll the active training job for completion/failure.

        Called from health monitor loop. Updates _active_job_id state
        and emits events on job completion or failure.
        """
        if not self._active_job_id:
            return

        try:
            job_data = await self.get_training_job(self._active_job_id)
            if not job_data:
                return

            status = job_data.get("status", "unknown")

            if status == "completed":
                metrics = job_data.get("metrics", {})
                logger.info(
                    f"[ReactorClient] Training job {self._active_job_id} completed: "
                    f"metrics={metrics}"
                )
                await self._emit_event("training_completed", job_data)
                await self._write_bridge_event("training_completed", job_data)
                self._active_job_id = None

            elif status == "failed":
                error = job_data.get("error", "unknown")
                logger.error(
                    f"[ReactorClient] Training job {self._active_job_id} failed: {error}"
                )
                await self._emit_event("training_failed", job_data)
                await self._write_bridge_event("training_failed", job_data)
                self._active_job_id = None

            elif status == "running":
                stage = job_data.get("stage", "unknown")
                logger.debug(
                    f"[ReactorClient] Training job {self._active_job_id}: {status}/{stage}"
                )

        except Exception as e:
            logger.warning(f"[ReactorClient] Job poll error: {e}")
```

**Step 4: Wire into health monitor loop**

In `_health_monitor_loop`, after the auto-trigger call, add:

```python
                    # v2.1: Poll active training job
                    await self._poll_active_job()
```

**Step 5: Run tests to verify they pass**

Run:
```bash
cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent && python3 -m pytest tests/unit/backend/clients/test_reactor_core_client.py::TestJobPolling -v -x 2>&1 | head -40
```
Expected: All 4 tests PASS.

**Step 6: Also fix `get_training_job` path if needed**

Verify the endpoint path used by `get_training_job()` matches Reactor-Core's route. Expected: `/api/v1/jobs/{job_id}`.

**Step 7: Commit**

```bash
cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent && git add backend/clients/reactor_core_client.py tests/unit/backend/clients/test_reactor_core_client.py && git commit -m "feat: add training job polling to health monitor loop

Health monitor now polls active training jobs for completion/failure.
Emits events and writes bridge data on state changes. Clears active
job when completed or failed, allowing next auto-trigger cycle.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 6: Add Deployment Feedback from Prime (Disconnect #5)

**Files:**
- Modify: `jarvis-prime/jarvis_prime/docker/reactor_core_watcher.py`
- Test: `jarvis-prime/tests/test_deployment_feedback.py` (create)

**Step 1: Read the current ReactorCoreWatcher deploy logic**

Read `jarvis-prime/jarvis_prime/docker/reactor_core_watcher.py` to find the exact method that completes a deployment (likely `_deploy_model` or `_handle_new_model`). Identify where to insert the feedback write.

**Step 2: Write test for deployment feedback**

Create: `jarvis-prime/tests/test_deployment_feedback.py`

```python
"""Tests for deployment feedback writing after model deployment."""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch


@pytest.fixture
def tmp_cross_repo(tmp_path):
    """Temporary cross-repo directory."""
    cross_repo = tmp_path / "cross_repo"
    cross_repo.mkdir()
    return cross_repo


class TestDeploymentFeedback:
    """Verify deployment status is written after model deploy."""

    def test_write_deployment_status_on_success(self, tmp_cross_repo):
        """Successful deployment should write deployment_status.json."""
        from jarvis_prime.docker.reactor_core_watcher import write_deployment_feedback

        write_deployment_feedback(
            cross_repo_dir=tmp_cross_repo,
            model_id="qwen2.5-coder-7b-v3",
            status="success",
            previous_model="qwen2.5-coder-7b-v2",
            reactor_job_id="job-123",
            first_inference_latency_ms=3200.0,
        )

        status_file = tmp_cross_repo / "deployment_status.json"
        assert status_file.exists()
        data = json.loads(status_file.read_text())
        assert data["model_id"] == "qwen2.5-coder-7b-v3"
        assert data["deployment_status"] == "success"
        assert data["previous_model"] == "qwen2.5-coder-7b-v2"
        assert data["reactor_job_id"] == "job-123"
        assert data["health_check_passed"] is True
        assert "deployed_at" in data

    def test_write_deployment_status_on_failure(self, tmp_cross_repo):
        """Failed deployment should write deployment_status.json with error."""
        from jarvis_prime.docker.reactor_core_watcher import write_deployment_feedback

        write_deployment_feedback(
            cross_repo_dir=tmp_cross_repo,
            model_id="qwen2.5-coder-7b-v3",
            status="failed",
            previous_model="qwen2.5-coder-7b-v2",
            error="GGUF header invalid",
        )

        status_file = tmp_cross_repo / "deployment_status.json"
        data = json.loads(status_file.read_text())
        assert data["deployment_status"] == "failed"
        assert data["error"] == "GGUF header invalid"
        assert data["health_check_passed"] is False
```

**Step 3: Run tests to verify they fail**

Run:
```bash
cd /Users/djrussell23/Documents/repos/jarvis-prime && python3 -m pytest tests/test_deployment_feedback.py -v -x 2>&1 | head -40
```
Expected: FAIL — `write_deployment_feedback` doesn't exist.

**Step 4: Implement `write_deployment_feedback`**

Add to `jarvis_prime/docker/reactor_core_watcher.py`:

```python
def write_deployment_feedback(
    cross_repo_dir: Path,
    model_id: str,
    status: str,  # "success", "failed", "rollback"
    previous_model: Optional[str] = None,
    reactor_job_id: Optional[str] = None,
    first_inference_latency_ms: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    """
    Write deployment feedback to cross-repo directory for Reactor-Core.

    This closes the deployment feedback loop: Reactor-Core trains a model,
    Prime deploys it, and this function reports the result back.
    """
    feedback = {
        "model_id": model_id,
        "deployment_status": status,
        "deployed_at": datetime.now().isoformat(),
        "previous_model": previous_model,
        "health_check_passed": status == "success",
        "first_inference_latency_ms": first_inference_latency_ms,
        "error": error,
        "reactor_job_id": reactor_job_id,
    }

    status_file = cross_repo_dir / "deployment_status.json"
    # Atomic write via temp file
    tmp_file = status_file.with_suffix(".tmp")
    tmp_file.write_text(json.dumps(feedback, indent=2, default=str))
    tmp_file.rename(status_file)

    logger.info(f"[ReactorWatcher] Deployment feedback written: {status} for {model_id}")
```

**Step 5: Call `write_deployment_feedback` from the deployment handler**

Find the existing deploy success/failure paths in `ReactorCoreWatcher` and call the function at each exit point. The `cross_repo_dir` should be `Path.home() / ".jarvis" / "cross_repo"` (or from environment `JARVIS_CROSS_REPO_DIR`).

**Step 6: Run tests to verify they pass**

Run:
```bash
cd /Users/djrussell23/Documents/repos/jarvis-prime && python3 -m pytest tests/test_deployment_feedback.py -v -x 2>&1 | head -40
```
Expected: All 2 tests PASS.

**Step 7: Commit**

```bash
cd /Users/djrussell23/Documents/repos/jarvis-prime && git add jarvis_prime/docker/reactor_core_watcher.py tests/test_deployment_feedback.py && git commit -m "feat: add deployment feedback from Prime to Reactor-Core

After deploying a GGUF model, Prime now writes deployment_status.json
to ~/.jarvis/cross_repo/ with status, model lineage, and metrics.
This closes the deployment feedback loop — Reactor-Core can now know
whether its trained models deployed successfully.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 7: Add Job Persistence in Reactor-Core (Survives Supervisor Restart)

**Files:**
- Modify: `reactor-core/reactor_core/api/server.py` (TrainingJobManager class ~line 304)
- Test: `reactor-core/tests/test_job_persistence.py` (create)

**Step 1: Create tests directory and test file**

```bash
mkdir -p /Users/djrussell23/Documents/repos/reactor-core/tests
```

Create: `reactor-core/tests/test_job_persistence.py`

```python
"""Tests for training job persistence across restarts."""

import json
import pytest
from pathlib import Path
from datetime import datetime


@pytest.fixture
def tmp_jobs_dir(tmp_path):
    """Temporary directory for job persistence."""
    return tmp_path / "reactor_state"


class TestJobPersistence:
    """Verify jobs survive process restarts via file persistence."""

    def test_create_job_persists_to_file(self, tmp_jobs_dir):
        """Creating a job should write to jobs.json."""
        from reactor_core.api.server import TrainingJobManager

        mgr = TrainingJobManager(persist_dir=tmp_jobs_dir)

        import asyncio
        job = asyncio.get_event_loop().run_until_complete(
            mgr.create_job(experience_count=100, priority="normal", sources=["jarvis"], metadata={}, triggered_by="test")
        )

        jobs_file = tmp_jobs_dir / "jobs.json"
        assert jobs_file.exists()
        data = json.loads(jobs_file.read_text())
        assert job["job_id"] in data

    def test_load_jobs_on_init(self, tmp_jobs_dir):
        """New TrainingJobManager should load existing jobs from disk."""
        tmp_jobs_dir.mkdir(parents=True, exist_ok=True)
        jobs_file = tmp_jobs_dir / "jobs.json"
        jobs_file.write_text(json.dumps({
            "restored-job-1": {
                "job_id": "restored-job-1",
                "status": "running",
                "created_at": datetime.now().isoformat(),
            }
        }))

        from reactor_core.api.server import TrainingJobManager

        mgr = TrainingJobManager(persist_dir=tmp_jobs_dir)
        assert "restored-job-1" in mgr.jobs
        assert mgr.jobs["restored-job-1"]["status"] == "running"

    def test_update_job_persists(self, tmp_jobs_dir):
        """Updating job status should persist the change."""
        from reactor_core.api.server import TrainingJobManager

        mgr = TrainingJobManager(persist_dir=tmp_jobs_dir)

        import asyncio
        loop = asyncio.get_event_loop()
        job = loop.run_until_complete(
            mgr.create_job(experience_count=50, priority="high", sources=[], metadata={}, triggered_by="test")
        )

        loop.run_until_complete(mgr.update_job(job["job_id"], status="completed", metrics={"loss": 0.42}))

        # Re-read from disk
        data = json.loads((tmp_jobs_dir / "jobs.json").read_text())
        assert data[job["job_id"]]["status"] == "completed"
```

**Step 2: Run tests to verify they fail**

Run:
```bash
cd /Users/djrussell23/Documents/repos/reactor-core && python3 -m pytest tests/test_job_persistence.py -v -x 2>&1 | head -40
```
Expected: FAIL — `TrainingJobManager` doesn't accept `persist_dir`, no file persistence.

**Step 3: Add persistence to TrainingJobManager**

In `reactor_core/api/server.py`, modify the `TrainingJobManager` class:

- Add `persist_dir: Optional[Path] = None` parameter to `__init__`
- Default to `Path.home() / ".jarvis" / "reactor_state"` if not provided
- On init: load existing `jobs.json` if it exists
- On `create_job()`: persist after adding
- Add `update_job()` method that updates status and persists
- On any status change: atomic write to `jobs.json`

```python
    def __init__(self, persist_dir: Optional[Path] = None):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.current_job_id: Optional[str] = None
        self.experiences: List[Dict[str, Any]] = []
        self.last_training: Optional[datetime] = None
        self.start_time = datetime.now()
        self._lock = asyncio.Lock()

        # v2.1: Job persistence
        self._persist_dir = Path(persist_dir) if persist_dir else Path.home() / ".jarvis" / "reactor_state"
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._load_jobs()

    def _load_jobs(self) -> None:
        """Load persisted jobs from disk."""
        jobs_file = self._persist_dir / "jobs.json"
        if jobs_file.exists():
            try:
                data = json.loads(jobs_file.read_text())
                if isinstance(data, dict):
                    self.jobs = data
                    logger.info(f"[JobManager] Loaded {len(data)} persisted jobs")
            except Exception as e:
                logger.warning(f"[JobManager] Failed to load jobs: {e}")

    def _persist_jobs(self) -> None:
        """Persist jobs to disk atomically."""
        jobs_file = self._persist_dir / "jobs.json"
        tmp_file = jobs_file.with_suffix(".tmp")
        try:
            tmp_file.write_text(json.dumps(self.jobs, indent=2, default=str))
            tmp_file.rename(jobs_file)
        except Exception as e:
            logger.warning(f"[JobManager] Failed to persist jobs: {e}")

    async def update_job(self, job_id: str, **kwargs) -> None:
        """Update a job's fields and persist."""
        async with self._lock:
            if job_id in self.jobs:
                self.jobs[job_id].update(kwargs)
                self.jobs[job_id]["updated_at"] = datetime.now().isoformat()
                self._persist_jobs()
```

Also modify existing `create_job` to call `self._persist_jobs()` after adding the job.

**Step 4: Run tests to verify they pass**

Run:
```bash
cd /Users/djrussell23/Documents/repos/reactor-core && python3 -m pytest tests/test_job_persistence.py -v -x 2>&1 | head -40
```
Expected: All 3 tests PASS.

**Step 5: Commit**

```bash
cd /Users/djrussell23/Documents/repos/reactor-core && git add reactor_core/api/server.py tests/test_job_persistence.py && git commit -m "feat: add job persistence to TrainingJobManager

Training jobs are now persisted to ~/.jarvis/reactor_state/jobs.json
with atomic writes. Jobs survive Reactor-Core restarts. On startup,
existing jobs are loaded from disk. Status updates are immediately
persisted. This ensures training continues if the supervisor dies.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Phase 3: Deployment Gate + Close the Loop

### Task 8: Create Deployment Gate in Reactor-Core

**Files:**
- Create: `reactor-core/reactor_core/deployment/gate.py`
- Create: `reactor-core/reactor_core/deployment/__init__.py`
- Test: `reactor-core/tests/test_deployment_gate.py` (create)

**Step 1: Create tests for deployment gate**

Create: `reactor-core/tests/test_deployment_gate.py`

```python
"""Tests for GGUF model deployment gate."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.fixture
def tmp_model(tmp_path):
    """Create a fake GGUF file for testing."""
    model_file = tmp_path / "test_model.gguf"
    # Write valid GGUF magic bytes (0x46475547 = "GGUF" little-endian) + version 3
    import struct
    header = struct.pack("<I", 0x46475547)  # magic
    header += struct.pack("<I", 3)  # version
    # Pad to make it look like a real model (> 100MB check needs to be configurable for tests)
    model_file.write_bytes(header + b"\x00" * 1024)
    return model_file


@pytest.fixture
def small_model(tmp_path):
    """Create a suspiciously small GGUF file."""
    model_file = tmp_path / "tiny_model.gguf"
    model_file.write_bytes(b"\x00" * 100)  # Too small
    return model_file


class TestDeploymentGate:
    """Verify deployment gate catches bad models."""

    async def test_valid_gguf_header_passes(self, tmp_model):
        from reactor_core.deployment.gate import DeploymentGate, GateResult
        gate = DeploymentGate(min_file_size_bytes=512)  # Lower threshold for test
        result = await gate._check_gguf_header(tmp_model)
        assert result.passed is True

    async def test_invalid_gguf_header_fails(self, tmp_path):
        bad_file = tmp_path / "bad.gguf"
        bad_file.write_bytes(b"NOT_GGUF_DATA" + b"\x00" * 1024)
        from reactor_core.deployment.gate import DeploymentGate
        gate = DeploymentGate(min_file_size_bytes=512)
        result = await gate._check_gguf_header(bad_file)
        assert result.passed is False

    async def test_small_file_fails(self, small_model):
        from reactor_core.deployment.gate import DeploymentGate
        gate = DeploymentGate(min_file_size_bytes=500)
        result = await gate._check_file_size(small_model)
        assert result.passed is False

    async def test_validate_aggregates_checks(self, tmp_model):
        from reactor_core.deployment.gate import DeploymentGate
        gate = DeploymentGate(min_file_size_bytes=512, skip_inference_check=True)
        result = await gate.validate(tmp_model)
        assert result.passed is True
        assert len(result.checks) >= 2  # At least header + size checks
```

**Step 2: Run tests to verify they fail**

Run:
```bash
cd /Users/djrussell23/Documents/repos/reactor-core && python3 -m pytest tests/test_deployment_gate.py -v -x 2>&1 | head -40
```
Expected: FAIL — module doesn't exist.

**Step 3: Create the deployment gate module**

Create directory:
```bash
mkdir -p /Users/djrussell23/Documents/repos/reactor-core/reactor_core/deployment
```

Create: `reactor-core/reactor_core/deployment/__init__.py`
```python
"""Deployment gate for model validation before deployment."""
```

Create: `reactor-core/reactor_core/deployment/gate.py`

```python
"""
Deployment Gate — validates GGUF models before deployment to JARVIS Prime.

Uses the existing Gatekeeper/ApprovalCriterion framework from reactor_core.eval.gatekeeper
for extensible multi-criteria evaluation. v1 includes smoke tests (header, size, inference).
v2 will add ScoringEngine quality checks and regression detection.
"""

import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

GGUF_MAGIC = 0x46475547  # "GGUF" in little-endian


@dataclass
class CheckResult:
    """Result of a single deployment check."""
    name: str
    passed: bool
    severity: str = "critical"  # "critical" or "warning"
    reason: str = ""
    value: Optional[float] = None


@dataclass
class GateResult:
    """Aggregated result of all deployment checks."""
    passed: bool
    checks: List[CheckResult] = field(default_factory=list)
    model_path: Optional[str] = None

    @property
    def critical_failures(self) -> List[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "critical"]

    def summary(self) -> str:
        passed = sum(1 for c in self.checks if c.passed)
        total = len(self.checks)
        status = "APPROVED" if self.passed else "REJECTED"
        failures = "; ".join(c.reason for c in self.critical_failures)
        return f"{status} ({passed}/{total} checks passed){': ' + failures if failures else ''}"


class DeploymentGate:
    """
    Validates a GGUF model before deployment to JARVIS Prime.

    v1 checks (smoke tests):
    - GGUF header magic + version
    - File size within expected range
    - Model loads and generates text (optional, requires llama-cpp-python)

    v2 checks (planned):
    - ScoringEngine multi-criteria quality
    - Regression detection vs previous model
    - Safety score
    """

    def __init__(
        self,
        min_file_size_bytes: int = 100 * 1024 * 1024,  # 100MB default
        max_file_size_bytes: int = 20 * 1024 * 1024 * 1024,  # 20GB default
        skip_inference_check: bool = False,
        test_prompts: Optional[List[str]] = None,
    ):
        self.min_file_size_bytes = min_file_size_bytes
        self.max_file_size_bytes = max_file_size_bytes
        self.skip_inference_check = skip_inference_check
        self.test_prompts = test_prompts or [
            "What is 2 + 2?",
            "Write a Python function that adds two numbers.",
            "Explain what machine learning is in one sentence.",
            "Hello, how are you?",
            "List three colors.",
        ]

    async def validate(self, model_path: Path, manifest: Optional[dict] = None) -> GateResult:
        """
        Run all deployment checks on a GGUF model.

        Returns GateResult with PASS only if all critical checks pass.
        """
        checks: List[CheckResult] = []

        checks.append(await self._check_gguf_header(model_path))
        checks.append(await self._check_file_size(model_path))

        if not self.skip_inference_check:
            checks.append(await self._check_generates_text(model_path))

        passed = all(c.passed for c in checks if c.severity == "critical")
        result = GateResult(passed=passed, checks=checks, model_path=str(model_path))

        logger.info(f"[DeploymentGate] {result.summary()}")
        return result

    async def _check_gguf_header(self, model_path: Path) -> CheckResult:
        """Verify GGUF magic bytes and version are valid."""
        try:
            with open(model_path, "rb") as f:
                header = f.read(8)
                if len(header) < 8:
                    return CheckResult(
                        name="gguf_header", passed=False, severity="critical",
                        reason=f"File too small for GGUF header ({len(header)} bytes)"
                    )
                magic = struct.unpack("<I", header[:4])[0]
                version = struct.unpack("<I", header[4:8])[0]

                if magic != GGUF_MAGIC:
                    return CheckResult(
                        name="gguf_header", passed=False, severity="critical",
                        reason=f"Invalid GGUF magic: 0x{magic:08X} (expected 0x{GGUF_MAGIC:08X})"
                    )
                if version not in (1, 2, 3):
                    return CheckResult(
                        name="gguf_header", passed=False, severity="warning",
                        reason=f"Unexpected GGUF version: {version}",
                        value=float(version),
                    )
                return CheckResult(
                    name="gguf_header", passed=True,
                    reason=f"Valid GGUF v{version}",
                    value=float(version),
                )
        except Exception as e:
            return CheckResult(
                name="gguf_header", passed=False, severity="critical",
                reason=f"Failed to read GGUF header: {e}"
            )

    async def _check_file_size(self, model_path: Path) -> CheckResult:
        """Verify file size is within expected range."""
        try:
            size = model_path.stat().st_size
            if size < self.min_file_size_bytes:
                return CheckResult(
                    name="file_size", passed=False, severity="critical",
                    reason=f"File too small: {size / 1024 / 1024:.1f}MB (min: {self.min_file_size_bytes / 1024 / 1024:.0f}MB)",
                    value=float(size),
                )
            if size > self.max_file_size_bytes:
                return CheckResult(
                    name="file_size", passed=False, severity="critical",
                    reason=f"File too large: {size / 1024 / 1024 / 1024:.1f}GB (max: {self.max_file_size_bytes / 1024 / 1024 / 1024:.0f}GB)",
                    value=float(size),
                )
            return CheckResult(
                name="file_size", passed=True,
                reason=f"Size OK: {size / 1024 / 1024:.1f}MB",
                value=float(size),
            )
        except Exception as e:
            return CheckResult(
                name="file_size", passed=False, severity="critical",
                reason=f"Failed to check file size: {e}"
            )

    async def _check_generates_text(self, model_path: Path) -> CheckResult:
        """Load model and verify it generates non-garbage text."""
        try:
            from llama_cpp import Llama
        except ImportError:
            return CheckResult(
                name="generates_text", passed=True, severity="warning",
                reason="llama-cpp-python not available, skipping inference check"
            )

        try:
            import asyncio
            loop = asyncio.get_event_loop()

            def _test_inference():
                model = Llama(
                    model_path=str(model_path),
                    n_ctx=512,
                    n_gpu_layers=0,  # CPU only for gate check
                    verbose=False,
                )
                try:
                    results = []
                    for prompt in self.test_prompts[:3]:  # Test first 3 prompts
                        output = model.create_completion(
                            prompt,
                            max_tokens=50,
                            temperature=0.1,
                        )
                        text = output["choices"][0]["text"].strip()
                        results.append(text)

                    # Check: non-empty outputs
                    empty_count = sum(1 for r in results if len(r) < 2)
                    if empty_count > 1:
                        return CheckResult(
                            name="generates_text", passed=False, severity="critical",
                            reason=f"{empty_count}/3 outputs were empty",
                        )

                    # Check: not all identical (sign of degenerate model)
                    if len(set(results)) == 1 and len(results) > 1:
                        return CheckResult(
                            name="generates_text", passed=False, severity="critical",
                            reason="All outputs identical (degenerate model)",
                        )

                    return CheckResult(
                        name="generates_text", passed=True,
                        reason=f"Generated {len(results)} distinct non-empty outputs",
                    )
                finally:
                    del model

            result = await loop.run_in_executor(None, _test_inference)
            return result

        except Exception as e:
            return CheckResult(
                name="generates_text", passed=False, severity="critical",
                reason=f"Inference check failed: {e}"
            )
```

**Step 4: Run tests to verify they pass**

Run:
```bash
cd /Users/djrussell23/Documents/repos/reactor-core && python3 -m pytest tests/test_deployment_gate.py -v -x 2>&1 | head -40
```
Expected: All 4 tests PASS.

**Step 5: Commit**

```bash
cd /Users/djrussell23/Documents/repos/reactor-core && git add reactor_core/deployment/ tests/test_deployment_gate.py && git commit -m "feat: add DeploymentGate for GGUF model validation

Validates GGUF models before deployment: header magic/version, file
size range, and optional inference test (generates non-empty, non-
degenerate text). Extensible interface for future ScoringEngine and
regression checks. Blocks corrupt or broken models from reaching Prime.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 9: Wire Deployment Gate into Training Pipeline

**Files:**
- Modify: `reactor-core/reactor_core/training/unified_pipeline.py` (after GGUF export step)

**Step 1: Find the GGUF export step in the pipeline**

Read `reactor_core/training/unified_pipeline.py` and locate where the trained model is exported to GGUF. This is the insertion point for the gate.

**Step 2: Insert gate check after export**

After the GGUF export produces a file, add:

```python
        # v2.1: Deployment gate - validate GGUF before deployment
        from reactor_core.deployment.gate import DeploymentGate
        gate = DeploymentGate(
            skip_inference_check=bool(os.getenv("REACTOR_SKIP_INFERENCE_CHECK", "")),
        )
        gate_result = await gate.validate(gguf_path)

        if not gate_result.passed:
            logger.error(f"[Pipeline] Deployment gate REJECTED: {gate_result.summary()}")
            # Move to failed directory instead of deploy directory
            failed_dir = self.config.output_dir / "failed"
            failed_dir.mkdir(parents=True, exist_ok=True)
            gguf_path.rename(failed_dir / gguf_path.name)
            return {"status": "gate_rejected", "gate_result": gate_result.summary()}

        logger.info(f"[Pipeline] Deployment gate APPROVED: {gate_result.summary()}")
```

**Step 3: Commit**

```bash
cd /Users/djrussell23/Documents/repos/reactor-core && git add reactor_core/training/unified_pipeline.py && git commit -m "feat: wire DeploymentGate into training pipeline after GGUF export

GGUF models must now pass the deployment gate before being placed
in the deploy directory. Rejected models are moved to failed/ with
logged reasons. Prevents corrupt or degenerate models from deploying.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 10: Add Correlation IDs to Telemetry Emitter

**Files:**
- Modify: `backend/core/telemetry_emitter.py` (~line 472-525, emit method)
- Test: `tests/unit/backend/clients/test_reactor_core_client.py` (append)

**Step 1: Add correlation_id to TelemetryEvent**

Find the `TelemetryEvent` dataclass in `telemetry_emitter.py` and add:

```python
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
```

This ensures every emitted experience has a unique correlation ID that follows it through the pipeline.

**Step 2: Verify the correlation_id is included in JSONL output and HTTP payload**

Check the serialization methods (`to_dict()` or equivalent). The `correlation_id` should be included automatically if using `dataclasses.asdict()` or similar.

**Step 3: Commit**

```bash
cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent && git add backend/core/telemetry_emitter.py && git commit -m "feat: add correlation_id to TelemetryEvent for pipeline observability

Every emitted experience now carries a UUID correlation_id that follows
it through ingestion, training, deployment, and probation. Enables
end-to-end tracing across all three repos.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 11: End-to-End Integration Test

**Files:**
- Create: `tests/integration/test_reactor_pipeline_e2e.py`

**Step 1: Write end-to-end test**

```python
"""
End-to-end integration test for the Reactor Core training pipeline.

Requires: Reactor-Core running on port 8090
Marks: @pytest.mark.integration, @pytest.mark.e2e
"""

import aiohttp
import pytest
from datetime import datetime


@pytest.mark.integration
@pytest.mark.e2e
class TestReactorPipelineE2E:
    """End-to-end tests that verify the full pipeline loop."""

    async def test_experience_accepted_by_reactor(self):
        """Verify a manually sent experience is accepted by Reactor-Core."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "experience": {
                    "event_type": "INTERACTION",
                    "user_input": "What is 2+2?",
                    "assistant_output": "4",
                    "source": "JARVIS_BODY",
                    "timestamp": datetime.now().isoformat(),
                    "confidence": 0.95,
                    "task_type": "math_simple",
                },
                "timestamp": datetime.now().isoformat(),
                "source": "jarvis_agent",
            }
            async with session.post(
                "http://localhost:8090/api/v1/experiences/stream",
                json=payload,
            ) as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["accepted"] is True
                assert data["count"] >= 1

    async def test_reactor_health_reports_training_ready(self):
        """Verify Reactor-Core health includes training readiness."""
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8090/health") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert "training_ready" in data
                assert "phase" in data

    async def test_client_paths_match_server(self):
        """Verify ReactorCoreClient endpoint paths resolve (no 404s)."""
        from backend.clients.reactor_core_client import ReactorCoreClient, ReactorCoreConfig
        config = ReactorCoreConfig(api_url="http://localhost:8090")
        client = ReactorCoreClient(config)
        await client.start()
        try:
            # Health should work
            healthy = await client.health_check()
            assert healthy is True

            # Experience count should not 404
            count = await client.get_experience_count()
            assert isinstance(count, int)
        finally:
            await client.close()
```

**Step 2: Run integration tests (requires Reactor-Core running)**

Run:
```bash
cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent && python3 -m pytest tests/integration/test_reactor_pipeline_e2e.py -v -x -m "integration and e2e" 2>&1 | head -40
```

**Step 3: Commit**

```bash
cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent && git add tests/integration/test_reactor_pipeline_e2e.py && git commit -m "test: add end-to-end integration test for Reactor Core pipeline

Verifies experiences flow to Reactor-Core, health endpoint returns
training readiness, and ReactorCoreClient paths match server routes.
Requires Reactor-Core running on port 8090.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Phase 4: Beef Up (Advanced Hardening)

### Task 12: Training Circuit Breaker

**Files:**
- Modify: `backend/clients/reactor_core_client.py`
- Test: `tests/unit/backend/clients/test_reactor_core_client.py` (append)

**Summary:**
- Import `CircuitBreaker` from `backend.kernel.circuit_breaker`
- Create `"reactor_training"` circuit breaker in `ReactorCoreClient.__init__`
- Config: `failure_threshold=3`, `recovery_timeout_seconds=3600` (1 hour)
- Guard `trigger_training()` with `circuit_breaker.can_execute()`
- Record failure on: job failure, gate rejection, post-deploy rollback
- Record success on: job completed + deployment approved
- In `_check_and_auto_trigger()`: skip if circuit is OPEN

### Task 13: Quality-Weighted Experience Scoring

**Files:**
- Create: `backend/clients/experience_scorer.py`
- Test: `tests/unit/backend/clients/test_experience_scorer.py`

**Summary:**
- `ExperienceScorer` class with weight table (CORRECTION=10x, FEEDBACK_NEGATIVE=5x, etc.)
- `score(experience: Dict) -> float` — returns weighted value
- `WeightedExperienceTracker` — accumulates weighted scores, checks threshold
- Deduplication via bloom filter on `(user_input, task_type)` hash
- Wire into `_check_and_auto_trigger()` — use weighted score instead of raw count
- All weights configurable via environment variables

### Task 14: Resource-Aware Training Gate

**Files:**
- Modify: `backend/clients/reactor_core_client.py` (`_check_and_auto_trigger`)

**Summary:**
- Import `MemoryQuantizer` from `backend.core.memory_quantizer`
- Before triggering: check current `MemoryTier`
- ABUNDANT/OPTIMAL: full training
- ELEVATED: reduced metadata flag (`"reduced_batch": true`)
- CONSTRAINED: defer (log "deferred to Night Shift", don't trigger)
- CRITICAL/EMERGENCY: skip entirely

### Task 15: Model Lineage Tracking

**Files:**
- Modify: `reactor-core/reactor_core/api/server.py` (after training completes in `run_training_pipeline`)

**Summary:**
- Import `DataHash` from `reactor_core.data.versioning`
- After GGUF export: compute `DataHash.from_file(gguf_path)`
- Write lineage record to `~/.jarvis/reactor/models/lineage.jsonl`
- Include: model_id, model_hash, parent_model, training config, eval scores, gate decision, dataset hash

### Task 16: Atomic Experience Snapshots

**Files:**
- Modify: `reactor-core/reactor_core/api/server.py` (`run_training_pipeline` function)

**Summary:**
- Before passing experiences to trainer: acquire lock, copy, clear, release
- Write snapshot to `~/.jarvis/reactor/training_data/snapshot_{job_id}.jsonl`
- Compute `DataHash.from_file(snapshot_path)` for dataset versioning
- Store hash in job metadata

---

## Phase 5: Post-Deployment Safety

### Task 17: Post-Deployment Probation

**Files:**
- Modify: `jarvis-prime/jarvis_prime/docker/reactor_core_watcher.py`

**Summary:**
- After successful deployment: start 30-minute probation window
- Probe every 60s: inference latency, error rate, memory
- Use `ProbationStatus` pattern from `backend.core.supervisor.rollback_manager`
- `health_score >= 0.8` → COMMITTED (update deployment_status.json)
- `health_score < 0.5` → ROLLING_BACK (restore previous GGUF)
- Emergency: error rate > 5x baseline → immediate rollback
- Keep previous model in `~/.jarvis/reactor/models/previous/`

### Task 18: TrinityEventBus Integration

**Files:**
- Modify: `backend/clients/reactor_core_client.py`
- Modify: `reactor-core/reactor_core/api/server.py`
- Modify: `jarvis-prime/jarvis_prime/docker/reactor_core_watcher.py`

**Summary:**
- Import `TrinityEventBus` from `backend.core.trinity_event_bus`
- Emit events at each pipeline stage with `correlation_id` and `causation_id`:
  - `experience.emitted`, `training.started`, `training.completed`, `training.failed`
  - `gate.evaluated`, `model.deployed`, `probation.started`, `probation.committed`, `probation.rollback`
- Subscribe supervisor to `training.*` and `model.*` topics
- Subscribe Reactor-Core to `experience.*` topics

---

## Run All Tests

After all tasks are complete:

```bash
# JARVIS tests
cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent && python3 -m pytest tests/unit/backend/clients/test_reactor_core_client.py -v 2>&1 | tail -20

# Reactor-Core tests
cd /Users/djrussell23/Documents/repos/reactor-core && python3 -m pytest tests/ -v 2>&1 | tail -20

# JARVIS Prime tests
cd /Users/djrussell23/Documents/repos/jarvis-prime && python3 -m pytest tests/test_deployment_feedback.py -v 2>&1 | tail -20

# Integration (requires Reactor-Core running)
cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent && python3 -m pytest tests/integration/test_reactor_pipeline_e2e.py -v -m "integration" 2>&1 | tail -20
```
