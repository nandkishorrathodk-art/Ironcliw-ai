# Startup Warning Root Fix — v244.0 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Eliminate 3 startup warning clusters at their root cause + fix brain vacuum classification hardcoding.

**Context:** Commit `167fcecb` intentionally deleted `tiered_vbia_adapter.py` and `tiered_command_router.py` (replaced by J-Prime's Phi classifier) but left orphaned imports, dead wiring code, and dead API endpoints. Cloud SQL proxy has a redundant settling delay after its log-signal gating already confirmed readiness. Brain vacuum hardcodes `intent="answer"` making all commands text-only during J-Prime downtime.

**Tech Stack:** Python 3, asyncio, FastAPI, unified_supervisor.py (76K+ line kernel)

**Design doc:** `docs/plans/2026-02-22-startup-warning-root-fix-design.md`

---

## Section 1: Dead Code Surgical Removal

### Task 1: Remove Dead Instance Vars and Status Keys from `__init__`

**Why:** `self._vbia_adapter` and `self._tiered_router` are never set to anything meaningful (their initialization closures always hit ImportError). Dead status keys in `_two_tier_status` cause confusion.

**Files:**
- Modify: `unified_supervisor.py:60555-60583`

**Implementation:**

1. At lines 60555-60564, remove the comment block and dead instance vars. Keep `_agentic_watchdog`, `_agentic_runner`, and `_cross_repo_initialized`. Replace with:

```python
        # v244.0: Integration components
        # - Agentic Watchdog (safety kill-switch for Computer Use)
        # - Cross-Repo State (JARVIS ↔ Prime ↔ Reactor coordination)
        # - AgenticTaskRunner (agentic execution engine)
        # NOTE: TieredVBIAAdapter and TieredCommandRouter were replaced by
        # J-Prime's Phi classifier (commit 167fcecb). Voice auth uses
        # SpeakerVerificationService directly.
        self._agentic_watchdog: Optional[Any] = None  # AgenticWatchdog
        self._agentic_runner: Optional[Any] = None  # AgenticTaskRunner
        self._cross_repo_initialized: bool = False
```

2. At lines 60577-60583, remove dead keys from `_two_tier_status`. Keep only live keys:

```python
        self._two_tier_status: Dict[str, Any] = {
            "watchdog": {"status": "pending", "mode": None},
            "cross_repo": {"status": "pending", "initialized": False},
            "runner_wired": False,
        }
```

**Test:** `python3 -c "import unified_supervisor"` — no ImportError. Verify `_vbia_adapter` and `_tiered_router` no longer exist as instance vars.

**Commit message:** `fix: remove dead _vbia_adapter and _tiered_router instance vars from __init__`

---

### Task 2: Remove Dead Closures from `_initialize_two_tier_security` and Rename

**Why:** `_init_vbia()`, `_init_router()`, and `_chain_vbia_then_router()` are dead — they try to import deleted modules. The parallel gather runs 3 tasks but only `_init_cross_repo()` does real work. The execute_tier2 wiring block (lines 68384-68583) is dead because `self._tiered_router` is always None. The method name `_initialize_two_tier_security` is misleading now.

**Files:**
- Modify: `unified_supervisor.py`
  - Method definition at line 68048
  - Dead closures: `_init_vbia()` (68136-68171), `_init_router()` (68223-68298), `_chain_vbia_then_router()` (68301-68303)
  - Parallel gather block (68305-68383)
  - Dead wiring block (68384-68583)
  - Status block at end (68585-68620+)

**Implementation:**

Rewrite the method. The new version:
- Renames to `_initialize_integration_components`
- Keeps: Step 1 (watchdog), Step 3 (`_init_cross_repo`), Step 5 (AgenticTaskRunner creation — this is LIVE and works independently)
- Removes: Step 2 (`_init_vbia`), Step 4 (`_init_router`), `_chain_vbia_then_router`, the parallel gather wrapper, the heartbeat task, the execute_tier2 wiring block (68459-68576)
- Keeps: The AgenticTaskRunner creation (lines 68399-68456) but REMOVES the `if self._tiered_router and self._agentic_runner:` wiring block and all 3 branches after it (68459-68576)

**CRITICAL — What to keep from Step 5 (lines 68384-68583):**
- KEEP: The AgenticTaskRunner import and creation block (68399-68456). This creates the runner independently.
- KEEP: `self._two_tier_status["runner_wired"] = False` — but only as final state since wiring no longer happens.
- REMOVE: The 3-branch conditional (68459-68576) that wires `execute_tier2` to the router — `self._tiered_router` is always None so the `else` branch always fires, logging the warning we're eliminating.

**Parallel gather simplification:** With only watchdog + cross_repo (both independent), use `asyncio.gather()` directly without the heartbeat wrapper. The heartbeat was needed because the old gather had 3 slow tasks; now it's 2 fast ones.

**Progress points to update:**
- Keep: 56 (parallel start), 58 (cross_repo)
- Remove: 57 (vbia — dead), 59 (router — dead)
- Keep: 60 (wiring → repurpose for runner creation), 61 (ready)

**Also update any callers:** Search for `_initialize_two_tier_security` — it's called from `_phase_security()` or similar. Update the call site to use the new name `_initialize_integration_components`.

**Test:** Start supervisor. No warnings about `tiered_vbia_adapter`, `TieredCommandRouter`, or `execute_tier2`. AgenticTaskRunner still initializes.

**Commit message:** `fix: remove dead VBIA/Router closures, rename to _initialize_integration_components`

---

### Task 3: Remove Dead Code from `agentic_api.py`

**Why:** `/route` and `/tier1` endpoints always return 503 (router is always None after module deletion). `get_tiered_router()`, `get_vbia_adapter()`, `RouteCommandRequest`, `RouteCommandResponse` are all dead.

**Files:**
- Modify: `backend/api/agentic_api.py`
  - Remove: `get_tiered_router()` (166-181)
  - Remove: `get_vbia_adapter()` (184-200)
  - Remove: `RouteCommandRequest` (85-90)
  - Remove: `RouteCommandResponse` (93-106)
  - Remove: `/route` endpoint (412-493)
  - Remove: `/tier1` endpoint (496-517)
  - Update: `/health` endpoint (520-563) — remove `router_obj` and `vbia` checks
  - Update: `/metrics` endpoint (566-579) — remove `router_obj` metric

**Implementation:**

1. Remove the 4 dead classes/functions and 2 dead endpoints listed above.

2. Update `/health` endpoint — remove references to router and vbia. The health check should only check runner and watchdog:

```python
@router.get("/health")
async def agentic_health(request: Request) -> Dict[str, Any]:
    """Health check for the unified agentic system."""
    runner = get_agentic_runner(request)
    watchdog = get_watchdog(request)

    components = {
        "runner": runner is not None and (runner.is_ready if runner else False),
        "watchdog": watchdog is not None,
    }

    if not runner:
        return {
            "status": "initializing",
            "message": "Agentic system is starting up",
            "components": components,
        }

    if watchdog and not watchdog.is_agentic_allowed():
        return {
            "status": "restricted",
            "message": "Watchdog has restricted agentic execution",
            "components": components,
        }

    return {
        "status": "healthy",
        "message": "Unified agentic system operational",
        "components": components,
        "capabilities": {
            "tier2": runner.is_ready if runner else False,
            "watchdog": watchdog is not None,
        }
    }
```

3. Update `/metrics` — remove router metric:

```python
@router.get("/metrics")
async def get_metrics(request: Request) -> Dict[str, Any]:
    """Get execution metrics for the agentic system."""
    runner = get_agentic_runner(request)
    return {"runner": runner.get_stats() if runner else {}}
```

4. Remove any now-unused imports (`CommandTier` is only used in `/route`).

**Test:** `python3 -c "from backend.api.agentic_api import router"` — no ImportError. Verify `/route` and `/tier1` no longer exist in the router's routes.

**Commit message:** `fix: remove dead /route, /tier1 endpoints and tiered router references from agentic_api`

---

### Task 4: Remove Dead Import from `voice_authentication_layer.py`

**Why:** Line 260 imports `get_tiered_vbia_adapter` from the deleted `core.tiered_vbia_adapter` module, causing a startup warning every time.

**Files:**
- Modify: `backend/core/voice_authentication_layer.py:255-279`

**Implementation:**

Remove the entire PHASE 1 block (lines 255-279) that tries to import and connect the dead VBIA adapter. The `_vbia_adapter` field on this class is only used by `verify_for_tier2()` and `verify_for_tier1()` which have ZERO callers (confirmed — they were exclusively called by the deleted TieredCommandRouter).

Replace with a comment explaining what happened:

```python
                # v244.0: TieredVBIAAdapter removed (commit 167fcecb).
                # Voice biometric auth now uses SpeakerVerificationService directly.
                # verify_for_tier2() and verify_for_tier1() retained for API
                # compatibility but have zero callers.
```

**Test:** `python3 -c "from backend.core.voice_authentication_layer import VoiceAuthenticationLayer"` — no warning about `tiered_vbia_adapter`.

**Commit message:** `fix: remove dead tiered_vbia_adapter import from voice_authentication_layer`

---

### Task 5: Delete Dead Test File and Update Stale Comment

**Why:** `test_two_tier_security.py` tests 100% deleted components. Google workspace agent has a stale comment referencing a deleted file.

**Files:**
- Delete: `test_two_tier_security.py`
- Modify: `backend/neural_mesh/agents/google_workspace_agent.py:3782`

**Implementation:**

1. Delete `test_two_tier_security.py` entirely.

2. Update the stale comment at line 3782 in `google_workspace_agent.py`:

Old:
```python
# v237.0: Singleton getter (required by tiered_command_router.py:1072)
```

New:
```python
# v237.0: Singleton getter for GoogleWorkspaceAgent
```

**Test:** `ls test_two_tier_security.py` should fail (file deleted). Grep codebase for `tiered_command_router.py` — should have no remaining references.

**Commit message:** `fix: delete dead test_two_tier_security.py, update stale comment`

---

### Task 6: Update Health Status Endpoint in Supervisor

**Why:** The health status endpoint at line 76843-76851 reports dead keys (`vbia_adapter`, `router`) that are always in `unavailable` state.

**Files:**
- Modify: `unified_supervisor.py:76843-76851`

**Implementation:**

Replace the `status["two_tier"]` block:

```python
        # v244.0: Integration component status (was "two_tier" pre-v244)
        status["two_tier"] = {
            "enabled": self.config.two_tier_security_enabled,
            "watchdog": self._two_tier_status.get("watchdog", {}),
            "cross_repo": self._two_tier_status.get("cross_repo", {}),
            "runner_wired": self._two_tier_status.get("runner_wired", False),
        }
```

(Removed `vbia_adapter` and `router` keys — they were always `{"status": "unavailable"}`.)

**Test:** Start supervisor, hit health endpoint. `two_tier` section no longer reports dead `vbia_adapter` or `router` keys.

**Commit message:** `fix: remove dead vbia_adapter and router from health status endpoint`

---

## Section 2: Cloud SQL Proxy Readiness Gate Fix

### Task 7: Skip Settling Delay When `start()` Confirmed Readiness

**Why:** `ProxyReadinessGate.ensure_proxy_ready()` applies a 2s settling delay (line 3319-3328) even when `proxy_manager.start()` already confirmed readiness via log-signal gating ("ready for new connections"). The proxy IS authenticated when `start()` returns success — the settling delay is redundant and wastes 2s on every startup.

**Files:**
- Modify: `backend/intelligence/cloud_sql_connection_manager.py:3316-3329`

**Implementation:**

The settling delay at lines 3319-3328 should only apply when reusing an already-running proxy (where we don't know its auth state). When `just_started_proxy=True`, `start()` already waited for the log signal confirming GCP auth — no settling needed.

Replace the settling block (lines 3316-3328):

```python
            # v244.0: Post-TCP settling delay — only needed when REUSING an
            # already-running proxy whose auth state we haven't verified.
            # When we just started the proxy, start() already confirmed
            # readiness via log-signal gating ("ready for new connections").
            if just_started_proxy:
                just_started_proxy = False
                # No settling delay — start() already confirmed GCP auth
                logger.debug(
                    "[ReadinessGate v244.0] Skipping settling delay "
                    "(start() confirmed readiness via log signal)"
                )
                elapsed = time.time() - start_time
            elif proxy_running and proxy_settling_delay > 0:
                # Reusing existing proxy — apply settling delay since we
                # haven't verified its auth state
                remaining = timeout - (time.time() - start_time)
                settle = min(proxy_settling_delay, remaining * 0.5)
                if settle > 0:
                    logger.debug(
                        "[ReadinessGate v3.2] Post-TCP settling delay %.1fs "
                        "(reusing existing proxy)", settle
                    )
                    await asyncio.sleep(settle)
                elapsed = time.time() - start_time
```

**Test:** Start supervisor with cloud_sql_proxy. Logs should show `Skipping settling delay` instead of `Post-TCP settling delay 2.0s`. Proxy should be ready in <5s instead of ~47s.

**Commit message:** `fix: skip redundant settling delay when proxy start() confirmed readiness`

---

### Task 8: Extend TCP-Only Fallback Timeout and Fix `recover_proxy()` Sleep

**Why:** TCP-only fallback fires after just 3s (line 1921: `elif i >= 6` at 0.5s intervals). This defeats log-signal gating on slow GCP auth. Also, `_auto_heal_reconnect()` has hardcoded `asyncio.sleep(3)` at line 2729 — should use the same log-signal-aware readiness check.

**Files:**
- Modify: `backend/intelligence/cloud_sql_proxy_manager.py:1921` (TCP fallback)
- Modify: `backend/intelligence/cloud_sql_proxy_manager.py:2727-2736` (heal reconnect sleep)

**Implementation:**

1. At line 1921, extend the TCP-only fallback from 3s to 10s (configurable):

```python
                        _tcp_fallback_timeout = float(os.environ.get(
                            "CLOUDSQL_LOG_READY_FALLBACK_TIMEOUT", "10.0"
                        ))
                        _tcp_fallback_iters = int(_tcp_fallback_timeout / 0.5)
                        elif i >= _tcp_fallback_iters:  # Configurable (default 10s)
```

Actually, looking at the loop structure, the `elif i >= 6` is inside a `for` loop. The cleanest change is to read the env var before the loop and compute the iteration threshold:

Before the readiness polling loop (find the `for i in range(...)` near line 1921), add:

```python
        _tcp_fallback_s = float(os.environ.get("CLOUDSQL_LOG_READY_FALLBACK_TIMEOUT", "10.0"))
        _tcp_fallback_iters = int(_tcp_fallback_s / 0.5)
```

Then change line 1921 from:
```python
                        elif i >= 6:  # After 3s with TCP open but no log signal
```
to:
```python
                        elif i >= _tcp_fallback_iters:  # v244.0: Configurable (default 10s)
```

2. At lines 2727-2736, replace the hardcoded `asyncio.sleep(3)` in `_auto_heal_reconnect()` with a call to `wait_for_proxy_ready()` (the log-signal-aware method) if available, or at minimum use the same configurable timeout:

```python
            if success:
                # v244.0: Use log-signal-aware readiness check instead of hardcoded sleep.
                # start() already waits for "ready for new connections" log signal,
                # so additional sleep is wasteful.
                logger.info("[CLOUDSQL] Verifying reconnection...")
                test_result = await self.check_connection_health()
                if test_result.get('connection_active'):
                    logger.info("[CLOUDSQL] ✅ Reconnection successful")
                    self.reconnect_count = 0
                    return True
                else:
                    logger.error("[CLOUDSQL] ❌ Reconnection failed: connection not active")
                    return False
```

**Test:** Set `CLOUDSQL_LOG_READY_FALLBACK_TIMEOUT=10` in env. Proxy startup should wait up to 10s for log signal before TCP-only fallback. Heal reconnect should not have 3s hardcoded sleep.

**Commit message:** `fix: extend TCP-only fallback to 10s, remove hardcoded sleep in reconnect`

---

### Task 9: Remove Redundant `check_connection_health()` from Supervisor

**Why:** After `proxy_manager.start()` and before `ensure_proxy_ready()`, the supervisor calls `check_connection_health()` (line 78180). This is redundant — `ensure_proxy_ready()` does its own DB verification. The double check wastes 5-10s on startup.

**Files:**
- Modify: `unified_supervisor.py:78178-78193`

**Implementation:**

Replace the redundant health check block (lines 78178-78193) with a comment:

```python
                        # v244.0: Removed redundant check_connection_health() here.
                        # ensure_proxy_ready() (called next) is the single source of
                        # truth for proxy readiness — it does TCP + DB verification.
```

**Test:** Start supervisor with proxy. No double DB verification in logs. `ensure_proxy_ready()` is the only verification step.

**Commit message:** `fix: remove redundant check_connection_health between proxy start and readiness gate`

---

## Section 3: Brain Vacuum Classification Fix

### Task 10: Add Classification Prompt to Brain Vacuum Fallback

**Why:** `_brain_vacuum_fallback()` at line 1563 hardcodes `intent="answer"` and `domain="general"`. When J-Prime is down, all commands become text responses — "lock my screen" gets a text answer instead of executing a screen lock.

**Files:**
- Modify: `backend/core/jarvis_prime_client.py:1563-1620`

**Implementation:**

Modify `_brain_vacuum_fallback()` to include a classification prompt prefix in the system prompt. The Claude/Gemini response should include a `CLASSIFICATION: {...}` JSON block that we parse to populate StructuredResponse fields.

**Valid intent values** (from `unified_command_processor.py:1930-2024`): `"answer"`, `"conversation"`, `"action"`, `"vision_needed"`, `"multi_step_action"`, `"clarify"`

**Valid domain values** (from usage patterns): `"general"`, `"system"`, `"security"`, `"workspace"`, `"development"`, `"media"`, `"smart_home"`

Replace the method body:

```python
    async def _brain_vacuum_fallback(
        self,
        query: str,
        system_prompt: Optional[str],
        max_tokens: int,
    ) -> StructuredResponse:
        """Fallback when J-Prime is completely unreachable.

        This covers startup windows, network failures, and circuit-breaker
        open states.  v242.2: Routes to Claude API first (preferred), then
        Gemini as last resort, to maintain responsiveness while offline.

        v244.0: Includes classification prompt so commands are properly
        classified (not hardcoded to intent="answer"). This ensures
        "lock my screen" executes as an action, not a text response.
        """
        # v244.0: Classification prompt — ask the LLM to classify the
        # command AND generate a response in a single call.
        _classification_prefix = (
            "IMPORTANT: Before your response, output a classification line in this exact format:\n"
            "CLASSIFICATION: {\"intent\": \"<intent>\", \"domain\": \"<domain>\", "
            "\"requires_action\": <true/false>, \"suggested_actions\": [\"<action>\"]}\n\n"
            "Valid intents: answer, conversation, action, vision_needed, multi_step_action, clarify\n"
            "Valid domains: general, system, security, workspace, development, media, smart_home\n\n"
            "- Use intent=\"action\" for commands that DO something (lock screen, open app, etc.)\n"
            "- Use intent=\"answer\" for questions that need information\n"
            "- Use intent=\"conversation\" for casual chat\n"
            "- suggested_actions: list specific actions like [\"lock_screen\"], [\"open_browser\"], etc.\n\n"
            "After the CLASSIFICATION line, provide your normal response.\n\n"
        )

        try:
            messages: List[ChatMessage] = []
            effective_system = (
                _classification_prefix +
                (system_prompt or "You are JARVIS, a helpful AI assistant.")
            )
            messages.append(ChatMessage(role="system", content=effective_system))
            messages.append(ChatMessage(role="user", content=query))

            # v242.2: Try Claude API first (preferred fallback)
            response = await self._execute_completion(
                mode=RoutingMode.CLAUDE_API,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            if not response.success:
                # Claude failed -- fall back to Gemini as last resort
                logger.warning(f"[v242.2] Claude brain vacuum failed ({response.error}), trying Gemini")
                response = await self._execute_completion(
                    mode=RoutingMode.GEMINI_API,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.7,
                )

            if response.success:
                content = response.content or ""
                # v244.0: Parse classification from response
                intent, domain, requires_action, suggested_actions = self._parse_classification(content)
                # Strip the CLASSIFICATION line from the content shown to user
                clean_content = self._strip_classification_line(content)

                return StructuredResponse(
                    content=clean_content,
                    intent=intent,
                    domain=domain,
                    requires_action=requires_action,
                    suggested_actions=suggested_actions,
                    generator_model=response.backend,
                    generation_ms=int(response.latency_ms),
                    source="claude_fallback",
                )

            logger.error(
                f"[v242] Brain vacuum: API fallback also failed: {response.error}"
            )
        except Exception as e:
            logger.error(f"[v242] Brain vacuum: all backends failed: {e}")

        return StructuredResponse(
            content="I'm still starting up. Please try again in a moment.",
            intent="answer",
            source="error",
        )

    def _parse_classification(self, content: str) -> tuple:
        """Parse CLASSIFICATION JSON from LLM response.

        v244.0: Returns (intent, domain, requires_action, suggested_actions).
        Falls back to conservative defaults if parsing fails.
        """
        import json as _json
        import re as _re

        _valid_intents = {"answer", "conversation", "action", "vision_needed", "multi_step_action", "clarify"}
        _valid_domains = {"general", "system", "security", "workspace", "development", "media", "smart_home"}

        try:
            match = _re.search(r'CLASSIFICATION:\s*(\{[^}]+\})', content)
            if match:
                data = _json.loads(match.group(1))
                intent = data.get("intent", "answer")
                domain = data.get("domain", "general")
                requires_action = data.get("requires_action", False)
                suggested_actions = data.get("suggested_actions", [])

                # Validate
                if intent not in _valid_intents:
                    intent = "answer"
                if domain not in _valid_domains:
                    domain = "general"
                if not isinstance(suggested_actions, list):
                    suggested_actions = []

                return intent, domain, bool(requires_action), suggested_actions
        except Exception as e:
            logger.debug(f"[v244.0] Classification parse failed: {e}")

        return "answer", "general", False, []

    def _strip_classification_line(self, content: str) -> str:
        """Remove the CLASSIFICATION: {...} line from content.

        v244.0: Strips the machine-readable classification prefix so the
        user only sees the natural language response.
        """
        import re as _re
        cleaned = _re.sub(r'CLASSIFICATION:\s*\{[^}]+\}\s*\n?', '', content, count=1)
        return cleaned.strip()
```

**Test:** Call `_brain_vacuum_fallback("lock my screen", None, 1024)` with mocked Claude response containing `CLASSIFICATION: {"intent": "action", "domain": "system", "requires_action": true, "suggested_actions": ["lock_screen"]}`. Verify returned StructuredResponse has `intent="action"`, not `"answer"`.

**Commit message:** `fix: add classification prompt to brain vacuum fallback for proper intent routing`

---

## Execution Order

```
Task 1 — Dead instance vars in __init__ (prerequisite for Task 2)
Task 2 — Dead closures + method rename (depends on Task 1)
Task 3 — Dead code in agentic_api.py (independent)
Task 4 — Dead import in voice_auth (independent)
Task 5 — Delete dead test + stale comment (independent)
Task 6 — Health status endpoint cleanup (depends on Task 1)
Task 7 — Cloud SQL settling delay fix (independent of Section 1)
Task 8 — TCP fallback + reconnect fix (independent)
Task 9 — Redundant health check removal (independent)
Task 10 — Brain vacuum classification (independent of Sections 1-2)
```

**Recommended sequence:** 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10

Tasks 3, 4, 5 could run in parallel after Task 1, but sequential is safer for review quality.

---

## Verification Criteria

After all tasks:

1. **Zero warnings at startup** from `tiered_vbia_adapter`, `TieredCommandRouter`, or `execute_tier2`
2. **Cloud SQL proxy completes in <10s** (down from 47s) when proxy binary is installed and GCP is reachable
3. **No double DB verification** — `ensure_proxy_ready()` is the single source of truth
4. **Brain vacuum commands classify correctly** — "lock my screen" gets `intent="action"`, not `"answer"`
5. **AgenticTaskRunner still initializes** — runner creation is preserved (only dead wiring removed)
6. **No regressions** in normal operation (J-Prime up, proxy working)
7. **Health endpoint** no longer reports dead `vbia_adapter` or `router` keys
8. **No remaining references** to `tiered_command_router.py` or `tiered_vbia_adapter` in codebase (except the v244.0 explanatory comments)
