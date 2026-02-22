# Startup Warning Root Fix — v244.0 Design

**Date:** 2026-02-22
**Status:** Approved
**Scope:** 3 root cause fixes eliminating startup warnings + 1 brain vacuum classification enhancement

---

## Problem Statement

Three warning clusters appear on every JARVIS startup:

1. `No module named 'core.tiered_vbia_adapter'` — dead import after intentional deletion (commit 167fcecb)
2. `Cannot wire execute_tier2 (missing: TieredCommandRouter)` — same commit, dead wiring code
3. `cloud_sql_proxy timeout (47s)` — redundant settling delay after proxy is already authenticated

Plus a latent issue discovered during analysis:
4. Brain vacuum fallback hardcodes `intent="answer"` — all commands become text responses when J-Prime is down

---

## Design

### Section 1: Dead Code Surgical Removal

**Approach:** Finish the cleanup started in commit 167fcecb. Remove dead import paths, preserve live dependencies, verify fallback safety.

**Files:**

| File | Action |
|------|--------|
| `unified_supervisor.py` | Remove `_init_vbia()`, `_init_router()` closures, `execute_tier2` wiring block. Keep `_init_cross_repo()` and AgenticTaskRunner init. Rename `_initialize_two_tier_security()` → `_initialize_integration_components()`. |
| `backend/api/agentic_api.py` | Remove dead `/route` endpoint entirely (always returns 503). Remove `get_tiered_router`, `get_vbia_adapter`, `CommandTier`, `RouteCommandRequest`, `RouteCommandResponse` dead imports. |
| `backend/core/voice_authentication_layer.py` | Remove dead `get_tiered_vbia_adapter` import. Auth bypass methods (`verify_for_tier2`, `verify_for_tier1`) have zero callers — safe. |
| `test_two_tier_security.py` | Delete entirely (100% tests deleted components). |
| `backend/neural_mesh/agents/google_workspace_agent.py` | Update stale comment referencing `tiered_command_router.py:1072`. |

**Safety verifications (all passed during design):**
- `execute_tier2` wiring is guarded by `if self._tiered_router and self._agentic_runner:` — always False since module deleted. AgenticTaskRunner works independently.
- `/route` endpoint in agentic_api.py already returns HTTP 503 (get_tiered_router() returns None). Entirely dead.
- `_vbia_adapter = None` bypasses auth, but `verify_for_tier2()`/`verify_for_tier1()` have zero callers. Voice unlock uses separate `SpeakerVerificationService`.

### Section 2: Cloud SQL Proxy Readiness Gate Fix

**Root cause:** Log-signal gating already exists in `proxy_manager.start()`. The proxy IS authenticated when `start()` returns. The bug is `ProxyReadinessGate.ensure_proxy_ready()` applying a redundant 2s settling delay, then running 5x SELECT 1 checks against an already-ready proxy. Plus double DB verification in the supervisor wastes startup time.

**Files:**

| File | Action |
|------|--------|
| `intelligence/cloud_sql_connection_manager.py` | In `ensure_proxy_ready()`: when `just_started_proxy=True` and start() returned success, set `proxy_settling_delay = 0`. Only apply settling delay when reusing an already-running proxy. |
| `unified_supervisor.py` | Remove redundant `check_connection_health()` call between `start()` and `ensure_proxy_ready()`. Gate is single source of truth. |
| `intelligence/cloud_sql_proxy_manager.py` | Extend TCP-only fallback from 3s to 10s (configurable via `CLOUDSQL_LOG_READY_FALLBACK_TIMEOUT`). Fix `recover_proxy()` hardcoded 3s sleep to use same log-signal-aware logic. |

**Expected outcome:** Startup from ~47s → ~3-5s for cloud_sql_proxy.

### Section 3: Brain Vacuum Classification Fix

**Root cause:** `_brain_vacuum_fallback()` in `jarvis_prime_client.py` hardcodes `intent="answer"`. All commands become text responses during brain vacuum (J-Prime down).

**Files:**

| File | Action |
|------|--------|
| `backend/core/jarvis_prime_client.py` | Modify `_brain_vacuum_fallback()` to include classification prompt prefix in Claude system prompt. Parse `CLASSIFICATION: {...}` JSON from response. Populate StructuredResponse fields. Fallback to `intent="answer"` if parsing fails. |
| `backend/core/jarvis_prime_client.py` | Add `metadata` field to `CompletionResponse` returned by `_complete_claude()` and `_complete_gemini()` when routing info is available. |

---

## Gap Tracker (13 Gaps from Critique)

| # | Problem | P | Gap | Resolution |
|---|---------|---|-----|------------|
| 1 | Dead Code | P0 | `_initialize_two_tier_security()` has live deps | Keep `_init_cross_repo()` + AgenticTaskRunner; remove only dead code |
| 2 | Dead Code | P1 | `agentic_api.py` Tier 2 branch silently skipped | Remove entire dead `/route` endpoint |
| 3 | Dead Code | P1 | Voice auth bypassed when adapter is None | Bypass methods have zero callers; safe to leave |
| 4 | Dead Code | P2 | `test_two_tier_security.py` 100% dead | Delete entirely |
| 5 | Dead Code | P2 | Stale comment in google_workspace_agent.py | Update comment |
| 6 | SQL Proxy | P0 | Settling delay redundant, not insufficient | Skip delay when start() confirmed readiness |
| 7 | SQL Proxy | P1 | Double DB verification | Remove redundant check_connection_health() |
| 8 | SQL Proxy | P1 | 3s TCP-only fallback defeats log gating | Extend to 10s, make configurable |
| 9 | SQL Proxy | P2 | `_check_log_ready` reads entire file | Not fixing (negligible risk for small log) |
| 10 | SQL Proxy | P2 | `recover_proxy()` has hardcoded 3s sleep | Apply same log-signal-aware logic |
| 11 | Cross-Repo | P1 | Brain vacuum hardcodes `intent="answer"` | Classification prompt in brain vacuum |
| 12 | Cross-Repo | P1 | Claude/Gemini return no metadata | Add metadata field to CompletionResponse |
| 13 | Cross-Repo | P2 | Commit evidence conflated | Documentation note only |

---

## Verification Criteria

After implementation:

1. **Zero warnings at startup** from tiered_vbia_adapter, TieredCommandRouter, or execute_tier2
2. **Cloud SQL proxy completes in <10s** (down from 47s) when proxy binary is installed and GCP is reachable
3. **learning_database initializes** instead of being fast-forward skipped
4. **Brain vacuum commands execute** ("lock my screen" during J-Prime downtime actually locks the screen)
5. **No regressions** in normal operation (J-Prime up, proxy working)
