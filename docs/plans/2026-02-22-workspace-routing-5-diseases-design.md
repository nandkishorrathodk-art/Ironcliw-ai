# Google Workspace Routing: 5-Disease Cure Design

**Date:** 2026-02-22
**Status:** Design
**Repos:** jarvis-prime, Ironcliw-AI-Agent

## Problem Statement

The Google Workspace integration has 5 root diseases causing silent data loss, double classification, uninitialized agents, missing fallback routing, and zombie API calls. The most visible symptom: users see "Workspace command completed successfully" even when Gmail returns perfect email data.

## Root Diseases

| # | Disease | Root Cause |
|---|---------|------------|
| 1 | Double classification | UCP sends `action: "handle_workspace_query"` forcing agent to re-parse what J-Prime already classified |
| 2 | Summarizer intent mismatch | `_summarize_workspace_result()` receives `intent="action"` but expects `"check_email"` |
| 3 | Lazy singleton skips lifecycle | `GoogleWorkspaceAgent()` constructed without `on_initialize()` |
| 4 | Brain vacuum drops workspace actions | Fallback classification prompt has no workspace `suggested_actions` examples |
| 5 | No deadline propagation | `deadline` never passed to `execute_task()` — zombie API calls |

## Architecture

**Principle: Brain decides, Body executes, Reactor learns.**

```
J-Prime Phi classifier
  -> suggested_actions: ["fetch_unread_emails"]
  -> x_jarvis_routing includes suggested_actions in ALL response paths

UCP._handle_workspace_action()
  -> reads response.suggested_actions[0]
  -> validates against agent.capabilities (single source of truth)
  -> wraps execute_task() in asyncio.wait_for(deadline)

GoogleWorkspaceAgent.execute_task({action: "fetch_unread_emails"})
  -> direct dispatch (no re-classification)
  -> returns workspace_action field in result dict

UCP._summarize_workspace_result()
  -> keys off result["workspace_action"], NOT response.intent
```

The keyword detector `_handle_natural_query()` is NOT deleted — it becomes the graceful degradation fallback when `suggested_actions` is empty.

## Files Changed

| # | File | Repo | Changes |
|---|------|------|---------|
| 1 | `jarvis_prime/core/classification_schema.py` | jarvis-prime | Add `suggested_actions` to schema + prompt examples |
| 2 | `run_server.py` | jarvis-prime | Add `suggested_actions` to escalation, self-serve, circuit breaker routing dicts; use `SCHEMA_VERSION` constant |
| 3 | `backend/api/unified_command_processor.py` | Ironcliw | Fix workspace action routing, summarizer, lazy singleton, deadline |
| 4 | `backend/core/jarvis_prime_client.py` | Ironcliw | Add workspace examples to brain vacuum prompt |
| 5 | `backend/neural_mesh/agents/google_workspace_agent.py` | Ironcliw | Add `workspace_action` to all handler results |

No new files. No deletions. 5 files edited across 2 repos.

## Disease 1: Phi Schema Extension + Direct Action Routing

### classification_schema.py

Add `suggested_actions` as optional property (not in `required` — backward compatible):

```python
CLASSIFICATION_SCHEMA["properties"]["suggested_actions"] = {
    "type": "array",
    "items": {"type": "string"},
}
```

Add workspace action examples to `build_classifier_system_prompt()`:

```
- "check my email" -> intent=action, domain=workspace, suggested_actions=["fetch_unread_emails"]
- "what's on my calendar today" -> intent=action, domain=workspace, suggested_actions=["check_calendar_events"]
- "send an email to John" -> intent=action, domain=workspace, suggested_actions=["send_email"]
- "draft a reply to that email" -> intent=action, domain=workspace, suggested_actions=["draft_email_reply"]
- "schedule a meeting tomorrow" -> intent=action, domain=workspace, suggested_actions=["create_calendar_event"]
- "give me my briefing" -> intent=action, domain=workspace, suggested_actions=["daily_briefing"]
```

### run_server.py (Gap 1 fix: all routing paths)

Add `"suggested_actions": _classification.get("suggested_actions", [])` to:
- Escalation early return (line 1554-1563)
- Phi self-serve (line 1614-1627)
- Circuit breaker fallback (line 1529-1538) — empty list default

Hollow Client routes (lines 1328, 1395) are NOT changed — they proxy to GCP and don't carry classification metadata.

### run_server.py (Gap 6 fix: schema version)

Replace hardcoded `"schema_version": 1` with `SCHEMA_VERSION` in:
- Circuit breaker fallback (line 1529)
- Escalation early return (line 1555)

### unified_command_processor.py

In `_handle_workspace_action()`:

```python
# Primary: J-Prime suggested action (Disease 1 cure)
workspace_action = "handle_workspace_query"  # fallback
if hasattr(response, 'suggested_actions') and response.suggested_actions:
    candidate = response.suggested_actions[0]
    # Gap 4 fix: validate against agent's capabilities (single source of truth)
    agent_capabilities = getattr(agent, 'capabilities', set()) if agent else set()
    if candidate in agent_capabilities:
        workspace_action = candidate
```

No static `_VALID_WORKSPACE_ACTIONS` set — the agent instance IS the authority.

## Disease 2: Summarizer Keys Off Executed Action

### unified_command_processor.py

Change the summarizer call from:

```python
summary = self._summarize_workspace_result(
    result, getattr(response, 'intent', 'workspace')
)
```

To:

```python
workspace_intent = result.get("workspace_action") or workspace_action
summary = self._summarize_workspace_result(result, workspace_intent)
```

### google_workspace_agent.py

Each handler adds `workspace_action` to its return dict:

```python
# _fetch_unread_emails: result["workspace_action"] = "fetch_unread_emails"
# _check_calendar:      result["workspace_action"] = "check_calendar_events"
# _get_workspace_summary: result["workspace_action"] = "workspace_summary"
# _send_email:          result["workspace_action"] = "send_email"
# _draft_email:         result["workspace_action"] = "draft_email_reply"
# _create_event:        result["workspace_action"] = "create_calendar_event"
# _get_contacts:        result["workspace_action"] = "get_contacts"
# _create_document:     result["workspace_action"] = "create_document"
# _handle_natural_query: result["workspace_action"] = detected_intent_action
```

### Summarizer expansion (Gap 7)

Add formatting for write operations:

```python
elif intent in ("send_email",):
    return result.get("message", "Email sent successfully.")
elif intent in ("draft_email_reply",):
    return result.get("message", "Email draft created.")
elif intent in ("create_calendar_event",):
    return result.get("message", "Calendar event created.")
elif intent in ("get_contacts",):
    contacts = result.get("contacts", [])
    if not contacts:
        return "No contacts found."
    lines = [f"Found {len(contacts)} contact(s):"]
    for c in contacts[:5]:
        name = c.get("name", "Unknown")
        email = c.get("email", "")
        lines.append(f"  - {name} ({email})")
    return "\n".join(lines)
elif intent in ("create_document",):
    return result.get("message", "Document created successfully.")
```

## Disease 3: Lazy Singleton Calls on_initialize()

### unified_command_processor.py

```python
if agent is None:
    if not hasattr(self, '_workspace_agent_singleton'):
        try:
            from neural_mesh.agents.google_workspace_agent import GoogleWorkspaceAgent
            _agent = GoogleWorkspaceAgent()
            # Disease 3 cure: initialize the agent lifecycle.
            # Gap 3 note: The lazy singleton intentionally lacks a message bus.
            # on_initialize() checks `if self.message_bus:` before subscribing,
            # so this is safe. The agent operates in standalone mode (no mesh).
            await _agent.on_initialize()
            self._workspace_agent_singleton = _agent
        except Exception:
            logger.warning("[v242] GoogleWorkspaceAgent initialization failed", exc_info=True)
            self._workspace_agent_singleton = None
    agent = self._workspace_agent_singleton
```

## Disease 4: Brain Vacuum Workspace Examples

### jarvis_prime_client.py

Add workspace action examples to the brain vacuum classification prefix:

```python
"- suggested_actions: list specific actions like [\"lock_screen\"], [\"open_browser\"], etc.\n"
"  For workspace domain: [\"fetch_unread_emails\"], [\"check_calendar_events\"], "
"[\"send_email\"], [\"draft_email_reply\"], [\"create_calendar_event\"], [\"daily_briefing\"]\n\n"
```

## Disease 5: Deadline Propagation via asyncio.wait_for()

### unified_command_processor.py (Gap 5 fix)

Wrap `execute_task()` at UCP level instead of per-handler:

```python
import time as _time

remaining = (deadline - _time.monotonic()) if deadline else 30.0
timeout = max(remaining, 1.0)

try:
    result = await asyncio.wait_for(
        agent.execute_task({
            "action": workspace_action,
            "query": command_text,
        }),
        timeout=timeout,
    )
except asyncio.TimeoutError:
    return {
        "success": False,
        "response": "Workspace request timed out. Please try again.",
        "command_type": "WORKSPACE",
        "error": "deadline_exceeded",
    }
```

This matches the pattern in `_handle_system_action_via_jprime` (line 2107: `asyncio.wait_for(..., timeout=20.0)`).

No changes needed inside GoogleWorkspaceAgent handlers — the UCP-level timeout cancels the coroutine on expiry.

## Edge Cases & Failure Modes

1. **Old J-Prime without schema change**: `suggested_actions` defaults to `[]` -> UCP falls back to `"handle_workspace_query"` -> keyword detector handles it
2. **Phi hallucinates unknown action**: `agent.capabilities` set rejects it -> falls back to keyword detection
3. **Agent not initialized**: `on_initialize()` fails -> singleton is None -> J-Prime text response returned
4. **Deadline expires during waterfall**: `asyncio.wait_for()` cancels the task -> user gets timeout message
5. **Brain vacuum + workspace**: Claude sees examples -> outputs suggested_actions -> UCP routes correctly. If not -> empty list -> keyword fallback
6. **Escalated workspace query**: `suggested_actions` now included in escalation routing (Gap 1 fix) -> Claude escalation path has the action
7. **Concurrent singleton init**: Safe — asyncio single-threaded, hasattr+await is atomic within one coroutine
8. **Grammar compilation**: `suggested_actions` as optional array-of-strings may affect grammar size — test on GCP VM (Gap 8)

## Gap 2 Acknowledgment

`_handle_system_action_via_jprime` accepts `suggested_actions` but never reads it. The workspace fix is the FIRST domain where `suggested_actions` is consumed end-to-end. This is acceptable — the workspace implementation proves the pattern, and system actions can follow the same pattern later.

## Testing Strategy

1. Grammar compilation: verify `LlamaGrammar.from_json_schema()` handles optional array property
2. Phi classification: verify workspace queries produce correct `suggested_actions` values
3. End-to-end: "check my email" -> emails displayed (not "Workspace command completed successfully")
4. Brain vacuum: J-Prime down -> "check my email" -> still routes correctly via Claude fallback
5. Escalation: complex workspace query -> escalation path preserves suggested_actions
6. Deadline: slow Gmail API -> user sees timeout within deadline, not hang
7. Lazy singleton: Neural Mesh unavailable -> singleton initializes with on_initialize() -> works
