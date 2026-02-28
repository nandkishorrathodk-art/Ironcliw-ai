# Workspace Routing 5-Disease Cure — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Cure 5 root diseases in Google Workspace routing so commands like "check my email" produce formatted email results instead of "Workspace command completed successfully."

**Architecture:** J-Prime's Phi classifier outputs `suggested_actions` in the classification schema. The UCP reads that action and passes it directly to `GoogleWorkspaceAgent.execute_task()`, bypassing the keyword re-classifier. Each handler returns a `workspace_action` field that the summarizer uses for formatting. Deadline is enforced via `asyncio.wait_for()` at the UCP level.

**Tech Stack:** Python 3, asyncio, llama-cpp-python (grammar-constrained decoding), jarvis-prime Phi classifier, Neural Mesh agent framework

**Design Doc:** `docs/plans/2026-02-22-workspace-routing-5-diseases-design.md`

---

### Task 1: Add `suggested_actions` to Phi Classification Schema (Disease 1 — jarvis-prime)

**Files:**
- Modify: `/Users/djrussell23/Documents/repos/jarvis-prime/jarvis_prime/core/classification_schema.py:13-49` (schema dict)
- Modify: `/Users/djrussell23/Documents/repos/jarvis-prime/jarvis_prime/core/classification_schema.py:85-111` (prompt builder)

**Step 1: Add `suggested_actions` property to `CLASSIFICATION_SCHEMA`**

In `classification_schema.py`, add after line 42 (`"confidence": {"type": "number"},`):

```python
        "suggested_actions": {
            "type": "array",
            "items": {"type": "string"},
        },
```

This is NOT added to `required` — it's optional for backward compatibility. `LlamaGrammar.from_json_schema()` will auto-generate GBNF rules for the optional array property.

**Step 2: Add `suggested_actions` field description and workspace examples to classifier prompt**

In `build_classifier_system_prompt()`, after line 100 (the `confidence` field description), add a new field description:

```python
- suggested_actions: optional list of specific action names the Body should execute. Use for action/multi_step_action intents.
  System actions: ["lock_screen"], ["unlock_screen"], ["open_browser"], ["volume_up"], ["volume_down"]
  Workspace actions: ["fetch_unread_emails"], ["check_calendar_events"], ["send_email"], ["draft_email_reply"], ["create_calendar_event"], ["daily_briefing"], ["workspace_summary"], ["get_contacts"], ["create_document"]
```

Then update the existing workspace examples (lines 109-111) to include `suggested_actions`:

```
- "check my email" -> intent=action, domain=workspace, complexity=simple, confidence=0.93, suggested_actions=["fetch_unread_emails"]
- "what's on my calendar today" -> intent=action, domain=workspace, complexity=simple, confidence=0.91, suggested_actions=["check_calendar_events"]
- "send an email to John" -> intent=action, domain=workspace, complexity=moderate, confidence=0.90, suggested_actions=["send_email"]
```

And add new examples:

```
- "draft a reply to that email" -> intent=action, domain=workspace, complexity=moderate, confidence=0.88, suggested_actions=["draft_email_reply"]
- "schedule a meeting for tomorrow" -> intent=action, domain=workspace, complexity=moderate, confidence=0.89, suggested_actions=["create_calendar_event"]
- "give me my morning briefing" -> intent=action, domain=workspace, complexity=simple, confidence=0.92, suggested_actions=["daily_briefing"]
```

**Step 3: Verify syntax**

Run: `cd /Users/djrussell23/Documents/repos/jarvis-prime && python3 -c "from jarvis_prime.core.classification_schema import CLASSIFICATION_SCHEMA, build_classifier_system_prompt; print('Schema OK:', 'suggested_actions' in CLASSIFICATION_SCHEMA['properties']); print('Prompt length:', len(build_classifier_system_prompt()))"`

Expected: `Schema OK: True` and a prompt length number (no import errors).

**Step 4: Commit**

```bash
cd /Users/djrussell23/Documents/repos/jarvis-prime
git add jarvis_prime/core/classification_schema.py
git commit -m "feat(classification): add suggested_actions to Phi schema for direct workspace routing

Disease 1 cure: Phi classifier now outputs suggested_actions field
with workspace-specific action names (fetch_unread_emails, etc.).
This eliminates double classification — the UCP can route directly
to the correct handler without re-parsing natural language."
```

---

### Task 2: Fix `x_jarvis_routing` in ALL Response Paths (Gap 1 + Gap 6 — jarvis-prime)

**Files:**
- Modify: `/Users/djrussell23/Documents/repos/jarvis-prime/run_server.py:1529-1538` (circuit breaker fallback)
- Modify: `/Users/djrussell23/Documents/repos/jarvis-prime/run_server.py:1554-1563` (escalation early return)
- Modify: `/Users/djrussell23/Documents/repos/jarvis-prime/run_server.py:1594-1601` (Phi self-serve)

**Step 1: Fix circuit breaker fallback (line 1529-1538)**

Add `SCHEMA_VERSION` import at top of file (verify it's already imported — it should be from the existing `from jarvis_prime.core.classification_schema import CLASSIFICATION_SCHEMA, ...` line).

Replace `"schema_version": 1,` with `"schema_version": SCHEMA_VERSION,` and add `"suggested_actions": [],`:

```python
        _classification = {
            "schema_version": SCHEMA_VERSION,
            "intent": "answer",
            "domain": "general",
            "complexity": "simple",
            "confidence": 0.0,
            "requires_vision": False,
            "requires_action": False,
            "escalate_to_claude": False,
            "suggested_actions": [],
        }
```

**Step 2: Fix escalation early return (line 1554-1563)**

Replace `"schema_version": 1,` with `"schema_version": SCHEMA_VERSION,` and add `"suggested_actions"`:

```python
            "x_jarvis_routing": {
                "schema_version": SCHEMA_VERSION,
                "intent": _classification.get("intent", "answer"),
                "domain": _classification.get("domain", "general"),
                "escalate_to_claude": True,
                "escalation_reason": _classification.get("escalation_reason", "complexity"),
                "confidence": _classification.get("confidence", 0.0),
                "classifier_model": "phi-3.5-mini-q4km",
                "classification_ms": _classification_ms,
                "suggested_actions": _classification.get("suggested_actions", []),
            },
```

**Step 3: Fix Phi self-serve path (line 1594-1601)**

Add `"suggested_actions"` to the self-serve `x_jarvis_routing` dict:

```python
                "x_jarvis_routing": {
                    "schema_version": _classification.get("schema_version", SCHEMA_VERSION),
                    "intent": _classification.get("intent", "conversation"),
                    "domain": _classification.get("domain", "conversation"),
                    "complexity": _classification.get("complexity", "trivial"),
                    "confidence": _classification.get("confidence", 0.0),
                    "requires_vision": False,
                    "requires_action": False,
                    "escalate_to_claude": False,
                    "source": "phi_self_serve",
                    "classifier_model": "phi-3.5-mini-q4km",
                    "classification_ms": _classification_ms,
                    "generation_ms": _phi_gen_ms,
                    "suggested_actions": _classification.get("suggested_actions", []),
                },
```

**Step 4: Verify `SCHEMA_VERSION` is imported**

Run: `cd /Users/djrussell23/Documents/repos/jarvis-prime && grep -n "SCHEMA_VERSION" run_server.py | head -5`

If not imported, add it to the existing import line from `classification_schema`.

**Step 5: Commit**

```bash
cd /Users/djrussell23/Documents/repos/jarvis-prime
git add run_server.py
git commit -m "fix(routing): propagate suggested_actions through ALL x_jarvis_routing paths

Gap 1: Escalation + self-serve paths were dropping suggested_actions.
Gap 6: Circuit breaker + escalation had hardcoded schema_version: 1.

Now all 5 routing construction sites include suggested_actions and
use SCHEMA_VERSION constant for version consistency."
```

---

### Task 3: Rewrite `_handle_workspace_action()` — Direct Routing + Lifecycle + Deadline (Diseases 1, 3, 5 — Ironcliw)

**Files:**
- Modify: `/Users/djrussell23/Documents/repos/Ironcliw-AI-Agent/backend/api/unified_command_processor.py:2205-2270`

**Step 1: Rewrite `_handle_workspace_action()` method**

Replace lines 2205-2270 with:

```python
    async def _handle_workspace_action(
        self, command_text: str, response: Any,
        deadline: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Route workspace commands to GoogleWorkspaceAgent with direct action dispatch.

        v266.0: Cures 3 root diseases:
        - Disease 1: Reads response.suggested_actions[0] for direct routing
          instead of forcing agent to re-classify via keyword waterfall.
        - Disease 3: Lazy singleton calls on_initialize() for proper lifecycle.
        - Disease 5: Wraps execute_task() in asyncio.wait_for() for deadline
          enforcement, preventing zombie API calls.

        Falls back to 'handle_workspace_query' (keyword detector) when
        suggested_actions is empty (brain vacuum, old J-Prime, etc).
        """
        import time as _time

        if deadline:
            _remaining = deadline - _time.monotonic()
            if _remaining <= 0:
                return {"success": False, "response": "Request timed out",
                        "command_type": "WORKSPACE", "error": "deadline_exceeded"}
        self._v242_metrics["workspace_requests"] += 1
        try:
            # Prefer existing agent instance from Neural Mesh coordinator
            agent = None
            coordinator = self._get_neural_mesh_coordinator()
            if coordinator:
                agent = coordinator.get_agent("google_workspace_agent")

            if agent is None:
                # Lazy singleton fallback — Disease 3 cure: call on_initialize()
                # The lazy singleton intentionally lacks a message bus.
                # on_initialize() checks `if self.message_bus:` before subscribing,
                # so this is safe. The agent operates in standalone mode (no mesh).
                if not hasattr(self, '_workspace_agent_singleton'):
                    try:
                        from neural_mesh.agents.google_workspace_agent import GoogleWorkspaceAgent
                        _agent = GoogleWorkspaceAgent()
                        await _agent.on_initialize()
                        self._workspace_agent_singleton = _agent
                    except Exception:
                        logger.warning("[v266] GoogleWorkspaceAgent initialization failed", exc_info=True)
                        self._workspace_agent_singleton = None
                agent = self._workspace_agent_singleton

            if agent is None:
                # No workspace agent available — return J-Prime's text response
                return {
                    "success": True,
                    "response": response.content or "Workspace features are not currently available.",
                    "command_type": "WORKSPACE",
                    "source": getattr(response, 'source', 'unknown'),
                }

            # Disease 1 cure: read suggested_actions from J-Prime classification.
            # Validate against agent's capabilities (single source of truth — Gap 4).
            workspace_action = "handle_workspace_query"  # fallback to keyword detector
            if hasattr(response, 'suggested_actions') and response.suggested_actions:
                candidate = response.suggested_actions[0]
                agent_capabilities = getattr(agent, 'capabilities', set())
                if candidate in agent_capabilities:
                    workspace_action = candidate
                else:
                    logger.debug(
                        f"[v266] suggested_action '{candidate}' not in agent capabilities, "
                        f"falling back to handle_workspace_query"
                    )

            # Disease 5 cure: enforce deadline via asyncio.wait_for() at UCP level.
            # This cancels the coroutine on timeout, preventing zombie API calls.
            remaining = (deadline - _time.monotonic()) if deadline else 30.0
            timeout = max(remaining, 1.0)

            result = await asyncio.wait_for(
                agent.execute_task({
                    "action": workspace_action,
                    "query": command_text,
                }),
                timeout=timeout,
            )

            # Disease 2 cure: key summarizer off the executed action, not J-Prime's intent.
            result_dict = result if isinstance(result, dict) else {"response": str(result)}
            workspace_intent = result_dict.get("workspace_action") or workspace_action
            summary = self._summarize_workspace_result(result_dict, workspace_intent)
            return {
                "success": True,
                "response": summary,
                "command_type": "WORKSPACE",
                "source": getattr(response, 'source', 'unknown'),
            }
        except asyncio.TimeoutError:
            logger.warning(f"[v266] Workspace action timed out after deadline")
            return {
                "success": False,
                "response": "Workspace request timed out. Please try again.",
                "command_type": "WORKSPACE",
                "error": "deadline_exceeded",
            }
        except Exception as e:
            logger.error(f"[v266] Workspace action failed: {e}", exc_info=True)
            # Fallback: return J-Prime's text response
            return {
                "success": True,
                "response": getattr(response, 'content', None) or "I couldn't complete that workspace action.",
                "command_type": "WORKSPACE",
                "source": getattr(response, 'source', 'unknown'),
            }
```

**Step 2: Verify syntax**

Run: `cd /Users/djrussell23/Documents/repos/Ironcliw-AI-Agent && python3 -c "import ast; ast.parse(open('backend/api/unified_command_processor.py').read()); print('Syntax OK')"`

Expected: `Syntax OK`

**Step 3: Commit**

```bash
cd /Users/djrussell23/Documents/repos/Ironcliw-AI-Agent
git add backend/api/unified_command_processor.py
git commit -m "fix(workspace): rewrite _handle_workspace_action for direct routing + lifecycle + deadline

Disease 1: Reads response.suggested_actions[0] — no more double classification.
Disease 3: Lazy singleton now calls on_initialize() for proper lifecycle.
Disease 5: asyncio.wait_for() enforces deadline on execute_task().
Gap 4: Validates actions against agent.capabilities (single source of truth).
Gap 5: Deadline at UCP level, not per-handler."
```

---

### Task 4: Fix `_summarize_workspace_result()` (Disease 2 + Gap 7 — Ironcliw)

**Files:**
- Modify: `/Users/djrussell23/Documents/repos/Ironcliw-AI-Agent/backend/api/unified_command_processor.py:1675-1706`

**Step 1: Expand the summarizer with formatting for all workspace actions**

Replace lines 1675-1706:

```python
    @staticmethod
    def _summarize_workspace_result(result: dict, intent: str) -> str:
        """Generate a human-readable response from structured workspace data.

        v266.0: Keys off the workspace_action field (from the agent's result dict
        or the UCP's resolved action), NOT J-Prime's high-level intent.
        Added formatting for send_email, draft_email_reply, create_calendar_event,
        get_contacts, create_document, and handle_workspace_query.
        """
        # Email check
        if intent in ("check_email", "fetch_unread_emails"):
            count = result.get("count", 0)
            total = result.get("total_unread", count)
            if count == 0:
                return "No unread emails found."
            emails = result.get("emails", [])
            lines = [f"You have {total} unread email{'s' if total != 1 else ''}. Here are the latest {count}:"]
            for em in emails[:5]:
                subj = em.get("subject", "(no subject)")
                sender = em.get("from", "unknown")
                lines.append(f"  - {subj} (from {sender})")
            if count > 5:
                lines.append(f"  ...and {count - 5} more")
            return "\n".join(lines)

        # Calendar check
        elif intent in ("check_calendar", "check_calendar_events"):
            events = result.get("events", [])
            if not events:
                return "No events on your calendar for this time period."
            lines = [f"You have {len(events)} event{'s' if len(events) != 1 else ''}:"]
            for ev in events[:5]:
                summary = ev.get("summary", ev.get("title", "(untitled)"))
                start = ev.get("start", "")
                lines.append(f"  - {summary} ({start})")
            return "\n".join(lines)

        # Workspace summary / daily briefing
        elif intent in ("workspace_summary", "daily_briefing"):
            return result.get("brief") or result.get("summary", "Workspace summary completed.")

        # Send email
        elif intent == "send_email":
            if result.get("error"):
                return f"Failed to send email: {result['error']}"
            return result.get("message", "Email sent successfully.")

        # Draft email
        elif intent == "draft_email_reply":
            if result.get("error"):
                return f"Failed to create draft: {result['error']}"
            return result.get("message", "Email draft created.")

        # Create calendar event
        elif intent == "create_calendar_event":
            if result.get("error"):
                return f"Failed to create event: {result['error']}"
            return result.get("message", "Calendar event created.")

        # Get contacts
        elif intent == "get_contacts":
            contacts = result.get("contacts", [])
            if result.get("error"):
                return f"Failed to fetch contacts: {result['error']}"
            if not contacts:
                return "No contacts found."
            lines = [f"Found {len(contacts)} contact{'s' if len(contacts) != 1 else ''}:"]
            for c in contacts[:5]:
                name = c.get("name", "Unknown")
                email = c.get("email", "")
                lines.append(f"  - {name} ({email})" if email else f"  - {name}")
            if len(contacts) > 5:
                lines.append(f"  ...and {len(contacts) - 5} more")
            return "\n".join(lines)

        # Create document
        elif intent == "create_document":
            if result.get("error"):
                return f"Failed to create document: {result['error']}"
            return result.get("message", "Document created successfully.")

        # Search email
        elif intent == "search_email":
            emails = result.get("emails", [])
            if not emails:
                return "No emails found matching your search."
            lines = [f"Found {len(emails)} email{'s' if len(emails) != 1 else ''}:"]
            for em in emails[:5]:
                subj = em.get("subject", "(no subject)")
                sender = em.get("from", "unknown")
                lines.append(f"  - {subj} (from {sender})")
            return "\n".join(lines)

        # handle_workspace_query (keyword detector fallback) — pass through
        elif intent == "handle_workspace_query":
            # The keyword detector routes to a handler which returns its own result.
            # Try to extract a meaningful response from common result fields.
            if result.get("brief"):
                return result["brief"]
            if result.get("message"):
                return result["message"]
            if result.get("response"):
                return str(result["response"])
            if result.get("emails"):
                # Keyword detector routed to email check — format it
                return UnifiedCommandProcessor._summarize_workspace_result(result, "fetch_unread_emails")
            if result.get("events"):
                return UnifiedCommandProcessor._summarize_workspace_result(result, "check_calendar_events")
            return "Workspace command completed."

        # Unknown intent — generic success
        else:
            if result.get("error"):
                return f"Workspace action failed: {result['error']}"
            return result.get("message") or result.get("response") or "Workspace command completed successfully."
```

**Step 2: Verify syntax**

Run: `cd /Users/djrussell23/Documents/repos/Ironcliw-AI-Agent && python3 -c "import ast; ast.parse(open('backend/api/unified_command_processor.py').read()); print('Syntax OK')"`

Expected: `Syntax OK`

**Step 3: Commit**

```bash
cd /Users/djrussell23/Documents/repos/Ironcliw-AI-Agent
git add backend/api/unified_command_processor.py
git commit -m "fix(workspace): expand summarizer to format all workspace action results

Disease 2 cure: Summarizer now keys off workspace_action field instead
of J-Prime's high-level intent. Added formatting for send_email,
draft_email_reply, create_calendar_event, get_contacts, create_document,
search_email, and handle_workspace_query fallback path."
```

---

### Task 5: Add `workspace_action` to All Agent Handler Results (Disease 2 — Ironcliw)

**Files:**
- Modify: `/Users/djrussell23/Documents/repos/Ironcliw-AI-Agent/backend/neural_mesh/agents/google_workspace_agent.py`

This task adds `result["workspace_action"] = "<action_name>"` to each handler's return path. Each edit is a single-line addition before the `return result` statement.

**Step 1: `_fetch_unread_emails()` — 3 return paths**

At line 2623 (before `return result` in the success path), add:
```python
                result["workspace_action"] = "fetch_unread_emails"
```

At line 2625-2628 (the error return), change to:
```python
                return {
                    "error": exec_result.error or "All email check methods failed",
                    "emails": [],
                    "workspace_action": "fetch_unread_emails",
                }
```

At line 2632-2634 (direct client fallback + no-method return), add `"workspace_action"` to both:
```python
        if self._client:
            _result = await self._client.fetch_unread_emails(limit=limit)
            if isinstance(_result, dict):
                _result["workspace_action"] = "fetch_unread_emails"
            return _result

        return {"error": "No execution method available", "emails": [], "workspace_action": "fetch_unread_emails"}
```

**Step 2: `_check_calendar()` — 3 return paths**

At line 3126 (before `return result` in success path), add:
```python
                result["workspace_action"] = "check_calendar_events"
```

At line 3128-3132 (error return), add:
```python
                return {
                    "error": exec_result.error or "All calendar check methods failed",
                    "events": [],
                    "count": 0,
                    "workspace_action": "check_calendar_events",
                }
```

At line 3134-3138 (direct client + no-method fallbacks), add `"workspace_action"` similarly.

**Step 3: `_get_workspace_summary()` — 1 return path**

At line 3343 (before `return summary`), add:
```python
        summary["workspace_action"] = "workspace_summary"
```

**Step 4: `_send_email()` — 1 return path**

At line 3031 (before `return result`), add:
```python
        result["workspace_action"] = "send_email"
```

Also add to the early error returns (lines 3010-3014):
```python
            return {"error": "Recipient 'to' is required", "workspace_action": "send_email"}
```
(And similarly for subject/body error returns.)

**Step 5: `_draft_email()` — 1 main return path**

At line 2797 (before `return result`), add:
```python
        result["workspace_action"] = "draft_email_reply"
```

Also add to the early error returns (lines 2682-2684):
```python
            return {"error": "Recipient 'to' is required", "success": False, "workspace_action": "draft_email_reply"}
```

**Step 6: `_create_event()` — 1 return path**

At line 3179 (before `return result`), add:
```python
        result["workspace_action"] = "create_calendar_event"
```

Also add to error returns (lines 3150-3152).

**Step 7: `_get_contacts()` — 2 return paths**

At line 3187 (the client call), wrap:
```python
        if self._client:
            _result = await self._client.get_contacts(query=query, limit=limit)
            if isinstance(_result, dict):
                _result["workspace_action"] = "get_contacts"
            return _result
        return {"error": "Google API client not available", "contacts": [], "workspace_action": "get_contacts"}
```

**Step 8: `_create_document()` — add to success and error paths**

After line 3224 (success path `result = exec_result.data`), add:
```python
                result["workspace_action"] = "create_document"
```

**Step 9: `_handle_natural_query()` — add to all return paths**

Each return dict in `_handle_natural_query()` (lines 3345-3413) should include `"workspace_action": "handle_workspace_query"`. However, the paths that call other handlers (like `_fetch_unread_emails`, `_check_calendar`) will already have `workspace_action` set by those handlers. Only the direct return dicts need it:

Line 3379-3384 (draft_ready):
```python
                return {
                    "status": "draft_ready",
                    "message": "Ready to draft email",
                    "detected_recipient": names[0] if names else None,
                    "instructions": "Please provide: to, subject, and body",
                    "workspace_action": "draft_email_reply",
                }
```

Line 3387-3391 (send_ready):
```python
                return {
                    "status": "send_ready",
                    "message": "Ready to send email",
                    "instructions": "Please provide: to, subject, and body",
                    "workspace_action": "send_email",
                }
```

Line 3401-3405 (event_ready):
```python
                return {
                    "status": "event_ready",
                    "message": "Ready to create calendar event",
                    "instructions": "Please provide: title, start, and optionally end, description, location, attendees",
                    "workspace_action": "create_calendar_event",
                }
```

Line 3408-3413 (unknown_intent):
```python
                return {
                    "status": "unknown_intent",
                    "detected_intent": intent.value,
                    "confidence": confidence,
                    "message": "I'm not sure what workspace action you'd like. Try asking about emails, calendar, or contacts.",
                    "workspace_action": "handle_workspace_query",
                }
```

**Step 10: Verify syntax**

Run: `cd /Users/djrussell23/Documents/repos/Ironcliw-AI-Agent && python3 -c "import ast; ast.parse(open('backend/neural_mesh/agents/google_workspace_agent.py').read()); print('Syntax OK')"`

Expected: `Syntax OK`

**Step 11: Commit**

```bash
cd /Users/djrussell23/Documents/repos/Ironcliw-AI-Agent
git add backend/neural_mesh/agents/google_workspace_agent.py
git commit -m "feat(workspace): add workspace_action field to all handler results

Disease 2 support: Every handler now returns workspace_action in its
result dict. The UCP summarizer uses this to determine formatting
instead of J-Prime's high-level intent (which was always 'action')."
```

---

### Task 6: Add Workspace Examples to Brain Vacuum Classification (Disease 4 — Ironcliw)

**Files:**
- Modify: `/Users/djrussell23/Documents/repos/Ironcliw-AI-Agent/backend/core/jarvis_prime_client.py:1581-1591`

**Step 1: Expand classification prefix with workspace examples**

Replace lines 1590-1591 (the `suggested_actions` guidance line):

```python
        "- suggested_actions: list specific actions like [\"lock_screen\"], [\"open_browser\"], etc.\n"
```

With:

```python
        "- suggested_actions: list specific actions the Body should execute.\n"
        "  System: [\"lock_screen\"], [\"unlock_screen\"], [\"open_browser\"]\n"
        "  Workspace: [\"fetch_unread_emails\"], [\"check_calendar_events\"], [\"send_email\"], "
        "[\"draft_email_reply\"], [\"create_calendar_event\"], [\"daily_briefing\"]\n"
```

**Step 2: Verify syntax**

Run: `cd /Users/djrussell23/Documents/repos/Ironcliw-AI-Agent && python3 -c "import ast; ast.parse(open('backend/core/jarvis_prime_client.py').read()); print('Syntax OK')"`

Expected: `Syntax OK`

**Step 3: Commit**

```bash
cd /Users/djrussell23/Documents/repos/Ironcliw-AI-Agent
git add backend/core/jarvis_prime_client.py
git commit -m "fix(brain-vacuum): add workspace action examples to fallback classification

Disease 4 cure: When J-Prime is down, the brain vacuum (Claude/Gemini)
now sees workspace-specific suggested_actions examples. This ensures
'check my email' routes correctly even without J-Prime."
```

---

### Task 7: Integration Verification

**Step 1: Verify all files parse correctly across both repos**

```bash
cd /Users/djrussell23/Documents/repos/jarvis-prime && python3 -c "
import ast
ast.parse(open('jarvis_prime/core/classification_schema.py').read())
print('classification_schema.py: OK')
"
```

```bash
cd /Users/djrussell23/Documents/repos/Ironcliw-AI-Agent && python3 -c "
import ast
for f in ['backend/api/unified_command_processor.py', 'backend/core/jarvis_prime_client.py', 'backend/neural_mesh/agents/google_workspace_agent.py']:
    ast.parse(open(f).read())
    print(f'{f}: OK')
"
```

**Step 2: Verify the classification schema produces valid grammar**

```bash
cd /Users/djrussell23/Documents/repos/jarvis-prime && python3 -c "
import json
from jarvis_prime.core.classification_schema import CLASSIFICATION_SCHEMA
schema_json = json.dumps(CLASSIFICATION_SCHEMA, indent=2)
print('Schema has suggested_actions:', 'suggested_actions' in schema_json)
print('Schema JSON length:', len(schema_json))
print(schema_json)
"
```

**Step 3: Verify workspace_action appears in agent capabilities**

```bash
cd /Users/djrussell23/Documents/repos/Ironcliw-AI-Agent && python3 -c "
import sys; sys.path.insert(0, 'backend')
from neural_mesh.agents.google_workspace_agent import GoogleWorkspaceAgent
agent = GoogleWorkspaceAgent()
print('Agent capabilities:', sorted(agent.capabilities))
print()
print('Key actions present:')
for action in ['fetch_unread_emails', 'check_calendar_events', 'send_email', 'draft_email_reply', 'create_calendar_event', 'daily_briefing', 'handle_workspace_query']:
    print(f'  {action}: {action in agent.capabilities}')
"
```

**Step 4: Verify classifier prompt includes workspace examples**

```bash
cd /Users/djrussell23/Documents/repos/jarvis-prime && python3 -c "
from jarvis_prime.core.classification_schema import build_classifier_system_prompt
prompt = build_classifier_system_prompt()
print('Contains fetch_unread_emails:', 'fetch_unread_emails' in prompt)
print('Contains check_calendar_events:', 'check_calendar_events' in prompt)
print('Contains suggested_actions field description:', 'suggested_actions' in prompt)
"
```

**Step 5: Commit verification results (optional)**

No code changes — this is validation only.

---

### Task 8: Final Commit — All Changes Together

If any tasks above weren't committed individually, create a single combined commit:

```bash
cd /Users/djrussell23/Documents/repos/Ironcliw-AI-Agent
git add backend/api/unified_command_processor.py backend/core/jarvis_prime_client.py backend/neural_mesh/agents/google_workspace_agent.py
git commit -m "feat(workspace): cure 5 root diseases in Google Workspace routing

Disease 1: Direct action routing via suggested_actions (no double classification)
Disease 2: Summarizer keys off workspace_action field (not J-Prime intent)
Disease 3: Lazy singleton calls on_initialize() for proper lifecycle
Disease 4: Brain vacuum includes workspace action examples
Disease 5: asyncio.wait_for() deadline enforcement on execute_task()

Gap 1: suggested_actions propagated through all x_jarvis_routing paths
Gap 4: Action validated against agent.capabilities (single source of truth)
Gap 5: Deadline at UCP level via asyncio.wait_for()
Gap 6: schema_version uses SCHEMA_VERSION constant
Gap 7: Summarizer formats all workspace action types"
```

```bash
cd /Users/djrussell23/Documents/repos/jarvis-prime
git add jarvis_prime/core/classification_schema.py run_server.py
git commit -m "feat(classification): add suggested_actions to Phi schema + fix all routing paths

Supports Disease 1 cure: Phi classifier outputs workspace-specific
suggested_actions. All x_jarvis_routing construction sites now include
the field. schema_version uses SCHEMA_VERSION constant everywhere."
```
