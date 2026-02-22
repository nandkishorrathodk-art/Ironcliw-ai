# Trinity Cognitive Architecture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace ~11,000 lines of keyword-based classification waterfall with an LLM-first Spinal Reflex Arc architecture where Phi-3.5-mini classifies every query and specialist models generate responses.

**Architecture:** Dual-model inside J-Prime (Phi-3.5-mini permanently resident as classifier, specialist models swappable for generation). Body's `process_command()` interface preserved (Liskov); internals hollowed out from 6,747 lines to ~800 lines. Reflex manifest published by J-Prime for <5ms local commands.

**Tech Stack:** llama-cpp-python (LlamaGrammar.from_json_schema for constrained decoding), FastAPI, asyncio, aiohttp, JSON-based shared state files.

**Design Doc:** `docs/plans/2026-02-21-trinity-cognitive-architecture-design.md`

---

## Task 1: J-Prime — Phi Classifier Engine (llama_cpp_executor.py)

**Repo:** `/Users/djrussell23/Documents/repos/jarvis-prime`

**Files:**
- Modify: `jarvis_prime/core/llama_cpp_executor.py` (1,710 lines)

**Context:** The `LlamaCppExecutor` currently manages ONE model at a time. We need it to support a permanently-resident Phi-3.5-mini classifier alongside the swappable specialist model. `LlamaGrammar.from_json_schema()` is confirmed available in the installed llama-cpp-python v0.3.16.

**Step 1:** Add a `_classifier` field to `LlamaCppExecutor.__init__` (line ~789-833). This holds a separate `Llama` instance for Phi-3.5-mini that never gets unloaded during model swaps. Add a `_classifier_grammar` field that holds the pre-compiled `LlamaGrammar` for the classification schema.

```python
# In __init__, after existing fields (around line 833):
# Phi classifier (permanently resident, never swapped)
self._classifier: Optional["Llama"] = None
self._classifier_grammar: Optional["LlamaGrammar"] = None
self._classifier_model_path: Optional[Path] = None
self._classifier_lock = threading.RLock()
```

**Step 2:** Add `async def load_classifier(self, model_path: Path, schema: dict)` method. This loads Phi-3.5-mini with minimal context (n_ctx=1024 is enough for classification — input is short prompt + short query, output is ~30 tokens of JSON). Compiles the JSON schema into a `LlamaGrammar`.

```python
async def load_classifier(self, model_path: Path, schema: dict) -> None:
    """Load the permanently-resident Phi classifier with JSON grammar."""
    from llama_cpp import Llama, LlamaGrammar
    import json

    loop = asyncio.get_event_loop()

    def _load_sync():
        with self._classifier_lock:
            if self._classifier is not None:
                return  # Already loaded

            grammar = LlamaGrammar.from_json_schema(json.dumps(schema))

            model = Llama(
                model_path=str(model_path),
                n_ctx=1024,
                n_gpu_layers=-1,  # Metal GPU
                n_threads=2,
                verbose=False,
                chat_format="chatml",  # Phi-3.5 uses chatml
            )

            self._classifier = model
            self._classifier_grammar = grammar
            self._classifier_model_path = model_path

    await loop.run_in_executor(self._executor, _load_sync)
```

**Step 3:** Add `async def classify(self, query: str, system_prompt: str) -> dict` method. Runs Phi-3.5-mini with grammar constraint. Returns parsed JSON classification.

```python
async def classify(self, query: str, system_prompt: str) -> dict:
    """Classify a query using the permanently-resident Phi classifier.

    Returns parsed JSON matching CLASSIFICATION_SCHEMA.
    """
    import json

    if self._classifier is None:
        raise RuntimeError("Classifier not loaded. Call load_classifier() first.")

    loop = asyncio.get_event_loop()

    def _classify_sync():
        with self._classifier_lock:
            result = self._classifier.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                max_tokens=256,
                temperature=0.0,  # Deterministic classification
                grammar=self._classifier_grammar,
            )
            text = result["choices"][0]["message"]["content"]
            return json.loads(text)

    return await loop.run_in_executor(self._executor, _classify_sync)
```

**Step 4:** Modify `unload()` (find in file) to NOT unload the classifier. Only unload the specialist model (`self._model`). The classifier stays resident across all model swaps.

**Step 5:** Add `@property def classifier_loaded(self) -> bool` that checks `self._classifier is not None`.

**Step 6:** Commit.

```bash
cd /Users/djrussell23/Documents/repos/jarvis-prime
git add jarvis_prime/core/llama_cpp_executor.py
git commit -m "feat: add permanently-resident Phi classifier to LlamaCppExecutor

Phi-3.5-mini stays loaded alongside specialist models. Uses
LlamaGrammar.from_json_schema() for constrained JSON output.
Classifier is never unloaded during model swaps."
```

---

## Task 2: J-Prime — Classification Schema & System Prompt

**Repo:** `/Users/djrussell23/Documents/repos/jarvis-prime`

**Files:**
- Create: `jarvis_prime/core/classification_schema.py`

**Context:** This file defines the classification JSON schema (the Mind-Body contract), the Phi system prompt with action registry, and domain-to-TaskType mapping so the coordinator knows which specialist to load.

**Step 1:** Create the schema module.

```python
"""
Trinity Cognitive Architecture — Classification Schema v1

The Mind-Body contract: Phi-3.5-mini outputs this schema via grammar-constrained
decoding. The Body reads x_jarvis_routing metadata to decide how to render/execute.
"""

import json
import os
from typing import Dict, Any, Optional

SCHEMA_VERSION = 1

CLASSIFICATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "schema_version": {"type": "integer"},
        "intent": {
            "type": "string",
            "enum": [
                "answer",
                "action",
                "multi_step_action",
                "vision_needed",
                "clarify",
                "conversation",
            ],
        },
        "domain": {
            "type": "string",
            "enum": [
                "math", "code", "reasoning", "creative", "general",
                "system", "vision", "agentic", "translation",
                "conversation", "surveillance",
            ],
        },
        "complexity": {
            "type": "string",
            "enum": ["trivial", "simple", "moderate", "complex", "expert"],
        },
        "requires_vision": {"type": "boolean"},
        "requires_action": {"type": "boolean"},
        "escalate_to_claude": {"type": "boolean"},
        "confidence": {"type": "number"},
    },
    "required": [
        "schema_version", "intent", "domain", "complexity",
        "confidence", "requires_vision", "requires_action",
        "escalate_to_claude",
    ],
}

# Domain -> TaskType mapping (feeds into GCPModelSwapCoordinator)
DOMAIN_TO_TASK_TYPE: Dict[str, str] = {
    "math": "math_complex",
    "code": "code_complex",
    "reasoning": "reason_complex",
    "creative": "creative_write",
    "general": "general_chat",
    "system": "voice_command",
    "vision": "multimodal",
    "agentic": "reason_complex",
    "translation": "translate",
    "conversation": "greeting",
    "surveillance": "voice_command",
}

# Domains where Phi can both classify AND respond (no specialist needed)
PHI_SELF_SERVE_DOMAINS = frozenset({"conversation", "system"})

# Minimum confidence to trust classification (below this -> escalate)
MIN_CONFIDENCE_THRESHOLD = float(
    os.environ.get("JARVIS_PHI_MIN_CONFIDENCE", "0.5")
)


def build_classifier_system_prompt(action_registry: Optional[Dict] = None) -> str:
    """Build the Phi classifier system prompt with optional action registry."""

    actions_section = ""
    if action_registry:
        actions_list = ", ".join(sorted(action_registry.keys()))
        actions_section = f"\n\nAvailable actions the Body can execute: {actions_list}"

    return f"""You are a query classifier for the JARVIS AI assistant. Your ONLY job is to classify the user's query into a structured JSON object. Do NOT answer the query — only classify it.

Output JSON with these fields:
- schema_version: always {SCHEMA_VERSION}
- intent: one of [answer, action, multi_step_action, vision_needed, clarify, conversation]
  - "answer": user is asking a question that needs a text response
  - "action": user wants a single system action (open app, lock screen, volume change)
  - "multi_step_action": user wants a sequence of actions (open browser, navigate, click)
  - "vision_needed": user is asking about something visual (screen content, image analysis)
  - "clarify": query is ambiguous, ask user to clarify
  - "conversation": greeting, small talk, or casual interaction
- domain: one of [math, code, reasoning, creative, general, system, vision, agentic, translation, conversation, surveillance]
- complexity: one of [trivial, simple, moderate, complex, expert]
- requires_vision: true if the query needs to see the screen or an image
- requires_action: true if the query needs the Body to execute a system action
- escalate_to_claude: true if this query is too complex for a 7B local model (multi-step agentic planning, computer use, safety-critical decisions)
- confidence: 0.0 to 1.0, how confident you are in this classification{actions_section}

Examples:
- "what's today" -> intent=answer, domain=general, complexity=trivial, confidence=0.95
- "lock my screen" -> intent=action, domain=system, complexity=trivial, confidence=0.99
- "what's on my screen" -> intent=vision_needed, domain=vision, requires_vision=true, confidence=0.92
- "what's the derivative of x squared" -> intent=answer, domain=math, complexity=moderate, confidence=0.93
- "open Safari and go to GitHub" -> intent=multi_step_action, domain=agentic, escalate_to_claude=true, confidence=0.88
- "hello" -> intent=conversation, domain=conversation, complexity=trivial, confidence=0.99
- "watch all Chrome windows for changes" -> intent=action, domain=surveillance, requires_action=true, confidence=0.91"""
```

**Step 2:** Commit.

```bash
cd /Users/djrussell23/Documents/repos/jarvis-prime
git add jarvis_prime/core/classification_schema.py
git commit -m "feat: add Trinity classification schema and Phi system prompt

Defines the Mind-Body contract (CLASSIFICATION_SCHEMA v1),
domain-to-TaskType mapping, and classifier system prompt
with action registry support."
```

---

## Task 3: J-Prime — Dual-Model Routing in run_server.py

**Repo:** `/Users/djrussell23/Documents/repos/jarvis-prime`

**Files:**
- Modify: `run_server.py` (3,437 lines)

**Context:** The `chat_completions` endpoint (line 1397) currently extracts `task_type` from `request.metadata` (sent by the Body). We replace this with: (1) Phi classifies the query, (2) classification determines the specialist model, (3) specialist generates the response, (4) response includes `x_jarvis_routing` metadata.

**Step 1:** At module level (near line 794 where `_model_coordinator` is defined), add classifier initialization state:

```python
_classifier_loaded = False  # v242.0: Phi classifier state
_classifier_system_prompt = ""  # Cached system prompt
```

**Step 2:** In the startup/initialization section (near lines 3057-3075 where the coordinator is initialized), add Phi classifier loading AFTER the coordinator initializes:

```python
# v242.0: Load Phi-3.5-mini as permanent classifier
from jarvis_prime.core.classification_schema import (
    CLASSIFICATION_SCHEMA,
    build_classifier_system_prompt,
    DOMAIN_TO_TASK_TYPE,
    PHI_SELF_SERVE_DOMAINS,
    MIN_CONFIDENCE_THRESHOLD,
)

global _classifier_loaded, _classifier_system_prompt

phi_path = models_dir / "phi-3.5-mini-instruct.Q4_K_M.gguf"
if phi_path.exists():
    try:
        await _executor.load_classifier(phi_path, CLASSIFICATION_SCHEMA)
        _classifier_system_prompt = build_classifier_system_prompt()
        _classifier_loaded = True
        logger.info("[v242] Phi-3.5-mini classifier loaded (permanently resident)")
    except Exception as e:
        logger.warning(f"[v242] Phi classifier failed to load: {e}. Falling back to metadata hints.")
else:
    logger.warning(f"[v242] Phi model not found at {phi_path}. Using metadata hints.")
```

**Step 3:** Replace the task_type extraction in `chat_completions` (lines 1407-1420) with Phi classification:

```python
# v242.0: Phi-first classification (replaces Body's keyword hints)
_classification = None
_classification_ms = 0

if _classifier_loaded:
    import time as _time
    _t0 = _time.monotonic()
    try:
        _classification = await _executor.classify(
            query=messages[-1]["content"] if messages else "",
            system_prompt=_classifier_system_prompt,
        )
        _classification_ms = int((_time.monotonic() - _t0) * 1000)
        _v241_task_type = DOMAIN_TO_TASK_TYPE.get(
            _classification.get("domain", "general"), "general_chat"
        )
        logger.debug(
            f"[v242] Phi classified: intent={_classification.get('intent')}, "
            f"domain={_classification.get('domain')}, "
            f"confidence={_classification.get('confidence'):.2f}, "
            f"ms={_classification_ms}"
        )
    except Exception as e:
        logger.warning(f"[v242] Phi classification failed: {e}. Falling back.")
        _v241_task_type = request.metadata.get("task_type") if request.metadata else None
else:
    # Fallback: use Body's metadata hint (pre-v242 compatibility)
    _v241_task_type = request.metadata.get("task_type") if request.metadata else None
```

**Step 4:** After the specialist generates the response (find the response assembly section), inject `x_jarvis_routing` metadata:

```python
# v242.0: Attach x_jarvis_routing metadata to response
if _classification:
    response_dict["x_jarvis_routing"] = {
        "schema_version": _classification.get("schema_version", 1),
        "intent": _classification.get("intent", "answer"),
        "domain": _classification.get("domain", "general"),
        "complexity": _classification.get("complexity", "simple"),
        "confidence": _classification.get("confidence", 0.0),
        "requires_vision": _classification.get("requires_vision", False),
        "requires_action": _classification.get("requires_action", False),
        "escalate_to_claude": _classification.get("escalate_to_claude", False),
        "classifier_model": "phi-3.5-mini-q4km",
        "generator_model": _v241_active_model_id or "unknown",
        "classification_ms": _classification_ms,
        "generation_ms": int(generation_elapsed * 1000) if 'generation_elapsed' in dir() else 0,
        "suggested_actions": _classification.get("suggested_actions", []),
    }
```

**Step 5:** Handle Phi self-serve domains (conversation, simple actions). When the domain is in `PHI_SELF_SERVE_DOMAINS` and intent is `conversation` or simple `action`, Phi generates the response directly without a specialist swap:

Add this BEFORE the `ensure_model` call:

```python
# v242.0: Phi self-serve for trivial domains (no specialist swap needed)
_phi_self_served = False
if (_classification
    and _classification.get("domain") in PHI_SELF_SERVE_DOMAINS
    and _classification.get("intent") in ("conversation", "action")
    and _classification.get("complexity") in ("trivial", "simple")):

    try:
        # Phi generates directly (it's already loaded and fast)
        _phi_response = await _executor.classify(
            query=messages[-1]["content"] if messages else "",
            system_prompt="You are JARVIS, a helpful AI assistant. Respond briefly and naturally.",
        )
        # For self-serve, the "classification" IS the response
        # But we need free text, so use generate without grammar
        _phi_text = await _executor.generate(
            prompt=_executor.format_messages(messages),
            max_tokens=128,
            temperature=0.7,
        )
        _phi_self_served = True
        # Build response with _phi_text as content
    except Exception:
        _phi_self_served = False  # Fall through to specialist
```

**Note:** The self-serve path is an optimization. If it fails, fall through to normal specialist routing.

**Step 6:** Handle escalation signal. When `escalate_to_claude: true`, return immediately with the classification metadata so the Body knows to route to Claude API:

```python
# v242.0: Escalation signal
if (_classification
    and _classification.get("escalate_to_claude")
    and _classification.get("confidence", 0) > MIN_CONFIDENCE_THRESHOLD):

    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "jarvis-prime",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": ""},
            "finish_reason": "escalated",
        }],
        "x_jarvis_routing": {
            "schema_version": 1,
            "intent": _classification.get("intent", "answer"),
            "domain": _classification.get("domain", "general"),
            "escalate_to_claude": True,
            "escalation_reason": _classification.get("escalation_reason", "complexity"),
            "confidence": _classification.get("confidence", 0.0),
            "classifier_model": "phi-3.5-mini-q4km",
            "classification_ms": _classification_ms,
        },
    })
```

**Step 7:** Commit.

```bash
cd /Users/djrussell23/Documents/repos/jarvis-prime
git add run_server.py
git commit -m "feat: integrate Phi-3.5-mini dual-model classification in chat_completions

Phi classifies every query via JSON grammar (100-150ms), determines
specialist model via domain-to-TaskType mapping, returns x_jarvis_routing
metadata with every response. Self-serve for trivial domains.
Escalation signal for Claude-bound queries."
```

---

## Task 4: J-Prime — Seed Reflex Manifest

**Repo:** `/Users/djrussell23/Documents/repos/jarvis-prime`

**Files:**
- Create: `jarvis_prime/config/seed_reflex_manifest.json`
- Modify: `run_server.py` (publish manifest on startup)

**Step 1:** Create the seed manifest:

```json
{
  "version": 1,
  "published_by": "jarvis-prime",
  "published_at": null,
  "reflexes": {
    "lock_screen": {
      "patterns": ["lock my screen", "lock screen", "lock the screen", "lock my mac", "lock my computer"],
      "action": "system_command",
      "executor": "macos_controller.lock_screen",
      "requires_auth": false
    },
    "volume_up": {
      "patterns": ["volume up", "turn it up", "louder"],
      "action": "system_command",
      "executor": "macos_controller.volume_up",
      "requires_auth": false
    },
    "volume_down": {
      "patterns": ["volume down", "turn it down", "quieter"],
      "action": "system_command",
      "executor": "macos_controller.volume_down",
      "requires_auth": false
    },
    "mute_toggle": {
      "patterns": ["mute", "unmute"],
      "action": "system_command",
      "executor": "macos_controller.mute_toggle",
      "requires_auth": false
    },
    "brightness_up": {
      "patterns": ["brighter", "brightness up"],
      "action": "system_command",
      "executor": "macos_controller.brightness_up",
      "requires_auth": false
    },
    "brightness_down": {
      "patterns": ["dimmer", "brightness down"],
      "action": "system_command",
      "executor": "macos_controller.brightness_down",
      "requires_auth": false
    },
    "greeting": {
      "patterns": ["hello jarvis", "hi jarvis", "hey jarvis", "good morning jarvis", "good evening jarvis"],
      "action": "canned_response",
      "response_pool": ["Hello, Sir.", "At your service.", "Good to see you, Sir."],
      "requires_auth": false
    }
  }
}
```

**Step 2:** In `run_server.py` startup (after classifier initialization), add manifest publishing:

```python
# v242.0: Publish reflex manifest to shared state
import shutil
from pathlib import Path
from datetime import datetime, timezone

trinity_dir = Path.home() / ".jarvis" / "trinity"
trinity_dir.mkdir(parents=True, exist_ok=True)

seed_manifest = Path(__file__).parent / "jarvis_prime" / "config" / "seed_reflex_manifest.json"
target_manifest = trinity_dir / "reflex_manifest.json"

if seed_manifest.exists():
    manifest_data = json.loads(seed_manifest.read_text())
    manifest_data["published_at"] = datetime.now(timezone.utc).isoformat()
    target_manifest.write_text(json.dumps(manifest_data, indent=2))
    logger.info(f"[v242] Reflex manifest published to {target_manifest}")
```

**Step 3:** Commit.

```bash
cd /Users/djrussell23/Documents/repos/jarvis-prime
git add jarvis_prime/config/seed_reflex_manifest.json run_server.py
git commit -m "feat: add seed reflex manifest and publish on startup

Conservative manifest with 7 reflexes (lock, volume, brightness,
mute, greeting). Published to ~/.jarvis/trinity/ on J-Prime boot."
```

---

## Task 5: Body — Extend JarvisPrimeClient

**Repo:** `/Users/djrussell23/Documents/repos/JARVIS-AI-Agent`

**Files:**
- Modify: `backend/core/jarvis_prime_client.py` (1,916 lines)

**Context:** The existing `JarvisPrimeClient` has `complete()` (line 1042) with memory-aware routing, circuit breakers, and fallback chain. We add `classify_and_complete()` that reads `x_jarvis_routing` from J-Prime's response and returns a structured result the Body can act on.

**Step 1:** Add a `StructuredResponse` dataclass near the existing `CompletionResponse` (around line 531):

```python
@dataclasses.dataclass
class StructuredResponse:
    """Response from J-Prime with routing metadata (Trinity v242)."""
    content: str
    intent: str = "answer"
    domain: str = "general"
    complexity: str = "simple"
    confidence: float = 0.0
    requires_vision: bool = False
    requires_action: bool = False
    escalated: bool = False
    escalation_reason: str = ""
    suggested_actions: list = dataclasses.field(default_factory=list)
    classifier_model: str = ""
    generator_model: str = ""
    classification_ms: int = 0
    generation_ms: int = 0
    schema_version: int = 1
    source: str = "jprime"  # "jprime", "claude_fallback", "local_fallback"
```

**Step 2:** Add `classify_and_complete()` method after `complete()` (around line 1220):

```python
async def classify_and_complete(
    self,
    query: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    context_metadata: Optional[Dict[str, Any]] = None,
) -> StructuredResponse:
    """Send query to J-Prime, get classified + generated response.

    J-Prime's Phi classifier determines intent/domain. Specialist model
    generates content. Returns StructuredResponse with routing metadata.

    Falls back to Claude API if J-Prime is unreachable (brain vacuum).
    """
    # Try J-Prime first (normal path)
    try:
        response = await self.complete(
            prompt=query,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[ChatMessage(role="user", content=query)],
        )

        # Parse x_jarvis_routing from response metadata
        routing = {}
        if hasattr(response, 'raw_response') and isinstance(response.raw_response, dict):
            routing = response.raw_response.get("x_jarvis_routing", {})

        # Check for escalation signal
        if routing.get("escalate_to_claude"):
            return await self._escalate_to_claude(
                query=query,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                routing=routing,
            )

        return StructuredResponse(
            content=response.text if response.text else "",
            intent=routing.get("intent", "answer"),
            domain=routing.get("domain", "general"),
            complexity=routing.get("complexity", "simple"),
            confidence=routing.get("confidence", 0.0),
            requires_vision=routing.get("requires_vision", False),
            requires_action=routing.get("requires_action", False),
            escalated=False,
            suggested_actions=routing.get("suggested_actions", []),
            classifier_model=routing.get("classifier_model", ""),
            generator_model=routing.get("generator_model", ""),
            classification_ms=routing.get("classification_ms", 0),
            generation_ms=routing.get("generation_ms", 0),
            schema_version=routing.get("schema_version", 1),
            source="jprime",
        )

    except Exception as e:
        logger.warning(f"[v242] J-Prime unreachable: {e}. Brain vacuum fallback to Claude.")
        return await self._brain_vacuum_fallback(
            query=query,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )
```

**Step 3:** Add `_escalate_to_claude()` and `_brain_vacuum_fallback()` helper methods:

```python
async def _escalate_to_claude(
    self,
    query: str,
    system_prompt: Optional[str],
    max_tokens: int,
    routing: dict,
) -> StructuredResponse:
    """Route to Claude API when J-Prime signals escalation."""
    try:
        # Use the existing Claude/Gemini fallback path
        response = await self._execute_completion(
            prompt=query,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            mode=RoutingMode.GEMINI_API,  # Claude/Gemini API
        )
        return StructuredResponse(
            content=response.text if response.text else "",
            intent=routing.get("intent", "answer"),
            domain=routing.get("domain", "general"),
            escalated=True,
            escalation_reason=routing.get("escalation_reason", ""),
            confidence=routing.get("confidence", 0.0),
            source="claude_escalation",
        )
    except Exception as e:
        logger.error(f"[v242] Claude escalation also failed: {e}")
        return StructuredResponse(
            content="I'm having trouble processing that right now.",
            intent="answer",
            source="error",
        )

async def _brain_vacuum_fallback(
    self,
    query: str,
    system_prompt: Optional[str],
    max_tokens: int,
) -> StructuredResponse:
    """Fallback when J-Prime is completely unreachable (startup, network failure)."""
    try:
        response = await self._execute_completion(
            prompt=query,
            system_prompt=system_prompt or "You are JARVIS, a helpful AI assistant.",
            max_tokens=max_tokens,
            mode=RoutingMode.GEMINI_API,
        )
        return StructuredResponse(
            content=response.text if response.text else "",
            intent="answer",  # Default: treat as question
            domain="general",
            source="claude_fallback",
        )
    except Exception as e:
        logger.error(f"[v242] Brain vacuum: all backends failed: {e}")
        return StructuredResponse(
            content="I'm still starting up. Please try again in a moment.",
            intent="answer",
            source="error",
        )
```

**Step 4:** Commit.

```bash
cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent
git add backend/core/jarvis_prime_client.py
git commit -m "feat: add classify_and_complete() to JarvisPrimeClient

Reads x_jarvis_routing from J-Prime responses. Handles escalation
signals and brain vacuum fallback to Claude API. Returns
StructuredResponse with full routing metadata."
```

---

## Task 6: Body — Hollow Out UCP (THE BIG SURGERY)

**Repo:** `/Users/djrussell23/Documents/repos/JARVIS-AI-Agent`

**Files:**
- Modify: `backend/api/unified_command_processor.py` (6,747 lines)

**Context:** This is the heart surgery. We keep `process_command()` signature and return shape identical. We replace ~5,000 lines of classification with ~200 lines of reflex check + J-Prime call + action executor. We PRESERVE the execution handlers (MacOSController, IntelligentCommandHandler, voice unlock, self-voice suppression, telemetry).

**This task is the largest and most delicate. It should be done in sub-steps with careful verification between each.**

**Step 1: Add reflex manifest reader.** Add a new internal method `_check_reflex_manifest()` to the `UnifiedCommandProcessor` class:

```python
async def _check_reflex_manifest(self, command_text: str) -> Optional[Dict[str, Any]]:
    """Check if command matches a reflex in the J-Prime-published manifest.

    Returns reflex dict if matched, None otherwise.
    Checks inhibition signals before executing.
    """
    manifest_path = Path.home() / ".jarvis" / "trinity" / "reflex_manifest.json"
    inhibition_path = Path.home() / ".jarvis" / "trinity" / "reflex_inhibition.json"

    if not manifest_path.exists():
        return None

    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    # Check each reflex for pattern match
    normalized = command_text.lower().strip()
    for reflex_id, reflex in manifest.get("reflexes", {}).items():
        patterns = reflex.get("patterns", [])
        if any(normalized == p.lower() for p in patterns):
            # Check inhibition before executing
            if inhibition_path.exists():
                try:
                    inhibition = json.loads(inhibition_path.read_text())
                    inhibited = inhibition.get("inhibit_reflexes", [])
                    published = inhibition.get("published_at", "")
                    ttl = inhibition.get("ttl_seconds", 0)
                    if reflex_id in inhibited:
                        from datetime import datetime, timezone
                        pub_time = datetime.fromisoformat(published)
                        if (datetime.now(timezone.utc) - pub_time).total_seconds() < ttl:
                            logger.info(f"[v242] Reflex '{reflex_id}' inhibited: {inhibition.get('reason')}")
                            return None  # Inhibited — send to J-Prime instead
                except (json.JSONDecodeError, OSError, ValueError):
                    pass  # Inhibition check failed — execute reflex anyway

            return {"reflex_id": reflex_id, **reflex}

    return None
```

**Step 2: Add J-Prime call wrapper.** This calls `classify_and_complete()` on the existing client:

```python
async def _call_jprime(
    self, command_text: str, deadline: Optional[float] = None,
) -> Optional[StructuredResponse]:
    """Send query to J-Prime for classification and response.

    Returns StructuredResponse or None on failure.
    """
    try:
        client = self._get_prime_client()
        if client is None:
            return None

        timeout = 30.0
        if deadline:
            import time
            remaining = deadline - time.monotonic()
            timeout = max(2.0, remaining - 1.0)

        return await asyncio.wait_for(
            client.classify_and_complete(
                query=command_text,
                max_tokens=512,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning("[v242] J-Prime call timed out")
        return None
    except Exception as e:
        logger.warning(f"[v242] J-Prime call failed: {e}")
        return None
```

**Step 3: Add action executor.** This takes a `StructuredResponse` and executes the appropriate action:

```python
async def _execute_action(
    self, response: StructuredResponse, command_text: str,
    websocket=None, audio_data: bytes = None, speaker_name: str = None,
) -> Dict[str, Any]:
    """Execute the action determined by J-Prime's classification.

    Maps intent + domain to existing execution handlers.
    Returns the standard process_command() response dict.
    """
    intent = response.intent
    domain = response.domain

    # Intent: answer / conversation — just return the text
    if intent in ("answer", "conversation"):
        return {
            "success": True,
            "response": response.content,
            "command_type": "QUERY",
            "source": response.source,
            "x_jarvis_routing": {
                "intent": intent, "domain": domain,
                "confidence": response.confidence,
            },
        }

    # Intent: action — execute a system command
    if intent == "action":
        if domain == "surveillance":
            return await self._handle_surveillance(command_text, websocket)
        elif domain == "system":
            return await self._handle_system_action(
                command_text, response.suggested_actions
            )
        else:
            return {
                "success": True,
                "response": response.content,
                "command_type": "SYSTEM",
                "source": response.source,
            }

    # Intent: vision_needed — capture screenshot and re-call J-Prime
    if intent == "vision_needed":
        return await self._handle_vision_query(command_text, websocket)

    # Intent: multi_step_action — execute step by step
    if intent == "multi_step_action":
        return await self._handle_multi_step(
            command_text, response.suggested_actions, websocket
        )

    # Intent: clarify — ask user to clarify
    if intent == "clarify":
        return {
            "success": True,
            "response": response.content or "Could you clarify what you'd like me to do?",
            "command_type": "QUERY",
            "source": response.source,
        }

    # Fallback: treat as answer
    return {
        "success": True,
        "response": response.content,
        "command_type": "QUERY",
        "source": response.source,
    }
```

**Step 4: Replace the core of process_command().** The existing `process_command()` (line 1243) stays as the entry point with the same signature. Replace the INTERNAL flow (after self-voice suppression and voice unlock check) with the new reflex + J-Prime + action executor flow:

The new internal flow inside `process_command()`:

```python
# === PRESERVED: Self-voice suppression (existing code, ~lines 1256-1280) ===

# === PRESERVED: Voice unlock detection (existing protected local op) ===
# Keep the existing VOICE_UNLOCK / SCREEN_LOCK fast-path check

# === NEW: Reflex manifest check ===
reflex = await self._check_reflex_manifest(command_text)
if reflex:
    result = await self._execute_reflex(reflex, command_text)
    # Async notify J-Prime for telemetry (fire-and-forget)
    asyncio.create_task(self._notify_reflex_executed(command_text, reflex))
    return result

# === NEW: J-Prime call (the brain) ===
response = await self._call_jprime(command_text, deadline=deadline)
if response:
    result = await self._execute_action(
        response, command_text, websocket, audio_data, speaker_name
    )
    # Telemetry
    await self._log_telemetry(command_text, response, result)
    return result

# === PRESERVED: Brain vacuum fallback ===
# If J-Prime is unreachable, fall through to basic response
return {
    "success": False,
    "response": "I'm having trouble connecting to my brain. Please try again.",
    "command_type": "UNKNOWN",
}
```

**Step 5: Add reflex executor helper:**

```python
async def _execute_reflex(
    self, reflex: dict, command_text: str,
) -> Dict[str, Any]:
    """Execute a matched reflex immediately."""
    reflex_id = reflex.get("reflex_id", "unknown")
    action = reflex.get("action", "")

    if action == "canned_response":
        import random
        pool = reflex.get("response_pool", ["Done."])
        return {
            "success": True,
            "response": random.choice(pool),
            "command_type": "REFLEX",
            "reflex_id": reflex_id,
        }

    if action == "system_command":
        executor_path = reflex.get("executor", "")
        # Execute via existing MacOS controller infrastructure
        try:
            result = await self._execute_system_reflex(executor_path)
            return {
                "success": True,
                "response": result or f"Done: {reflex_id}",
                "command_type": "REFLEX",
                "reflex_id": reflex_id,
            }
        except Exception as e:
            logger.error(f"[v242] Reflex execution failed: {e}")
            return {"success": False, "response": str(e), "command_type": "REFLEX"}

    return {"success": False, "response": "Unknown reflex action", "command_type": "REFLEX"}
```

**Step 6:** Remove the old classification internals. This means deleting:
- The entire `_classify_command()` method (~lines 2405-2700+)
- The tiered router integration (~lines 1293-1387)
- The surveillance detection block (~lines 1405-1630)
- The command type switch/handler dispatch (~lines 1633-6700+)
- Keep ONLY: execution handlers that are called by the new `_execute_action()`

**WARNING:** This step removes thousands of lines. Verify the golden test corpus before committing.

**Step 7:** Commit.

```bash
cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent
git add backend/api/unified_command_processor.py
git commit -m "feat: hollow out UCP — replace classification with Spinal Reflex Arc

process_command() signature preserved. 5,000 lines of keyword
classification replaced with ~200 lines of reflex check + J-Prime
call + action executor. Zero caller changes."
```

---

## Task 7: Body — Clean Up Caller Pre-Classification

**Repo:** `/Users/djrussell23/Documents/repos/JARVIS-AI-Agent`

**Files:**
- Modify: `backend/api/jarvis_voice_api.py` (5,450 lines)
- Modify: `main.py`

**Context:** `jarvis_voice_api.py` has ~250 lines of pre-classification (math guard, workspace bypass, surveillance regex, vision handler routing) that intercept queries before they reach `process_command()`. These are now redundant — J-Prime classifies everything.

**Step 1:** In `jarvis_voice_api.py`, remove:
- Math equation guard regex patterns and `_MathBypassSignal` exception handling (~lines 100-113, and wherever they're caught ~lines 2270-2475)
- `_WorkspaceBypassSignal` exception handling
- Vision handler routing block (~lines 2308-2475 where `vision_command_handler` is called as a pre-filter)
- Surveillance regex detection (if any before `process_command` call)

Replace with a clean pass-through: the voice pipeline should transcribe speech, call `process_command()`, and speak the response. No pre-classification.

**Step 2:** In `main.py`, remove the lock-screen regex fast-path (~lines 7818-7881). Lock screen detection is now handled by the reflex manifest inside `process_command()`.

**Step 3:** Commit.

```bash
cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent
git add backend/api/jarvis_voice_api.py main.py
git commit -m "feat: remove pre-classification from voice API and main.py

Math guards, workspace bypasses, vision routing, and lock-screen
regex fast-paths removed. All classification now flows through
J-Prime via process_command(). ~320 lines removed."
```

---

## Task 8: Body — Delete Dead Files

**Repo:** `/Users/djrussell23/Documents/repos/JARVIS-AI-Agent`

**Files:**
- Delete: `backend/core/tiered_command_router.py` (1,356 lines)
- Delete: `backend/core/tiered_vbia_adapter.py` (772 lines)
- Delete: `backend/api/unified_command_processor_pure.py` (328 lines)

**Context:** These files are now dead code. The tiered command router's classification is replaced by J-Prime. The VBIA adapter bridged the tiered router to voice auth — no longer needed since voice unlock is a protected local op. The pure UCP variant is a stripped-down copy — no longer needed.

**Step 1:** Before deleting, search for imports of these files to ensure nothing still depends on them:

```bash
grep -rn "tiered_command_router" backend/ --include="*.py"
grep -rn "tiered_vbia_adapter" backend/ --include="*.py"
grep -rn "unified_command_processor_pure" backend/ --include="*.py"
```

Remove any remaining imports in other files.

**Step 2:** Delete the files.

**Step 3:** Commit.

```bash
cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent
git rm backend/core/tiered_command_router.py backend/core/tiered_vbia_adapter.py backend/api/unified_command_processor_pure.py
git commit -m "feat: delete dead classification files

tiered_command_router.py (1,356 lines), tiered_vbia_adapter.py (772 lines),
unified_command_processor_pure.py (328 lines) removed. Classification
now handled entirely by J-Prime's Phi classifier."
```

---

## Task 9: Reactor Core — Telemetry Integration

**Repo:** `/Users/djrussell23/Documents/repos/reactor-core`

**Files:**
- Modify: `reactor_core/ingestion/telemetry_ingestor.py` (296 lines)

**Context:** The `TelemetryIngestor._parse_item()` (lines 161-217) parses telemetry events into `RawInteraction` format. We extend it to parse the new `x_jarvis_routing` metadata that the Body writes to telemetry after every query.

**Step 1:** In `_parse_item()`, add parsing for `x_jarvis_routing` fields:

```python
# In _parse_item(), after existing field extraction:

# v242.0: Parse Trinity classification metadata
routing = event.get("x_jarvis_routing", {})
if routing:
    properties["phi_intent"] = routing.get("intent", "")
    properties["phi_domain"] = routing.get("domain", "")
    properties["phi_complexity"] = routing.get("complexity", "")
    properties["phi_confidence"] = routing.get("confidence", 0.0)
    properties["escalated_to_claude"] = routing.get("escalated", False)
    properties["classifier_model"] = routing.get("classifier_model", "")
    properties["generator_model"] = routing.get("generator_model", "")
    properties["classification_ms"] = routing.get("classification_ms", 0)
    properties["generation_ms"] = routing.get("generation_ms", 0)

    # Override confidence with Phi's classification confidence
    if routing.get("confidence"):
        confidence = routing["confidence"]
```

**Step 2:** Add DPO pair detection. When a query was escalated to Claude AND the user accepted the response (no correction), this is a positive training signal:

```python
# In _parse_item(), in the tags section:
if routing.get("escalated_to_claude"):
    tags.append("escalated")
    tags.append(f"escalation_domain:{routing.get('domain', 'unknown')}")
    # This is a DPO training signal: local model couldn't handle it
    properties["dpo_candidate"] = True
```

**Step 3:** Commit.

```bash
cd /Users/djrussell23/Documents/repos/reactor-core
git add reactor_core/ingestion/telemetry_ingestor.py
git commit -m "feat: parse x_jarvis_routing telemetry from Trinity architecture

Extracts Phi classification metadata (intent, domain, confidence,
escalation signals) from Body telemetry. Tags escalated queries
as DPO training candidates for local model improvement."
```

---

## Task 10: Verification — Golden Test Corpus

**Repo:** `/Users/djrussell23/Documents/repos/JARVIS-AI-Agent`

**Context:** Verify the surgery worked. Create a test corpus of diverse queries and verify each produces correct behavior through the new routing.

**Step 1:** Create a verification script:

```python
# tests/verify_trinity_routing.py
"""
Golden test corpus for Trinity Cognitive Architecture.
Verifies that every query type routes correctly through the
Spinal Reflex Arc (reflex vs J-Prime vs protected local op).
"""

GOLDEN_CORPUS = [
    # Reflexes (should match manifest, execute locally)
    {"query": "lock my screen", "expected_type": "REFLEX"},
    {"query": "volume up", "expected_type": "REFLEX"},
    {"query": "mute", "expected_type": "REFLEX"},
    {"query": "hello jarvis", "expected_type": "REFLEX"},

    # Questions (should go to J-Prime, intent=answer)
    {"query": "what's today's date", "expected_intent": "answer"},
    {"query": "what's the derivative of x squared", "expected_intent": "answer", "expected_domain": "math"},
    {"query": "explain async await in Python", "expected_intent": "answer", "expected_domain": "code"},

    # Vision (should get intent=vision_needed)
    {"query": "what's on my screen", "expected_intent": "vision_needed"},
    {"query": "read the error message", "expected_intent": "vision_needed"},

    # System actions (should get intent=action)
    {"query": "open Safari", "expected_intent": "action", "expected_domain": "system"},

    # Surveillance (should get intent=action, domain=surveillance)
    {"query": "watch all Chrome windows for changes", "expected_intent": "action", "expected_domain": "surveillance"},

    # Conversation (should get intent=conversation)
    {"query": "how are you doing", "expected_intent": "conversation"},

    # Complex/agentic (should escalate to Claude)
    {"query": "open Safari, go to GitHub, find my repo, and star it", "expected_escalate": True},
]
```

**Step 2:** Run verification manually by sending each query through `process_command()` and checking the response matches expectations.

**Step 3:** Commit test corpus.

```bash
cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent
git add tests/verify_trinity_routing.py
git commit -m "test: add golden test corpus for Trinity routing verification

14 diverse queries covering reflexes, questions, vision, system
actions, surveillance, conversation, and agentic commands."
```

---

## Verification Criteria (from design doc)

The surgery is complete when:

1. `python3 unified_supervisor.py` boots all three repos as one system
2. Every query type produces correct behavior
3. `unified_command_processor.py` is <1,000 lines
4. `tiered_command_router.py` no longer exists
5. No keyword-based classification exists in the Body
6. All classification decisions flow through J-Prime's Phi classifier
7. `x_jarvis_routing` metadata present on every J-Prime response
8. Brain vacuum (J-Prime unreachable) degrades gracefully to Claude API
9. Reflexes fire in <5ms and are logged to telemetry
10. Golden test corpus passes with expected behavior
