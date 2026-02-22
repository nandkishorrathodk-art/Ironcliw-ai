#!/usr/bin/env python3
"""
Golden test corpus for Trinity Cognitive Architecture.

Verifies that every query type routes correctly through the
Spinal Reflex Arc (reflex vs J-Prime vs protected local op).

Usage:
    python3 tests/verify_trinity_routing.py

    Or with PYTHONPATH for import checks:
    PYTHONPATH=./backend python3 tests/verify_trinity_routing.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add project root and backend to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "backend"))


# ---------------------------------------------------------------------------
# Golden Corpus — diverse queries covering every Trinity routing path
# ---------------------------------------------------------------------------

GOLDEN_CORPUS = [
    # -----------------------------------------------------------------------
    # Reflexes (should match manifest patterns, execute locally — never
    # leave the Body, sub-ms latency)
    # -----------------------------------------------------------------------
    {"query": "lock my screen",       "expected_type": "REFLEX", "expected_reflex": "lock_screen"},
    {"query": "lock screen",          "expected_type": "REFLEX", "expected_reflex": "lock_screen"},
    {"query": "lock my mac",          "expected_type": "REFLEX", "expected_reflex": "lock_screen"},
    {"query": "volume up",            "expected_type": "REFLEX", "expected_reflex": "volume_up"},
    {"query": "louder",               "expected_type": "REFLEX", "expected_reflex": "volume_up"},
    {"query": "volume down",          "expected_type": "REFLEX", "expected_reflex": "volume_down"},
    {"query": "quieter",              "expected_type": "REFLEX", "expected_reflex": "volume_down"},
    {"query": "mute",                 "expected_type": "REFLEX", "expected_reflex": "mute_toggle"},
    {"query": "unmute",               "expected_type": "REFLEX", "expected_reflex": "mute_toggle"},
    {"query": "brightness up",        "expected_type": "REFLEX", "expected_reflex": "brightness_up"},
    {"query": "brightness down",      "expected_type": "REFLEX", "expected_reflex": "brightness_down"},
    {"query": "hello jarvis",         "expected_type": "REFLEX", "expected_reflex": "greeting"},
    {"query": "hi jarvis",            "expected_type": "REFLEX", "expected_reflex": "greeting"},
    {"query": "good morning jarvis",  "expected_type": "REFLEX", "expected_reflex": "greeting"},

    # -----------------------------------------------------------------------
    # Questions (should go to J-Prime, intent=answer)
    # -----------------------------------------------------------------------
    {"query": "what's today's date",                  "expected_intent": "answer"},
    {"query": "what's the derivative of x squared",   "expected_intent": "answer", "expected_domain": "math"},
    {"query": "explain async await in Python",        "expected_intent": "answer", "expected_domain": "code"},
    {"query": "what is the capital of France",         "expected_intent": "answer"},
    {"query": "how does TCP three way handshake work", "expected_intent": "answer", "expected_domain": "code"},

    # -----------------------------------------------------------------------
    # Vision (should get intent=vision_needed)
    # -----------------------------------------------------------------------
    {"query": "what's on my screen",    "expected_intent": "vision_needed"},
    {"query": "read the error message", "expected_intent": "vision_needed"},
    {"query": "describe what you see",  "expected_intent": "vision_needed"},

    # -----------------------------------------------------------------------
    # System actions (should get intent=action, domain=system)
    # -----------------------------------------------------------------------
    {"query": "open Safari",          "expected_intent": "action", "expected_domain": "system"},
    {"query": "close this window",    "expected_intent": "action", "expected_domain": "system"},

    # -----------------------------------------------------------------------
    # Surveillance (should get intent=action, domain=surveillance)
    # -----------------------------------------------------------------------
    {"query": "watch all Chrome windows for changes", "expected_intent": "action", "expected_domain": "surveillance"},
    {"query": "monitor my screen for updates",        "expected_intent": "action", "expected_domain": "surveillance"},

    # -----------------------------------------------------------------------
    # Conversation (should get intent=conversation)
    # -----------------------------------------------------------------------
    {"query": "how are you doing",   "expected_intent": "conversation"},
    {"query": "tell me a joke",      "expected_intent": "conversation"},

    # -----------------------------------------------------------------------
    # Complex / agentic (should escalate to Claude)
    # -----------------------------------------------------------------------
    {"query": "open Safari, go to GitHub, find my repo, and star it", "expected_escalate": True},
    {"query": "refactor my utils.py file to use async patterns",      "expected_escalate": True},
]


# ---------------------------------------------------------------------------
# Verification functions
# ---------------------------------------------------------------------------

_MANIFEST_PATH = Path.home() / ".jarvis" / "trinity" / "reflex_manifest.json"

# Track global results
_results = {"passed": 0, "failed": 0, "skipped": 0}


def _record(status: str, msg: str):
    """Print and tally a result line."""
    _results[status] += 1
    prefix = {"passed": "PASS", "failed": "FAIL", "skipped": "SKIP"}[status]
    print(f"  {prefix}: {msg}")


def verify_reflex_manifest():
    """Verify the reflex manifest exists, is valid JSON, and contains required reflexes."""
    if not _MANIFEST_PATH.exists():
        _record("skipped", f"Reflex manifest not found at {_MANIFEST_PATH}")
        print("        (J-Prime must be started first to publish the manifest)")
        return False

    try:
        manifest = json.loads(_MANIFEST_PATH.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        _record("failed", f"Invalid manifest JSON: {exc}")
        return False

    # Structure checks
    if "reflexes" not in manifest:
        _record("failed", "Manifest missing 'reflexes' key")
        return False

    required_reflexes = {
        "lock_screen", "volume_up", "volume_down",
        "mute_toggle", "brightness_up", "brightness_down", "greeting",
    }
    actual_reflexes = set(manifest["reflexes"].keys())
    missing = required_reflexes - actual_reflexes
    if missing:
        _record("failed", f"Missing reflexes: {missing}")
        return False

    # Validate each reflex has required fields
    for reflex_id, reflex in manifest["reflexes"].items():
        if "patterns" not in reflex or not isinstance(reflex["patterns"], list):
            _record("failed", f"Reflex '{reflex_id}' missing or invalid 'patterns'")
            return False
        if "action" not in reflex:
            _record("failed", f"Reflex '{reflex_id}' missing 'action'")
            return False

    # Check published_at timestamp
    published_at = manifest.get("published_at")
    if published_at:
        try:
            ts = datetime.fromisoformat(published_at)
            age_hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
            if age_hours > 24:
                print(f"  WARN: Manifest is {age_hours:.1f} hours old (published: {published_at})")
        except (ValueError, TypeError):
            print(f"  WARN: Could not parse published_at: {published_at}")

    _record("passed", f"Manifest valid with {len(actual_reflexes)} reflexes")
    extra = actual_reflexes - required_reflexes
    if extra:
        print(f"        Extra reflexes present: {extra}")
    return True


def verify_reflex_routing():
    """Verify that every REFLEX query in the golden corpus matches the manifest."""
    if not _MANIFEST_PATH.exists():
        _record("skipped", "No manifest to test against")
        return

    manifest = json.loads(_MANIFEST_PATH.read_text())
    reflexes = manifest.get("reflexes", {})

    reflex_tests = [t for t in GOLDEN_CORPUS if t.get("expected_type") == "REFLEX"]

    for test in reflex_tests:
        query = test["query"].lower().strip()
        expected_reflex = test.get("expected_reflex")
        matched_id = None

        for reflex_id, reflex in reflexes.items():
            patterns = [p.lower() for p in reflex.get("patterns", [])]
            if query in patterns:
                matched_id = reflex_id
                break

        if matched_id is None:
            _record("failed", f"'{test['query']}' did not match any reflex pattern")
        elif expected_reflex and matched_id != expected_reflex:
            _record(
                "failed",
                f"'{test['query']}' matched '{matched_id}' but expected '{expected_reflex}'",
            )
        else:
            _record("passed", f"'{test['query']}' -> reflex '{matched_id}'")


def verify_no_reflex_false_positives():
    """Verify that non-reflex queries do NOT accidentally match a reflex pattern."""
    if not _MANIFEST_PATH.exists():
        _record("skipped", "No manifest to test against")
        return

    manifest = json.loads(_MANIFEST_PATH.read_text())
    reflexes = manifest.get("reflexes", {})

    non_reflex_tests = [t for t in GOLDEN_CORPUS if t.get("expected_type") != "REFLEX"]

    for test in non_reflex_tests:
        query = test["query"].lower().strip()
        for reflex_id, reflex in reflexes.items():
            patterns = [p.lower() for p in reflex.get("patterns", [])]
            if query in patterns:
                _record(
                    "failed",
                    f"FALSE POSITIVE: '{test['query']}' matched reflex '{reflex_id}' "
                    f"but expected_type is not REFLEX",
                )
                break
        else:
            _record("passed", f"'{test['query'][:50]}' correctly avoids reflex match")


def verify_imports():
    """Verify all required Trinity components are importable."""
    components = [
        ("StructuredResponse", "core.jarvis_prime_client"),
        ("UnifiedCommandProcessor", "api.unified_command_processor"),
    ]

    for name, module_path in components:
        try:
            mod = __import__(module_path, fromlist=[name])
            if hasattr(mod, name):
                _record("passed", f"{name} importable from {module_path}")
            else:
                _record("failed", f"{name} not found in {module_path}")
        except ImportError as exc:
            _record("skipped", f"{module_path} not importable ({exc})")


def verify_structured_response_fields():
    """Verify StructuredResponse has all fields the golden corpus expects."""
    try:
        from core.jarvis_prime_client import StructuredResponse
    except ImportError:
        _record("skipped", "Cannot import StructuredResponse for field check")
        return

    required_fields = {
        "content", "intent", "domain", "complexity", "confidence",
        "requires_vision", "requires_action", "escalated",
        "escalation_reason", "suggested_actions",
        "classifier_model", "generator_model",
        "classification_ms", "generation_ms",
        "schema_version", "source",
    }

    # StructuredResponse is a dataclass -- check __dataclass_fields__ or annotations
    actual_fields = set()
    if hasattr(StructuredResponse, "__dataclass_fields__"):
        actual_fields = set(StructuredResponse.__dataclass_fields__.keys())
    elif hasattr(StructuredResponse, "__annotations__"):
        actual_fields = set(StructuredResponse.__annotations__.keys())

    missing = required_fields - actual_fields
    if missing:
        _record("failed", f"StructuredResponse missing fields: {missing}")
    else:
        _record("passed", f"StructuredResponse has all {len(required_fields)} expected fields")

    extra = actual_fields - required_fields
    if extra:
        print(f"        Extra fields present: {extra}")


def verify_dead_files_removed():
    """Verify that dead classification files from pre-Trinity era have been deleted."""
    dead_files = [
        "backend/core/tiered_command_router.py",
        "backend/core/tiered_vbia_adapter.py",
        "backend/api/unified_command_processor_pure.py",
    ]

    for f in dead_files:
        path = _PROJECT_ROOT / f
        if path.exists():
            _record("failed", f"Dead file still exists: {f}")
        else:
            _record("passed", f"{f} removed")


def verify_ucp_size():
    """Verify the UCP has been hollowed out (classification logic removed)."""
    ucp_path = _PROJECT_ROOT / "backend" / "api" / "unified_command_processor.py"

    if not ucp_path.exists():
        _record("failed", f"UCP not found at {ucp_path}")
        return False

    content = ucp_path.read_text()
    line_count = len(content.splitlines())
    print(f"  INFO: unified_command_processor.py is {line_count} lines")

    # Check for significant reduction from the original 6,747 lines.
    # The surgery preserved execution handlers but removed classification logic,
    # so we expect a meaningful reduction.
    if line_count < 6000:
        reduction = 6747 - line_count
        pct = (reduction / 6747) * 100
        _record("passed", f"Reduced from 6,747 to {line_count} lines ({reduction} removed, {pct:.1f}%)")
        return True
    else:
        _record("failed", f"Expected significant reduction from 6,747 lines, got {line_count}")
        return False


def verify_spinal_reflex_arc_in_ucp():
    """Verify the UCP contains the Spinal Reflex Arc entry point (v242 routing)."""
    ucp_path = _PROJECT_ROOT / "backend" / "api" / "unified_command_processor.py"

    if not ucp_path.exists():
        _record("skipped", "UCP not found")
        return

    content = ucp_path.read_text()

    checks = {
        "_check_reflex_manifest": "Reflex manifest check method",
        "_call_jprime": "J-Prime call method",
        "_execute_action": "Action execution method",
        "StructuredResponse": "StructuredResponse import/usage",
        "reflex_manifest.json": "Manifest file path reference",
    }

    for symbol, description in checks.items():
        if symbol in content:
            _record("passed", f"UCP contains {description} ({symbol})")
        else:
            _record("failed", f"UCP missing {description} ({symbol})")


def verify_trinity_directory_structure():
    """Verify the ~/.jarvis/trinity/ directory structure exists."""
    trinity_dir = Path.home() / ".jarvis" / "trinity"

    if trinity_dir.exists() and trinity_dir.is_dir():
        _record("passed", f"Trinity directory exists: {trinity_dir}")
        contents = list(trinity_dir.iterdir())
        if contents:
            for item in contents:
                print(f"        {item.name}")
        else:
            print("        (empty)")
    else:
        _record("skipped", f"Trinity directory not found at {trinity_dir}")
        print("        (Created when J-Prime publishes manifest)")


def verify_golden_corpus_coverage():
    """Verify the golden corpus covers all expected routing categories."""
    expected_categories = {
        "REFLEX": 0,
        "answer": 0,
        "vision_needed": 0,
        "action_system": 0,
        "action_surveillance": 0,
        "conversation": 0,
        "escalate": 0,
    }

    for test in GOLDEN_CORPUS:
        if test.get("expected_type") == "REFLEX":
            expected_categories["REFLEX"] += 1
        elif test.get("expected_intent") == "answer":
            expected_categories["answer"] += 1
        elif test.get("expected_intent") == "vision_needed":
            expected_categories["vision_needed"] += 1
        elif test.get("expected_intent") == "action" and test.get("expected_domain") == "system":
            expected_categories["action_system"] += 1
        elif test.get("expected_intent") == "action" and test.get("expected_domain") == "surveillance":
            expected_categories["action_surveillance"] += 1
        elif test.get("expected_intent") == "conversation":
            expected_categories["conversation"] += 1
        elif test.get("expected_escalate"):
            expected_categories["escalate"] += 1

    all_covered = True
    for category, count in expected_categories.items():
        if count == 0:
            _record("failed", f"Golden corpus missing category: {category}")
            all_covered = False
        else:
            _record("passed", f"Category '{category}' covered ({count} test cases)")

    total = len(GOLDEN_CORPUS)
    print(f"\n  Total golden corpus entries: {total}")
    return all_covered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run all Trinity verification checks."""
    print("=" * 60)
    print("Trinity Cognitive Architecture -- Verification Suite")
    print("=" * 60)
    print(f"Run at: {datetime.now().isoformat()}")
    print(f"Project root: {_PROJECT_ROOT}")

    print("\n1. Reflex Manifest Validation")
    print("-" * 40)
    verify_reflex_manifest()

    print("\n2. Reflex Routing Verification")
    print("-" * 40)
    verify_reflex_routing()

    print("\n3. Reflex False Positive Check")
    print("-" * 40)
    verify_no_reflex_false_positives()

    print("\n4. Component Import Check")
    print("-" * 40)
    verify_imports()

    print("\n5. StructuredResponse Field Validation")
    print("-" * 40)
    verify_structured_response_fields()

    print("\n6. Dead File Removal Verification")
    print("-" * 40)
    verify_dead_files_removed()

    print("\n7. UCP Size Verification")
    print("-" * 40)
    verify_ucp_size()

    print("\n8. Spinal Reflex Arc in UCP")
    print("-" * 40)
    verify_spinal_reflex_arc_in_ucp()

    print("\n9. Trinity Directory Structure")
    print("-" * 40)
    verify_trinity_directory_structure()

    print("\n10. Golden Corpus Coverage")
    print("-" * 40)
    verify_golden_corpus_coverage()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = _results["passed"] + _results["failed"] + _results["skipped"]
    print(f"  Passed:  {_results['passed']}")
    print(f"  Failed:  {_results['failed']}")
    print(f"  Skipped: {_results['skipped']}")
    print(f"  Total:   {total}")

    if _results["failed"] == 0:
        print("\n  All checks passed (or were skipped due to runtime dependencies).")
    else:
        print(f"\n  {_results['failed']} check(s) failed -- review above output.")

    print("\n" + "=" * 60)
    print("Verification complete.")
    print("=" * 60)
    print("\nNote: Full end-to-end testing requires J-Prime running.")
    print("Start J-Prime, then run individual queries through process_command()")
    print("to verify intent/domain classification matches GOLDEN_CORPUS expectations.")

    # Exit with non-zero if any failures
    sys.exit(1 if _results["failed"] > 0 else 0)


if __name__ == "__main__":
    main()
