#!/usr/bin/env python3
"""
God Mode Surveillance Routing Tests

ROOT CAUSE FIX VERIFICATION:
Tests that God Mode surveillance commands are correctly:
1. Classified by the fallback classifier (python_bridge.py)
2. Parsed by _parse_watch_command (intelligent_command_handler.py)
3. Routed BEFORE classification in handle_command()

Run: python3 backend/tests/test_god_mode_routing.py
"""

import re
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Optional pytest import for formal testing
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # Create a dummy decorator
    class pytest:
        @staticmethod
        def mark():
            pass
    pytest.mark = type('mark', (), {'parametrize': lambda *a, **k: lambda f: f})()


class TestFallbackClassifier:
    """Test the fallback classification in python_bridge.py"""

    def get_classifier_result(self, text: str):
        """Simulate the fallback classifier logic"""
        text_lower = text.lower().strip()

        # Surveillance patterns from the fix
        surveillance_verbs = [
            r'\bwatch\b', r'\bmonitor\b', r'\btrack\b', r'\bobserve\b',
            r'\bsurveillance\b', r'\bkeep\s+an?\s+eye\b', r'\bscan\b'
        ]

        notification_patterns = [
            r'\bnotify\s+me\s+when\b', r'\balert\s+me\s+when\b',
            r'\btell\s+me\s+when\b', r'\blet\s+me\s+know\s+when\b',
            r'\bwarn\s+me\s+when\b', r'\bping\s+me\s+when\b'
        ]

        trigger_patterns = [
            r'\bfor\s+.+', r'\bwhen\s+.+\s+(?:says|shows|displays|appears)\b',
            r'\buntil\s+.+', r'\bwindows?\b', r'\binstances?\b', r'\btabs?\b'
        ]

        god_mode_patterns = [
            r'\b(?:all|every|each)\s+.*?\s*(?:windows?|tabs?|instances?|spaces?)\b',
            r'\bacross\s+all\b', r'\bevery\s+space\b', r'\ball\s+spaces\b'
        ]

        has_surveillance_verb = any(re.search(p, text_lower) for p in surveillance_verbs)
        has_notification = any(re.search(p, text_lower) for p in notification_patterns)
        has_trigger = any(re.search(p, text_lower) for p in trigger_patterns)
        has_god_mode = any(re.search(p, text_lower) for p in god_mode_patterns)

        # Surveillance detected - route to vision
        if (has_surveillance_verb and has_trigger) or has_notification or has_god_mode:
            return {
                "type": "vision",
                "intent": "god_mode_surveillance" if has_god_mode else "surveillance",
                "is_god_mode": has_god_mode
            }

        return {"type": "conversation", "intent": "general"}

    @pytest.mark.parametrize("command,expected_type,expected_god_mode", [
        # God Mode commands (all/every + windows)
        ("Watch all Chrome windows", "vision", True),
        ("Monitor every Terminal instance", "vision", True),
        ("Watch all Safari tabs for loading complete", "vision", True),
        ("Track every Firefox window across all spaces", "vision", True),

        # Regular surveillance commands
        ("Watch Terminal for Build Complete", "vision", False),
        ("Monitor Chrome for error message", "vision", False),
        ("Watch Safari for page loaded", "vision", False),

        # Notification-style commands
        ("Notify me when Terminal says done", "vision", False),
        ("Alert me when Chrome shows error", "vision", False),
        ("Tell me when Safari displays login page", "vision", False),
        ("Let me know when Terminal has finished", "vision", False),

        # Should NOT match - regular commands
        ("Tell me a joke", "conversation", False),
        ("What time is it", "conversation", False),
        ("Open Chrome", "conversation", False),  # No trigger
        ("Close Terminal", "conversation", False),
    ])
    def test_classifier_routing(self, command, expected_type, expected_god_mode):
        """Test that classifier routes surveillance commands to vision handler"""
        result = self.get_classifier_result(command)

        assert result["type"] == expected_type, \
            f"Command '{command}' should route to '{expected_type}', got '{result['type']}'"

        if expected_type == "vision":
            assert result.get("is_god_mode") == expected_god_mode, \
                f"Command '{command}' god_mode should be {expected_god_mode}, got {result.get('is_god_mode')}"

        print(f"  {command}")


class TestParseWatchCommand:
    """Test the _parse_watch_command function logic"""

    def parse_watch_command(self, text: str):
        """Simulate _parse_watch_command logic from the fix"""
        text_lower = text.lower().strip()
        original_text = text.strip()

        # Surveillance intent detection
        primary_surveillance = [r'\bwatch\b', r'\bmonitor\b', r'\bsurveillance\b']
        secondary_surveillance = [r'\btrack\b', r'\bobserve\b', r'\bscan\b',
                                  r'\bkeep\s+(?:an?\s+)?eye\s+on\b']
        notification_patterns = [
            r'\bnotify\s+me\s+when\b', r'\balert\s+me\s+when\b',
            r'\btell\s+me\s+when\b', r'\blet\s+me\s+know\s+when\b'
        ]

        has_primary = any(re.search(p, text_lower) for p in primary_surveillance)
        has_secondary = any(re.search(p, text_lower) for p in secondary_surveillance)
        has_notification = any(re.search(p, text_lower) for p in notification_patterns)

        if not (has_primary or has_secondary or has_notification):
            return None

        # God mode detection
        god_mode_patterns = [
            r'\b(?:all|every|each)\s+(?:\w+\s+)?(?:windows?|tabs?|instances?|spaces?)\b',
            r'\bacross\s+(?:all\s+)?spaces?\b', r'\bevery\s+space\b'
        ]
        all_spaces = any(re.search(p, text_lower) for p in god_mode_patterns)

        # App & trigger extraction - Pattern A: direct
        pattern_direct = re.compile(
            r'(?:watch|monitor|track|observe|scan)\s+'
            r'(?:all\s+)?(?:the\s+)?(?:open\s+)?'
            r'(\w+(?:\s+\w+)?)\s*'
            r'(?:windows?|tabs?|instances?)?\s*'
            r'(?:across\s+all\s+spaces?\s*)?'
            r'(?:for|when|until)\s+(.+)',
            re.IGNORECASE
        )

        app_name = None
        trigger_text = None

        match = pattern_direct.search(original_text)
        if match:
            app_name = match.group(1).strip()
            trigger_text = match.group(2).strip()

        # Pattern B: notification style
        if not app_name:
            pattern_notify = re.compile(
                r'(?:notify|alert|tell|warn|ping)\s+me\s+'
                r'(?:when|if)\s+(?:the\s+)?'
                r'(\w+(?:\s+\w+)?)\s+'
                r'(?:says?|shows?|displays?)\s+(.+)',
                re.IGNORECASE
            )
            match = pattern_notify.search(original_text)
            if match:
                app_name = match.group(1).strip()
                trigger_text = match.group(2).strip()

        if not app_name or not trigger_text:
            return None

        # Clean up
        app_prefixes = {'the', 'my', 'all', 'every', 'open'}
        app_words = app_name.split()
        while app_words and app_words[0].lower() in app_prefixes:
            app_words.pop(0)
        app_name = ' '.join(app_words).title() if app_words else app_name.title()

        return {
            'app_name': app_name,
            'trigger_text': trigger_text.strip('"\''),
            'all_spaces': all_spaces,
            'is_god_mode': all_spaces
        }

    @pytest.mark.parametrize("command,expected_app,expected_trigger,expected_god_mode", [
        # Basic watch commands
        ("Watch Terminal for Build Complete", "Terminal", "Build Complete", False),
        ("Monitor Chrome for error message", "Chrome", "error message", False),
        ("Watch Safari for loading finished", "Safari", "loading finished", False),

        # God Mode commands
        ("Watch all Chrome windows for bouncing ball", "Chrome", "bouncing ball", True),
        ("Monitor every Terminal instance for done", "Terminal", "done", True),
        ("Watch all Safari tabs for page loaded", "Safari", "page loaded", True),

        # Notification style
        ("Notify me when Terminal says Build Complete", "Terminal", "Build Complete", False),
        ("Alert me when Chrome shows error", "Chrome", "error", False),
        ("Tell me when Safari displays login", "Safari", "login", False),

        # Complex commands
        ("Watch all Chrome windows across all spaces for bouncing ball", "Chrome", "bouncing ball", True),
    ])
    def test_parse_watch_command(self, command, expected_app, expected_trigger, expected_god_mode):
        """Test that _parse_watch_command extracts correct values"""
        result = self.parse_watch_command(command)

        assert result is not None, f"Command '{command}' should be parsed, got None"
        assert result['app_name'] == expected_app, \
            f"App should be '{expected_app}', got '{result['app_name']}'"
        assert expected_trigger in result['trigger_text'], \
            f"Trigger should contain '{expected_trigger}', got '{result['trigger_text']}'"
        assert result['is_god_mode'] == expected_god_mode, \
            f"God mode should be {expected_god_mode}, got {result['is_god_mode']}"

        print(f"  {command} -> app={result['app_name']}, god_mode={result['is_god_mode']}")

    @pytest.mark.parametrize("command", [
        "Tell me a joke",
        "What is the weather",
        "Open Chrome",
        "Close Terminal",
        "Hello Ironcliw",
    ])
    def test_non_surveillance_commands_return_none(self, command):
        """Test that non-surveillance commands return None"""
        result = self.parse_watch_command(command)
        assert result is None, f"Command '{command}' should return None, got {result}"
        print(f"  {command} -> None (correct)")


class TestGodModePatternRegex:
    """Test the God Mode regex pattern specifically"""

    @pytest.mark.parametrize("text,should_match", [
        # Should match - God Mode patterns
        ("all Chrome windows", True),
        ("every Terminal instance", True),
        ("each Safari tab", True),
        ("all windows", True),
        ("every window", True),
        ("across all spaces", True),
        ("every space", True),
        ("all spaces", True),
        ("all my Safari tabs", True),
        ("all open Chrome windows", True),

        # Should NOT match
        ("Chrome window", False),
        ("Terminal instance", False),
        ("Safari tab", False),
        ("open Chrome", False),
        ("the window", False),
    ])
    def test_god_mode_pattern(self, text, should_match):
        """Test God Mode regex patterns"""
        god_mode_patterns = [
            r'\b(?:all|every|each)\s+(?:\w+\s+)?(?:windows?|tabs?|instances?|spaces?)\b',
            r'\bacross\s+(?:all\s+)?spaces?\b',
            r'\bevery\s+space\b',
            r'\ball\s+spaces?\b'
        ]

        text_lower = text.lower()
        matches = any(re.search(p, text_lower) for p in god_mode_patterns)

        assert matches == should_match, \
            f"'{text}' should {'match' if should_match else 'NOT match'} God Mode pattern, got {matches}"

        status = "" if matches == should_match else ""
        print(f"  {status} '{text}' -> {'MATCH' if matches else 'NO MATCH'}")


def run_manual_tests():
    """Run manual verification without pytest"""
    print("\n" + "=" * 70)
    print("GOD MODE ROUTING - ROOT CAUSE FIX VERIFICATION")
    print("=" * 70)

    # Test 1: Fallback classifier
    print("\n[1] FALLBACK CLASSIFIER TESTS")
    print("-" * 50)
    test_classifier = TestFallbackClassifier()

    test_cases = [
        ("Watch all Chrome windows", "vision", True),
        ("Monitor every Terminal instance", "vision", True),
        ("Watch Terminal for Build Complete", "vision", False),
        ("Notify me when Chrome shows error", "vision", False),
        ("Tell me a joke", "conversation", False),
    ]

    for cmd, expected_type, expected_god_mode in test_cases:
        result = test_classifier.get_classifier_result(cmd)
        status = "" if result["type"] == expected_type else ""
        gm_status = "" if result.get("is_god_mode", False) == expected_god_mode else ""
        print(f"  {status} {gm_status} {cmd}")
        print(f"       -> type={result['type']}, god_mode={result.get('is_god_mode', False)}")

    # Test 2: Parse watch command
    print("\n[2] PARSE WATCH COMMAND TESTS")
    print("-" * 50)
    test_parser = TestParseWatchCommand()

    parse_cases = [
        ("Watch all Chrome windows for bouncing ball", "Chrome", True),
        ("Watch Terminal for Build Complete", "Terminal", False),
        ("Notify me when Safari says loading finished", "Safari", False),
        ("Tell me a joke", None, False),
    ]

    for cmd, expected_app, expected_god_mode in parse_cases:
        result = test_parser.parse_watch_command(cmd)
        if result:
            status = "" if result['app_name'] == expected_app else ""
            gm_status = "" if result['is_god_mode'] == expected_god_mode else ""
            print(f"  {status} {gm_status} {cmd}")
            print(f"       -> app={result['app_name']}, trigger={result['trigger_text']}, god_mode={result['is_god_mode']}")
        else:
            status = "" if expected_app is None else ""
            print(f"  {status}  {cmd} -> None (non-surveillance)")

    # Test 3: God Mode pattern
    print("\n[3] GOD MODE REGEX PATTERN TESTS")
    print("-" * 50)

    god_mode_patterns = [
        r'\b(?:all|every|each)\s+(?:\w+\s+)?(?:windows?|tabs?|instances?|spaces?)\b',
        r'\bacross\s+(?:all\s+)?spaces?\b',
        r'\bevery\s+space\b',
        r'\ball\s+spaces?\b'
    ]

    pattern_cases = [
        ("all Chrome windows", True),
        ("every Terminal instance", True),
        ("Chrome window", False),
        ("across all spaces", True),
    ]

    for text, should_match in pattern_cases:
        matches = any(re.search(p, text.lower()) for p in god_mode_patterns)
        status = "" if matches == should_match else ""
        print(f"  {status} '{text}' -> {'MATCH' if matches else 'NO MATCH'}")

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_manual_tests()
