from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict


class _CallbackSet:
    def __init__(self):
        self._items = set()

    def add(self, callback: Callable) -> None:
        self._items.add(callback)

    def discard(self, callback: Callable) -> None:
        self._items.discard(callback)

    def __len__(self) -> int:
        return len(self._items)


class _FakeAnalyzer:
    def __init__(self):
        callback_names = [
            "error_detected",
            "content_changed",
            "app_changed",
            "user_needs_help",
            "memory_warning",
            "notification_detected",
            "meeting_detected",
            "security_concern",
            "screen_captured",
        ]
        self.event_callbacks: Dict[str, _CallbackSet] = {
            name: _CallbackSet() for name in callback_names
        }

    def register_callback(self, event_type: str, callback: Callable) -> None:
        self.event_callbacks[event_type].add(callback)

    def unregister_callback(self, event_type: str, callback: Callable) -> None:
        self.event_callbacks[event_type].discard(callback)

    def get_event_stats(self) -> Dict[str, Any]:
        return {"emitted": {}, "suppressed": {}}


class _FakeEventStream:
    async def emit(self, event: Any) -> None:
        return None


class _FakeResponseUsage:
    input_tokens = 10
    output_tokens = 5


class _FakeResponseContent:
    def __init__(self, text: str):
        self.text = text


class _FakeResponse:
    def __init__(self, text: str = "SUMMARY: test\nSUGGESTIONS: none\nCONFIDENCE: 0.9"):
        self.content = [_FakeResponseContent(text)]
        self.usage = _FakeResponseUsage()


class _DelayedMessages:
    def __init__(self, delay_seconds: float = 0.0, error: Exception | None = None):
        self._delay_seconds = delay_seconds
        self._error = error
        self.calls = 0

    async def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls += 1
        if self._delay_seconds > 0:
            await asyncio.sleep(self._delay_seconds)
        if self._error is not None:
            raise self._error
        return _FakeResponse()


class _FakeClaudeClient:
    def __init__(self, messages: _DelayedMessages):
        self.messages = messages
        self.closed = False

    async def close(self) -> None:
        self.closed = True


async def test_bridge_registers_and_unregisters_all_callbacks():
    from backend.agi_os.jarvis_integration import ScreenAnalyzerBridge

    bridge = ScreenAnalyzerBridge()
    analyzer = _FakeAnalyzer()
    bridge._analyzer = analyzer

    await bridge._register_callbacks()
    stats = bridge.get_stats()

    assert stats["callback_expected_count"] == 9
    assert stats["callback_registered_count"] == 9
    assert stats["callback_missing_types"] == []
    for callback_set in analyzer.event_callbacks.values():
        assert len(callback_set) == 1

    bridge._connected = True
    await bridge.disconnect()

    post_stats = bridge.get_stats()
    assert post_stats["callback_registered_count"] == 0
    for callback_set in analyzer.event_callbacks.values():
        assert len(callback_set) == 0


async def test_error_detected_queues_vision_analysis_without_blocking_callback():
    from backend.agi_os.jarvis_integration import ScreenAnalyzerBridge

    messages = _DelayedMessages(delay_seconds=0.6)
    bridge = ScreenAnalyzerBridge()
    bridge._vision_enabled = True
    bridge._claude_client = _FakeClaudeClient(messages)
    bridge._event_stream = _FakeEventStream()
    bridge._connected = True
    bridge._ensure_vision_error_worker()

    started = time.perf_counter()
    await bridge._on_error_detected(
        {
            "error_type": "exception",
            "message": "something failed",
            "location": "dialog",
            "screenshot": b"fake-image-bytes",
        }
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 0.3

    await asyncio.sleep(0.8)
    stats = bridge.get_stats()
    assert messages.calls >= 1
    assert stats["vision_error_queue_enqueued"] >= 1
    assert stats["vision_error_queue_processed"] >= 1

    await bridge.disconnect()


async def test_analyze_screen_opens_circuit_after_consecutive_failures():
    from backend.agi_os.jarvis_integration import ScreenAnalyzerBridge

    messages = _DelayedMessages(error=RuntimeError("Request timed out or interrupted"))
    bridge = ScreenAnalyzerBridge()
    bridge._vision_enabled = True
    bridge._claude_client = _FakeClaudeClient(messages)
    bridge._vision_failure_threshold = 2
    bridge._vision_circuit_cooldown_seconds = 60.0

    await bridge.analyze_screen(screenshot=b"img-1")
    await bridge.analyze_screen(screenshot=b"img-2")
    calls_before_short_circuit = messages.calls
    await bridge.analyze_screen(screenshot=b"img-3")

    stats = bridge.get_stats()
    assert calls_before_short_circuit == 2
    assert messages.calls == calls_before_short_circuit
    assert stats["vision_circuit_open"] is True
    assert stats["vision_circuit_short_circuits"] >= 1
