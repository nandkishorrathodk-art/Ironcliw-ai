#!/usr/bin/env python3
"""
End-to-end integration test for the NeuralMeshBridge (v237.2-v237.3).

Tests the full bidirectional bridge between:
  AgentCommunicationBus ↔ NeuralMeshBridge ↔ ProactiveEventStream

Usage:
    cd backend && python3 -m tests.test_event_bridge_e2e
"""

import asyncio
import sys
import os
import time
import logging

# Ensure both backend/ and repo root are on sys.path.
# backend/ — for direct imports like `from agi_os.xxx import ...`
# repo root — for transitive `from backend.core.xxx import ...` in dependencies
_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_repo_root = os.path.dirname(_backend_dir)
sys.path.insert(0, _backend_dir)
sys.path.insert(0, _repo_root)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("test_bridge")
logger.setLevel(logging.INFO)

# Track test results
_results = []


def test(name):
    """Decorator to register and run a test."""
    def decorator(func):
        func._test_name = name
        _results.append(func)
        return func
    return decorator


class MockCoordinator:
    """Minimal mock of NeuralMeshCoordinator with a real bus."""

    def __init__(self, bus):
        self._bus = bus

    @property
    def bus(self):
        return self._bus


async def create_test_stream():
    """Create a ProactiveEventStream configured for testing.

    Disables voice narration to avoid slow imports of
    realtime_voice_communicator that would otherwise delay
    event delivery to handlers.
    """
    from agi_os.proactive_event_stream import ProactiveEventStream
    stream = ProactiveEventStream()
    stream._voice = False  # Sentinel: skip voice import
    await stream.start()
    return stream


# ==================== TESTS ====================


@test("Import all components")
async def test_imports():
    """Verify all bridge components can be imported."""
    from neural_mesh.communication.agent_communication_bus import AgentCommunicationBus
    from neural_mesh.data_models import MessageType, AgentMessage, MessagePriority
    from neural_mesh.config import CommunicationBusConfig
    from agi_os.proactive_event_stream import (
        ProactiveEventStream, AGIEvent, EventType, EventPriority,
    )
    from agi_os.jarvis_integration import NeuralMeshBridge, get_event_bridge
    return True


@test("Forward bridge: Bus ERROR_DETECTED → EventStream ERROR_DETECTED")
async def test_forward_error():
    """Emit an ERROR_DETECTED on the bus, verify it arrives at the event stream."""
    from neural_mesh.communication.agent_communication_bus import AgentCommunicationBus
    from neural_mesh.data_models import MessageType, MessagePriority
    from neural_mesh.config import CommunicationBusConfig
    from agi_os.proactive_event_stream import (
        ProactiveEventStream, AGIEvent, EventType, EventPriority,
    )
    from agi_os.jarvis_integration import NeuralMeshBridge

    # Create components
    bus_config = CommunicationBusConfig()
    bus = AgentCommunicationBus(config=bus_config)
    await bus.start()

    stream = await create_test_stream()

    bridge = NeuralMeshBridge()
    mock_coord = MockCoordinator(bus)

    # Track received events
    received = []

    async def handler(event: AGIEvent):
        received.append(event)

    stream.subscribe(
        event_types=[EventType.ERROR_DETECTED],
        handler=handler,
    )

    # Connect bridge (subscribes both directions)
    await bridge.connect(
        neural_mesh=mock_coord,
        event_stream=stream,
    )

    # Emit error on bus
    await bus.broadcast(
        from_agent="test_agent",
        message_type=MessageType.ERROR_DETECTED,
        payload={"error": "disk full", "severity": "high"},
    )

    # Give the event time to propagate through async handlers
    await asyncio.sleep(0.3)

    # Verify
    assert len(received) >= 1, f"Expected ≥1 event, got {len(received)}"
    evt = received[0]
    assert evt.event_type == EventType.ERROR_DETECTED, f"Wrong type: {evt.event_type}"
    assert evt.source.startswith("neural_mesh_bridge."), f"Wrong source: {evt.source}"
    assert evt.data.get("error") == "disk full", f"Wrong data: {evt.data}"
    assert evt.metadata.get("bridged") is True, "Missing bridged flag"
    assert evt.requires_narration is True, "ERROR_DETECTED should require narration"
    assert bridge.get_stats()['forward'] >= 1, "Forward counter not incremented"

    # Cleanup
    await bridge.stop()
    await bus.stop()

    return True


@test("Forward bridge: Bus TASK_COMPLETED → EventStream ACTION_COMPLETED")
async def test_forward_task_completed():
    """Verify TASK_COMPLETED translates to ACTION_COMPLETED."""
    from neural_mesh.communication.agent_communication_bus import AgentCommunicationBus
    from neural_mesh.data_models import MessageType
    from neural_mesh.config import CommunicationBusConfig
    from agi_os.proactive_event_stream import (
        ProactiveEventStream, AGIEvent, EventType,
    )
    from agi_os.jarvis_integration import NeuralMeshBridge

    bus = AgentCommunicationBus(config=CommunicationBusConfig())
    await bus.start()
    stream = await create_test_stream()
    bridge = NeuralMeshBridge()

    received = []

    async def on_action_completed(event):
        received.append(event)

    stream.subscribe(EventType.ACTION_COMPLETED, on_action_completed)

    await bridge.connect(neural_mesh=MockCoordinator(bus), event_stream=stream)

    await bus.broadcast(
        from_agent="worker_agent",
        message_type=MessageType.TASK_COMPLETED,
        payload={"task_id": "T-123", "result": "success"},
    )
    await asyncio.sleep(0.3)

    assert len(received) >= 1, f"Expected ≥1, got {len(received)}"
    assert received[0].event_type == EventType.ACTION_COMPLETED
    assert received[0].requires_narration is False  # TASK_COMPLETED doesn't narrate

    await bridge.stop()
    await bus.stop()
    return True


@test("Reverse bridge: EventStream ACTION_PROPOSED → Bus ANNOUNCEMENT")
async def test_reverse_action_proposed():
    """Emit ACTION_PROPOSED on event stream, verify it reaches the bus."""
    from neural_mesh.communication.agent_communication_bus import AgentCommunicationBus
    from neural_mesh.data_models import MessageType
    from neural_mesh.config import CommunicationBusConfig
    from agi_os.proactive_event_stream import (
        ProactiveEventStream, AGIEvent, EventType, EventPriority,
    )
    from agi_os.jarvis_integration import NeuralMeshBridge

    bus = AgentCommunicationBus(config=CommunicationBusConfig())
    await bus.start()
    stream = await create_test_stream()
    bridge = NeuralMeshBridge()

    # Subscribe to bus broadcasts for ANNOUNCEMENT
    bus_received = []

    async def bus_handler(msg):
        bus_received.append(msg)

    await bus.subscribe_broadcast(MessageType.ANNOUNCEMENT, bus_handler)

    await bridge.connect(neural_mesh=MockCoordinator(bus), event_stream=stream)

    # Emit ACTION_PROPOSED on event stream (from orchestrator, not from bridge)
    event = AGIEvent(
        event_type=EventType.ACTION_PROPOSED,
        source="intelligent_action_orchestrator",
        data={"action": "clear_disk_space", "confidence": 0.85},
        priority=EventPriority.NORMAL,
    )
    await stream.emit(event)
    await asyncio.sleep(0.3)

    assert len(bus_received) >= 1, f"Expected ≥1 bus messages, got {len(bus_received)}"
    msg = bus_received[0]
    assert msg.from_agent == "agi_os_bridge", f"Wrong from_agent: {msg.from_agent}"
    assert msg.message_type == MessageType.ANNOUNCEMENT, f"Wrong type: {msg.message_type}"
    assert msg.payload.get("bridged") is True, "Missing bridged flag in payload"
    assert msg.payload.get("action") == "clear_disk_space"
    assert bridge.get_stats()['reverse'] >= 1, "Reverse counter not incremented"

    await bridge.stop()
    await bus.stop()
    return True


@test("Loop prevention: forward-bridged event NOT re-bridged back")
async def test_loop_prevention_forward():
    """A bus message bridged forward should NOT be bridged back to the bus."""
    from neural_mesh.communication.agent_communication_bus import AgentCommunicationBus
    from neural_mesh.data_models import MessageType
    from neural_mesh.config import CommunicationBusConfig
    from agi_os.proactive_event_stream import (
        ProactiveEventStream, AGIEvent, EventType,
    )
    from agi_os.jarvis_integration import NeuralMeshBridge

    bus = AgentCommunicationBus(config=CommunicationBusConfig())
    await bus.start()
    stream = await create_test_stream()
    bridge = NeuralMeshBridge()

    # Track ALL bus messages (not just from bridge)
    bus_announcements = []

    async def bus_handler(msg):
        bus_announcements.append(msg)

    await bus.subscribe_broadcast(MessageType.ANNOUNCEMENT, bus_handler)

    await bridge.connect(neural_mesh=MockCoordinator(bus), event_stream=stream)

    # Emit ERROR_DETECTED on bus → gets forward-bridged to stream as ERROR_DETECTED
    # ERROR_DETECTED is in the forward map but NOT in the reverse map,
    # so it won't come back. But let's also test with ACTION_FAILED which IS in both maps.
    await bus.broadcast(
        from_agent="test_agent",
        message_type=MessageType.ALERT_RAISED,
        payload={"alert": "test"},
    )
    await asyncio.sleep(0.5)

    # The forward bridge translates ALERT_RAISED → WARNING_DETECTED (stream)
    # WARNING_DETECTED is NOT in the reverse map, so nothing should come back to bus
    # Even if it were, the source "neural_mesh_bridge.test_agent" would be caught
    assert len(bus_announcements) == 0, (
        f"Loop detected! Got {len(bus_announcements)} bus messages back from bridge"
    )
    assert bridge.get_stats()['loop_prevented'] == 0 or bridge.get_stats()['forward'] >= 1

    await bridge.stop()
    await bus.stop()
    return True


@test("Loop prevention: reverse-bridged message NOT re-bridged forward")
async def test_loop_prevention_reverse():
    """A reverse-bridged bus message (from 'agi_os_bridge') should NOT be forward-bridged.

    Uses ACTION_FAILED because it maps reverse to ALERT_RAISED (bus),
    and ALERT_RAISED IS in the forward map. So the forward handler
    receives it but must reject it (from_agent='agi_os_bridge').
    """
    from neural_mesh.communication.agent_communication_bus import AgentCommunicationBus
    from neural_mesh.config import CommunicationBusConfig
    from agi_os.proactive_event_stream import (
        AGIEvent, EventType, EventPriority,
    )
    from agi_os.jarvis_integration import NeuralMeshBridge

    bus = AgentCommunicationBus(config=CommunicationBusConfig())
    await bus.start()
    stream = await create_test_stream()
    bridge = NeuralMeshBridge()

    # Track events that reach the stream
    stream_events = []

    async def on_stream_event(event):
        stream_events.append(event)

    stream.subscribe(None, on_stream_event)  # subscribe to ALL

    await bridge.connect(neural_mesh=MockCoordinator(bus), event_stream=stream)

    # Emit ACTION_FAILED on stream → reverse bridge → ALERT_RAISED on bus
    # ALERT_RAISED IS in the forward map, so forward handler receives it.
    # Forward handler sees from_agent="agi_os_bridge" → loop prevention fires.
    event = AGIEvent(
        event_type=EventType.ACTION_FAILED,
        source="orchestrator",
        data={"action": "test", "error": "simulated"},
        priority=EventPriority.NORMAL,
    )
    await stream.emit(event)
    await asyncio.sleep(0.5)

    # Count events from "neural_mesh_bridge.*" (forward-bridged events)
    rebridged = [e for e in stream_events
                 if getattr(e, 'source', '').startswith('neural_mesh_bridge.')]
    assert len(rebridged) == 0, (
        f"Loop detected! {len(rebridged)} events were re-bridged from bus back to stream"
    )
    assert bridge.get_stats()['loop_prevented'] >= 1, (
        f"Loop prevention counter should be >=1, got {bridge.get_stats()['loop_prevented']}"
    )

    await bridge.stop()
    await bus.stop()
    return True


@test("Rate limiting: rapid ERROR_DETECTED only emits once per 2s window")
async def test_rate_limiting():
    """Emit 10 ERROR_DETECTED in rapid succession, verify rate limiting."""
    from neural_mesh.communication.agent_communication_bus import AgentCommunicationBus
    from neural_mesh.data_models import MessageType
    from neural_mesh.config import CommunicationBusConfig
    from agi_os.proactive_event_stream import (
        ProactiveEventStream, EventType,
    )
    from agi_os.jarvis_integration import NeuralMeshBridge

    bus = AgentCommunicationBus(config=CommunicationBusConfig())
    await bus.start()
    stream = await create_test_stream()
    bridge = NeuralMeshBridge()

    received = []

    async def on_error(event):
        received.append(event)

    stream.subscribe(EventType.ERROR_DETECTED, on_error)

    await bridge.connect(neural_mesh=MockCoordinator(bus), event_stream=stream)

    # Rapid fire 10 ERROR_DETECTED messages
    for i in range(10):
        await bus.broadcast(
            from_agent=f"agent_{i}",
            message_type=MessageType.ERROR_DETECTED,
            payload={"error": f"error_{i}"},
        )

    await asyncio.sleep(0.5)

    # Only 1 should pass through (2s rate limit per MessageType)
    assert len(received) == 1, (
        f"Rate limiting failed: expected 1 event, got {len(received)}"
    )
    stats = bridge.get_stats()
    assert stats['rate_limited'] == 9, (
        f"Expected 9 rate-limited, got {stats['rate_limited']}"
    )

    await bridge.stop()
    await bus.stop()
    return True


@test("CUSTOM messages only bridge when payload declares intent")
async def test_custom_selective_bridging():
    """CUSTOM messages without explicit intent keys should NOT be bridged."""
    from neural_mesh.communication.agent_communication_bus import AgentCommunicationBus
    from neural_mesh.data_models import MessageType
    from neural_mesh.config import CommunicationBusConfig
    from agi_os.proactive_event_stream import (
        ProactiveEventStream, EventType,
    )
    from agi_os.jarvis_integration import NeuralMeshBridge

    bus = AgentCommunicationBus(config=CommunicationBusConfig())
    await bus.start()
    stream = await create_test_stream()
    bridge = NeuralMeshBridge()

    received = []

    async def on_any_event(event):
        received.append(event)

    stream.subscribe(None, on_any_event)

    await bridge.connect(neural_mesh=MockCoordinator(bus), event_stream=stream)

    # CUSTOM without intent keys → should NOT bridge
    await bus.broadcast(
        from_agent="test",
        message_type=MessageType.CUSTOM,
        payload={"random_data": "hello"},
    )
    await asyncio.sleep(0.2)

    bridged_from_mesh = [e for e in received
                         if getattr(e, 'source', '').startswith('neural_mesh_bridge.')]
    assert len(bridged_from_mesh) == 0, "CUSTOM without intent should not bridge"

    # CUSTOM with severity key → SHOULD bridge
    await bus.broadcast(
        from_agent="test",
        message_type=MessageType.CUSTOM,
        payload={"severity": "error", "message": "disk full"},
    )
    await asyncio.sleep(0.2)

    bridged_from_mesh = [e for e in received
                         if getattr(e, 'source', '').startswith('neural_mesh_bridge.')]
    assert len(bridged_from_mesh) == 1, (
        f"CUSTOM with severity should bridge, got {len(bridged_from_mesh)}"
    )
    assert bridged_from_mesh[0].event_type == EventType.ERROR_DETECTED

    await bridge.stop()
    await bus.stop()
    return True


@test("Bridge stats tracking")
async def test_stats():
    """Verify stats are tracked correctly across forward/reverse/rate-limited events."""
    from agi_os.jarvis_integration import NeuralMeshBridge

    bridge = NeuralMeshBridge()
    stats = bridge.get_stats()

    assert stats['connected'] is False
    assert stats['forward'] == 0
    assert stats['reverse'] == 0
    assert stats['rate_limited'] == 0
    assert stats['loop_prevented'] == 0
    assert stats['forward_subscribed'] is False
    assert stats['reverse_subscriptions'] == 0
    return True


@test("Backward compatibility: emit_agent_event() still works")
async def test_emit_agent_event():
    """The legacy emit_agent_event() API should still work."""
    from agi_os.proactive_event_stream import EventType
    from agi_os.jarvis_integration import NeuralMeshBridge

    stream = await create_test_stream()
    bridge = NeuralMeshBridge()

    received = []

    async def on_legacy_event(event):
        received.append(event)

    # Subscribe to ERROR_DETECTED specifically (not ALL — avoids catching SYSTEM_STARTED)
    stream.subscribe(EventType.ERROR_DETECTED, on_legacy_event)

    # Connect without a coordinator (legacy mode)
    await bridge.connect(neural_mesh=None, event_stream=stream)

    # Use legacy API
    event_id = await bridge.emit_agent_event(
        agent_name="test_agent",
        event_type="error",
        data={"message": "something broke"},
    )

    await asyncio.sleep(0.2)

    assert event_id is not None, "emit_agent_event should return an event ID"
    assert len(received) >= 1, "Event should reach stream"
    assert received[0].event_type == EventType.ERROR_DETECTED, (
        f"Expected ERROR_DETECTED, got {received[0].event_type}"
    )

    await bridge.stop()
    return True


# ==================== RUNNER ====================

async def main():
    passed = 0
    failed = 0
    errors = []

    print(f"\n{'='*60}")
    print("  NeuralMeshBridge End-to-End Integration Tests")
    print(f"{'='*60}\n")

    for test_func in _results:
        name = test_func._test_name
        try:
            result = await asyncio.wait_for(test_func(), timeout=10.0)
            if result:
                print(f"  PASS  {name}")
                passed += 1
            else:
                print(f"  FAIL  {name} (returned False)")
                failed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}")
            print(f"        {e}")
            failed += 1
            errors.append((name, str(e)))
        except asyncio.TimeoutError:
            print(f"  FAIL  {name} (timeout)")
            failed += 1
            errors.append((name, "timeout after 10s"))
        except Exception as e:
            print(f"  ERR   {name}")
            print(f"        {type(e).__name__}: {e}")
            failed += 1
            errors.append((name, f"{type(e).__name__}: {e}"))

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'='*60}\n")

    if errors:
        print("Failures:")
        for name, err in errors:
            print(f"  - {name}: {err}")
        print()

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
