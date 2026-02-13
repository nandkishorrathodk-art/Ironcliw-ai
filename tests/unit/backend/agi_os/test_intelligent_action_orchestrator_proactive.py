from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock


class _FakeEventStream:
    def __init__(self):
        self._next_sub = 0
        self.subscriptions: Dict[str, Dict[str, Any]] = {}
        self.unsubscribed: List[str] = []
        self.action_proposals: List[Dict[str, Any]] = []
        self._corr = 0

    def subscribe(self, event_types, handler, min_priority=None, filter_func=None):
        self._next_sub += 1
        sub_id = f"sub-{self._next_sub}"
        self.subscriptions[sub_id] = {
            "event_types": event_types,
            "handler": handler,
            "min_priority": min_priority,
            "filter_func": filter_func,
        }
        return sub_id

    def unsubscribe(self, subscription_id: str) -> bool:
        self.unsubscribed.append(subscription_id)
        return self.subscriptions.pop(subscription_id, None) is not None

    def create_correlation_id(self) -> str:
        self._corr += 1
        return f"corr-{self._corr}"

    def get_recent_events(self, event_type=None, source=None, minutes=5):
        return []

    async def emit_action_proposed(
        self,
        action: str,
        target: str,
        reason: str,
        confidence: float,
        correlation_id: Optional[str] = None,
        source: str = "decision_engine",
        **kwargs,
    ) -> str:
        self.action_proposals.append(
            {
                "action": action,
                "target": target,
                "reason": reason,
                "confidence": confidence,
                "correlation_id": correlation_id,
                "source": source,
                "extra": kwargs,
            }
        )
        return f"proposal-{len(self.action_proposals)}"


class _FakeInterventionEngine:
    async def generate_goal(self, context: Optional[Dict[str, Any]] = None):
        return None


async def test_setup_event_handlers_replaces_prior_subscriptions():
    from backend.agi_os.intelligent_action_orchestrator import IntelligentActionOrchestrator

    orchestrator = IntelligentActionOrchestrator()
    stream = _FakeEventStream()
    orchestrator._event_stream = stream

    await orchestrator._setup_event_handlers()
    assert len(orchestrator._subscription_ids) == 3
    assert len(stream.subscriptions) == 3

    await orchestrator._setup_event_handlers()
    assert len(orchestrator._subscription_ids) == 3
    assert len(stream.subscriptions) == 3
    assert len(stream.unsubscribed) == 3


async def test_start_stop_manage_proactive_task_and_subscriptions():
    from backend.agi_os.intelligent_action_orchestrator import (
        IntelligentActionOrchestrator,
        OrchestratorState,
    )

    orchestrator = IntelligentActionOrchestrator()
    stream = _FakeEventStream()

    async def _fake_init_components():
        orchestrator._event_stream = stream
        orchestrator._approval_manager = None
        orchestrator._voice = None
        orchestrator._decision_engine = object()
        orchestrator._intervention_engine = _FakeInterventionEngine()
        orchestrator._action_executor = None

    orchestrator._init_components = _fake_init_components  # type: ignore[method-assign]

    await orchestrator.start()
    assert orchestrator._state == OrchestratorState.RUNNING
    assert orchestrator._proactive_task is not None
    assert not orchestrator._proactive_task.done()
    assert len(orchestrator._subscription_ids) == 3

    await orchestrator.stop()
    assert orchestrator._state == OrchestratorState.STOPPED
    assert orchestrator._proactive_task is None
    assert orchestrator._executor_task is None
    assert len(orchestrator._subscription_ids) == 0
    assert len(stream.unsubscribed) == 3


async def test_proactive_cycle_routes_goal_without_inbound_events():
    from backend.agi_os.intelligent_action_orchestrator import IntelligentActionOrchestrator

    orchestrator = IntelligentActionOrchestrator()
    stream = _FakeEventStream()
    orchestrator._event_stream = stream
    orchestrator._approval_manager = None
    orchestrator._voice = None
    orchestrator._generate_proactive_goal = AsyncMock(  # type: ignore[method-assign]
        return_value={
            "description": "You have a meeting in 15 minutes",
            "priority": "high",
            "context": {
                "situation_type": "time_management",
                "severity": 0.82,
                "confidence": 0.88,
            },
        }
    )

    await orchestrator._run_proactive_cycle()

    stats = orchestrator.get_stats()
    assert stats["proactive_goals_generated"] == 1
    assert stats["actions_proposed"] == 1
    assert stats["issues_detected"] == 1
    assert len(stream.action_proposals) == 1

    proposal = stream.action_proposals[0]
    assert proposal["action"] == "prepare_meeting"
    assert proposal["target"] == "workspace"
    assert "Proactive scheduler generated this action" in proposal["reason"]
    assert len(orchestrator._pending_issues) == 1
