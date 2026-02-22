import pytest

from backend.api.vision_command_handler import (
    VisionCommandHandler,
    VisionDescribeResult,
)
from backend.vision.continuous_screen_analyzer import MemoryAwareScreenAnalyzer


@pytest.mark.asyncio
async def test_describe_screen_returns_backward_compatible_result(monkeypatch):
    handler = VisionCommandHandler()

    async def fake_analyze_screen(command_text):
        return {
            "handled": True,
            "response": f"analysis:{command_text}",
            "metadata": {"source": "test"},
        }

    monkeypatch.setattr(handler, "analyze_screen", fake_analyze_screen)

    result = await handler.describe_screen({"query": "status check"})

    assert isinstance(result, VisionDescribeResult)
    assert isinstance(result, dict)
    assert result.success is True
    assert result["success"] is True
    assert result.description == "analysis:status check"
    assert result.data["handled"] is True
    assert result.data["query"] == "status check"
    assert result.error is None


@pytest.mark.asyncio
async def test_describe_screen_handles_errors_without_raising(monkeypatch):
    handler = VisionCommandHandler()

    async def fake_analyze_screen(_command_text):
        raise RuntimeError("boom")

    monkeypatch.setattr(handler, "analyze_screen", fake_analyze_screen)

    result = await handler.describe_screen({"query": "trigger error"})

    assert result.success is False
    assert result.error == "boom"
    assert "error" in result.description.lower()


def test_continuous_analyzer_validates_handler_contract():
    class InvalidHandler:
        async def capture_screen(self):
            return None

    with pytest.raises(TypeError, match="describe_screen"):
        MemoryAwareScreenAnalyzer(InvalidHandler())

