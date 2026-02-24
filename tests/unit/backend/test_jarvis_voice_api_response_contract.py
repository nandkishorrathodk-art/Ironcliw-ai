from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.api.jarvis_voice_api import JARVISCommand, JARVISVoiceAPI


_RAW_PROCESS_COMMAND = JARVISVoiceAPI.process_command.__wrapped__.__wrapped__


class _FakePipeline:
    def __init__(self, result):
        self._result = result

    async def process_async(self, *args, **kwargs):
        return self._result


@pytest.fixture
def api() -> JARVISVoiceAPI:
    inst = JARVISVoiceAPI()
    inst.jarvis_available = True
    inst._jarvis_initialized = True
    inst._jarvis = SimpleNamespace(running=True, user_name="Sir")
    inst.system_control_enabled = False

    async def _noop_record(*args, **kwargs):
        return -1

    inst._record_interaction = _noop_record
    return inst


def test_coerce_text_handles_none_and_empty() -> None:
    assert JARVISVoiceAPI._coerce_text(None, fallback="fallback") == "fallback"
    assert JARVISVoiceAPI._coerce_text("   ", fallback="fallback") == "fallback"
    assert JARVISVoiceAPI._coerce_text("  hello  ", fallback="fallback") == "hello"


@pytest.mark.asyncio
async def test_process_command_rejects_null_text_before_pipeline(api: JARVISVoiceAPI) -> None:
    bad_command = SimpleNamespace(text=None, audio_data=None, deadline=None)

    result = await _RAW_PROCESS_COMMAND(api, bad_command)

    assert result["status"] == "error"
    assert "didn't catch" in result["response"].lower()


@pytest.mark.asyncio
async def test_process_command_normalizes_none_pipeline_response(api: JARVISVoiceAPI) -> None:
    api._pipeline = _FakePipeline({"response": None, "type": "generic"})

    result = await _RAW_PROCESS_COMMAND(api, JARVISCommand(text="open safari"))

    assert result["success"] is True
    assert result["response"] == "I processed your command, Sir."
