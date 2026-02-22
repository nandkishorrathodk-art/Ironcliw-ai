import json

import numpy as np
import pytest

import aiohttp

from backend.voice_unlock.ml_engine_registry import MLEngineRegistry


class _FakeResponse:
    def __init__(self, status, body, headers=None):
        self.status = status
        self._body = body
        self.headers = headers or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        if isinstance(self._body, dict):
            return self._body
        raise ValueError("Response body is not JSON")

    async def text(self):
        if isinstance(self._body, str):
            return self._body
        return json.dumps(self._body)


class _FakeSession:
    def __init__(self, response_plan, call_log):
        self._response_plan = response_plan
        self._call_log = call_log
        self._idx = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, url, json=None, timeout=None):  # noqa: A002
        self._call_log.append(url)
        status, body, headers = self._response_plan[self._idx]
        self._idx += 1
        return _FakeResponse(status=status, body=body, headers=headers)


@pytest.fixture(autouse=True)
def _reset_registry_singleton():
    MLEngineRegistry._instance = None
    yield
    MLEngineRegistry._instance = None


def _patch_cloud_primitives(monkeypatch, registry):
    async def _ready():
        return True

    monkeypatch.setattr(registry, "_check_cloud_readiness", _ready)
    monkeypatch.setattr(registry._cloud_embedding_cb, "can_execute", lambda: (True, ""))
    monkeypatch.setattr(registry._cloud_embedding_cb, "record_success", lambda: None)
    monkeypatch.setattr(registry._cloud_embedding_cb, "record_failure", lambda *_a, **_k: None)
    monkeypatch.setattr(registry, "_mark_cloud_api_success", lambda: None)
    monkeypatch.setattr(registry, "_mark_cloud_api_failure", lambda *_a, **_k: None)
    monkeypatch.setattr(registry, "_record_cloud_endpoint_failure", lambda *_a, **_k: None)
    monkeypatch.setattr(registry, "_log_cloud_cooldown", lambda *_a, **_k: None)
    monkeypatch.setattr(registry, "_apply_cloud_retry_after", lambda *_a, **_k: None)


@pytest.mark.asyncio
async def test_extract_embedding_falls_back_to_root_route(monkeypatch):
    registry = MLEngineRegistry()
    registry._cloud_endpoint = "https://unit.test"
    registry._cloud_endpoint_source = "unit"
    registry._cloud_embedding_route = "/api/ml/speaker_embedding"
    _patch_cloud_primitives(monkeypatch, registry)

    call_log = []
    response_plan = [
        (500, "Internal Server Error", {}),
        (200, {"success": True, "embedding": [0.1, 0.2, 0.3]}, {}),
    ]
    monkeypatch.setattr(
        aiohttp,
        "ClientSession",
        lambda *args, **kwargs: _FakeSession(response_plan, call_log),
    )

    embedding = await registry.extract_speaker_embedding_cloud(b"\x00" * 32, timeout=1.0)

    assert embedding is not None
    assert tuple(embedding.shape) == (1, 3)
    assert call_log[0].endswith("/api/ml/speaker_embedding")
    assert call_log[1].endswith("/speaker_embedding")
    assert registry._cloud_embedding_route == "/speaker_embedding"


@pytest.mark.asyncio
async def test_verify_speaker_falls_back_to_root_route(monkeypatch):
    registry = MLEngineRegistry()
    registry._cloud_endpoint = "https://unit.test"
    registry._cloud_endpoint_source = "unit"
    registry._cloud_verify_route = "/api/ml/speaker_verify"
    _patch_cloud_primitives(monkeypatch, registry)

    call_log = []
    response_plan = [
        (404, "Not Found", {}),
        (200, {"verified": True, "confidence": 0.95}, {}),
    ]
    monkeypatch.setattr(
        aiohttp,
        "ClientSession",
        lambda *args, **kwargs: _FakeSession(response_plan, call_log),
    )

    result = await registry.verify_speaker_cloud(
        b"\x00" * 32,
        reference_embedding=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        timeout=1.0,
    )

    assert result is not None
    assert result.get("verified") is True
    assert call_log[0].endswith("/api/ml/speaker_verify")
    assert call_log[1].endswith("/speaker_verify")
    assert registry._cloud_verify_route == "/speaker_verify"

