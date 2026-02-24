import asyncio

from backend.core.voice_sidecar_contract import (
    VoiceSidecarClient,
    VoiceSidecarContractConfig,
    contract_config_from_env,
    wait_for_sidecar_health,
)


def test_contract_config_from_env_parses_command(monkeypatch):
    monkeypatch.setenv("JARVIS_VOICE_SIDECAR_ENABLED", "true")
    monkeypatch.setenv("JARVIS_VOICE_SIDECAR_REQUIRED", "true")
    monkeypatch.setenv("JARVIS_VOICE_SIDECAR_TRANSPORT", "http")
    monkeypatch.setenv("JARVIS_VOICE_SIDECAR_BASE_URL", "http://127.0.0.1:9999")
    monkeypatch.setenv(
        "JARVIS_VOICE_SIDECAR_COMMAND",
        'python3 -m backend.voice.voice_worker_service --flag "quoted value"',
    )

    cfg = contract_config_from_env()

    assert cfg.enabled is True
    assert cfg.required is True
    assert cfg.transport == "http"
    assert cfg.base_url == "http://127.0.0.1:9999"
    assert cfg.command[:3] == ["python3", "-m", "backend.voice.voice_worker_service"]
    assert cfg.command[-2:] == ["--flag", "quoted value"]


def test_contract_client_object_constructs():
    cfg = VoiceSidecarContractConfig(
        enabled=True,
        required=False,
        transport="http",
        base_url="http://127.0.0.1:9860",
        unix_socket_path="",
        control_timeout=2.0,
        health_timeout=1.0,
        command=["voice-sidecar"],
    )
    client = VoiceSidecarClient(cfg)
    assert client.config.base_url.endswith(":9860")


def test_wait_for_sidecar_health_timeout():
    class AlwaysFailClient:
        async def health(self):
            raise RuntimeError("not ready")

    async def _run():
        try:
            await wait_for_sidecar_health(
                AlwaysFailClient(),
                timeout_seconds=0.2,
                poll_interval_seconds=0.05,
            )
        except TimeoutError as exc:
            assert "health timeout" in str(exc)
            return
        raise AssertionError("expected timeout")

    asyncio.run(_run())


def test_wait_for_sidecar_health_success_after_retry():
    class FlakyClient:
        def __init__(self):
            self.calls = 0

        async def health(self):
            self.calls += 1
            if self.calls < 2:
                raise RuntimeError("warming up")
            return {"status": "ok"}

    async def _run():
        client = FlakyClient()
        result = await wait_for_sidecar_health(
            client,
            timeout_seconds=0.5,
            poll_interval_seconds=0.05,
        )
        assert result["status"] == "ok"
        assert client.calls >= 2

    asyncio.run(_run())
