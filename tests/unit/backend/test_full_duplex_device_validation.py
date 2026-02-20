"""Validation tests for FullDuplexDevice device preflight checks."""

import types

import pytest

import backend.audio.full_duplex_device as fd_mod
from backend.audio.full_duplex_device import DeviceConfig, FullDuplexDevice


def _make_fake_sd(default_device=(1, 2), check_error=None):
    """Create a fake sounddevice module surface for validation tests."""
    fake = types.SimpleNamespace()
    fake.default = types.SimpleNamespace(device=default_device)
    fake.query_devices = lambda: [{"name": "mic"}, {"name": "speaker"}]

    def _check_input_settings(**_kwargs):
        if check_error is not None:
            raise check_error

    def _check_output_settings(**_kwargs):
        if check_error is not None:
            raise check_error

    fake.check_input_settings = _check_input_settings
    fake.check_output_settings = _check_output_settings
    return fake


def test_validate_device_selection_uses_defaults(monkeypatch):
    """Validation should resolve and persist default duplex devices."""
    monkeypatch.delenv("JARVIS_AUDIO_INPUT_DEVICE", raising=False)
    monkeypatch.delenv("JARVIS_AUDIO_OUTPUT_DEVICE", raising=False)
    monkeypatch.setattr(fd_mod, "sd", _make_fake_sd(default_device=(3, 4)))

    device = FullDuplexDevice(DeviceConfig())
    device._validate_device_selection()

    assert device.config.input_device == 3
    assert device.config.output_device == 4


def test_validate_device_selection_rejects_invalid_defaults(monkeypatch):
    """Missing default devices should fail early with explicit error."""
    monkeypatch.setattr(fd_mod, "sd", _make_fake_sd(default_device=(-1, -1)))

    device = FullDuplexDevice(DeviceConfig())
    with pytest.raises(RuntimeError, match="No valid default input device available"):
        device._validate_device_selection()


def test_validate_device_selection_surfaces_settings_error(monkeypatch):
    """Driver/settings validation failures should be wrapped with context."""
    monkeypatch.setattr(
        fd_mod,
        "sd",
        _make_fake_sd(default_device=(1, 2), check_error=ValueError("invalid settings")),
    )

    device = FullDuplexDevice(DeviceConfig())
    with pytest.raises(RuntimeError, match="Audio device validation failed"):
        device._validate_device_selection()

