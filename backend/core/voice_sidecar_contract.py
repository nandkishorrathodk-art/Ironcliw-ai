"""Voice sidecar contract client used by unified_supervisor.

This module contains only control-plane integration logic (HTTP/Unix-socket
contract calls). Voice model loading and inference remain in existing modules.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class VoiceSidecarContractConfig:
    enabled: bool
    required: bool
    transport: str
    base_url: str
    unix_socket_path: str
    control_timeout: float
    health_timeout: float
    command: List[str]


class VoiceSidecarClient:
    """Async contract client for the Go voice sidecar."""

    def __init__(self, config: VoiceSidecarContractConfig):
        self.config = config

    async def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        allow_error_status: Optional[set[int]] = None,
    ) -> Dict[str, Any]:
        if self.config.transport == "unix":
            return await self._request_unix(
                method,
                path,
                payload=payload,
                timeout=timeout,
                allow_error_status=allow_error_status,
            )
        return await self._request_http(
            method,
            path,
            payload=payload,
            timeout=timeout,
            allow_error_status=allow_error_status,
        )

    async def _request_http(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        allow_error_status: Optional[set[int]] = None,
    ) -> Dict[str, Any]:
        import aiohttp

        base = self.config.base_url.rstrip("/")
        url = f"{base}{path}"
        total_timeout = aiohttp.ClientTimeout(total=timeout or self.config.control_timeout)
        async with aiohttp.ClientSession(timeout=total_timeout) as session:
            async with session.request(method, url, json=payload) as resp:
                text = await resp.text()
                if resp.status >= 400 and (allow_error_status is None or resp.status not in allow_error_status):
                    raise RuntimeError(f"{method} {url} -> {resp.status}: {text[:300]}")
                if not text.strip():
                    return {}
                return json.loads(text)

    async def _request_unix(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        allow_error_status: Optional[set[int]] = None,
    ) -> Dict[str, Any]:
        import aiohttp

        socket_path = self.config.unix_socket_path
        if not socket_path:
            raise RuntimeError("unix_socket_path not configured for unix transport")

        total_timeout = aiohttp.ClientTimeout(total=timeout or self.config.control_timeout)
        connector = aiohttp.UnixConnector(path=socket_path)
        url = f"http://localhost{path}"
        async with aiohttp.ClientSession(connector=connector, timeout=total_timeout) as session:
            async with session.request(method, url, json=payload) as resp:
                text = await resp.text()
                if resp.status >= 400 and (allow_error_status is None or resp.status not in allow_error_status):
                    raise RuntimeError(f"{method} {path} -> {resp.status}: {text[:300]}")
                if not text.strip():
                    return {}
                return json.loads(text)

    async def health(self) -> Dict[str, Any]:
        return await self._request("GET", "/v1/health", timeout=self.config.health_timeout)

    async def status(self) -> Dict[str, Any]:
        return await self._request("GET", "/v1/observer/state", timeout=self.config.health_timeout)

    async def heavy_load_gate(self) -> Dict[str, Any]:
        return await self._request(
            "GET",
            "/v1/gates/heavy-load",
            timeout=self.config.health_timeout,
            allow_error_status={429},
        )

def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}



def _env_command(name: str) -> List[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return []
    return shlex.split(raw)



def contract_config_from_env() -> VoiceSidecarContractConfig:
    transport = os.getenv("JARVIS_VOICE_SIDECAR_TRANSPORT", "http").strip().lower() or "http"
    if transport not in {"http", "unix"}:
        transport = "http"

    return VoiceSidecarContractConfig(
        enabled=_env_bool("JARVIS_VOICE_SIDECAR_ENABLED", False),
        required=_env_bool("JARVIS_VOICE_SIDECAR_REQUIRED", False),
        transport=transport,
        base_url=os.getenv("JARVIS_VOICE_SIDECAR_BASE_URL", "http://127.0.0.1:9860").strip(),
        unix_socket_path=os.getenv("JARVIS_VOICE_SIDECAR_SOCKET", "").strip(),
        control_timeout=float(os.getenv("JARVIS_VOICE_SIDECAR_CONTROL_TIMEOUT", "5.0")),
        health_timeout=float(os.getenv("JARVIS_VOICE_SIDECAR_HEALTH_TIMEOUT", "2.5")),
        command=_env_command("JARVIS_VOICE_SIDECAR_COMMAND"),
    )


async def wait_for_sidecar_health(
    client: VoiceSidecarClient,
    timeout_seconds: float,
    poll_interval_seconds: float = 0.25,
) -> Dict[str, Any]:
    """Wait until sidecar responds to health endpoint or timeout."""
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    last_error = "unavailable"

    while True:
        try:
            return await client.health()
        except Exception as exc:
            last_error = str(exc)

        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(f"voice sidecar health timeout: {last_error}")

        await asyncio.sleep(max(0.05, poll_interval_seconds))
