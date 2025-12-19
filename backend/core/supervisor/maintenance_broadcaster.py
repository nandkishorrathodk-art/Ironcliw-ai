#!/usr/bin/env python3
"""
Maintenance Mode Broadcaster
=============================

Sends maintenance mode events to connected WebSocket clients before
supervisor actions (update, restart, rollback).

This allows the frontend to show a polished "Maintenance Mode" overlay
instead of a "Connection Error" banner.

Usage:
    from core.supervisor.maintenance_broadcaster import broadcast_maintenance_mode
    
    # Before shutting down JARVIS for update
    await broadcast_maintenance_mode('updating', 'Downloading updates...')

Author: JARVIS System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


async def broadcast_maintenance_mode(
    reason: str,
    message: str,
    estimated_time: int = 30,
    backend_url: str = "http://localhost:8010",
) -> bool:
    """
    Broadcast maintenance mode event to all connected WebSocket clients.
    
    Sends a message to the backend's broadcast endpoint, which relays
    it to all connected WebSocket clients.
    
    Args:
        reason: 'updating' | 'restarting' | 'rollback'
        message: Human-readable status message
        estimated_time: Estimated seconds until back online
        backend_url: Backend URL to send broadcast to
        
    Returns:
        True if broadcast successful
    """
    event_type = f"system_{reason}"
    
    payload = {
        "type": event_type,
        "message": message,
        "estimated_time": estimated_time,
        "reason": reason,
    }
    
    # Try multiple potential broadcast endpoints
    endpoints = [
        f"{backend_url}/api/broadcast",
        f"{backend_url}/broadcast",
        f"{backend_url}/ws/broadcast",
    ]
    
    for endpoint in endpoints:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    json={"event": event_type, "data": payload},
                    timeout=aiohttp.ClientTimeout(total=2.0),
                ) as response:
                    if response.status in (200, 201, 202, 204):
                        logger.info(f"📡 Broadcast {event_type}: {message}")
                        return True
        except aiohttp.ClientConnectorError:
            # Backend not responding - expected during shutdown
            continue
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            logger.debug(f"Broadcast to {endpoint} failed: {e}")
            continue
    
    # Fallback: Try WebSocket direct send
    try:
        async with aiohttp.ClientSession() as session:
            ws_url = backend_url.replace("http://", "ws://").replace("https://", "wss://")
            async with session.ws_connect(f"{ws_url}/ws", timeout=2.0) as ws:
                await ws.send_json(payload)
                logger.info(f"📡 Broadcast via WS {event_type}: {message}")
                return True
    except Exception as e:
        logger.debug(f"WebSocket broadcast failed: {e}")
    
    logger.debug(f"Could not broadcast {event_type} (backend may be down)")
    return False


async def broadcast_system_online(
    message: str = "JARVIS is back online",
    backend_url: str = "http://localhost:8010",
) -> bool:
    """
    Broadcast that system is back online after maintenance.
    
    Args:
        message: Status message
        backend_url: Backend URL to send broadcast to
        
    Returns:
        True if broadcast successful
    """
    return await broadcast_maintenance_mode(
        reason="online",
        message=message,
        estimated_time=0,
        backend_url=backend_url,
    )


async def announce_and_broadcast(
    reason: str,
    tts_message: str,
    ws_message: Optional[str] = None,
) -> None:
    """
    Announce via TTS and broadcast to WebSocket clients.
    
    Args:
        reason: 'updating' | 'restarting' | 'rollback'
        tts_message: Message to speak via TTS
        ws_message: Message for WebSocket (defaults to tts_message)
    """
    from .narrator import get_narrator
    
    narrator = get_narrator()
    
    # Parallel: TTS + WebSocket broadcast
    await asyncio.gather(
        narrator.speak(tts_message),
        broadcast_maintenance_mode(reason, ws_message or tts_message),
        return_exceptions=True,
    )
