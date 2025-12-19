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


async def broadcast_update_available(
    commits_behind: int,
    summary: str,
    priority: str = "medium",
    highlights: Optional[list[str]] = None,
    security_update: bool = False,
    breaking_changes: bool = False,
    remote_sha: Optional[str] = None,
    local_sha: Optional[str] = None,
    backend_url: str = "http://localhost:8010",
) -> bool:
    """
    Broadcast update available notification to frontend.
    
    This enables the frontend to display an "Update Available" badge
    or notification modal with rich information about what's changed.
    
    Args:
        commits_behind: Number of commits behind remote
        summary: Human-readable summary of changes
        priority: 'low' | 'medium' | 'high' | 'critical'
        highlights: List of key changes (bullet points)
        security_update: True if this includes security fixes
        breaking_changes: True if this includes breaking changes
        remote_sha: Short SHA of remote HEAD
        local_sha: Short SHA of local HEAD
        backend_url: Backend URL for broadcasting
        
    Returns:
        True if broadcast successful
    """
    from datetime import datetime
    
    payload = {
        "type": "update_available",
        "data": {
            "available": True,
            "commits_behind": commits_behind,
            "summary": summary,
            "priority": priority,
            "highlights": highlights or [],
            "security_update": security_update,
            "breaking_changes": breaking_changes,
            "remote_sha": remote_sha,
            "local_sha": local_sha,
            "timestamp": datetime.now().isoformat(),
        }
    }
    
    # Try multiple potential broadcast endpoints
    endpoints = [
        f"{backend_url}/api/broadcast",
        f"{backend_url}/api/system/broadcast",
        f"{backend_url}/ws/broadcast",
    ]
    
    for endpoint in endpoints:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    json={"event": "update_available", "data": payload["data"]},
                    timeout=aiohttp.ClientTimeout(total=3.0),
                ) as response:
                    if response.status in (200, 201, 202, 204):
                        logger.info(f"📡 Broadcast update_available: {commits_behind} commits, {summary}")
                        return True
        except aiohttp.ClientConnectorError:
            continue
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            logger.debug(f"Broadcast to {endpoint} failed: {e}")
            continue
    
    logger.debug("Could not broadcast update_available (backend may be starting)")
    return False


async def broadcast_update_dismissed(
    backend_url: str = "http://localhost:8010",
) -> bool:
    """
    Broadcast that user dismissed the update notification.
    
    Allows frontend to hide the notification badge/modal.
    
    Args:
        backend_url: Backend URL for broadcasting
        
    Returns:
        True if broadcast successful
    """
    payload = {
        "type": "update_dismissed",
        "data": {
            "available": False,
        }
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{backend_url}/api/broadcast",
                json={"event": "update_dismissed", "data": payload["data"]},
                timeout=aiohttp.ClientTimeout(total=2.0),
            ) as response:
                if response.status in (200, 201, 202, 204):
                    logger.info("📡 Broadcast update_dismissed")
                    return True
    except Exception as e:
        logger.debug(f"Broadcast update_dismissed failed: {e}")
    
    return False


async def broadcast_update_progress(
    phase: str,
    message: str,
    progress_percent: int = 0,
    backend_url: str = "http://localhost:8010",
) -> bool:
    """
    Broadcast update progress to frontend.
    
    Enables frontend to show a progress indicator during updates.
    
    Args:
        phase: 'downloading' | 'installing' | 'building' | 'verifying'
        message: Human-readable progress message
        progress_percent: 0-100 progress value
        backend_url: Backend URL for broadcasting
        
    Returns:
        True if broadcast successful
    """
    payload = {
        "type": "update_progress",
        "data": {
            "phase": phase,
            "message": message,
            "progress": progress_percent,
        }
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{backend_url}/api/broadcast",
                json={"event": "update_progress", "data": payload["data"]},
                timeout=aiohttp.ClientTimeout(total=2.0),
            ) as response:
                if response.status in (200, 201, 202, 204):
                    return True
    except Exception as e:
        logger.debug(f"Broadcast update_progress failed: {e}")
    
    return False
