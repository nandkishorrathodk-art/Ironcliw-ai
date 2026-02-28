"""
Ironcliw Service Registry v1.0.0
===============================

Dynamic service discovery and registration for the Trinity ecosystem.

Provides:
1. Service registration with metadata
2. Dynamic port discovery
3. Health endpoint tracking
4. Service dependency mapping
5. Heartbeat-based liveness detection

Architecture:
    The ServiceRegistry maintains a catalog of all Trinity services,
    their current status, and connection information. Services can:
    - Register themselves on startup
    - Be discovered via port scanning
    - Be monitored via heartbeat files

Service Discovery Methods:
    1. Explicit registration (service calls register())
    2. Port-based discovery (scan known ports)
    3. Heartbeat file discovery (~/.jarvis/heartbeats/)
    4. HTTP health endpoint probing

Author: Ironcliw AI System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class ServiceStatus(Enum):
    """Service status states."""
    UNKNOWN = "unknown"
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STOPPED = "stopped"


class ServiceType(Enum):
    """Trinity service types."""
    Ironcliw_BODY = "jarvis_body"
    Ironcliw_PRIME = "jarvis_prime"
    REACTOR_CORE = "reactor_core"
    LOADING_SERVER = "loading_server"
    FRONTEND = "frontend"
    WEBSOCKET = "websocket"
    EXTERNAL = "external"


# =============================================================================
# SERVICE INFO
# =============================================================================

@dataclass
class ServiceInfo:
    """
    Information about a registered service.
    """
    name: str
    service_type: ServiceType
    host: str = "localhost"
    port: int = 0
    health_path: str = "/health"
    status: ServiceStatus = ServiceStatus.UNKNOWN
    
    # Process info
    pid: Optional[int] = None
    started_at: Optional[datetime] = None
    
    # Health tracking
    last_health_check: Optional[datetime] = None
    last_healthy: Optional[datetime] = None
    consecutive_failures: int = 0
    
    # Metadata
    version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    @property
    def health_url(self) -> str:
        return f"{self.base_url}{self.health_path}"
    
    @property
    def is_healthy(self) -> bool:
        return self.status == ServiceStatus.HEALTHY
    
    @property
    def is_running(self) -> bool:
        return self.status in (ServiceStatus.STARTING, ServiceStatus.HEALTHY, ServiceStatus.DEGRADED)
    
    @property
    def uptime_seconds(self) -> float:
        if self.started_at:
            return (datetime.now() - self.started_at).total_seconds()
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "type": self.service_type.value,
            "host": self.host,
            "port": self.port,
            "status": self.status.value,
            "pid": self.pid,
            "base_url": self.base_url,
            "health_url": self.health_url,
            "uptime_seconds": self.uptime_seconds,
            "consecutive_failures": self.consecutive_failures,
            "last_health_check": (
                self.last_health_check.isoformat() if self.last_health_check else None
            ),
            "version": self.version,
        }


# =============================================================================
# DEFAULT SERVICE CONFIGURATION
# =============================================================================

DEFAULT_SERVICES: Dict[str, Dict[str, Any]] = {
    "jarvis-body": {
        "type": ServiceType.Ironcliw_BODY,
        "port": 8010,
        "health_path": "/health",
        "required": True,
    },
    "jarvis-prime": {
        "type": ServiceType.Ironcliw_PRIME,
        "port": 8001,
        "health_path": "/health",
        "required": False,
    },
    "reactor-core": {
        "type": ServiceType.REACTOR_CORE,
        "port": 8090,
        "health_path": "/health",
        "required": False,
    },
    "loading-server": {
        "type": ServiceType.LOADING_SERVER,
        "port": 3001,
        "health_path": "/health",
        "required": False,
    },
    "frontend": {
        "type": ServiceType.FRONTEND,
        "port": 3000,
        "health_path": "/",
        "required": False,
    },
}


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

class ServiceRegistry:
    """
    Dynamic service registry for Trinity services.
    
    Manages service discovery, registration, and health tracking.
    
    Usage:
        registry = ServiceRegistry()
        
        # Register a service
        registry.register("jarvis-body", ServiceType.Ironcliw_BODY, port=8010)
        
        # Get service info
        info = registry.get("jarvis-body")
        
        # Update status
        registry.update_status("jarvis-body", ServiceStatus.HEALTHY)
        
        # Discover running services
        await registry.discover_services()
    """
    
    _instance: Optional["ServiceRegistry"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "ServiceRegistry":
        """Singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._services: Dict[str, ServiceInfo] = {}
        self._callbacks: List[Callable[[str, ServiceStatus], None]] = []
        self._discovery_lock = asyncio.Lock()
        self._heartbeat_dir = Path.home() / ".jarvis" / "heartbeats"
        self._initialized = True
        
        # Ensure heartbeat directory exists
        self._heartbeat_dir.mkdir(parents=True, exist_ok=True)
        
        # Register default services
        self._register_defaults()
        
        logger.debug("[ServiceRegistry] Initialized")
    
    def _register_defaults(self) -> None:
        """Register default Trinity services."""
        for name, config in DEFAULT_SERVICES.items():
            self._services[name] = ServiceInfo(
                name=name,
                service_type=config["type"],
                port=config["port"],
                health_path=config["health_path"],
            )
    
    def register(
        self,
        name: str,
        service_type: ServiceType,
        host: str = "localhost",
        port: int = 0,
        health_path: str = "/health",
        pid: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ServiceInfo:
        """
        Register a service with the registry.
        
        Args:
            name: Service name
            service_type: Type of service
            host: Service host
            port: Service port
            health_path: Health endpoint path
            pid: Process ID
            metadata: Additional metadata
        
        Returns:
            ServiceInfo for the registered service
        """
        info = ServiceInfo(
            name=name,
            service_type=service_type,
            host=host,
            port=port,
            health_path=health_path,
            pid=pid,
            started_at=datetime.now() if pid else None,
            status=ServiceStatus.STARTING if pid else ServiceStatus.UNKNOWN,
            metadata=metadata or {},
        )
        
        self._services[name] = info
        logger.info(f"[ServiceRegistry] Registered {name} on port {port}")
        
        # Write heartbeat file
        self._write_heartbeat(info)
        
        return info
    
    def unregister(self, name: str) -> None:
        """
        Unregister a service from the registry.
        
        Args:
            name: Service name to unregister
        """
        if name in self._services:
            del self._services[name]
            self._remove_heartbeat(name)
            logger.info(f"[ServiceRegistry] Unregistered {name}")
    
    def get(self, name: str) -> Optional[ServiceInfo]:
        """Get service info by name."""
        return self._services.get(name)
    
    def get_all(self) -> Dict[str, ServiceInfo]:
        """Get all registered services."""
        return dict(self._services)
    
    def get_by_type(self, service_type: ServiceType) -> List[ServiceInfo]:
        """Get all services of a specific type."""
        return [s for s in self._services.values() if s.service_type == service_type]
    
    def get_healthy(self) -> List[ServiceInfo]:
        """Get all healthy services."""
        return [s for s in self._services.values() if s.is_healthy]
    
    def update_status(
        self,
        name: str,
        status: ServiceStatus,
        error: Optional[str] = None,
    ) -> None:
        """
        Update service status.
        
        Args:
            name: Service name
            status: New status
            error: Optional error message
        """
        if name not in self._services:
            return
        
        service = self._services[name]
        old_status = service.status
        service.status = status
        service.last_health_check = datetime.now()
        
        if status == ServiceStatus.HEALTHY:
            service.last_healthy = datetime.now()
            service.consecutive_failures = 0
        elif status in (ServiceStatus.UNHEALTHY, ServiceStatus.STOPPED):
            service.consecutive_failures += 1
        
        # Log status change
        if old_status != status:
            logger.info(
                f"[ServiceRegistry] {name}: {old_status.value} -> {status.value}"
            )
            
            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(name, status)
                except Exception as e:
                    logger.warning(f"[ServiceRegistry] Callback failed: {e}")
        
        # Update heartbeat
        self._write_heartbeat(service)
    
    def add_status_callback(
        self,
        callback: Callable[[str, ServiceStatus], None],
    ) -> None:
        """Add a callback for status changes."""
        self._callbacks.append(callback)
    
    # =========================================================================
    # DISCOVERY
    # =========================================================================
    
    async def discover_services(self) -> Dict[str, bool]:
        """
        Discover running services by probing known ports.
        
        Returns:
            Dictionary mapping service names to discovery success
        """
        async with self._discovery_lock:
            results = {}
            
            for name, service in self._services.items():
                is_running = await self._probe_service(service)
                results[name] = is_running
                
                if is_running:
                    service.status = ServiceStatus.HEALTHY
                else:
                    service.status = ServiceStatus.STOPPED
            
            return results
    
    async def _probe_service(self, service: ServiceInfo) -> bool:
        """
        Probe a service to check if it's running.
        
        Args:
            service: Service to probe
        
        Returns:
            True if service is responding
        """
        try:
            import aiohttp
            
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5.0)
            ) as session:
                async with session.get(service.health_url) as resp:
                    return resp.status in (200, 204)
        except Exception:
            # Try socket-level check
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.0)
                result = sock.connect_ex((service.host, service.port))
                sock.close()
                return result == 0
            except Exception:
                return False
    
    def discover_from_heartbeats(self) -> Dict[str, ServiceInfo]:
        """
        Discover services from heartbeat files.
        
        Returns:
            Dictionary of discovered services
        """
        discovered = {}
        
        if not self._heartbeat_dir.exists():
            return discovered
        
        for heartbeat_file in self._heartbeat_dir.glob("*.heartbeat.json"):
            try:
                data = json.loads(heartbeat_file.read_text())
                name = data.get("name")
                if not name:
                    continue
                
                # Check if heartbeat is recent (within last 5 minutes)
                last_beat = datetime.fromisoformat(data.get("timestamp", ""))
                if datetime.now() - last_beat > timedelta(minutes=5):
                    continue  # Stale heartbeat
                
                info = ServiceInfo(
                    name=name,
                    service_type=ServiceType(data.get("type", "external")),
                    host=data.get("host", "localhost"),
                    port=data.get("port", 0),
                    pid=data.get("pid"),
                    status=ServiceStatus(data.get("status", "unknown")),
                )
                
                discovered[name] = info
                
                # Update registry
                self._services[name] = info
                
            except Exception as e:
                logger.debug(f"[ServiceRegistry] Failed to read heartbeat {heartbeat_file}: {e}")
        
        return discovered
    
    # =========================================================================
    # HEARTBEAT
    # =========================================================================
    
    def _write_heartbeat(self, service: ServiceInfo) -> None:
        """Write heartbeat file for a service."""
        try:
            heartbeat_path = self._heartbeat_dir / f"{service.name}.heartbeat.json"
            data = {
                "name": service.name,
                "type": service.service_type.value,
                "host": service.host,
                "port": service.port,
                "pid": service.pid,
                "status": service.status.value,
                "timestamp": datetime.now().isoformat(),
            }
            heartbeat_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"[ServiceRegistry] Failed to write heartbeat: {e}")
    
    def _remove_heartbeat(self, name: str) -> None:
        """Remove heartbeat file for a service."""
        try:
            heartbeat_path = self._heartbeat_dir / f"{name}.heartbeat.json"
            if heartbeat_path.exists():
                heartbeat_path.unlink()
        except Exception:
            pass
    
    # =========================================================================
    # PRE-FLIGHT CLEANUP (v210.0)
    # =========================================================================
    
    async def pre_flight_cleanup(self) -> Dict[str, Any]:
        """
        v210.0: Clean up stale service entries and resources before startup.
        
        This method is called during kernel pre-flight to ensure a clean state:
        1. Remove stale heartbeat files from crashed sessions
        2. Reset service statuses to UNKNOWN
        3. Verify no orphaned port allocations
        4. Clean up any temporary service state
        
        Returns:
            Dict with cleanup statistics:
            - total_entries: Total services in registry
            - valid_entries: Services that appear valid
            - stale_heartbeats_removed: List of removed heartbeat files
            - ports_freed: List of ports that were freed
        """
        stats = {
            "total_entries": len(self._services),
            "valid_entries": 0,
            "stale_heartbeats_removed": [],
            "ports_freed": [],
        }
        
        try:
            # 1. Clean up stale heartbeat files
            if self._heartbeat_dir.exists():
                cutoff_time = time.time() - 300  # 5 minutes stale threshold
                
                for hb_file in self._heartbeat_dir.glob("*.heartbeat"):
                    try:
                        file_mtime = hb_file.stat().st_mtime
                        if file_mtime < cutoff_time:
                            # Stale heartbeat - service likely crashed
                            service_name = hb_file.stem
                            hb_file.unlink()
                            stats["stale_heartbeats_removed"].append(service_name)
                            logger.debug(f"[ServiceRegistry] Removed stale heartbeat: {service_name}")
                    except Exception as e:
                        logger.debug(f"[ServiceRegistry] Error cleaning heartbeat {hb_file}: {e}")
            
            # 2. Reset all service statuses to UNKNOWN
            for name, service in self._services.items():
                service.status = ServiceStatus.UNKNOWN
                service.consecutive_failures = 0
                service.last_health_check = None
            
            # 3. Check for port conflicts (ports that are in use by non-Ironcliw processes)
            ports_in_use = set()
            for name, service in self._services.items():
                if service.port > 0:
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(0.1)
                        result = sock.connect_ex(('127.0.0.1', service.port))
                        sock.close()
                        
                        if result == 0:
                            # Port is in use - check if it's a Ironcliw process
                            ports_in_use.add(service.port)
                    except Exception:
                        pass
            
            # 4. Count valid entries (services with proper configuration)
            for name, service in self._services.items():
                if service.port > 0 and service.health_path:
                    stats["valid_entries"] += 1
            
            logger.debug(
                f"[ServiceRegistry] Pre-flight cleanup complete: "
                f"{stats['total_entries']} services, "
                f"{len(stats['stale_heartbeats_removed'])} stale heartbeats removed"
            )
            
        except Exception as e:
            logger.warning(f"[ServiceRegistry] Pre-flight cleanup error: {e}")
        
        return stats
    
    # =========================================================================
    # SERIALIZATION
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert registry to dictionary."""
        return {
            name: service.to_dict()
            for name, service in self._services.items()
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of service statuses."""
        statuses = {}
        for status in ServiceStatus:
            statuses[status.value] = len([
                s for s in self._services.values()
                if s.status == status
            ])
        
        return {
            "total_services": len(self._services),
            "healthy": statuses.get(ServiceStatus.HEALTHY.value, 0),
            "unhealthy": statuses.get(ServiceStatus.UNHEALTHY.value, 0),
            "stopped": statuses.get(ServiceStatus.STOPPED.value, 0),
            "statuses": statuses,
        }


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_registry_instance: Optional[ServiceRegistry] = None


def get_service_registry() -> ServiceRegistry:
    """Get the singleton ServiceRegistry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ServiceRegistry()
    return _registry_instance


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger.debug("[ServiceRegistry] Module loaded")
