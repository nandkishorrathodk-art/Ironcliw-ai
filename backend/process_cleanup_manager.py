#!/usr/bin/env python3
"""
Intelligent Process Cleanup Manager for JARVIS
Dynamically identifies and cleans up stuck/hanging processes without hardcoding
Uses Swift performance monitoring for minimal overhead
Enhanced with code change detection to ensure only latest instance runs
Integrated with GCP VM session tracking for cloud resource cleanup
"""

import asyncio
import hashlib
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Set, Tuple

# Check for Swift availability
try:
    pass

    SWIFT_AVAILABLE = True
except (ImportError, OSError):
    SWIFT_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# DYNAMIC RESTART CONFIGURATION - No Hardcoding
# =============================================================================

@dataclass
class DynamicRestartConfig:
    """
    Dynamic configuration for restart operations.
    Loads ports and settings from config file - NO HARDCODING.
    """
    # Primary port (loaded from config)
    primary_api_port: int = 8011

    # Fallback ports (loaded from config)
    fallback_ports: List[int] = field(default_factory=lambda: [8010, 8000, 8001, 8080, 8888])

    # Frontend and auxiliary ports
    frontend_port: int = 3000
    loading_server_port: int = 3001
    database_port: int = 5432

    # UE state indicators for macOS/Linux
    ue_state_indicators: List[str] = field(default_factory=lambda: [
        'disk-sleep', 'uninterruptible', 'D', 'U',
        'D+', 'U+', 'Ds', 'Us',  # With additional flags
    ])

    # Cleanup timeouts (seconds)
    graceful_timeout: float = 2.0
    force_kill_timeout: float = 1.0
    port_verification_timeout: float = 1.0
    parallel_cleanup_timeout: float = 10.0

    # Self-healing settings
    enable_self_healing: bool = True
    blacklist_ue_ports: bool = True
    max_kill_attempts: int = 3

    # Config file path
    config_file_path: Optional[Path] = None

    def __post_init__(self):
        """Load configuration from file if available."""
        if self.config_file_path is None:
            self.config_file_path = Path(__file__).parent / 'config' / 'startup_progress_config.json'
        self._load_from_config()

    def _load_from_config(self) -> None:
        """Load dynamic configuration from JSON config file."""
        try:
            if self.config_file_path and self.config_file_path.exists():
                with open(self.config_file_path, 'r') as f:
                    config = json.load(f)

                # Load backend config
                backend_config = config.get('backend_config', {})
                self.primary_api_port = backend_config.get('port', self.primary_api_port)
                self.fallback_ports = backend_config.get('fallback_ports', self.fallback_ports)

                # Load loading server config
                loading_config = config.get('loading_server', {})
                self.loading_server_port = loading_config.get('port', self.loading_server_port)

                logger.debug(f"Loaded restart config: primary={self.primary_api_port}, fallbacks={self.fallback_ports}")
        except Exception as e:
            logger.warning(f"Failed to load restart config from {self.config_file_path}: {e}")

    def get_all_ports(self) -> List[int]:
        """Get all ports that need to be managed during restart."""
        ports = {self.primary_api_port, self.frontend_port, self.loading_server_port, self.database_port}
        ports.update(self.fallback_ports)
        return sorted(list(ports))

    def get_api_ports(self) -> List[int]:
        """Get only API-related ports (primary + fallbacks)."""
        ports = {self.primary_api_port}
        ports.update(self.fallback_ports)
        return sorted(list(ports))


class UEStateDetector:
    """
    Detects processes in Uninterruptible Sleep (UE) state on macOS.
    These processes CANNOT be killed and must be avoided/blacklisted.
    """

    # macOS process state indicators for unkillable processes
    # Note: psutil returns verbose states like 'disk-sleep', 'running', 'sleeping'
    # ps command returns single-letter codes: D=uninterruptible sleep, U=uninterruptible wait
    PSUTIL_UE_STATES = ['disk-sleep', 'uninterruptible']
    PS_UE_CODES = {'D', 'U', 'D+', 'U+', 'Ds', 'Us'}  # Single-letter ps codes

    def __init__(self, config: Optional[DynamicRestartConfig] = None):
        self.config = config or DynamicRestartConfig()
        self._blacklisted_pids: Set[int] = set()
        self._blacklisted_ports: Set[int] = set()

    def is_ue_state(self, status: str, is_ps_status: bool = False) -> bool:
        """
        Check if a process status indicates UE (Uninterruptible Sleep) state.

        Args:
            status: The process status string
            is_ps_status: If True, status is from ps command (single letter codes)
                         If False, status is from psutil (verbose states)
        """
        if is_ps_status:
            # ps command returns single-letter codes like 'D', 'U', 'D+', etc.
            # Check exact match against known UE codes
            status_stripped = status.strip()
            return status_stripped in self.PS_UE_CODES
        else:
            # psutil returns verbose states like 'disk-sleep', 'running'
            status_lower = status.lower()
            return any(ue_state in status_lower for ue_state in self.PSUTIL_UE_STATES)

    def check_process_state(self, pid: int) -> Tuple[bool, str]:
        """
        Check if a process is in UE state using multiple methods.
        Returns: (is_ue_state, detailed_status)
        """
        try:
            # Method 1: Use psutil (returns verbose states like 'disk-sleep', 'running')
            proc = psutil.Process(pid)
            status = proc.status()

            if self.is_ue_state(status, is_ps_status=False):
                return True, f"psutil:{status}"

            # Method 2: Use ps command for more detailed status (macOS-specific)
            # ps returns single-letter codes like D, U, S, R
            try:
                result = subprocess.run(
                    ['ps', '-o', 'stat=', '-p', str(pid)],
                    capture_output=True, text=True, timeout=2.0
                )
                if result.returncode == 0:
                    ps_status = result.stdout.strip()
                    if self.is_ue_state(ps_status, is_ps_status=True):
                        return True, f"ps:{ps_status}"
                    return False, f"ps:{ps_status}"
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

            return False, f"psutil:{status}"

        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            return False, f"error:{str(e)}"

    def check_port_for_ue_process(self, port: int) -> Tuple[bool, Optional[int], str]:
        """
        Check if a port is being held by a UE-state process.
        Returns: (has_ue_process, pid_if_ue, detailed_status)
        """
        try:
            # Find process using the port
            for conn in psutil.net_connections(kind='inet'):
                if conn.laddr.port == port and conn.status == 'LISTEN':
                    pid = conn.pid
                    if pid:
                        is_ue, status = self.check_process_state(pid)
                        if is_ue:
                            self._blacklisted_pids.add(pid)
                            self._blacklisted_ports.add(port)
                            return True, pid, status
                        return False, pid, status
            return False, None, "no_listener"
        except psutil.AccessDenied:
            # Access denied on macOS is normal for some ports - treat as no UE process
            return False, None, "access_denied"
        except OSError as e:
            return False, None, f"os_error:{str(e)}"

    def is_pid_blacklisted(self, pid: int) -> bool:
        """Check if a PID is blacklisted due to UE state."""
        return pid in self._blacklisted_pids

    def is_port_blacklisted(self, port: int) -> bool:
        """Check if a port is blacklisted due to UE process."""
        return port in self._blacklisted_ports

    def get_blacklisted_ports(self) -> Set[int]:
        """Get all blacklisted ports."""
        return self._blacklisted_ports.copy()

    def get_blacklisted_pids(self) -> Set[int]:
        """Get all blacklisted PIDs."""
        return self._blacklisted_pids.copy()

    def scan_all_ports(self, ports: List[int]) -> Dict[int, Dict[str, Any]]:
        """Scan all given ports for UE processes."""
        results = {}
        for port in ports:
            has_ue, pid, status = self.check_port_for_ue_process(port)
            results[port] = {
                'has_ue_process': has_ue,
                'pid': pid,
                'status': status,
                'blacklisted': has_ue
            }
        return results


class AsyncRestartManager:
    """
    Async-capable restart manager with parallel cleanup and self-healing.
    """

    def __init__(self, config: Optional[DynamicRestartConfig] = None):
        self.config = config or DynamicRestartConfig()
        self.ue_detector = UEStateDetector(config=self.config)
        self._cleanup_results: List[Dict[str, Any]] = []

    async def _kill_process_async(
        self,
        proc: psutil.Process,
        ptype: str,
        timeout: float
    ) -> Dict[str, Any]:
        """
        Kill a single process with UE state detection.
        Returns detailed result dict.
        """
        pid = proc.pid
        result = {
            'pid': pid,
            'type': ptype,
            'status': 'pending',
            'ue_state': False,
            'attempts': 0,
            'error': None
        }

        try:
            cmdline = " ".join(proc.cmdline()) if proc.cmdline() else ""
            result['cmdline'] = cmdline[:150]
            result['name'] = proc.name()
            result['age_minutes'] = (time.time() - proc.create_time()) / 60
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            result['status'] = 'already_dead'
            return result

        # Check for UE state BEFORE attempting to kill
        is_ue, ue_status = self.ue_detector.check_process_state(pid)
        if is_ue:
            result['ue_state'] = True
            result['status'] = 'ue_state_detected'
            result['ue_status'] = ue_status
            logger.warning(f"⚠️ PID {pid} is in UE state ({ue_status}) - CANNOT KILL, blacklisting")
            return result

        # Attempt graceful termination
        for attempt in range(self.config.max_kill_attempts):
            result['attempts'] = attempt + 1

            try:
                if attempt == 0:
                    # Try SIGTERM first
                    proc.terminate()
                    try:
                        await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(
                                None, proc.wait, timeout
                            ),
                            timeout=timeout + 0.5
                        )
                        result['status'] = 'terminated'
                        return result
                    except (asyncio.TimeoutError, psutil.TimeoutExpired):
                        pass

                elif attempt == 1:
                    # Try SIGKILL
                    proc.kill()
                    await asyncio.sleep(0.2)
                    if not proc.is_running():
                        result['status'] = 'force_killed'
                        return result

                else:
                    # Last resort: OS-level SIGKILL
                    os.kill(pid, signal.SIGKILL)
                    await asyncio.sleep(0.3)
                    if not psutil.pid_exists(pid):
                        result['status'] = 'sigkill'
                        return result

                    # Check if it became a UE process
                    is_ue_now, ue_status = self.ue_detector.check_process_state(pid)
                    if is_ue_now:
                        result['ue_state'] = True
                        result['status'] = 'became_ue_state'
                        result['ue_status'] = ue_status
                        logger.error(f"⚠️ PID {pid} became UE state after kill attempt - blacklisting")
                        return result

            except psutil.NoSuchProcess:
                result['status'] = 'terminated'
                return result
            except Exception as e:
                result['error'] = str(e)

        result['status'] = 'failed'
        return result

    async def cleanup_processes_parallel(
        self,
        processes: List[Tuple[psutil.Process, str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Clean up multiple processes in parallel using asyncio.

        Args:
            processes: List of (process, cmdline, type) tuples

        Returns:
            List of cleanup result dicts
        """
        if not processes:
            return []

        tasks = []
        for proc, cmdline, ptype in processes:
            timeout = self.config.graceful_timeout
            task = asyncio.create_task(
                self._kill_process_async(proc, ptype, timeout)
            )
            tasks.append(task)

        # Wait for all tasks with overall timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.parallel_cleanup_timeout
            )
        except asyncio.TimeoutError:
            logger.warning("Parallel cleanup timed out, some processes may still be running")
            results = []
            for task in tasks:
                if task.done():
                    try:
                        results.append(task.result())
                    except Exception as e:
                        results.append({'status': 'error', 'error': str(e)})
                else:
                    task.cancel()
                    results.append({'status': 'timeout'})

        # Process results
        cleaned = []
        for result in results:
            if isinstance(result, Exception):
                cleaned.append({'status': 'error', 'error': str(result)})
            elif isinstance(result, dict):
                cleaned.append(result)

        return cleaned

    async def verify_ports_free_async(self, ports: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Verify multiple ports are free in parallel.
        Returns dict of port -> status info.
        """
        async def check_port(port: int) -> Tuple[int, Dict[str, Any]]:
            # First check for UE process
            has_ue, pid, ue_status = self.ue_detector.check_port_for_ue_process(port)

            if has_ue:
                return port, {
                    'free': False,
                    'ue_process': True,
                    'pid': pid,
                    'status': ue_status,
                    'action': 'blacklisted'
                }

            try:
                # Try to connect to check if port is in use
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.config.port_verification_timeout)
                result = sock.connect_ex(('127.0.0.1', port))
                sock.close()

                if result == 0:
                    # Port still in use - try force kill
                    try:
                        kill_result = subprocess.run(
                            f"lsof -ti:{port} | xargs kill -9 2>/dev/null",
                            shell=True, capture_output=True, timeout=2.0
                        )
                        await asyncio.sleep(0.3)

                        # Recheck
                        sock2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock2.settimeout(0.5)
                        recheck = sock2.connect_ex(('127.0.0.1', port))
                        sock2.close()

                        if recheck != 0:
                            return port, {'free': True, 'action': 'force_freed'}
                        else:
                            # Check if it's now a UE process
                            has_ue_now, pid_now, _ = self.ue_detector.check_port_for_ue_process(port)
                            if has_ue_now:
                                return port, {
                                    'free': False,
                                    'ue_process': True,
                                    'pid': pid_now,
                                    'action': 'blacklisted_after_kill_attempt'
                                }
                            return port, {'free': False, 'action': 'still_in_use'}
                    except subprocess.TimeoutExpired:
                        return port, {'free': False, 'action': 'kill_timeout'}
                else:
                    return port, {'free': True, 'action': 'already_free'}

            except Exception as e:
                return port, {'free': False, 'error': str(e)}

        tasks = [check_port(port) for port in ports]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        port_status = {}
        for result in results:
            if isinstance(result, tuple):
                port, status = result
                port_status[port] = status
            elif isinstance(result, Exception):
                logger.error(f"Port check failed: {result}")

        return port_status

    def get_healthy_port(self, exclude_blacklisted: bool = True) -> Optional[int]:
        """
        Find a healthy port for the API to use.
        Excludes UE-blocked ports if requested.
        """
        blacklisted = self.ue_detector.get_blacklisted_ports() if exclude_blacklisted else set()

        # Try primary port first
        if self.config.primary_api_port not in blacklisted:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.5)
                result = sock.connect_ex(('127.0.0.1', self.config.primary_api_port))
                sock.close()
                if result != 0:  # Port is free
                    return self.config.primary_api_port
            except:
                pass

        # Try fallback ports
        for port in self.config.fallback_ports:
            if port in blacklisted:
                continue
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.5)
                result = sock.connect_ex(('127.0.0.1', port))
                sock.close()
                if result != 0:  # Port is free
                    return port
            except:
                continue

        return None


# Global instances for module-level access
_restart_config: Optional[DynamicRestartConfig] = None
_ue_detector: Optional[UEStateDetector] = None
_async_restart_manager: Optional[AsyncRestartManager] = None


def get_restart_config() -> DynamicRestartConfig:
    """Get or create the global restart config instance."""
    global _restart_config
    if _restart_config is None:
        _restart_config = DynamicRestartConfig()
    return _restart_config


def get_ue_detector() -> UEStateDetector:
    """Get or create the global UE state detector instance."""
    global _ue_detector
    if _ue_detector is None:
        _ue_detector = UEStateDetector(config=get_restart_config())
    return _ue_detector


def get_async_restart_manager() -> AsyncRestartManager:
    """Get or create the global async restart manager instance."""
    global _async_restart_manager
    if _async_restart_manager is None:
        _async_restart_manager = AsyncRestartManager(config=get_restart_config())
    return _async_restart_manager


class GCPVMSessionManager:
    """
    Manages GCP VM session tracking and cleanup.
    Interfaces with VMSessionTracker for cloud resource management.
    """

    def __init__(self):
        """Initialize GCP VM session manager"""
        self.temp_dir = Path(tempfile.gettempdir())
        self.vm_registry_file = self.temp_dir / "jarvis_vm_registry.json"
        self.session_file_pattern = "jarvis_session_*.json"

        # Load GCP configuration from environment
        self.gcp_project = os.getenv("GCP_PROJECT_ID", "jarvis-473803")
        self.default_zone = os.getenv("GCP_DEFAULT_ZONE", "us-central1-a")

    def _load_vm_registry(self) -> Dict[str, Any]:
        """Load the global VM registry"""
        if not self.vm_registry_file.exists():
            return {}

        try:
            with open(self.vm_registry_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load VM registry: {e}")
            return {}

    def _save_vm_registry(self, registry: Dict[str, Any]):
        """Save the global VM registry"""
        try:
            with open(self.vm_registry_file, "w") as f:
                json.dump(registry, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save VM registry: {e}")

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get all VM sessions from registry"""
        registry = self._load_vm_registry()
        return list(registry.values())

    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get sessions with active PIDs"""
        active = []
        for session in self.get_all_sessions():
            pid = session.get("pid")
            if pid and psutil.pid_exists(pid):
                try:
                    proc = psutil.Process(pid)
                    # Verify process is actually running start_system.py
                    cmdline = " ".join(proc.cmdline())
                    if "start_system.py" in cmdline or "main.py" in cmdline:
                        active.append(session)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        return active

    def get_orphaned_sessions(self) -> List[Dict[str, Any]]:
        """Get sessions where the PID is dead"""
        orphaned = []
        for session in self.get_all_sessions():
            pid = session.get("pid")
            hostname = session.get("hostname", "")
            current_hostname = socket.gethostname()

            # Check if session is from this machine
            if hostname != current_hostname:
                continue

            # Check if PID is dead or not running JARVIS
            is_dead = False
            if not pid or not psutil.pid_exists(pid):
                is_dead = True
            else:
                try:
                    proc = psutil.Process(pid)
                    cmdline = " ".join(proc.cmdline())
                    if "start_system.py" not in cmdline and "main.py" not in cmdline:
                        is_dead = True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    is_dead = True

            if is_dead:
                orphaned.append(session)

        return orphaned

    def get_stale_sessions(self, max_age_hours: float = 24.0) -> List[Dict[str, Any]]:
        """Get sessions older than max_age_hours"""
        stale = []
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        for session in self.get_all_sessions():
            created_at = session.get("created_at", 0)
            age_seconds = current_time - created_at

            if age_seconds > max_age_seconds:
                stale.append(session)

        return stale

    async def delete_vm_async(self, vm_id: str, zone: str) -> bool:
        """Delete a GCP VM asynchronously"""
        try:
            logger.info(f"Deleting GCP VM: {vm_id} in zone {zone}")

            proc = await asyncio.create_subprocess_exec(
                "gcloud",
                "compute",
                "instances",
                "delete",
                vm_id,
                "--project",
                self.gcp_project,
                "--zone",
                zone,
                "--quiet",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)

            if proc.returncode == 0:
                logger.info(f"✅ Successfully deleted VM: {vm_id}")
                return True
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                if "was not found" in error_msg:
                    logger.info(f"VM {vm_id} already deleted")
                    return True
                logger.error(f"Failed to delete VM {vm_id}: {error_msg}")
                return False

        except asyncio.TimeoutError:
            logger.error(f"Timeout deleting VM {vm_id}")
            return False
        except Exception as e:
            logger.error(f"Error deleting VM {vm_id}: {e}")
            return False

    def delete_vm_sync(self, vm_id: str, zone: str) -> bool:
        """Delete a GCP VM synchronously (for use in non-async contexts)"""
        try:
            logger.info(f"Deleting GCP VM: {vm_id} in zone {zone}")

            result = subprocess.run(
                [
                    "gcloud",
                    "compute",
                    "instances",
                    "delete",
                    vm_id,
                    "--project",
                    self.gcp_project,
                    "--zone",
                    zone,
                    "--quiet",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                logger.info(f"✅ Successfully deleted VM: {vm_id}")
                return True
            else:
                if "was not found" in result.stderr:
                    logger.info(f"VM {vm_id} already deleted")
                    return True
                logger.error(f"Failed to delete VM {vm_id}: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout deleting VM {vm_id}")
            return False
        except Exception as e:
            logger.error(f"Error deleting VM {vm_id}: {e}")
            return False

    async def cleanup_orphaned_vms(self) -> Dict[str, Any]:
        """
        Find and delete VMs from orphaned sessions.
        Returns cleanup report.
        """
        orphaned = self.get_orphaned_sessions()

        report = {
            "timestamp": datetime.now(),
            "orphaned_sessions": len(orphaned),
            "vms_deleted": [],
            "errors": [],
        }

        if not orphaned:
            logger.info("No orphaned VM sessions found")
            return report

        logger.warning(f"Found {len(orphaned)} orphaned VM sessions")

        # Delete VMs in parallel
        tasks = []
        for session in orphaned:
            vm_id = session.get("vm_id")
            zone = session.get("zone", self.default_zone)

            if vm_id:
                tasks.append(self._cleanup_session_vm(session, vm_id, zone))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, Exception):
                report["errors"].append(str(result))
            elif result:
                report["vms_deleted"].append(result)

        # Clean up registry
        self._remove_orphaned_from_registry(orphaned)

        logger.info(
            f"Orphaned VM cleanup: {len(report['vms_deleted'])} VMs deleted, {len(report['errors'])} errors"
        )

        return report

    async def _cleanup_session_vm(
        self, session: Dict[str, Any], vm_id: str, zone: str
    ) -> Optional[str]:
        """Helper to cleanup a single session's VM"""
        try:
            success = await self.delete_vm_async(vm_id, zone)
            if success:
                return vm_id
            return None
        except Exception as e:
            logger.error(f"Error cleaning up VM {vm_id}: {e}")
            return None

    def _remove_orphaned_from_registry(self, orphaned_sessions: List[Dict[str, Any]]):
        """Remove orphaned sessions from registry"""
        registry = self._load_vm_registry()

        orphaned_ids = {s.get("session_id") for s in orphaned_sessions if s.get("session_id")}

        # Filter out orphaned sessions
        active_registry = {sid: data for sid, data in registry.items() if sid not in orphaned_ids}

        if len(active_registry) != len(registry):
            logger.info(
                f"Removing {len(registry) - len(active_registry)} orphaned sessions from registry"
            )
            if active_registry:
                self._save_vm_registry(active_registry)
            else:
                # Delete registry file if empty
                try:
                    self.vm_registry_file.unlink()
                    logger.info("Deleted empty VM registry")
                except Exception:
                    pass

    async def cleanup_all_vms_for_user(self) -> Dict[str, Any]:
        """
        Emergency cleanup - delete ALL VMs registered to this machine.
        Returns cleanup report.
        """
        all_sessions = self.get_all_sessions()
        current_hostname = socket.gethostname()

        # Filter to only this machine's sessions
        local_sessions = [s for s in all_sessions if s.get("hostname") == current_hostname]

        report = {
            "timestamp": datetime.now(),
            "total_sessions": len(local_sessions),
            "vms_deleted": [],
            "errors": [],
        }

        if not local_sessions:
            logger.info("No VMs found for current machine")
            return report

        logger.warning(f"Emergency cleanup: Deleting {len(local_sessions)} VMs from this machine")

        # Delete all VMs in parallel
        tasks = []
        for session in local_sessions:
            vm_id = session.get("vm_id")
            zone = session.get("zone", self.default_zone)

            if vm_id:
                tasks.append(self._cleanup_session_vm(session, vm_id, zone))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, Exception):
                report["errors"].append(str(result))
            elif result:
                report["vms_deleted"].append(result)

        # Clear entire registry
        try:
            if self.vm_registry_file.exists():
                self.vm_registry_file.unlink()
                logger.info("Cleared VM registry")
        except Exception as e:
            logger.error(f"Failed to clear VM registry: {e}")

        logger.info(
            f"Emergency VM cleanup: {len(report['vms_deleted'])} VMs deleted, {len(report['errors'])} errors"
        )

        return report

    def get_vm_count(self) -> int:
        """Get total number of VMs in registry"""
        return len(self.get_all_sessions())

    def get_active_vm_count(self) -> int:
        """Get number of VMs with active PIDs"""
        return len(self.get_active_sessions())


class ProcessCleanupManager:
    """Manages cleanup of stuck or zombie processes with code change detection"""

    def __init__(self):
        """Initialize the cleanup manager"""
        self.swift_monitor = None  # Initialize as None

        # Initialize GCP VM session manager
        self.vm_manager = GCPVMSessionManager()

        # Default configuration - Aggressive memory target of 35%
        self.config = {
            "check_interval": 5.0,  # seconds
            "process_timeout": 30.0,  # seconds
            "stuck_process_time": 600.0,  # 10 minutes - consider process stuck (much more aggressive)
            "memory_threshold": 0.35,  # 35% memory usage target
            "memory_threshold_warning": 0.50,  # 50% warning threshold
            "memory_threshold_critical": 0.65,  # 65% critical threshold (reduced from 70%)
            "memory_threshold_single_process": 500,  # 500MB per process threshold
            "memory_threshold_jarvis_process": 1000,  # 1GB for JARVIS main process
            "cpu_threshold": 0.9,  # 90% CPU usage
            "cpu_threshold_system": 80.0,  # 80% total system CPU usage threshold
            "cpu_threshold_single": 40.0,  # 40% CPU for single process (reduced from 50%)
            "enable_cleanup": True,
            "aggressive_cleanup": True,  # Enable aggressive memory management
            "enable_ipc_cleanup": True,  # Enable semaphore and shared memory cleanup
            "ipc_cleanup_age_threshold": 60,  # 1 minute - more aggressive IPC cleanup
            # JARVIS-specific patterns (improved detection)
            "jarvis_patterns": [
                "jarvis",
                "main.py",
                "jarvis_backend",
                "jarvis_voice",
                "voice_unlock",
                "websocket_server",
                "jarvis-ai-agent",
                "unified_command_processor",
                "resource_manager",
                "wake_word_api",
                "document_writer",
                "neural_trinity",
                "jarvis_optimized",
                "backend.main",
                "jarvis_reload_manager",
                "vision_intelligence",
                "vision_websocket",
                "vision_manager",
            ],
            "jarvis_excluded_patterns": [
                # Exclude IDE/editor processes that may contain "jarvis" in path
                "vscode",
                "code helper",
                "cursor",
                "sublime",
                "pycharm",
                "codeium",
                "copilot",
                "node_modules",
                ".vscode",
                # Exclude macOS system processes
                "controlcenter",
                "ControlCenter.app",
                "CoreServices",
            ],
            "jarvis_port_patterns": [
                3000,
                8000,
                8001,
                8010,
                8080,
                8765,
                5000,
            ],  # Common JARVIS ports including frontend
            "system_critical": [
                "kernel_task",
                "WindowServer",
                "loginwindow",
                "launchd",
                "systemd",
                "init",
                "Finder",
                "Dock",
                "SystemUIServer",
                "python",
                "Python",  # Don't kill generic python processes
                # IDE and development tools (with all helper processes)
                "Cursor",
                "cursor",
                "Cursor Helper",
                "cursor helper",
                "Code",
                "Code Helper",
                "Visual Studio Code",
                "VSCode",
                "vscode",
                "Electron",
                "node",
                "Node",
                "codeium",
                "Codeium",
                # Browsers (with all helper processes)
                "Google Chrome",
                "Chrome",
                "chrome",
                "Google Chrome Helper",
                "Chrome Helper",
                "Chromium",
                "chromium",
                "Safari",
                "safari",
                "Safari Helper",
                "WebKit",
                "Firefox",
                "firefox",
                "Arc",
                "Brave",
                # System tools
                "Terminal",
                "iTerm",
                "iTerm2",
                "Warp",
                "Claude Code",
                # Media and analysis (often uses lots of memory legitimately)
                "mediaanalysisd",
                "photolibraryd",
                "photoanalysisd",
            ],
            # Critical files to monitor for changes
            "critical_files": [
                "main.py",
                "api/jarvis_voice_api.py",
                "api/unified_command_processor.py",
                "api/voice_unlock_integration.py",
                "voice/jarvis_voice.py",
                "voice/macos_voice.py",
                "engines/voice_engine.py",
            ],
        }

        # Learning patterns
        self.problem_patterns = {}
        self.cleanup_history = []

        # Code state tracking
        self.code_state_file = Path.home() / ".jarvis" / "code_state.json"
        self.code_state = self._load_code_state()

        # Load history if exists
        self._load_cleanup_history()

        # Backend base path
        self.backend_path = Path(__file__).parent.absolute()

    def _get_swift_monitor(self):
        """
        Lazy load the Swift monitor.
        DISABLED: Swift monitor is async and causes issues in sync contexts.
        Fallback to psutil is more reliable.
        """
        # DISABLED: Swift monitor is async, use psutil instead
        return None

    def _calculate_code_hash(self) -> str:
        """Calculate hash of critical JARVIS files to detect code changes"""
        hasher = hashlib.sha256()

        for file_path in self.config["critical_files"]:
            full_path = self.backend_path / file_path
            if full_path.exists():
                try:
                    with open(full_path, "rb") as f:
                        hasher.update(f.read())
                    # Also include file modification time
                    hasher.update(str(full_path.stat().st_mtime).encode())
                except Exception as e:
                    logger.error(f"Error hashing {file_path}: {e}")

        return hasher.hexdigest()

    def _clear_python_cache(self) -> int:
        """
        Aggressively clear all Python __pycache__ directories, .pyc files, AND runtime modules.
        This ensures fresh code is ALWAYS loaded on restart.

        Returns:
            Number of cache directories/files removed
        """
        cleared_count = 0

        logger.info("🧹 Clearing all Python cache to ensure fresh code loads...")

        # STEP 1: Clear runtime module cache from sys.modules
        # This is critical - even if .pyc files are deleted, Python caches modules in memory
        modules_to_remove = []
        for module_name in list(sys.modules.keys()):
            # Remove JARVIS-specific modules from cache
            if any(pattern in module_name for pattern in [
                'voice', 'intelligence', 'api', 'backend', 'core',
                'chatbots', 'display', 'memory', 'vision', 'engines',
                'context_intelligence', 'neural_trinity', 'macos_keychain_unlock'
            ]):
                modules_to_remove.append(module_name)

        for module_name in modules_to_remove:
            try:
                del sys.modules[module_name]
                cleared_count += 1
            except Exception as e:
                logger.debug(f"Could not remove module {module_name}: {e}")

        if modules_to_remove:
            logger.info(f"✅ Cleared {len(modules_to_remove)} cached modules from sys.modules")

        # STEP 2: Clear __pycache__ directories recursively
        try:
            for root, dirs, files in os.walk(self.backend_path):
                # Clear __pycache__ directories
                if "__pycache__" in dirs:
                    pycache_path = Path(root) / "__pycache__"
                    try:
                        import shutil
                        shutil.rmtree(pycache_path)
                        cleared_count += 1
                        logger.debug(f"Cleared cache: {pycache_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove {pycache_path}: {e}")

                # Also clear any .pyc files in the directory
                for file in files:
                    if file.endswith(".pyc"):
                        pyc_path = Path(root) / file
                        try:
                            pyc_path.unlink()
                            cleared_count += 1
                            logger.debug(f"Removed .pyc file: {pyc_path}")
                        except Exception as e:
                            logger.warning(f"Failed to remove {pyc_path}: {e}")

        except Exception as e:
            logger.error(f"Error during Python cache cleanup: {e}")

        if cleared_count > 0:
            logger.info(f"✅ Cleared {cleared_count} total cache items (modules + files)")
        else:
            logger.debug("No Python cache found to clear")

        return cleared_count

    def _detect_code_changes(self) -> bool:
        """Detect if JARVIS code has changed since last run"""
        current_hash = self._calculate_code_hash()
        last_hash = self.code_state.get("code_hash", "")

        if current_hash != last_hash:
            logger.info(
                f"Code changes detected! Current: {current_hash[:8]}... Last: {last_hash[:8]}..."
            )
            return True
        return False

    def _save_code_state(self):
        """Save current code state"""
        self.code_state["code_hash"] = self._calculate_code_hash()
        self.code_state["last_update"] = datetime.now().isoformat()
        self.code_state["pid"] = os.getpid()

        self.code_state_file.parent.mkdir(exist_ok=True)
        try:
            with open(self.code_state_file, "w") as f:
                json.dump(self.code_state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save code state: {e}")

    def _load_code_state(self) -> Dict:
        """Load saved code state"""
        if self.code_state_file.exists():
            try:
                with open(self.code_state_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load code state: {e}")
        return {}

    def force_restart_cleanup(self, skip_code_check: bool = True) -> List[Dict]:
        """
        FORCE restart cleanup - ALWAYS kills all JARVIS processes.
        Used when --restart flag is provided to ensure clean slate.

        ENHANCED with:
        - Dynamic port configuration (no hardcoding)
        - UE (Uninterruptible Sleep) state detection
        - Parallel async cleanup for better performance
        - Self-healing capabilities

        This is MORE AGGRESSIVE than cleanup_old_instances_on_code_change()
        because it doesn't check for code changes - it ALWAYS kills everything.

        Args:
            skip_code_check: If True, skip code change detection and force cleanup

        Returns:
            List of cleaned up processes
        """
        cleaned = []

        # Get dynamic configuration - NO HARDCODED PORTS
        restart_config = get_restart_config()
        ue_detector = get_ue_detector()
        async_manager = get_async_restart_manager()

        # Get ports dynamically from config
        all_ports = restart_config.get_all_ports()
        api_ports = restart_config.get_api_ports()

        logger.warning("🔥 ENHANCED FORCE RESTART MODE - Killing ALL JARVIS processes...")
        logger.info(f"   📋 Dynamic port configuration loaded:")
        logger.info(f"      Primary API: {restart_config.primary_api_port}")
        logger.info(f"      Fallback ports: {restart_config.fallback_ports}")
        logger.info(f"      All managed ports: {all_ports}")

        # Step 0: Scan for UE (Uninterruptible Sleep) processes FIRST
        logger.info("🔍 Scanning for UE (Uninterruptible Sleep) processes...")
        ue_scan_results = ue_detector.scan_all_ports(all_ports)
        ue_blocked_ports = []
        for port, result in ue_scan_results.items():
            if result['has_ue_process']:
                ue_blocked_ports.append(port)
                logger.warning(f"   ⚠️ Port {port} has UE process (PID {result['pid']}) - CANNOT KILL")
                logger.warning(f"      Status: {result['status']}")
                logger.warning(f"      This port will be blacklisted and avoided")

        if ue_blocked_ports:
            logger.warning(f"   🚫 {len(ue_blocked_ports)} ports blocked by UE processes: {ue_blocked_ports}")
            logger.info(f"   💡 System will use alternative ports for startup")

        # Step 1: Gracefully shutdown ML Learning Engine before killing processes
        try:
            logger.info("🧠 Attempting graceful ML Learning Engine shutdown...")
            from voice_unlock.continuous_learning_engine import shutdown_learning_engine

            # Use asyncio to run async shutdown
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(shutdown_learning_engine())
                else:
                    asyncio.run(shutdown_learning_engine())
                logger.info("✅ ML Learning Engine shutdown complete (models saved)")
            except RuntimeError:
                # No event loop, create one
                asyncio.run(shutdown_learning_engine())
                logger.info("✅ ML Learning Engine shutdown complete (models saved)")
            except Exception as e:
                logger.warning(f"ML Learning Engine shutdown warning: {e}")
        except ImportError:
            logger.debug("ML Learning Engine not available")
        except Exception as e:
            logger.warning(f"Error during ML Learning Engine shutdown: {e}")

        # Step 2: ALWAYS clear Python cache in force restart mode
        cache_cleared = self._clear_python_cache()
        if cache_cleared:
            logger.info(f"✅ Cleared {cache_cleared} total cache items (modules + files)")

        # Step 3: Clear importlib cache to prevent stale module usage
        import importlib
        importlib.invalidate_caches()
        logger.info("✅ Invalidated importlib caches")

        current_pid = os.getpid()
        current_ppid = os.getppid()
        current_time = time.time()

        # Track all JARVIS processes with enhanced categorization
        backend_processes = []
        frontend_processes = []
        related_processes = []
        websocket_processes = []
        python_processes = []
        node_processes = []
        proxy_processes = []
        ue_state_processes = []  # NEW: Track UE state processes separately

        # Step 4: Enhanced process detection with multiple strategies
        logger.info("🔍 Using multiple strategies to find ALL JARVIS processes...")

        # Extended pattern list for better detection - DYNAMIC based on config
        jarvis_patterns = [
            "main.py", "start_system.py", "jarvis", "JARVIS",
            "unified_websocket", "voice_unlock", "speaker_verification",
            "whisper_audio", "hybrid_stt", "cloud-sql-proxy",
            "npm start", "react-scripts", "webpack",
        ]
        # Add dynamic port patterns
        for port in api_ports:
            jarvis_patterns.append(f"python.*{port}")
            jarvis_patterns.append(f"websocket.*{port}")
            jarvis_patterns.append(f"node.*{port}")

        # Strategy 1: Process iteration with extended patterns
        for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time", "ppid", "connections"]):
            try:
                if proc.pid == current_pid or proc.pid == current_ppid:
                    continue  # Skip self and parent

                cmdline = " ".join(proc.cmdline()) if proc.cmdline() else ""
                cmdline_lower = cmdline.lower()

                # Check if process matches any JARVIS pattern
                is_jarvis = False
                if self._is_jarvis_process(proc):
                    is_jarvis = True
                else:
                    # Additional pattern matching
                    for pattern in jarvis_patterns:
                        if pattern.lower() in cmdline_lower:
                            is_jarvis = True
                            break

                # Also check by port connections - USING DYNAMIC PORTS
                if not is_jarvis:
                    try:
                        for conn in proc.connections(kind='inet'):
                            if conn.laddr.port in all_ports:
                                is_jarvis = True
                                logger.debug(f"Found process on port {conn.laddr.port}: PID {proc.pid}")
                                break
                    except (psutil.AccessDenied, psutil.NoSuchProcess):
                        pass

                if is_jarvis:
                    # Check for UE state BEFORE categorizing
                    is_ue, ue_status = ue_detector.check_process_state(proc.pid)
                    if is_ue:
                        ue_state_processes.append((proc, cmdline, ue_status))
                        logger.warning(f"   ⚠️ PID {proc.pid} is in UE state ({ue_status}) - will skip")
                        continue  # Don't try to kill UE processes

                    # Enhanced categorization
                    if "main.py" in cmdline:
                        backend_processes.append((proc, cmdline))
                    elif "start_system.py" in cmdline:
                        frontend_processes.append((proc, cmdline))
                    elif "websocket" in cmdline_lower or "ws_server" in cmdline_lower:
                        websocket_processes.append((proc, cmdline))
                    elif "cloud-sql-proxy" in cmdline_lower:
                        proxy_processes.append((proc, cmdline))
                    elif "node" in cmdline_lower or "npm" in cmdline_lower:
                        node_processes.append((proc, cmdline))
                    elif "python" in cmdline_lower:
                        python_processes.append((proc, cmdline))
                    else:
                        related_processes.append((proc, cmdline))

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Strategy 2: Port-based detection - USING DYNAMIC PORTS
        ports_to_check = [p for p in all_ports if p not in ue_blocked_ports]
        logger.info(f"🔍 Checking for processes on JARVIS ports: {ports_to_check}")
        for port in ports_to_check:
            try:
                for conn in psutil.net_connections(kind='inet'):
                    if conn.laddr.port == port and conn.status == 'LISTEN':
                        try:
                            proc = psutil.Process(conn.pid)
                            if proc.pid != current_pid and proc.pid != current_ppid:
                                # Check UE state
                                is_ue, ue_status = ue_detector.check_process_state(proc.pid)
                                if is_ue:
                                    logger.warning(f"   ⚠️ Port {port} PID {proc.pid} is UE state - skipping")
                                    continue

                                cmdline = " ".join(proc.cmdline()) if proc.cmdline() else ""
                                # Check if not already in our lists
                                all_pids = [p[0].pid for p in backend_processes + frontend_processes +
                                          related_processes + websocket_processes + proxy_processes +
                                          node_processes + python_processes]
                                if proc.pid not in all_pids:
                                    related_processes.append((proc, cmdline))
                                    logger.info(f"   Found process on port {port}: PID {proc.pid}")
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
            except Exception as e:
                logger.debug(f"Port scan error: {e}")

        # Report UE state processes
        if ue_state_processes:
            logger.warning(f"⚠️ {len(ue_state_processes)} processes in UE state (CANNOT KILL):")
            for proc, cmdline, ue_status in ue_state_processes:
                logger.warning(f"   • PID {proc.pid}: {ue_status}")
                logger.warning(f"     Command: {cmdline[:100]}")

        # Step 5: Kill processes - Try async parallel cleanup first
        all_killable_processes = []
        for proc, cmdline in backend_processes:
            all_killable_processes.append((proc, cmdline, "backend"))
        for proc, cmdline in websocket_processes:
            all_killable_processes.append((proc, cmdline, "websocket"))
        for proc, cmdline in python_processes:
            all_killable_processes.append((proc, cmdline, "python"))
        for proc, cmdline in proxy_processes:
            all_killable_processes.append((proc, cmdline, "cloud-sql-proxy"))
        for proc, cmdline in node_processes:
            all_killable_processes.append((proc, cmdline, "node"))
        for proc, cmdline in related_processes:
            all_killable_processes.append((proc, cmdline, "related"))
        for proc, cmdline in frontend_processes:
            all_killable_processes.append((proc, cmdline, "frontend"))

        if all_killable_processes:
            logger.info(f"🔪 Killing {len(all_killable_processes)} processes...")

            # Try async parallel cleanup
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Already in async context - use sync fallback
                    cleaned = self._sync_kill_processes(all_killable_processes, current_time, ue_detector)
                else:
                    # Run async cleanup
                    cleaned = asyncio.run(
                        async_manager.cleanup_processes_parallel(all_killable_processes)
                    )
            except RuntimeError:
                # No event loop - run async cleanup
                try:
                    cleaned = asyncio.run(
                        async_manager.cleanup_processes_parallel(all_killable_processes)
                    )
                except Exception as e:
                    logger.warning(f"Async cleanup failed ({e}), using sync fallback")
                    cleaned = self._sync_kill_processes(all_killable_processes, current_time, ue_detector)

            # Report results
            successful = [r for r in cleaned if r.get('status') in ['terminated', 'force_killed', 'sigkill']]
            ue_detected = [r for r in cleaned if r.get('ue_state')]
            failed = [r for r in cleaned if r.get('status') == 'failed']

            logger.info(f"   ✅ Successfully killed: {len(successful)}")
            if ue_detected:
                logger.warning(f"   ⚠️ UE state detected (skipped): {len(ue_detected)}")
            if failed:
                logger.error(f"   ❌ Failed to kill: {len(failed)}")

        # Step 6: Final verification - ensure ports are free (DYNAMIC PORTS)
        logger.info("🔍 Verifying all JARVIS ports are free...")

        # Use async port verification if possible
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                port_status = asyncio.run(async_manager.verify_ports_free_async(ports_to_check))
            else:
                # Sync fallback
                port_status = self._sync_verify_ports(ports_to_check, ue_detector)
        except RuntimeError:
            try:
                port_status = asyncio.run(async_manager.verify_ports_free_async(ports_to_check))
            except:
                port_status = self._sync_verify_ports(ports_to_check, ue_detector)

        for port, status in port_status.items():
            if status.get('free'):
                logger.info(f"   ✅ Port {port} is free ({status.get('action', 'ok')})")
            elif status.get('ue_process'):
                logger.warning(f"   ⚠️ Port {port} has UE process - blacklisted")
            else:
                logger.warning(f"   ⚠️ Port {port} still in use: {status}")

        # Step 7: Clear any remaining module references
        to_remove = []
        for module_name in list(sys.modules.keys()):
            if any(pattern in module_name for pattern in [
                'jarvis', 'voice', 'api.unified', 'backend', 'intelligence',
                'whisper_audio', 'speaker_verification', 'hybrid_stt'
            ]):
                to_remove.append(module_name)

        for module_name in to_remove:
            try:
                del sys.modules[module_name]
            except:
                pass

        if to_remove:
            logger.info(f"✅ Removed {len(to_remove)} additional JARVIS modules from sys.modules")

        # Step 8: Report healthy port for next startup
        healthy_port = async_manager.get_healthy_port(exclude_blacklisted=True)
        if healthy_port:
            logger.info(f"🎯 Recommended port for next startup: {healthy_port}")
        else:
            logger.warning("⚠️ No healthy ports available - may need system restart")

        if cleaned:
            successful_count = len([r for r in cleaned if r.get('status') != 'failed'])
            logger.warning(f"✅ ENHANCED FORCE RESTART: Processed {len(cleaned)} processes ({successful_count} killed)")
            logger.info("   System is now clean for fresh start")
            if ue_blocked_ports:
                logger.warning(f"   ⚠️ Note: {len(ue_blocked_ports)} ports remain blocked by UE processes")
        else:
            logger.info("✅ No JARVIS processes found - system already clean")

        # Final step: Small delay to ensure OS has released all resources
        time.sleep(0.5)

        return cleaned

    def _sync_kill_processes(
        self,
        processes: List[Tuple[psutil.Process, str, str]],
        current_time: float,
        ue_detector: UEStateDetector
    ) -> List[Dict]:
        """
        Synchronous process killing fallback with UE state detection.
        Used when async cleanup is not available.
        """
        cleaned = []

        for proc, cmdline, ptype in processes:
            try:
                age_seconds = current_time - proc.create_time()

                # Check UE state before attempting kill
                is_ue, ue_status = ue_detector.check_process_state(proc.pid)
                if is_ue:
                    cleaned.append({
                        "pid": proc.pid,
                        "name": proc.name() if proc.is_running() else "unknown",
                        "type": ptype,
                        "cmdline": cmdline[:100],
                        "age_minutes": age_seconds / 60,
                        "status": "ue_state_detected",
                        "ue_state": True,
                        "ue_status": ue_status
                    })
                    logger.warning(f"   ⚠️ PID {proc.pid} in UE state - skipping")
                    continue

                logger.info(f"   → Terminating {ptype} PID {proc.pid} (age: {age_seconds/60:.1f}min)")

                # Try graceful termination first
                try:
                    proc.terminate()
                    proc.wait(timeout=2.0)
                    cleaned.append({
                        "pid": proc.pid,
                        "name": proc.name(),
                        "type": ptype,
                        "cmdline": cmdline[:100],
                        "age_minutes": age_seconds / 60,
                        "status": "terminated",
                    })
                    logger.info(f"   ✅ Terminated {ptype} PID {proc.pid}")
                except psutil.TimeoutExpired:
                    # IMMEDIATE force kill if unresponsive
                    logger.warning(f"   ⚡ Process {proc.pid} unresponsive, FORCE KILLING...")
                    proc.kill()
                    time.sleep(0.1)
                    if not proc.is_running():
                        cleaned.append({
                            "pid": proc.pid,
                            "name": proc.name(),
                            "type": ptype,
                            "cmdline": cmdline[:100],
                            "age_minutes": age_seconds / 60,
                            "status": "force_killed",
                        })
                        logger.warning(f"   💀 Force killed {ptype} PID {proc.pid}")
                    else:
                        # Check if became UE state
                        is_ue_now, ue_status = ue_detector.check_process_state(proc.pid)
                        if is_ue_now:
                            cleaned.append({
                                "pid": proc.pid,
                                "name": proc.name(),
                                "type": ptype,
                                "cmdline": cmdline[:100],
                                "age_minutes": age_seconds / 60,
                                "status": "became_ue_state",
                                "ue_state": True,
                                "ue_status": ue_status
                            })
                            logger.error(f"   ⚠️ PID {proc.pid} became UE state - blacklisting")
                        else:
                            # Use OS-level kill as last resort
                            os.kill(proc.pid, signal.SIGKILL)
                            logger.error(f"   ☠️ OS-level SIGKILL sent to PID {proc.pid}")
                            cleaned.append({
                                "pid": proc.pid,
                                "name": proc.name(),
                                "type": ptype,
                                "cmdline": cmdline[:100],
                                "age_minutes": age_seconds / 60,
                                "status": "sigkill",
                            })
                except psutil.NoSuchProcess:
                    logger.debug(f"   Process {proc.pid} already dead")

            except Exception as e:
                # Last resort: shell command kill
                try:
                    subprocess.run(f"kill -9 {proc.pid}", shell=True, capture_output=True, timeout=2.0)
                    logger.warning(f"   🔨 Shell kill -9 executed for PID {proc.pid}")
                    cleaned.append({
                        "pid": proc.pid,
                        "name": "unknown",
                        "type": ptype,
                        "cmdline": cmdline[:100],
                        "status": "shell_killed",
                    })
                except:
                    logger.error(f"   ❌ Failed to kill {ptype} PID {proc.pid}: {e}")

        return cleaned

    def _sync_verify_ports(
        self,
        ports: List[int],
        ue_detector: UEStateDetector
    ) -> Dict[int, Dict[str, Any]]:
        """
        Synchronous port verification fallback with UE detection.
        """
        port_status = {}

        for port in ports:
            # Check for UE process first
            has_ue, pid, ue_status = ue_detector.check_port_for_ue_process(port)
            if has_ue:
                port_status[port] = {
                    'free': False,
                    'ue_process': True,
                    'pid': pid,
                    'status': ue_status,
                    'action': 'blacklisted'
                }
                continue

            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('127.0.0.1', port))
                sock.close()

                if result == 0:
                    # Port still in use - try force kill
                    subprocess.run(
                        f"lsof -ti:{port} | xargs kill -9 2>/dev/null",
                        shell=True, capture_output=True, timeout=2.0
                    )
                    time.sleep(0.3)

                    # Recheck
                    sock2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock2.settimeout(0.5)
                    recheck = sock2.connect_ex(('127.0.0.1', port))
                    sock2.close()

                    if recheck != 0:
                        port_status[port] = {'free': True, 'action': 'force_freed'}
                    else:
                        # Check if it's now a UE process
                        has_ue_now, pid_now, _ = ue_detector.check_port_for_ue_process(port)
                        if has_ue_now:
                            port_status[port] = {
                                'free': False,
                                'ue_process': True,
                                'pid': pid_now,
                                'action': 'blacklisted_after_kill_attempt'
                            }
                        else:
                            port_status[port] = {'free': False, 'action': 'still_in_use'}
                else:
                    port_status[port] = {'free': True, 'action': 'already_free'}

            except Exception as e:
                port_status[port] = {'free': False, 'error': str(e)}

        return port_status

    def cleanup_old_instances_on_code_change(self) -> List[Dict]:
        """
        Cleanup old JARVIS instances when code changes are detected.
        This ensures only the latest code is running.
        Includes GCP VM cleanup for orphaned sessions.

        Enhanced to kill both frontend (start_system.py) AND backend (main.py) processes,
        ensuring complete cache clearing and fresh code reload across all components.
        """
        cleaned = []

        # Check for code changes
        if not self._detect_code_changes():
            logger.info("No code changes detected, skipping old instance cleanup")
            return cleaned

        logger.warning("🔄 Code changes detected! Cleaning up old JARVIS instances...")

        # Step 0: Gracefully shutdown ML Learning Engine before killing processes
        try:
            logger.info("🧠 Attempting graceful ML Learning Engine shutdown...")
            from voice_unlock.continuous_learning_engine import shutdown_learning_engine

            # Use asyncio to run async shutdown
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(shutdown_learning_engine())
                else:
                    asyncio.run(shutdown_learning_engine())
                logger.info("✅ ML Learning Engine shutdown complete (models saved)")
            except RuntimeError:
                # No event loop, create one
                asyncio.run(shutdown_learning_engine())
                logger.info("✅ ML Learning Engine shutdown complete (models saved)")
            except Exception as e:
                logger.warning(f"ML Learning Engine shutdown warning: {e}")
        except ImportError:
            logger.debug("ML Learning Engine not available")
        except Exception as e:
            logger.warning(f"Error during ML Learning Engine shutdown: {e}")

        # Clear Python cache FIRST to ensure fresh code loads
        # This affects the frontend process, but backend subprocesses need to be killed
        cache_cleared = self._clear_python_cache()
        if cache_cleared:
            logger.info(f"✅ Cleared {cache_cleared} Python cache directories")

        current_pid = os.getpid()
        current_ppid = os.getppid()  # Track parent process too
        current_time = time.time()

        # Track PIDs that will be terminated (for VM cleanup)
        pids_to_terminate = []

        # Track backend processes separately for better logging
        backend_processes = []
        frontend_processes = []
        related_processes = []

        # Find all JARVIS processes
        for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time", "ppid"]):
            try:
                if proc.pid == current_pid or proc.pid == current_ppid:
                    continue  # Skip self and parent

                if self._is_jarvis_process(proc):
                    cmdline = " ".join(proc.cmdline())
                    logger.debug(f"Found JARVIS process: PID {proc.pid} - {cmdline[:100]}...")

                    # Categorize the process
                    if "main.py" in cmdline and "backend" in cmdline.lower():
                        # Backend process - ALWAYS kill these to reload fresh code
                        backend_processes.append((proc, cmdline))
                    elif "start_system.py" in cmdline:
                        # Frontend process
                        frontend_processes.append((proc, cmdline))
                    elif any(
                        pattern in cmdline
                        for pattern in ["voice_unlock", "websocket_server", "jarvis_", "jarvis_reload"]
                    ):
                        # Related helper processes
                        related_processes.append((proc, cmdline))

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Step 1: Kill ALL backend processes first (critical for fresh code)
        if backend_processes:
            logger.warning(f"🎯 Found {len(backend_processes)} backend process(es) to terminate")
            for proc, cmdline in backend_processes:
                try:
                    age_seconds = current_time - proc.create_time()
                    logger.warning(
                        f"Terminating backend process (PID: {proc.pid}, Age: {age_seconds/60:.1f} min)"
                    )
                    logger.info(f"   Command: {cmdline[:120]}")

                    pids_to_terminate.append(proc.pid)

                    # Be aggressive with backend - use shorter timeout
                    try:
                        proc.terminate()
                        proc.wait(timeout=3)  # Shorter timeout for backend
                        cleaned.append({
                            "pid": proc.pid,
                            "name": proc.name(),
                            "type": "backend",
                            "cmdline": cmdline[:100],
                            "age_minutes": age_seconds / 60,
                            "status": "terminated",
                        })
                        logger.info(f"✅ Terminated backend process {proc.pid}")
                    except psutil.TimeoutExpired:
                        # Force kill backend immediately if it doesn't respond
                        proc.kill()
                        cleaned.append({
                            "pid": proc.pid,
                            "name": proc.name(),
                            "type": "backend",
                            "cmdline": cmdline[:100],
                            "age_minutes": age_seconds / 60,
                            "status": "force_killed",
                        })
                        logger.warning(f"⚠️ Force killed unresponsive backend {proc.pid}")
                    except psutil.NoSuchProcess:
                        logger.info(f"✅ Backend process {proc.pid} already terminated")

                except Exception as e:
                    logger.error(f"❌ Failed to clean up backend PID {proc.pid}: {e}")

        # Step 2: Kill old frontend processes
        if frontend_processes:
            logger.info(f"Found {len(frontend_processes)} frontend process(es)")
            for proc, cmdline in frontend_processes:
                try:
                    age_seconds = current_time - proc.create_time()
                    logger.info(
                        f"Terminating frontend (PID: {proc.pid}, Age: {age_seconds/60:.1f} min)"
                    )

                    pids_to_terminate.append(proc.pid)

                    try:
                        proc.terminate()
                        proc.wait(timeout=5)
                        cleaned.append({
                            "pid": proc.pid,
                            "name": proc.name(),
                            "type": "frontend",
                            "cmdline": cmdline[:100],
                            "age_minutes": age_seconds / 60,
                            "status": "terminated",
                        })
                        logger.info(f"✅ Terminated frontend process {proc.pid}")
                    except psutil.TimeoutExpired:
                        proc.kill()
                        cleaned.append({
                            "pid": proc.pid,
                            "name": proc.name(),
                            "type": "frontend",
                            "cmdline": cmdline[:100],
                            "age_minutes": age_seconds / 60,
                            "status": "killed",
                        })
                        logger.warning(f"⚠️ Force killed frontend {proc.pid}")
                    except psutil.NoSuchProcess:
                        logger.info(f"✅ Frontend process {proc.pid} already terminated")

                except Exception as e:
                    logger.error(f"❌ Failed to clean up frontend PID {proc.pid}: {e}")

        # Step 3: Clean up related helper processes
        if related_processes:
            logger.info(f"Cleaning up {len(related_processes)} related process(es)")
            for proc, cmdline in related_processes:
                try:
                    logger.debug(f"Cleaning up related: {proc.name()} (PID: {proc.pid})")
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                        cleaned.append({
                            "pid": proc.pid,
                            "name": proc.name(),
                            "type": "related",
                            "cmdline": cmdline[:50],
                            "status": "terminated",
                        })
                    except psutil.TimeoutExpired:
                        proc.kill()
                        cleaned.append({
                            "pid": proc.pid,
                            "name": proc.name(),
                            "type": "related",
                            "cmdline": cmdline[:50],
                            "status": "killed",
                        })
                except Exception as e:
                    logger.debug(f"Could not clean up related process {proc.pid}: {e}")

        # Step 4: Wait a moment for processes to fully terminate and release resources
        if cleaned:
            logger.info("⏳ Waiting for processes to release resources...")
            time.sleep(2)  # Increased wait time for clean shutdown

        # Step 5: Clean up orphaned ports
        self._cleanup_orphaned_ports()

        # Step 6: Clean up IPC resources (semaphores, shared memory)
        ipc_cleaned = self._cleanup_ipc_resources()
        if sum(ipc_cleaned.values()) > 0:
            logger.info(
                f"Cleaned up {sum(ipc_cleaned.values())} IPC resources "
                f"({ipc_cleaned['semaphores']} semaphores, {ipc_cleaned['shared_memory']} shared memory)"
            )

        # Step 7: Clean up VMs for terminated PIDs
        if pids_to_terminate:
            logger.info(
                f"🌐 Checking for VMs associated with {len(pids_to_terminate)} terminated processes"
            )
            vms_cleaned = self._cleanup_vms_for_pids_sync(pids_to_terminate)
            if vms_cleaned:
                logger.info(f"🧹 Cleaned up {vms_cleaned} VMs from terminated processes")

        # Step 8: Clear Python cache AGAIN after killing backend to be absolutely sure
        # This handles any .pyc files that might have been regenerated
        final_cache_clear = self._clear_python_cache()
        if final_cache_clear:
            logger.debug(f"Final cache clear: {final_cache_clear} items removed")

        # Step 9: Save new code state after cleanup
        self._save_code_state()

        # Summary logging
        if cleaned:
            backend_count = len([c for c in cleaned if c.get("type") == "backend"])
            frontend_count = len([c for c in cleaned if c.get("type") == "frontend"])
            related_count = len([c for c in cleaned if c.get("type") == "related"])

            logger.info(
                f"🧹 Cleaned up {len(cleaned)} old JARVIS processes: "
                f"{backend_count} backend, {frontend_count} frontend, {related_count} related"
            )
            logger.info("✅ System ready for fresh code reload")
        else:
            logger.info("No old processes found to clean up")

        return cleaned

    def _cleanup_vms_for_pids_sync(self, pids: List[int]) -> int:
        """
        Synchronously cleanup VMs associated with specific PIDs.
        Used during code change cleanup (non-async context).
        """
        cleaned_count = 0

        # Get all sessions and filter by PIDs
        all_sessions = self.vm_manager.get_all_sessions()
        sessions_to_cleanup = [s for s in all_sessions if s.get("pid") in pids]

        if not sessions_to_cleanup:
            return 0

        logger.info(f"Found {len(sessions_to_cleanup)} VMs to cleanup for terminated PIDs")

        for session in sessions_to_cleanup:
            vm_id = session.get("vm_id")
            zone = session.get("zone", self.vm_manager.default_zone)

            if vm_id:
                success = self.vm_manager.delete_vm_sync(vm_id, zone)
                if success:
                    cleaned_count += 1

        # Clean up registry
        if sessions_to_cleanup:
            self.vm_manager._remove_orphaned_from_registry(sessions_to_cleanup)

        return cleaned_count

    def ensure_single_instance(self) -> bool:
        """
        Ensure only one instance of JARVIS is running on the same port.
        Returns True if this is the only instance, False otherwise.
        """
        current_pid = os.getpid()
        target_port = int(os.getenv("BACKEND_PORT", "8000"))

        # First, check for any JARVIS processes regardless of port
        jarvis_processes = self._find_jarvis_processes()
        other_jarvis_processes = [p for p in jarvis_processes if p["pid"] != current_pid]

        if other_jarvis_processes:
            logger.warning(f"Found {len(other_jarvis_processes)} other JARVIS processes running:")
            for proc in other_jarvis_processes:
                logger.warning(
                    f"  - PID {proc['pid']}: {proc['name']} (age: {proc['age_seconds']/60:.1f} min)"
                )

            # Check if we should take over (code changes or old instance)
            if self._detect_code_changes():
                logger.info("Code changes detected, terminating old instances...")
                for proc_info in other_jarvis_processes:
                    try:
                        proc = psutil.Process(proc_info["pid"])
                        logger.info(f"Terminating old JARVIS process {proc_info['pid']}")
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except psutil.TimeoutExpired:
                            proc.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                time.sleep(2)  # Give processes time to clean up
                return True
            else:
                # Check if any of these processes are using our target port
                for proc_info in other_jarvis_processes:
                    try:
                        proc = psutil.Process(proc_info["pid"])
                        for conn in proc.connections():
                            if conn.laddr.port == target_port and conn.status == "LISTEN":
                                logger.warning(
                                    f"JARVIS instance {proc_info['pid']} is using port {target_port}"
                                )
                                return False
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

        # DISABLED: net_connections hangs on macOS due to security restrictions
        # Skip directly to returning True to allow JARVIS to start
        return True

        # Original code disabled (everything below is commented out):
        """
        try:
            for conn in psutil.net_connections():
                if conn.laddr.port == target_port and conn.status == "LISTEN":
                    if conn.pid != current_pid:
                        try:
                            proc = psutil.Process(conn.pid)
                            if self._is_jarvis_process(proc):
                                logger.warning(
                                    f"Another JARVIS instance (PID: {conn.pid}) is already "
                                    f"running on port {target_port}"
                                )

                                # Check if we should take over (code changes or old instance)
                                if self._detect_code_changes():
                                    logger.info("Code changes detected, terminating old instance...")
                                    proc.terminate()
                                    try:
                                        proc.wait(timeout=5)
                                    except psutil.TimeoutExpired:
                                        proc.kill()
                                    time.sleep(1)
                                    return True
                                else:
                                    return False
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
        except (psutil.AccessDenied, PermissionError):
            # Fall back to checking specific port using lsof
            # DISABLED: lsof hangs on macOS - this was preventing JARVIS from starting
            logger.info(f"Skipping lsof check (macOS compatibility)")
            return True  # Allow JARVIS to start

            # Original code disabled:
            # result = subprocess.run(
            #     ["lsof", "-i", f":{target_port}", "-t"],
            #     capture_output=True,
            #     text=True
            # )
                if result.stdout.strip():
                    pids = [int(pid) for pid in result.stdout.strip().split('\n') if pid]
                    for pid in pids:
                        if pid != current_pid:
                            try:
                                proc = psutil.Process(pid)
                                if self._is_jarvis_process(proc):
                                    logger.warning(f"Found JARVIS instance on port {target_port} (PID: {pid})")
                                    if self._detect_code_changes():
                                        logger.info("Code changes detected, terminating old instance...")
                                        proc.terminate()
                                        try:
                                            proc.wait(timeout=5)
                                        except psutil.TimeoutExpired:
                                            proc.kill()
                                        time.sleep(1)
                                        return True
                                    else:
                                        return False
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
            except Exception as e:
                logger.error(f"Failed to check port with lsof: {e}")

        return True
        """

    def get_system_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of the system state using Swift if available"""
        monitor = self._get_swift_monitor()
        if monitor:
            return monitor.get_system_snapshot()
        else:
            # Fallback to psutil
            return {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_available_mb": psutil.virtual_memory().available // (1024 * 1024),
                "timestamp": datetime.now(),
            }

    def analyze_system_state(self) -> Dict[str, Any]:
        """Analyze system state for cleanup"""
        snapshot = self.get_system_snapshot()

        # Use Swift monitor for process details if available
        monitor = self._get_swift_monitor()
        if monitor:
            snapshot["all_processes"] = monitor.get_all_processes()

        state = {
            "cpu_percent": snapshot["cpu_percent"],
            "memory_percent": snapshot["memory_percent"],
            "memory_available_mb": snapshot["memory_available_mb"],
            "timestamp": snapshot["timestamp"],
        }

        # Find problematic processes
        state["high_cpu_processes"] = self._find_high_cpu_processes()
        state["high_memory_processes"] = self._find_high_memory_processes()
        state["stuck_processes"] = self._find_stuck_processes()
        state["zombie_processes"] = self._find_zombie_processes()
        state["jarvis_processes"] = self._find_jarvis_processes()

        return state

    def _find_high_cpu_processes(self) -> List[Dict]:
        """Find processes using excessive CPU"""
        high_cpu = []

        for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                # Get CPU usage over a short interval
                cpu = proc.cpu_percent(interval=0.1)
                memory_info = proc.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)

                if cpu > self.config["cpu_threshold_single"]:
                    proc_info = {
                        "pid": proc.pid,
                        "name": proc.name(),
                        "cpu_percent": cpu,
                        "memory_percent": proc.memory_percent(),
                        "memory_mb": memory_mb,
                        "cmdline": " ".join(proc.cmdline()[:5]),  # First 5 args
                        "create_time": datetime.fromtimestamp(proc.create_time()),
                        "is_jarvis": self._is_jarvis_process(proc),
                    }
                    high_cpu.append(proc_info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return sorted(high_cpu, key=lambda x: x["cpu_percent"], reverse=True)

    def _find_high_memory_processes(self) -> List[Dict]:
        """Find processes using excessive memory"""
        high_memory = []

        for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                memory_info = proc.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                is_jarvis = self._is_jarvis_process(proc)

                # Check against thresholds
                threshold = (
                    self.config["memory_threshold_jarvis_process"]
                    if is_jarvis
                    else self.config["memory_threshold_single_process"]
                )

                if memory_mb > threshold:
                    proc_info = {
                        "pid": proc.pid,
                        "name": proc.name(),
                        "cpu_percent": proc.cpu_percent(interval=0.1),
                        "memory_percent": proc.memory_percent(),
                        "memory_mb": memory_mb,
                        "cmdline": " ".join(proc.cmdline()[:5]),
                        "create_time": datetime.fromtimestamp(proc.create_time()),
                        "is_jarvis": is_jarvis,
                    }
                    high_memory.append(proc_info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return sorted(high_memory, key=lambda x: x["memory_mb"], reverse=True)

    def _find_stuck_processes(self) -> List[Dict]:
        """
        Find processes that appear to be stuck or hanging.
        Enhanced to detect async/sync deadlocks (like SpeechBrain model loading).
        """
        stuck = []
        current_time = time.time()

        for proc in psutil.process_iter(["pid", "name", "create_time", "status", "cmdline"]):
            try:
                # Check if process is in uninterruptible sleep or zombie state
                if proc.status() in [psutil.STATUS_DISK_SLEEP, psutil.STATUS_ZOMBIE]:
                    stuck.append(
                        {
                            "pid": proc.pid,
                            "name": proc.name(),
                            "status": proc.status(),
                            "age_seconds": current_time - proc.create_time(),
                            "reason": "uninterruptible_sleep_or_zombie",
                        }
                    )

                # Check for old JARVIS processes that might be stuck
                if self._is_jarvis_process(proc):
                    age = current_time - proc.create_time()
                    cmdline = " ".join(proc.cmdline())

                    # Enhanced: Detect async/sync deadlock scenarios
                    is_likely_deadlocked = False
                    deadlock_reason = ""

                    # Check for SpeechBrain-related deadlocks
                    if "speechbrain" in cmdline.lower() or "torch" in cmdline.lower():
                        cpu_usage = proc.cpu_percent(interval=0.5)
                        if cpu_usage < 0.1 and age > 120:  # 2 minutes idle
                            is_likely_deadlocked = True
                            deadlock_reason = "speechbrain_async_deadlock"

                    # Check for database connection hangs
                    if "database" in cmdline.lower() or "psycopg" in cmdline.lower():
                        cpu_usage = proc.cpu_percent(interval=0.5)
                        if cpu_usage < 0.1 and age > 180:  # 3 minutes idle
                            is_likely_deadlocked = True
                            deadlock_reason = "database_connection_hang"

                    # General stuck process detection
                    if age > self.config["stuck_process_time"] and not is_likely_deadlocked:
                        # Check if it's actually doing something
                        cpu_usage = proc.cpu_percent(interval=1.0)
                        if cpu_usage < 0.1:  # Less than 0.1% CPU - probably stuck
                            is_likely_deadlocked = True
                            deadlock_reason = "general_stuck_process"

                    if is_likely_deadlocked:
                        stuck.append(
                            {
                                "pid": proc.pid,
                                "name": proc.name(),
                                "status": "likely_stuck",
                                "age_seconds": age,
                                "cpu_percent": cpu_usage,
                                "reason": deadlock_reason,
                                "cmdline": cmdline[:100],
                            }
                        )
                        logger.warning(
                            f"🔴 Detected stuck process: PID {proc.pid} ({deadlock_reason}), "
                            f"Age: {age/60:.1f}m, CPU: {cpu_usage:.1f}%"
                        )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return stuck

    def _find_zombie_processes(self) -> List[Dict]:
        """Find zombie processes"""
        zombies = []

        for proc in psutil.process_iter(["pid", "name", "status", "ppid"]):
            try:
                if proc.status() == psutil.STATUS_ZOMBIE:
                    zombies.append({"pid": proc.pid, "name": proc.name(), "ppid": proc.ppid()})
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return zombies

    def _find_jarvis_processes(self) -> List[Dict]:
        """Find all JARVIS-related processes"""
        jarvis_procs = []

        for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
            try:
                if self._is_jarvis_process(proc):
                    jarvis_procs.append(
                        {
                            "pid": proc.pid,
                            "name": proc.name(),
                            "cmdline": " ".join(proc.cmdline()[:3]),
                            "age_seconds": time.time() - proc.create_time(),
                            "cpu_percent": proc.cpu_percent(interval=0.1),
                            "memory_mb": proc.memory_info().rss // (1024 * 1024),
                        }
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return jarvis_procs

    def _is_jarvis_process(self, proc: psutil.Process) -> bool:
        """Intelligently determine if a process is JARVIS-related"""
        try:
            # Check process name
            proc_name = proc.name().lower()

            # Check command line
            cmdline = " ".join(proc.cmdline()).lower()

            # First check exclusions - IDE/editor processes should be excluded
            for excluded_pattern in self.config.get("jarvis_excluded_patterns", []):
                if excluded_pattern.lower() in proc_name or excluded_pattern.lower() in cmdline:
                    return False

            # Check if it's a generic python/Python process without JARVIS context
            if proc_name in ["python", "python3", "python3.11", "python3.12"] and not any(
                pattern in cmdline for pattern in ["jarvis", "main.py", "backend"]
            ):
                return False

            # Check working directory for JARVIS project
            try:
                cwd = proc.cwd()
                # Be more specific - must be JARVIS-AI-Agent directory, not just any dir with "jarvis"
                if "jarvis-ai-agent" in cwd.lower():
                    # It's in JARVIS directory, now check if it's actually JARVIS code
                    if any(
                        pattern.lower() in cmdline for pattern in self.config["jarvis_patterns"]
                    ):
                        return True
                    # Also check if it's running our Python files
                    if proc_name.startswith("python") and (
                        "main.py" in cmdline or "start_system.py" in cmdline
                    ):
                        return True
            except (psutil.AccessDenied, PermissionError, psutil.NoSuchProcess):
                pass

            # Dynamic pattern matching
            for pattern in self.config["jarvis_patterns"]:
                if pattern.lower() in proc_name or pattern.lower() in cmdline:
                    return True

            # Check if it's using JARVIS ports (with permission handling)
            try:
                for conn in proc.connections():
                    if conn.laddr.port in self.config["jarvis_port_patterns"]:
                        return True
            except (psutil.AccessDenied, PermissionError):
                # Can't check connections, but that's okay
                pass

            return False
        except:
            return False

    def calculate_cleanup_priority(self, proc_info: Dict) -> float:
        """Calculate cleanup priority score (higher = more likely to clean)"""
        score = 0.0

        # CPU usage factor
        if proc_info.get("cpu_percent", 0) > self.config["cpu_threshold_single"]:
            score += proc_info["cpu_percent"] / 100.0 * 0.3

        # Memory usage factor (enhanced)
        memory_mb = proc_info.get("memory_mb", 0)
        is_jarvis = proc_info.get("is_jarvis", False)

        # Use appropriate threshold
        mem_threshold = (
            self.config["memory_threshold_jarvis_process"]
            if is_jarvis
            else self.config["memory_threshold_single_process"]
        )

        if memory_mb > mem_threshold:
            # Higher score for processes significantly over threshold
            excess_ratio = memory_mb / mem_threshold
            score += min(excess_ratio * 0.3, 0.5)  # Cap at 0.5

        # Memory percentage factor
        if proc_info.get("memory_percent", 0) > 5:
            score += min(proc_info["memory_percent"] / 100.0 * 0.2, 0.3)

        # Age factor (older = higher priority)
        age_hours = proc_info.get("age_seconds", 0) / 3600
        if age_hours > 1:
            score += min(age_hours / 24, 1.0) * 0.15

        # Stuck/zombie factor
        if proc_info.get("status") in ["zombie", "likely_stuck"]:
            score += 0.5

        # JARVIS process factor (be more aggressive with our own processes)
        if is_jarvis:
            score += 0.25

        # Learn from patterns
        proc_name = proc_info.get("name", "")
        if proc_name in self.problem_patterns:
            score += self.problem_patterns[proc_name] * 0.15

        return min(score, 1.0)

    async def smart_cleanup(self, dry_run: bool = False) -> Dict[str, any]:
        """Perform intelligent cleanup of problematic processes"""
        logger.info("🧹 Starting intelligent process cleanup...")

        # First, handle code change cleanup
        code_cleanup = self.cleanup_old_instances_on_code_change()

        # Analyze system state
        state = self.analyze_system_state()

        cleanup_report = {
            "timestamp": datetime.now(),
            "system_state": {
                "cpu_percent": state["cpu_percent"],
                "memory_percent": state["memory_percent"],
            },
            "code_changes_cleanup": code_cleanup,
            "actions": [],
            "freed_resources": {"cpu_percent": 0, "memory_mb": 0},
        }

        # Build list of cleanup candidates
        candidates = []

        # Add high CPU processes
        for proc in state["high_cpu_processes"]:
            proc["reason"] = "high_cpu"
            proc["priority"] = self.calculate_cleanup_priority(proc)
            candidates.append(proc)

        # Add high memory processes
        for proc in state["high_memory_processes"]:
            proc["reason"] = "high_memory"
            proc["priority"] = self.calculate_cleanup_priority(proc)
            # Increase priority if system memory is critical
            if state["memory_percent"] > self.config["memory_threshold_critical"] * 100:
                proc["priority"] = min(proc["priority"] + 0.3, 1.0)
            candidates.append(proc)

        # Add stuck processes
        for proc in state["stuck_processes"]:
            proc["reason"] = "stuck"
            proc["priority"] = self.calculate_cleanup_priority(proc)
            candidates.append(proc)

        # Add zombies
        for proc in state["zombie_processes"]:
            proc["reason"] = "zombie"
            proc["priority"] = 1.0  # Always clean zombies
            candidates.append(proc)

        # Sort by priority
        candidates.sort(key=lambda x: x["priority"], reverse=True)

        # Cleanup high-priority processes
        for candidate in candidates:
            if candidate["priority"] < 0.3:
                continue  # Skip low priority

            # Skip protected processes - check both exact match and substring
            should_skip = False
            candidate_name_lower = candidate["name"].lower()

            # Exact match check
            if candidate["name"] in self.config["system_critical"]:
                should_skip = True

            # Substring match check (for processes like "Cursor Helper", "Code Helper", etc.)
            for protected in self.config["system_critical"]:
                if (
                    protected.lower() in candidate_name_lower
                    or candidate_name_lower in protected.lower()
                ):
                    should_skip = True
                    break

            if should_skip:
                continue

            action = {
                "pid": candidate["pid"],
                "name": candidate["name"],
                "reason": candidate["reason"],
                "priority": candidate["priority"],
                "action": "none",
                "success": False,
            }

            if not dry_run:
                try:
                    proc = psutil.Process(candidate["pid"])

                    # Try graceful termination first
                    logger.info(
                        f"Terminating {candidate['name']} (PID: {candidate['pid']}, Reason: {candidate['reason']})"
                    )
                    proc.terminate()

                    # Wait for graceful shutdown
                    try:
                        proc.wait(timeout=5)
                        action["action"] = "terminated"
                        action["success"] = True
                    except psutil.TimeoutExpired:
                        # Force kill if needed
                        logger.warning(
                            f"Force killing {candidate['name']} (PID: {candidate['pid']})"
                        )
                        proc.kill()
                        action["action"] = "killed"
                        action["success"] = True

                    # Estimate freed resources
                    cleanup_report["freed_resources"]["cpu_percent"] += candidate.get(
                        "cpu_percent", 0
                    )
                    cleanup_report["freed_resources"]["memory_mb"] += candidate.get("memory_mb", 0)

                    # Learn from this
                    self._update_problem_patterns(candidate["name"], True)

                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    action["error"] = str(e)
                    logger.error(f"Failed to cleanup {candidate['name']}: {e}")
            else:
                action["action"] = "would_terminate"

            cleanup_report["actions"].append(action)

        # Clean up orphaned JARVIS ports
        if not dry_run:
            self._cleanup_orphaned_ports()

            # Clean up IPC resources (semaphores, shared memory, message queues)
            ipc_cleaned = self._cleanup_ipc_resources()
            cleanup_report["ipc_resources_cleaned"] = ipc_cleaned

        # Save cleanup history for learning
        self.cleanup_history.append(cleanup_report)
        self._save_cleanup_history()

        # Log summary
        logger.info(f"Cleanup complete: {len(cleanup_report['actions'])} processes handled")
        if code_cleanup:
            logger.info(f"Code change cleanup: {len(code_cleanup)} old instances terminated")
        logger.info(
            f"Freed approximately {cleanup_report['freed_resources']['cpu_percent']:.1f}% CPU, "
            f"{cleanup_report['freed_resources']['memory_mb']}MB memory"
        )

        return cleanup_report

    def _cleanup_orphaned_ports(self):
        """Clean up ports that might be stuck from previous JARVIS runs"""
        # DISABLED: net_connections hangs on macOS due to security restrictions
        logger.info("Skipping orphaned ports cleanup (macOS compatibility)")
        return

        # Original code disabled (commented out to prevent hanging):
        """
        for port in self.config["jarvis_port_patterns"]:
            try:
                for conn in psutil.net_connections():
                    if conn.laddr.port == port and conn.status == "LISTEN":
                        try:
                            proc = psutil.Process(conn.pid)
                            # Only kill if it's been running for a while and is idle
                            if time.time() - proc.create_time() > 300:  # 5 minutes
                                if proc.cpu_percent(interval=0.1) < 0.1:
                                    logger.info(
                                        f"Cleaning up orphaned port {port} (PID: {proc.pid})"
                                    )
                                    proc.terminate()
                        except:
                            pass
            except:
                pass
        """

    def _get_ipc_resources(self) -> Dict[str, List[Dict]]:
        """
        Get all IPC resources (semaphores, shared memory) for current user.
        Uses ipcs command to detect orphaned resources.
        """
        if not self.config.get("enable_ipc_cleanup", True):
            return {"semaphores": [], "shared_memory": [], "message_queues": []}

        resources = {"semaphores": [], "shared_memory": [], "message_queues": []}

        try:
            # Get current user
            current_user = os.getenv("USER", "")
            if not current_user:
                return resources

            # Query semaphores
            result = subprocess.run(["ipcs", "-s"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if current_user in line:
                        parts = line.split()
                        if len(parts) >= 2 and parts[0].startswith("0x"):
                            resources["semaphores"].append(
                                {
                                    "key": parts[0],
                                    "id": parts[1],
                                    "owner": parts[2] if len(parts) > 2 else current_user,
                                }
                            )

            # Query shared memory
            result = subprocess.run(["ipcs", "-m"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if current_user in line:
                        parts = line.split()
                        if len(parts) >= 2 and parts[0].startswith("0x"):
                            resources["shared_memory"].append(
                                {
                                    "key": parts[0],
                                    "id": parts[1],
                                    "owner": parts[2] if len(parts) > 2 else current_user,
                                    "size": parts[4] if len(parts) > 4 else "0",
                                }
                            )

            # Query message queues
            result = subprocess.run(["ipcs", "-q"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if current_user in line:
                        parts = line.split()
                        if len(parts) >= 2 and parts[0].startswith("0x"):
                            resources["message_queues"].append(
                                {
                                    "key": parts[0],
                                    "id": parts[1],
                                    "owner": parts[2] if len(parts) > 2 else current_user,
                                }
                            )

        except subprocess.TimeoutExpired:
            logger.warning("Timeout querying IPC resources")
        except FileNotFoundError:
            logger.debug("ipcs command not available on this system")
        except Exception as e:
            logger.error(f"Error querying IPC resources: {e}")

        return resources

    def _cleanup_ipc_resources(self) -> Dict[str, int]:
        """
        Clean up orphaned IPC resources (semaphores, shared memory, message queues).
        Only removes resources from current user that appear to be orphaned.
        """
        if not self.config.get("enable_ipc_cleanup", True):
            return {"semaphores": 0, "shared_memory": 0, "message_queues": 0}

        cleaned = {"semaphores": 0, "shared_memory": 0, "message_queues": 0}

        try:
            current_user = os.getenv("USER", "")
            if not current_user:
                return cleaned

            resources = self._get_ipc_resources()

            # Clean up semaphores
            for sem in resources["semaphores"]:
                try:
                    # Verify it's owned by current user and try to remove
                    result = subprocess.run(
                        ["ipcrm", "-s", sem["id"]], capture_output=True, text=True, timeout=2
                    )
                    if result.returncode == 0:
                        cleaned["semaphores"] += 1
                        logger.debug(f"Removed orphaned semaphore {sem['id']}")
                except subprocess.TimeoutExpired:
                    logger.warning(f"Timeout removing semaphore {sem['id']}")
                except Exception as e:
                    logger.debug(f"Could not remove semaphore {sem['id']}: {e}")

            # Clean up shared memory
            for shm in resources["shared_memory"]:
                try:
                    result = subprocess.run(
                        ["ipcrm", "-m", shm["id"]], capture_output=True, text=True, timeout=2
                    )
                    if result.returncode == 0:
                        cleaned["shared_memory"] += 1
                        logger.debug(
                            f"Removed orphaned shared memory {shm['id']} ({shm.get('size', '0')} bytes)"
                        )
                except subprocess.TimeoutExpired:
                    logger.warning(f"Timeout removing shared memory {shm['id']}")
                except Exception as e:
                    logger.debug(f"Could not remove shared memory {shm['id']}: {e}")

            # Clean up message queues
            for mq in resources["message_queues"]:
                try:
                    result = subprocess.run(
                        ["ipcrm", "-q", mq["id"]], capture_output=True, text=True, timeout=2
                    )
                    if result.returncode == 0:
                        cleaned["message_queues"] += 1
                        logger.debug(f"Removed orphaned message queue {mq['id']}")
                except subprocess.TimeoutExpired:
                    logger.warning(f"Timeout removing message queue {mq['id']}")
                except Exception as e:
                    logger.debug(f"Could not remove message queue {mq['id']}: {e}")

            # Log summary if anything was cleaned
            total_cleaned = sum(cleaned.values())
            if total_cleaned > 0:
                logger.info(
                    f"🧹 Cleaned {total_cleaned} orphaned IPC resources: "
                    f"{cleaned['semaphores']} semaphores, {cleaned['shared_memory']} shared memory segments, "
                    f"{cleaned['message_queues']} message queues"
                )
        except FileNotFoundError:
            logger.debug("ipcrm command not available on this system")
        except Exception as e:
            logger.error(f"Error cleaning up IPC resources: {e}")

        return cleaned

    def _update_problem_patterns(self, process_name: str, was_problematic: bool):
        """Learn from cleanup actions"""
        if process_name not in self.problem_patterns:
            self.problem_patterns[process_name] = 0.0

        # Exponential moving average
        alpha = 0.3
        self.problem_patterns[process_name] = (
            alpha * (1.0 if was_problematic else 0.0)
            + (1 - alpha) * self.problem_patterns[process_name]
        )

    def _save_cleanup_history(self):
        """Save cleanup history for learning"""
        history_file = Path.home() / ".jarvis" / "cleanup_history.json"
        history_file.parent.mkdir(exist_ok=True)

        # Keep only last 100 entries
        recent_history = self.cleanup_history[-100:]

        try:
            with open(history_file, "w") as f:
                json.dump(recent_history, f, default=str, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cleanup history: {e}")

    def get_cleanup_recommendations(self) -> List[str]:
        """Get recommendations based on current system state (macOS-aware)"""
        state = self.analyze_system_state()
        recommendations = []

        if state["cpu_percent"] > self.config["cpu_threshold_system"]:
            recommendations.append(
                f"System CPU is high ({state['cpu_percent']:.1f}%). Consider closing unnecessary applications."
            )

        # Use available memory instead of percentage (macOS-aware)
        # macOS typically shows 70-90% usage due to caching - this is normal!
        available_gb = state["memory_available_mb"] / 1024.0

        if available_gb < 0.5:  # Less than 500MB available
            recommendations.append(
                f"⚠️ CRITICAL: Very low available memory ({available_gb:.1f}GB). "
                f"Immediate action required!"
            )
        elif available_gb < 1.0:  # Less than 1GB available
            recommendations.append(
                f"⚡ WARNING: Low available memory ({available_gb:.1f}GB). "
                f"Consider closing applications."
            )
        elif available_gb < 2.0:  # Less than 2GB available
            recommendations.append(
                f"Memory is getting low ({available_gb:.1f}GB available). "
                f"Monitor for potential issues."
            )

        if len(state["zombie_processes"]) > 0:
            recommendations.append(
                f"Found {len(state['zombie_processes'])} zombie processes that should be cleaned."
            )

        if len(state["stuck_processes"]) > 0:
            recommendations.append(
                f"Found {len(state['stuck_processes'])} potentially stuck processes."
            )

        old_jarvis = [p for p in state["jarvis_processes"] if p["age_seconds"] > 3600]
        if old_jarvis:
            recommendations.append(
                f"Found {len(old_jarvis)} old JARVIS processes that may be stuck."
            )

        # Check for code changes
        if self._detect_code_changes():
            recommendations.append(
                "⚠️ CODE CHANGES DETECTED: Old JARVIS instances should be terminated!"
            )

        # Check for orphaned VMs
        orphaned_vms = self.vm_manager.get_orphaned_sessions()
        if orphaned_vms:
            recommendations.append(
                f"🌐 Found {len(orphaned_vms)} orphaned GCP VMs from dead sessions - should be cleaned up!"
            )

        # Check for stale VMs
        stale_vms = self.vm_manager.get_stale_sessions(max_age_hours=12.0)
        if stale_vms:
            recommendations.append(
                f"⏰ Found {len(stale_vms)} stale GCP VMs (>12 hours old) - consider cleanup"
            )

        # Report active VM count
        active_vms = self.vm_manager.get_active_vm_count()
        total_vms = self.vm_manager.get_vm_count()
        if total_vms > 0:
            recommendations.append(
                f"📊 GCP VM Status: {active_vms} active, {total_vms - active_vms} orphaned/stale"
            )

        return recommendations

    def _load_cleanup_history(self):
        """Load cleanup history from disk"""
        history_file = Path.home() / ".jarvis" / "cleanup_history.json"
        if history_file.exists():
            try:
                with open(history_file, "r") as f:
                    self.cleanup_history = json.load(f)
                    # Rebuild problem patterns from history
                    for entry in self.cleanup_history[-50:]:  # Last 50 entries
                        for action in entry.get("actions", []):
                            if action.get("success"):
                                self._update_problem_patterns(action.get("name", ""), True)
            except Exception as e:
                logger.error(f"Failed to load cleanup history: {e}")

    def cleanup_old_jarvis_processes(self, max_age_hours: float = 12.0) -> List[Dict]:
        """
        Specifically clean up old JARVIS processes that have been running too long

        Args:
            max_age_hours: Maximum age in hours before considering a JARVIS process stale

        Returns:
            List of cleaned up processes
        """
        cleaned = []
        current_time = time.time()

        for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
            try:
                if self._is_jarvis_process(proc):
                    age_hours = (current_time - proc.create_time()) / 3600

                    if age_hours > max_age_hours:
                        # Check if it's the current main process
                        cmdline = " ".join(proc.cmdline())
                        if "main.py" in cmdline:
                            # This is likely an old JARVIS main process
                            logger.warning(
                                f"Found stale JARVIS process (PID: {proc.pid}, "
                                f"Age: {age_hours:.1f} hours)"
                            )

                            try:
                                proc.terminate()
                                proc.wait(timeout=5)
                                cleaned.append(
                                    {
                                        "pid": proc.pid,
                                        "name": proc.name(),
                                        "age_hours": age_hours,
                                        "status": "terminated",
                                    }
                                )
                                logger.info(f"Terminated old JARVIS process {proc.pid}")
                            except psutil.TimeoutExpired:
                                proc.kill()
                                cleaned.append(
                                    {
                                        "pid": proc.pid,
                                        "name": proc.name(),
                                        "age_hours": age_hours,
                                        "status": "killed",
                                    }
                                )
                                logger.warning(f"Force killed old JARVIS process {proc.pid}")
                            except Exception as e:
                                logger.error(f"Failed to clean up PID {proc.pid}: {e}")

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return cleaned

    def get_jarvis_process_age(self) -> Optional[float]:
        """Get the age of the main JARVIS process in hours"""
        for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
            try:
                if self._is_jarvis_process(proc):
                    cmdline = " ".join(proc.cmdline())
                    if "main.py" in cmdline:
                        age_hours = (time.time() - proc.create_time()) / 3600
                        return age_hours
            except:  # nosec B112
                continue
        return None

    def _cleanup_cloudsql_connections(self) -> Dict[str, int]:
        """
        Clean up orphaned CloudSQL connections by:
        1. Terminating JARVIS processes with database connections
        2. Triggering singleton connection manager shutdown (if available)

        Returns:
            Dictionary with cleanup statistics
        """
        cleaned = {
            "processes_terminated": 0,
            "cloud_sql_proxy_restarted": False,
            "connection_manager_shutdown": False,
        }

        logger.info("🔌 Cleaning up CloudSQL connections...")

        # Step 1: Find and terminate JARVIS processes with database connections
        for proc in psutil.process_iter(["pid", "name", "cmdline", "connections"]):
            try:
                if self._is_jarvis_process(proc):
                    cmdline = " ".join(proc.cmdline())

                    # Check if process has connections to CloudSQL proxy port (5432)
                    has_db_connection = False
                    try:
                        for conn in proc.connections():
                            if conn.laddr.port == 5432 or conn.raddr.port == 5432:
                                has_db_connection = True
                                break
                    except (psutil.AccessDenied, AttributeError):
                        # Can't check connections, but if it's a backend process, assume it has DB connections
                        if "backend" in cmdline.lower() or "main.py" in cmdline:
                            has_db_connection = True

                    if has_db_connection:
                        logger.info(f"Terminating process with DB connection: PID {proc.pid}")
                        try:
                            proc.terminate()
                            proc.wait(timeout=3)
                            cleaned["processes_terminated"] += 1
                        except psutil.TimeoutExpired:
                            proc.kill()
                            cleaned["processes_terminated"] += 1
                        except psutil.NoSuchProcess:
                            pass

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Step 2: Trigger singleton connection manager shutdown (if available)
        try:
            from intelligence.cloud_sql_connection_manager import get_connection_manager

            manager = get_connection_manager()
            if manager.is_initialized:
                logger.info("🔌 Shutting down singleton CloudSQL connection manager...")
                # Use asyncio to run async shutdown
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(manager.shutdown())
                    else:
                        asyncio.run(manager.shutdown())
                    cleaned["connection_manager_shutdown"] = True
                except RuntimeError:
                    # No event loop, create one
                    asyncio.run(manager.shutdown())
                    cleaned["connection_manager_shutdown"] = True
                except Exception as e:
                    logger.warning(f"Failed to shutdown connection manager: {e}")
        except ImportError:
            logger.debug("Connection manager not available (expected during startup)")
        except Exception as e:
            logger.warning(f"Error shutting down connection manager: {e}")

        # Step 3: Give CloudSQL time to release connections
        if cleaned["processes_terminated"] > 0 or cleaned["connection_manager_shutdown"]:
            logger.info(f"⏳ Waiting 5 seconds for CloudSQL to release connections...")
            time.sleep(5)

        return cleaned

    def kill_hanging_start_system_processes(self, timeout_minutes: float = 5.0) -> List[Dict]:
        """
        Kill hanging start_system.py processes that have been running too long.

        A start_system.py process should complete within a few minutes. If it's been
        running longer than timeout_minutes, it's likely hung during initialization.

        Args:
            timeout_minutes: Kill processes running longer than this (default: 5 minutes)

        Returns:
            List of killed process info
        """
        killed = []
        timeout_seconds = timeout_minutes * 60
        current_time = time.time()

        for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
            try:
                cmdline = " ".join(proc.cmdline())

                # Check if it's start_system.py
                if "start_system.py" in cmdline and "python" in proc.name().lower():
                    # Calculate how long it's been running
                    age_seconds = current_time - proc.create_time()

                    if age_seconds > timeout_seconds:
                        logger.warning(
                            f"⏱️  Found hanging start_system.py (PID: {proc.pid}, "
                            f"running for {age_seconds/60:.1f} minutes)"
                        )

                        # Try graceful termination first
                        try:
                            proc.terminate()
                            try:
                                proc.wait(timeout=3)
                                killed.append({
                                    "pid": proc.pid,
                                    "age_minutes": age_seconds / 60,
                                    "method": "SIGTERM"
                                })
                                logger.info(f"✅ Terminated hanging start_system.py (PID: {proc.pid})")
                            except psutil.TimeoutExpired:
                                # Force kill if graceful fails
                                proc.kill()
                                killed.append({
                                    "pid": proc.pid,
                                    "age_minutes": age_seconds / 60,
                                    "method": "SIGKILL"
                                })
                                logger.info(f"✅ Force-killed hanging start_system.py (PID: {proc.pid})")
                        except psutil.NoSuchProcess:
                            pass  # Already gone

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return killed

    def emergency_cleanup_all_jarvis(self, force_kill: bool = False) -> Dict[str, Any]:
        """
        Emergency cleanup - kill ALL JARVIS-related processes and clean up resources.
        Use this when JARVIS has segfaulted or is in a bad state.
        Includes GCP VM cleanup for all sessions from this machine.
        Enhanced with CloudSQL connection cleanup.

        Args:
            force_kill: If True, skip graceful termination and go straight to SIGKILL

        Returns:
            Dictionary with cleanup results
        """
        logger.warning("🚨 EMERGENCY CLEANUP: Killing all JARVIS processes and VMs...")

        results = {
            "processes_killed": [],
            "ports_freed": [],
            "ipc_cleaned": {},
            "vms_deleted": [],
            "vm_errors": [],
            "cloudsql_cleanup": {},
            "ml_engine_shutdown": None,
            "hanging_start_system": [],
            "errors": [],
        }

        # Step 0: Kill any hanging start_system.py processes first
        try:
            hanging = self.kill_hanging_start_system_processes(timeout_minutes=5.0)
            results["hanging_start_system"] = hanging
            if hanging:
                logger.info(f"🧹 Killed {len(hanging)} hanging start_system.py process(es)")
        except Exception as e:
            logger.error(f"Failed to kill hanging start_system.py: {e}")

        # Step 1: Find and kill all JARVIS processes
        for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
            try:
                # Check if it's any kind of JARVIS-related process
                cmdline = " ".join(proc.cmdline()).lower()
                proc_name = proc.name().lower()

                # Look for any JARVIS patterns in command line or process name
                is_jarvis_related = False
                for pattern in self.config["jarvis_patterns"]:
                    if pattern.lower() in cmdline or pattern.lower() in proc_name:
                        # Check it's not an excluded process (IDE, etc)
                        is_excluded = any(
                            excl.lower() in cmdline or excl.lower() in proc_name
                            for excl in self.config["jarvis_excluded_patterns"]
                        )
                        if not is_excluded:
                            is_jarvis_related = True
                            break

                # Also check if it's a Python process running in JARVIS directory
                if "python" in proc_name and "jarvis-ai-agent" in cmdline:
                    is_jarvis_related = True

                # Check if it's a Node process for JARVIS frontend
                if ("node" in proc_name or "npm" in proc_name) and (
                    "jarvis" in cmdline or "localhost:3000" in cmdline
                ):
                    is_jarvis_related = True

                if is_jarvis_related:
                    logger.info(f"Killing JARVIS process: {proc.name()} (PID: {proc.pid})")
                    try:
                        if force_kill:
                            proc.kill()  # SIGKILL immediately
                        else:
                            proc.terminate()  # Try SIGTERM first
                            try:
                                proc.wait(timeout=2)
                            except psutil.TimeoutExpired:
                                proc.kill()  # Then SIGKILL if needed

                        results["processes_killed"].append(
                            {"pid": proc.pid, "name": proc.name(), "cmdline": cmdline[:100]}
                        )
                    except psutil.NoSuchProcess:
                        # Process already gone, that's fine
                        pass
                    except Exception as e:
                        results["errors"].append(f"Failed to kill PID {proc.pid}: {e}")

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Step 2: Force-free all JARVIS ports
        for port in self.config["jarvis_port_patterns"]:
            try:
                # Find any process using this port
                result = subprocess.run(
                    ["lsof", "-ti", f":{port}"], capture_output=True, text=True, timeout=2
                )
                if result.stdout.strip():
                    pids = result.stdout.strip().split("\n")
                    for pid_str in pids:
                        if pid_str:
                            try:
                                pid = int(pid_str)
                                os.kill(pid, signal.SIGKILL)
                                results["ports_freed"].append(port)
                                logger.info(f"Freed port {port} by killing PID {pid}")
                            except (ValueError, ProcessLookupError):
                                pass
            except Exception as e:
                logger.debug(f"Could not check/free port {port}: {e}")

        # Step 3: Clean up ALL IPC resources (semaphores, shared memory)
        try:
            # More aggressive IPC cleanup - remove ALL semaphores and shared memory for current user
            current_user = os.getenv("USER", "")
            if current_user:
                # Clean all semaphores
                result = subprocess.run(["ipcs", "-s"], capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if current_user in line:
                            parts = line.split()
                            if len(parts) >= 2:
                                sem_id = parts[1]
                                try:
                                    subprocess.run(["ipcrm", "-s", sem_id], timeout=1)
                                    results["ipc_cleaned"]["semaphores"] = (
                                        results["ipc_cleaned"].get("semaphores", 0) + 1
                                    )
                                except:
                                    pass

                # Clean all shared memory
                result = subprocess.run(["ipcs", "-m"], capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if current_user in line:
                            parts = line.split()
                            if len(parts) >= 2:
                                shm_id = parts[1]
                                try:
                                    subprocess.run(["ipcrm", "-m", shm_id], timeout=1)
                                    results["ipc_cleaned"]["shared_memory"] = (
                                        results["ipc_cleaned"].get("shared_memory", 0) + 1
                                    )
                                except:
                                    pass

        except Exception as e:
            logger.error(f"Failed to clean IPC resources: {e}")

        # Step 4: Clean up any leftover Python multiprocessing resources
        try:
            # Clean up resource tracker warnings
            import multiprocessing

            multiprocessing.set_start_method("spawn", force=True)
        except:
            pass

        # Step 5: Clean up CloudSQL connections
        try:
            cloudsql_cleanup = self._cleanup_cloudsql_connections()
            results["cloudsql_cleanup"] = cloudsql_cleanup
            if cloudsql_cleanup["processes_terminated"] > 0:
                logger.info(
                    f"🔌 Cleaned up {cloudsql_cleanup['processes_terminated']} processes with DB connections"
                )
        except Exception as e:
            logger.error(f"Failed to clean CloudSQL connections: {e}")

        # Step 5.5: Shutdown ML Continuous Learning Engine
        try:
            logger.info("🧠 Shutting down ML Continuous Learning Engine...")
            from voice_unlock.continuous_learning_engine import shutdown_learning_engine

            # Use asyncio to run async shutdown
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(shutdown_learning_engine())
                else:
                    asyncio.run(shutdown_learning_engine())
                logger.info("✅ ML Learning Engine shutdown complete (models saved)")
                results["ml_engine_shutdown"] = True
            except RuntimeError:
                # No event loop, create one
                asyncio.run(shutdown_learning_engine())
                logger.info("✅ ML Learning Engine shutdown complete (models saved)")
                results["ml_engine_shutdown"] = True
            except Exception as e:
                logger.warning(f"Failed to shutdown ML Learning Engine: {e}")
                results["ml_engine_shutdown"] = False
        except ImportError:
            logger.debug("ML Learning Engine not available (expected during startup)")
            results["ml_engine_shutdown"] = None
        except Exception as e:
            logger.error(f"Error during ML Learning Engine cleanup: {e}")
            results["ml_engine_shutdown"] = False

        # Step 6: Remove code state file to force fresh start
        if self.code_state_file.exists():
            try:
                self.code_state_file.unlink()
                logger.info("Removed code state file for fresh start")
            except:
                pass

        # Step 7: Clean up ALL GCP VMs for this machine (synchronous)
        logger.info("🌐 Cleaning up all GCP VMs from this machine...")
        try:
            # Get all sessions for this machine
            all_sessions = self.vm_manager.get_all_sessions()
            current_hostname = socket.gethostname()

            local_sessions = [s for s in all_sessions if s.get("hostname") == current_hostname]

            if local_sessions:
                logger.warning(f"Found {len(local_sessions)} VMs to delete")
                for session in local_sessions:
                    vm_id = session.get("vm_id")
                    zone = session.get("zone", self.vm_manager.default_zone)

                    if vm_id:
                        try:
                            success = self.vm_manager.delete_vm_sync(vm_id, zone)
                            if success:
                                results["vms_deleted"].append(vm_id)
                        except Exception as e:
                            error_msg = f"Failed to delete VM {vm_id}: {e}"
                            results["vm_errors"].append(error_msg)
                            logger.error(error_msg)

                # Clear VM registry
                try:
                    if self.vm_manager.vm_registry_file.exists():
                        self.vm_manager.vm_registry_file.unlink()
                        logger.info("Cleared VM registry")
                except Exception as e:
                    logger.error(f"Failed to clear VM registry: {e}")
            else:
                logger.info("No VMs found for this machine")

        except Exception as e:
            error_msg = f"Error during VM cleanup: {e}"
            results["vm_errors"].append(error_msg)
            logger.error(error_msg)

        # Log summary
        logger.info(f"🧹 Emergency cleanup complete:")
        logger.info(f"  • Killed {len(results['processes_killed'])} processes")
        logger.info(f"  • Freed {len(results['ports_freed'])} ports")
        if results["ipc_cleaned"]:
            logger.info(f"  • Cleaned {sum(results['ipc_cleaned'].values())} IPC resources")
        if results["cloudsql_cleanup"].get("processes_terminated", 0) > 0:
            logger.info(
                f"  • Terminated {results['cloudsql_cleanup']['processes_terminated']} processes with DB connections"
            )
        if results["ml_engine_shutdown"] is True:
            logger.info(f"  • ML Learning Engine shutdown complete")
        elif results["ml_engine_shutdown"] is False:
            logger.warning(f"  • ML Learning Engine shutdown failed")
        if results["vms_deleted"]:
            logger.info(f"  • Deleted {len(results['vms_deleted'])} GCP VMs")
        if results["vm_errors"]:
            logger.warning(f"  • {len(results['vm_errors'])} VM cleanup errors")
        if results["errors"]:
            logger.warning(f"  • {len(results['errors'])} errors occurred")

        return results

    def check_for_segfault_recovery(self) -> bool:
        # DISABLED: This causes infinite loops on macOS
        return False

    def check_for_segfault_recovery_disabled(self) -> bool:
        """
        Check if we need to recover from a segfault or crash.
        Returns True if recovery actions were taken.
        """
        recovery_needed = False

        # Check for leaked semaphores (sign of ungraceful shutdown)
        try:
            result = subprocess.run(["ipcs", "-s"], capture_output=True, text=True, timeout=2)
            if "leaked semaphore" in result.stdout.lower() or result.returncode != 0:
                recovery_needed = True
                logger.warning("Detected leaked semaphores - likely segfault or crash")
        except:
            pass

        # Check for zombie Python processes
        for proc in psutil.process_iter(["pid", "name", "status"]):
            try:
                if proc.status() == psutil.STATUS_ZOMBIE and "python" in proc.name().lower():
                    recovery_needed = True
                    logger.warning(f"Found zombie Python process: PID {proc.pid}")
                    break
            except:  # nosec B112
                continue

        # Check for stale lock files or PIDs
        jarvis_home = Path.home() / ".jarvis"
        if jarvis_home.exists():
            # Look for PID files older than 5 minutes
            for pid_file in jarvis_home.glob("*.pid"):
                try:
                    if (time.time() - pid_file.stat().st_mtime) > 300:  # 5 minutes
                        with open(pid_file, "r") as f:
                            old_pid = int(f.read().strip())
                        # Check if that PID is still running
                        if not psutil.pid_exists(old_pid):
                            recovery_needed = True
                            logger.warning(f"Found stale PID file: {pid_file}")
                            pid_file.unlink()  # Remove stale PID file
                except:
                    pass

        if recovery_needed:
            logger.warning("🔧 Initiating crash recovery...")
            # Perform emergency cleanup
            self.emergency_cleanup_all_jarvis()

        return recovery_needed


# Convenience functions for integration
async def cleanup_system_for_jarvis(dry_run: bool = False) -> Dict[str, any]:
    """
    Main entry point for cleaning up system before JARVIS starts.
    Includes orphaned VM cleanup.
    """
    manager = ProcessCleanupManager()

    # Check for crash recovery first
    if manager.check_for_segfault_recovery():
        logger.info("Performed crash recovery cleanup")

    # Always check for code changes and clean up old instances
    manager.cleanup_old_instances_on_code_change()

    # Clean up orphaned VMs (async)
    logger.info("🌐 Checking for orphaned GCP VMs...")
    vm_report = await manager.vm_manager.cleanup_orphaned_vms()
    if vm_report["vms_deleted"]:
        logger.info(f"Cleaned up {len(vm_report['vms_deleted'])} orphaned VMs")

    return await manager.smart_cleanup(dry_run=dry_run)


def get_system_recommendations() -> List[str]:
    """Get recommendations for system optimization"""
    manager = ProcessCleanupManager()
    return manager.get_cleanup_recommendations()


def ensure_fresh_jarvis_instance():
    """
    Ensure JARVIS is running fresh code. Call this at startup.
    Returns True if it's safe to start, False if another instance should be used.
    Includes orphaned VM cleanup (synchronous version).
    ALWAYS clears Python cache to guarantee fresh code loads.
    """
    manager = ProcessCleanupManager()

    # ALWAYS clear Python cache at startup to ensure fresh code
    logger.info("🔄 Ensuring fresh code by clearing Python cache at startup...")
    manager._clear_python_cache()

    # Check for crash recovery first
    if manager.check_for_segfault_recovery():
        logger.info("Performed crash recovery - safe to start fresh")
        return True

    # Clean up old instances if code has changed (includes VM cleanup)
    cleaned = manager.cleanup_old_instances_on_code_change()
    if cleaned:
        logger.info(f"Cleaned {len(cleaned)} old instances due to code changes")

    # Clean up orphaned VMs (synchronous version for startup)
    logger.info("🌐 Checking for orphaned GCP VMs...")
    orphaned = manager.vm_manager.get_orphaned_sessions()
    if orphaned:
        logger.warning(f"Found {len(orphaned)} orphaned VM sessions - cleaning up synchronously")
        for session in orphaned:
            vm_id = session.get("vm_id")
            zone = session.get("zone", manager.vm_manager.default_zone)
            if vm_id:
                manager.vm_manager.delete_vm_sync(vm_id, zone)
        manager.vm_manager._remove_orphaned_from_registry(orphaned)

    # Ensure single instance
    return manager.ensure_single_instance()


def prevent_multiple_jarvis_instances():
    """
    Comprehensive check to prevent multiple JARVIS instances from running.
    This is the main function to call at startup.
    ALWAYS clears Python cache to guarantee fresh code loads.

    Returns:
        Tuple[bool, str]: (can_start, message)
        - can_start: True if it's safe to start JARVIS
        - message: Human-readable status message
    """
    manager = ProcessCleanupManager()

    try:
        # Step 1: ALWAYS clear Python cache at startup to ensure fresh code
        logger.info("🔄 Ensuring fresh code by clearing Python cache at startup...")
        manager._clear_python_cache()

        # DISABLED: Check for crash recovery (causes loops on macOS)
        # if manager.check_for_segfault_recovery():
        #     return True, "System recovered from crash - safe to start fresh"

        # Step 2: Check for code changes and clean up old instances
        cleaned = manager.cleanup_old_instances_on_code_change()
        if cleaned:
            return True, f"Cleaned {len(cleaned)} old instances due to code changes - safe to start"

        # Step 3: Check for existing JARVIS processes
        jarvis_processes = manager._find_jarvis_processes()
        current_pid = os.getpid()
        other_processes = [p for p in jarvis_processes if p["pid"] != current_pid]

        if other_processes:
            # Check if any are using the target port
            target_port = int(os.getenv("BACKEND_PORT", "8000"))
            port_conflict = False

            for proc_info in other_processes:
                try:
                    proc = psutil.Process(proc_info["pid"])
                    for conn in proc.connections():
                        if conn.laddr.port == target_port and conn.status == "LISTEN":
                            port_conflict = True
                            break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            if port_conflict:
                return (
                    False,
                    f"JARVIS instance already running on port {target_port}. Use --emergency-cleanup to force restart.",
                )
            else:
                return (
                    True,
                    f"Found {len(other_processes)} other JARVIS processes but no port conflict - safe to start",
                )

        # Step 4: Final port check
        if not manager.ensure_single_instance():
            return False, "Port conflict detected. Another JARVIS instance may be running."

        return True, "No conflicts detected - safe to start JARVIS"

    except Exception as e:
        logger.error(f"Error checking for multiple instances: {e}")
        return False, f"Error during startup check: {e}"


def emergency_cleanup(force: bool = False):
    """
    Perform emergency cleanup of all JARVIS processes and GCP VMs.
    Use this when JARVIS is stuck or has segfaulted.

    Args:
        force: If True, use SIGKILL immediately instead of trying graceful shutdown

    Returns:
        Cleanup results dictionary
    """
    manager = ProcessCleanupManager()
    results = manager.emergency_cleanup_all_jarvis(force_kill=force)

    # Print summary to console
    print("\n🚨 Emergency Cleanup Results:")
    print(f"  • Processes killed: {len(results['processes_killed'])}")
    print(f"  • Ports freed: {len(results['ports_freed'])}")
    if results["ipc_cleaned"]:
        print(f"  • IPC resources cleaned: {sum(results['ipc_cleaned'].values())}")
    if results["ml_engine_shutdown"] is True:
        print(f"  • ML Learning Engine shutdown: ✅")
    elif results["ml_engine_shutdown"] is False:
        print(f"  • ML Learning Engine shutdown: ⚠️ Failed")
    if results["vms_deleted"]:
        print(f"  • GCP VMs deleted: {len(results['vms_deleted'])}")
        for vm_id in results["vms_deleted"][:5]:  # Show first 5 VMs
            print(f"    - {vm_id}")
        if len(results["vms_deleted"]) > 5:
            print(f"    ... and {len(results['vms_deleted']) - 5} more")
    if results["vm_errors"]:
        print(f"  • ⚠️ VM cleanup errors: {len(results['vm_errors'])}")
    if results["errors"]:
        print(f"  • ⚠️ Errors: {len(results['errors'])}")
        for error in results["errors"][:3]:  # Show first 3 errors
            print(f"    - {error}")

    return results


if __name__ == "__main__":
    # Test the cleanup manager
    import asyncio
    import sys

    async def test():
        print("🔍 Analyzing system state...")

        manager = ProcessCleanupManager()

        # Check for segfault recovery
        if manager.check_for_segfault_recovery():
            print("\n🔧 Performed crash recovery cleanup!")
            print("System should be clean for fresh start.\n")

        # Check for code changes
        if manager._detect_code_changes():
            print("\n⚠️  CODE CHANGES DETECTED!")
            print("Old JARVIS instances will be terminated.\n")

        state = manager.analyze_system_state()

        print(f"\n📊 System State:")
        print(f"  CPU: {state['cpu_percent']:.1f}%")
        print(f"  Memory: {state['memory_percent']:.1f}%")
        print(f"  High CPU processes: {len(state['high_cpu_processes'])}")
        print(f"  High memory processes: {len(state['high_memory_processes'])}")
        print(f"  Stuck processes: {len(state['stuck_processes'])}")
        print(f"  Zombie processes: {len(state['zombie_processes'])}")
        print(f"  JARVIS processes: {len(state['jarvis_processes'])}")

        print("\n💡 Recommendations:")
        for rec in manager.get_cleanup_recommendations():
            print(f"  • {rec}")

        print("\n🧹 Performing dry run cleanup...")
        report = await manager.smart_cleanup(dry_run=True)

        print(f"\nWould clean {len(report['actions'])} processes:")
        for action in report["actions"][:5]:  # Show first 5
            print(f"  • {action['name']} (PID: {action['pid']}) - {action['reason']}")

        if len(report["actions"]) > 5:
            print(f"  ... and {len(report['actions']) - 5} more")

        if report.get("code_changes_cleanup"):
            print(f"\nCode change cleanup: {len(report['code_changes_cleanup'])} old instances")

        # Ask if user wants to perform emergency cleanup
        if len(sys.argv) > 1 and sys.argv[1] == "--emergency":
            print("\n⚠️  EMERGENCY CLEANUP MODE")
            response = input("Perform emergency cleanup? (y/N): ")
            if response.lower() == "y":
                results = emergency_cleanup(force=True)
                print("\n✅ Emergency cleanup completed!")

    asyncio.run(test())
