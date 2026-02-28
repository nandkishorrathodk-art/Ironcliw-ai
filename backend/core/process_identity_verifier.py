"""
Process Identity Verifier - Intelligent Process Ownership Detection
===================================================================

Fixes the ROOT CAUSE of port allocation race conditions by:
1. Identifying process by PID, name, command line, and executable path
2. Detecting if a process is a Ironcliw/Trinity component vs external process
3. Cross-referencing with service registry for known registrations
4. Supporting process adoption for healthy existing instances
5. Providing forensic-level process analysis for debugging

Architecture:
    +------------------------------------------------------------------+
    |  ProcessIdentityVerifier                                         |
    |  +-- ProcessFingerprint (PID, name, cmdline, exe, ports, env)    |
    |  +-- ComponentMatcher (pattern-based Trinity component detection)|
    |  +-- RegistryCorrelator (cross-ref with service registry)        |
    |  +-- HealthProber (verify component health for adoption)         |
    |  +-- ProcessAncestryAnalyzer (parent/child relationships)        |
    +------------------------------------------------------------------+

Key Innovation:
- Before falling back to a different port, we now verify if the process
  holding the port is actually a Ironcliw component (which we can adopt)
  vs an external process (which requires fallback)

Author: Ironcliw Trinity v94.0 - Process Identity Verification
"""

from __future__ import annotations

import asyncio
import enum
import hashlib
import json
import logging
import os
import re
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# Psutil Import (Graceful Degradation)
# =============================================================================

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None  # type: ignore
    PSUTIL_AVAILABLE = False


# =============================================================================
# Enums and Types
# =============================================================================

class ComponentType(enum.Enum):
    """Known Trinity/Ironcliw component types."""
    Ironcliw_BODY = "jarvis_body"          # Main Ironcliw agent (port 8010)
    Ironcliw_PRIME = "jarvis_prime"        # Local LLM inference (port 8000)
    REACTOR_CORE = "reactor_core"        # Training pipeline (port 8090)
    CODING_COUNCIL = "coding_council"    # Coding assistance
    NEURAL_MESH = "neural_mesh"          # Multi-agent coordination
    UNKNOWN_Ironcliw = "unknown_jarvis"    # Ironcliw-related but type unknown
    EXTERNAL = "external"                # Not a Ironcliw component


class ProcessVerificationResult(enum.Enum):
    """Result of process identity verification."""
    Ironcliw_HEALTHY = "jarvis_healthy"           # Healthy Ironcliw component (can adopt)
    Ironcliw_UNHEALTHY = "jarvis_unhealthy"       # Unhealthy Ironcliw (should restart)
    Ironcliw_STARTING = "jarvis_starting"         # Ironcliw still initializing
    EXTERNAL_PROCESS = "external_process"       # Not Ironcliw (use fallback)
    SYSTEM_PROCESS = "system_process"           # System/protected process
    RELATED_PROCESS = "related_process"         # Parent/child/sibling
    PROCESS_GONE = "process_gone"               # Process no longer exists
    PERMISSION_DENIED = "permission_denied"     # Cannot inspect process


# =============================================================================
# Process Fingerprint
# =============================================================================

@dataclass
class ProcessFingerprint:
    """
    Comprehensive fingerprint of a process for identity verification.

    Contains all information needed to determine if a process is a
    Ironcliw component vs external process.
    """
    pid: int
    name: str = ""
    cmdline: str = ""
    exe_path: str = ""
    cwd: str = ""
    parent_pid: Optional[int] = None
    parent_name: str = ""
    create_time: float = 0.0
    status: str = ""
    username: str = ""
    memory_mb: float = 0.0

    # Computed fields
    component_type: ComponentType = ComponentType.EXTERNAL
    confidence: float = 0.0  # 0.0 - 1.0
    match_reasons: List[str] = field(default_factory=list)

    # Port information
    ports_listening: List[int] = field(default_factory=list)

    # Environment variables (selective)
    env_hints: Dict[str, str] = field(default_factory=dict)

    # Timing
    fingerprint_time_ms: float = 0.0

    @property
    def identity_hash(self) -> str:
        """Generate a hash for this process identity."""
        data = f"{self.pid}:{self.exe_path}:{self.create_time}"
        return hashlib.md5(data.encode()).hexdigest()[:12]

    @property
    def is_jarvis_component(self) -> bool:
        """Check if this is any Ironcliw component."""
        return self.component_type not in (
            ComponentType.EXTERNAL,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "pid": self.pid,
            "name": self.name,
            "exe_path": self.exe_path,
            "cwd": self.cwd,
            "component_type": self.component_type.value,
            "confidence": round(self.confidence, 3),
            "match_reasons": self.match_reasons,
            "ports_listening": self.ports_listening,
            "fingerprint_time_ms": round(self.fingerprint_time_ms, 1),
        }


@dataclass
class VerificationResult:
    """Complete result of process identity verification."""
    result: ProcessVerificationResult
    fingerprint: Optional[ProcessFingerprint] = None
    is_adoptable: bool = False
    should_use_fallback: bool = False
    health_check_passed: bool = False

    # Timing
    total_time_ms: float = 0.0

    # Error information
    error: Optional[str] = None

    # Recommendations
    recommendation: str = ""


# =============================================================================
# Component Pattern Matcher
# =============================================================================

class ComponentPatternMatcher:
    """
    Pattern-based matching for identifying Ironcliw components.

    Uses multiple signals:
    1. Process name patterns
    2. Command line patterns
    3. Executable path patterns
    4. Port associations
    5. Environment variable hints
    """

    # Patterns for each component type
    PATTERNS: Dict[ComponentType, Dict[str, Any]] = {
        ComponentType.Ironcliw_BODY: {
            "name": ["python", "python3", "uvicorn"],
            "cmdline": [
                "backend.main",
                "backend/main.py",
                "start_system.py",
                "jarvis.*backend",
                "uvicorn backend.main:app",
                "8010",  # Default port
            ],
            "env_vars": [
                "Ironcliw_",
                "TRINITY_",
            ],
            "ports": [8010, 8011, 8012, 8013],
        },
        ComponentType.Ironcliw_PRIME: {
            "name": ["python", "python3", "llama", "uvicorn"],
            "cmdline": [
                "jarvis.prime",
                "jarvis_prime",
                "jarvis-prime",
                "j-prime",
                "j_prime",
                "llama.*server",
                "gguf",
                "8000",  # Default port
                "8004", "8005", "8006",  # Fallback ports
            ],
            "env_vars": [
                "Ironcliw_PRIME_",
                "LLAMA_",
            ],
            "ports": [8000, 8004, 8005, 8006],
        },
        ComponentType.REACTOR_CORE: {
            "name": ["python", "python3", "uvicorn"],
            "cmdline": [
                "reactor.core",
                "reactor_core",
                "reactor-core",
                "training.*pipeline",
                "8090",  # Default port
                "8091", "8092", "8093",  # Fallback ports
            ],
            "env_vars": [
                "REACTOR_CORE_",
                "TRAINING_",
            ],
            "ports": [8090, 8091, 8092, 8093],
        },
        ComponentType.CODING_COUNCIL: {
            "name": ["python", "python3"],
            "cmdline": [
                "coding.council",
                "coding_council",
                "coding-council",
            ],
            "env_vars": [
                "CODING_COUNCIL_",
            ],
            "ports": [8100, 8101],
        },
        ComponentType.NEURAL_MESH: {
            "name": ["python", "python3"],
            "cmdline": [
                "neural.mesh",
                "neural_mesh",
                "neural-mesh",
            ],
            "env_vars": [
                "NEURAL_MESH_",
            ],
            "ports": [],
        },
    }

    # System process indicators
    SYSTEM_INDICATORS: Set[str] = {
        "kernel_task", "launchd", "windowserver", "loginwindow",
        "coreaudiod", "coreservicesd", "mds", "mds_stores",
        "mdworker", "hidd", "opendirectoryd", "configd",
        "securityd", "trustd", "apsd", "powerd", "diskmanagementd",
        "kextd", "notifyd", "syslogd", "fseventsd", "diskarbitrationd",
    }

    SYSTEM_PATHS: List[str] = [
        "/System/Library",
        "/usr/libexec",
        "/Library/Apple",
        "/usr/sbin",
        "/sbin",
    ]

    def match(self, fingerprint: ProcessFingerprint) -> Tuple[ComponentType, float, List[str]]:
        """
        Match a process fingerprint to a component type.

        Returns:
            Tuple of (component_type, confidence, match_reasons)
        """
        # Check for system process first
        if self._is_system_process(fingerprint):
            return (ComponentType.EXTERNAL, 1.0, ["system_process"])

        best_match = ComponentType.EXTERNAL
        best_confidence = 0.0
        best_reasons: List[str] = []

        for component_type, patterns in self.PATTERNS.items():
            confidence, reasons = self._match_patterns(fingerprint, patterns)

            if confidence > best_confidence:
                best_match = component_type
                best_confidence = confidence
                best_reasons = reasons

        # If we have partial Ironcliw matches but couldn't identify specific type
        if best_confidence > 0.2 and best_confidence < 0.6:
            # Check for generic Ironcliw indicators
            generic_reasons = self._check_generic_jarvis(fingerprint)
            if generic_reasons:
                if best_confidence < 0.5:
                    return (ComponentType.UNKNOWN_Ironcliw, 0.5, generic_reasons)

        return (best_match, best_confidence, best_reasons)

    def _match_patterns(
        self,
        fingerprint: ProcessFingerprint,
        patterns: Dict[str, List[str]],
    ) -> Tuple[float, List[str]]:
        """Match fingerprint against a set of patterns."""
        score = 0.0
        reasons: List[str] = []
        max_score = 4.0  # Maximum possible score

        name_lower = fingerprint.name.lower()
        cmdline_lower = fingerprint.cmdline.lower()

        # Name match (0.5 weight)
        for pattern in patterns.get("name", []):
            if pattern.lower() in name_lower:
                score += 0.5
                reasons.append(f"name_match:{pattern}")
                break

        # Command line match (1.5 weight - most important)
        for pattern in patterns.get("cmdline", []):
            pattern_lower = pattern.lower()
            if re.search(pattern_lower, cmdline_lower):
                score += 1.5
                reasons.append(f"cmdline_match:{pattern}")
                break

        # Port match (1.0 weight)
        for port in patterns.get("ports", []):
            if port in fingerprint.ports_listening:
                score += 1.0
                reasons.append(f"port_match:{port}")
                break

        # Environment variable match (1.0 weight)
        for env_prefix in patterns.get("env_vars", []):
            for env_key in fingerprint.env_hints.keys():
                if env_key.startswith(env_prefix):
                    score += 1.0
                    reasons.append(f"env_match:{env_prefix}")
                    break

        confidence = score / max_score
        return (min(confidence, 1.0), reasons)

    def _is_system_process(self, fingerprint: ProcessFingerprint) -> bool:
        """Check if process is a system process."""
        name_lower = fingerprint.name.lower()

        # Check system process names
        if name_lower in self.SYSTEM_INDICATORS:
            return True

        # Check system paths
        for path in self.SYSTEM_PATHS:
            if fingerprint.exe_path.startswith(path):
                return True

        # Low PID heuristic (but not definitive)
        if fingerprint.pid < 100:
            return True

        return False

    def _check_generic_jarvis(self, fingerprint: ProcessFingerprint) -> List[str]:
        """Check for generic Ironcliw indicators."""
        reasons: List[str] = []
        cmdline_lower = fingerprint.cmdline.lower()

        generic_patterns = [
            "jarvis",
            ".jarvis",
            "trinity",
            "claude.*vision",
            "voice.*unlock",
            "neural.*mesh",
        ]

        for pattern in generic_patterns:
            if re.search(pattern, cmdline_lower):
                reasons.append(f"generic_jarvis:{pattern}")

        # Check environment
        for key in fingerprint.env_hints.keys():
            if "Ironcliw" in key or "TRINITY" in key:
                reasons.append(f"env_hint:{key}")

        return reasons


# =============================================================================
# Process Identity Verifier
# =============================================================================

class ProcessIdentityVerifier:
    """
    Main class for verifying process identity on ports.

    This class answers the critical question:
    "Is the process on this port a Ironcliw component we can adopt,
    or an external process requiring fallback?"
    """

    def __init__(
        self,
        service_registry: Optional[Any] = None,  # ServiceRegistry
        health_probe_timeout: float = 5.0,
        health_probe_retries: int = 3,
        cache_ttl_seconds: float = 30.0,
    ):
        self.service_registry = service_registry
        self.health_probe_timeout = health_probe_timeout
        self.health_probe_retries = health_probe_retries
        self.cache_ttl = cache_ttl_seconds

        self.pattern_matcher = ComponentPatternMatcher()

        # Cache for fingerprints (avoid repeated expensive lookups)
        self._fingerprint_cache: Dict[int, Tuple[ProcessFingerprint, float]] = {}
        self._cache_lock = asyncio.Lock()

        logger.info("[ProcessIdentityVerifier] Initialized")

    async def verify_port_owner(
        self,
        port: int,
        expected_component: Optional[ComponentType] = None,
    ) -> VerificationResult:
        """
        Verify the identity of the process owning a port.

        Args:
            port: The port to check
            expected_component: Optional expected component type for matching

        Returns:
            VerificationResult with full analysis
        """
        start_time = time.perf_counter()

        # Step 1: Get PID on port
        pid = await self._get_pid_on_port(port)
        if pid is None:
            return VerificationResult(
                result=ProcessVerificationResult.PROCESS_GONE,
                is_adoptable=False,
                should_use_fallback=False,
                total_time_ms=(time.perf_counter() - start_time) * 1000,
                recommendation="Port is free, proceed with binding",
            )

        # Step 2: Get process fingerprint
        fingerprint = await self._get_fingerprint(pid, port)
        if fingerprint is None:
            return VerificationResult(
                result=ProcessVerificationResult.PERMISSION_DENIED,
                is_adoptable=False,
                should_use_fallback=True,
                total_time_ms=(time.perf_counter() - start_time) * 1000,
                error=f"Cannot inspect PID {pid}",
                recommendation="Use fallback port (cannot verify process)",
            )

        # Step 3: Match against patterns
        component_type, confidence, reasons = self.pattern_matcher.match(fingerprint)
        fingerprint.component_type = component_type
        fingerprint.confidence = confidence
        fingerprint.match_reasons = reasons

        # Step 4: Cross-reference with service registry
        registry_match = await self._check_service_registry(pid, port)
        if registry_match:
            fingerprint.match_reasons.append(f"registry_match:{registry_match}")
            fingerprint.confidence = min(fingerprint.confidence + 0.2, 1.0)

        # Step 5: Determine result based on component type
        result = await self._determine_result(fingerprint, port, expected_component)
        result.total_time_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            f"[ProcessIdentityVerifier] Port {port} verification: "
            f"result={result.result.value}, component={fingerprint.component_type.value}, "
            f"confidence={fingerprint.confidence:.2f}, adoptable={result.is_adoptable}, "
            f"took {result.total_time_ms:.1f}ms"
        )

        return result

    async def _get_fingerprint(
        self,
        pid: int,
        port: int,
    ) -> Optional[ProcessFingerprint]:
        """
        Get comprehensive fingerprint for a process.

        Uses caching to avoid repeated expensive lookups.
        """
        async with self._cache_lock:
            # Check cache
            if pid in self._fingerprint_cache:
                cached, timestamp = self._fingerprint_cache[pid]
                if time.time() - timestamp < self.cache_ttl:
                    return cached

        start_time = time.perf_counter()
        fingerprint = ProcessFingerprint(pid=pid)

        if PSUTIL_AVAILABLE:
            try:
                proc = psutil.Process(pid)

                fingerprint.name = proc.name()
                fingerprint.status = proc.status()
                fingerprint.create_time = proc.create_time()

                try:
                    fingerprint.cmdline = " ".join(proc.cmdline())
                except Exception:
                    fingerprint.cmdline = ""

                try:
                    fingerprint.exe_path = proc.exe()
                except Exception:
                    fingerprint.exe_path = ""

                try:
                    fingerprint.cwd = proc.cwd()
                except Exception:
                    fingerprint.cwd = ""

                try:
                    parent = proc.parent()
                    if parent:
                        fingerprint.parent_pid = parent.pid
                        fingerprint.parent_name = parent.name()
                except Exception:
                    pass

                try:
                    fingerprint.username = proc.username()
                except Exception:
                    fingerprint.username = ""

                try:
                    mem_info = proc.memory_info()
                    fingerprint.memory_mb = mem_info.rss / (1024 * 1024)
                except Exception:
                    pass

                # Get listening ports
                try:
                    # Use net_connections if available (psutil >= 5.3.0)
                    if hasattr(proc, 'net_connections'):
                        connections = proc.net_connections(kind='inet')
                    else:
                        connections = proc.connections(kind='inet')
                    fingerprint.ports_listening = [
                        conn.laddr.port for conn in connections
                        if conn.status == 'LISTEN'
                    ]
                except Exception:
                    fingerprint.ports_listening = [port]  # At least the port we're checking

                # Get selective environment variables
                try:
                    env = proc.environ()
                    for key, value in env.items():
                        if any(hint in key for hint in ["Ironcliw", "TRINITY", "REACTOR", "PRIME", "NEURAL"]):
                            fingerprint.env_hints[key] = value[:100]  # Truncate long values
                except Exception:
                    pass

            except Exception as psutil_error:
                error_name = type(psutil_error).__name__
                if "NoSuchProcess" in error_name:
                    return None
                elif "AccessDenied" in error_name:
                    # Can't inspect, return minimal fingerprint
                    fingerprint.status = "access_denied"
                else:
                    logger.debug(f"[ProcessIdentityVerifier] psutil error: {psutil_error}")

        else:
            # Fallback without psutil
            fingerprint = await self._fingerprint_via_shell(pid, port)
            if fingerprint is None:
                return None

        fingerprint.fingerprint_time_ms = (time.perf_counter() - start_time) * 1000

        # Update cache
        async with self._cache_lock:
            self._fingerprint_cache[pid] = (fingerprint, time.time())

        return fingerprint

    async def _fingerprint_via_shell(
        self,
        pid: int,
        port: int,
    ) -> Optional[ProcessFingerprint]:
        """Fallback fingerprinting using shell commands."""
        fingerprint = ProcessFingerprint(pid=pid)

        try:
            # Get process info via ps
            proc = await asyncio.create_subprocess_exec(
                "ps", "-p", str(pid), "-o", "comm=,ppid=,stat=,user=",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)

            if stdout:
                parts = stdout.decode().strip().split()
                if len(parts) >= 1:
                    fingerprint.name = parts[0]
                if len(parts) >= 2:
                    try:
                        fingerprint.parent_pid = int(parts[1])
                    except ValueError:
                        pass
                if len(parts) >= 3:
                    fingerprint.status = parts[2]
                if len(parts) >= 4:
                    fingerprint.username = parts[3]

            # Get full command line
            proc = await asyncio.create_subprocess_exec(
                "ps", "-p", str(pid), "-o", "command=",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)

            if stdout:
                fingerprint.cmdline = stdout.decode().strip()

            fingerprint.ports_listening = [port]
            return fingerprint

        except Exception as e:
            logger.debug(f"[ProcessIdentityVerifier] Shell fingerprint failed: {e}")
            return None

    async def _check_service_registry(
        self,
        pid: int,
        port: int,
    ) -> Optional[str]:
        """Check if process is registered in service registry."""
        if not self.service_registry:
            return None

        try:
            # Try to find service by PID or port
            services = await self.service_registry.list_services()

            for service in services:
                if service.pid == pid:
                    return f"pid:{service.service_name}"
                if service.port == port:
                    return f"port:{service.service_name}"

            return None

        except Exception as e:
            logger.debug(f"[ProcessIdentityVerifier] Registry check failed: {e}")
            return None

    async def _determine_result(
        self,
        fingerprint: ProcessFingerprint,
        port: int,
        expected_component: Optional[ComponentType],
    ) -> VerificationResult:
        """Determine the final verification result."""

        # Not a Ironcliw component - use fallback
        if not fingerprint.is_jarvis_component:
            return VerificationResult(
                result=ProcessVerificationResult.EXTERNAL_PROCESS,
                fingerprint=fingerprint,
                is_adoptable=False,
                should_use_fallback=True,
                recommendation=f"External process (PID {fingerprint.pid}: {fingerprint.name}) "
                               f"holding port {port}. Use fallback port.",
            )

        # Check for related process (can't kill)
        current_pid = os.getpid()
        if fingerprint.parent_pid == current_pid or fingerprint.pid == current_pid:
            return VerificationResult(
                result=ProcessVerificationResult.RELATED_PROCESS,
                fingerprint=fingerprint,
                is_adoptable=False,
                should_use_fallback=True,
                recommendation="Process is related to current process. Use fallback port.",
            )

        # It's a Ironcliw component - check if healthy for adoption
        health_passed = await self._probe_health(port)

        if health_passed:
            # Healthy Ironcliw component - can adopt
            return VerificationResult(
                result=ProcessVerificationResult.Ironcliw_HEALTHY,
                fingerprint=fingerprint,
                is_adoptable=True,
                should_use_fallback=False,
                health_check_passed=True,
                recommendation=f"Healthy {fingerprint.component_type.value} found on port {port}. "
                               f"Can adopt existing instance (PID {fingerprint.pid}).",
            )

        # Ironcliw component but unhealthy - need to restart or fallback
        # Check if still starting up
        startup_indicators = [
            "initializing" in fingerprint.status.lower(),
            fingerprint.memory_mb < 50,  # Low memory might indicate startup
            time.time() - fingerprint.create_time < 60,  # Started less than 60s ago
        ]

        if any(startup_indicators):
            return VerificationResult(
                result=ProcessVerificationResult.Ironcliw_STARTING,
                fingerprint=fingerprint,
                is_adoptable=False,
                should_use_fallback=True,
                health_check_passed=False,
                recommendation=f"{fingerprint.component_type.value} still starting up. "
                               f"Use fallback port and retry later.",
            )

        return VerificationResult(
            result=ProcessVerificationResult.Ironcliw_UNHEALTHY,
            fingerprint=fingerprint,
            is_adoptable=False,
            should_use_fallback=False,  # Try to restart it
            health_check_passed=False,
            recommendation=f"Unhealthy {fingerprint.component_type.value} on port {port}. "
                           f"Consider terminating PID {fingerprint.pid} and restarting.",
        )

    async def _probe_health(self, port: int) -> bool:
        """
        Probe health endpoint with retries.

        Checks for READY status, not just HTTP 200.
        """
        try:
            import aiohttp
        except ImportError:
            logger.warning("[ProcessIdentityVerifier] aiohttp not available")
            return False

        # Check multiple health endpoint variants
        endpoints = [
            f"http://127.0.0.1:{port}/health/ready",
            f"http://127.0.0.1:{port}/health/live",
            f"http://127.0.0.1:{port}/health",
            f"http://127.0.0.1:{port}/health/ping",
        ]

        timeout_per_request = self.health_probe_timeout / max(self.health_probe_retries, 1)

        for attempt in range(self.health_probe_retries):
            for endpoint in endpoints:
                try:
                    timeout = aiohttp.ClientTimeout(total=timeout_per_request)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.get(endpoint) as resp:
                            if resp.status == 200:
                                try:
                                    data = await resp.json()
                                    status = data.get("status", "").lower()
                                    ready = data.get("ready", data.get("is_ready", True))

                                    # Check for explicit readiness
                                    if "ready" in endpoint:
                                        return ready is True

                                    # For other endpoints, check status
                                    if status in ("ok", "healthy", "running", "ready"):
                                        return True
                                except:
                                    # If we can't parse JSON but got 200, consider it healthy
                                    return True

                            # 503 Service Unavailable often means "not ready yet"
                            elif resp.status == 503:
                                continue  # Try next endpoint

                except asyncio.TimeoutError:
                    continue
                except aiohttp.ClientConnectorError:
                    continue
                except Exception:
                    continue

            # Backoff between retries
            if attempt < self.health_probe_retries - 1:
                await asyncio.sleep(0.3 * (attempt + 1))

        return False

    async def _get_pid_on_port(self, port: int) -> Optional[int]:
        """Get PID of process listening on a port."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "lsof", "-t", "-i", f":{port}", "-sTCP:LISTEN",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)

            if stdout:
                pids = stdout.decode().strip().split("\n")
                if pids and pids[0]:
                    return int(pids[0])

            return None

        except Exception:
            return None

    def clear_cache(self) -> None:
        """Clear the fingerprint cache."""
        self._fingerprint_cache.clear()


# =============================================================================
# Singleton Access
# =============================================================================

_verifier_instance: Optional[ProcessIdentityVerifier] = None


async def get_process_identity_verifier(
    service_registry: Optional[Any] = None,
    **kwargs,
) -> ProcessIdentityVerifier:
    """Get or create the global process identity verifier."""
    global _verifier_instance

    if _verifier_instance is None:
        _verifier_instance = ProcessIdentityVerifier(
            service_registry=service_registry,
            **kwargs,
        )

    return _verifier_instance


async def verify_port_owner(
    port: int,
    expected_component: Optional[ComponentType] = None,
) -> VerificationResult:
    """Convenience function to verify port owner."""
    verifier = await get_process_identity_verifier()
    return await verifier.verify_port_owner(port, expected_component)
