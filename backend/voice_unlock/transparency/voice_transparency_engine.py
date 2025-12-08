"""
Voice Transparency Engine v1.0
==============================

Provides comprehensive voice communication and transparency for JARVIS
voice authentication system. Enables debugging, decision tracing, and
intelligent verbal feedback.

Features:
- Decision trace logging with full reasoning chain
- Verbose mode for detailed spoken feedback
- Cloud infrastructure status reporting (Docker, GCP, VM Spot)
- Hypothesis explanation for borderline cases
- Real-time debugging output via voice
- Environment-variable driven configuration
- Fully async with observability hooks

Configuration (Environment Variables):
- JARVIS_TRANSPARENCY_ENABLED: Enable transparency engine (default: true)
- JARVIS_VERBOSE_MODE: Enable verbose spoken feedback (default: false)
- JARVIS_DEBUG_VOICE: Speak debug info during auth (default: false)
- JARVIS_TRACE_RETENTION_HOURS: How long to keep traces (default: 24)
- JARVIS_CLOUD_STATUS_ENABLED: Report cloud infra status (default: true)
- JARVIS_EXPLAIN_DECISIONS: Explain why decisions were made (default: true)
- JARVIS_ANNOUNCE_CONFIDENCE: Always announce confidence (default: borderline)
- JARVIS_ANNOUNCE_LATENCY: Announce processing time (default: false)
- JARVIS_ANNOUNCE_INFRASTRUCTURE: Mention cloud status (default: false)

Author: JARVIS AI System
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Union
from uuid import uuid4

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class TransparencyConfig:
    """Environment-driven configuration for voice transparency."""

    @staticmethod
    def is_enabled() -> bool:
        """Whether transparency engine is enabled."""
        return os.getenv("JARVIS_TRANSPARENCY_ENABLED", "true").lower() == "true"

    @staticmethod
    def verbose_mode() -> bool:
        """Whether to use verbose spoken feedback."""
        return os.getenv("JARVIS_VERBOSE_MODE", "false").lower() == "true"

    @staticmethod
    def debug_voice() -> bool:
        """Whether to speak debug information during authentication."""
        return os.getenv("JARVIS_DEBUG_VOICE", "false").lower() == "true"

    @staticmethod
    def trace_retention_hours() -> int:
        """How many hours to retain decision traces."""
        return int(os.getenv("JARVIS_TRACE_RETENTION_HOURS", "24"))

    @staticmethod
    def cloud_status_enabled() -> bool:
        """Whether to check and report cloud infrastructure status."""
        return os.getenv("JARVIS_CLOUD_STATUS_ENABLED", "true").lower() == "true"

    @staticmethod
    def explain_decisions() -> bool:
        """Whether to explain WHY decisions were made."""
        return os.getenv("JARVIS_EXPLAIN_DECISIONS", "true").lower() == "true"

    @staticmethod
    def announce_confidence() -> str:
        """When to announce confidence: always, never, borderline."""
        return os.getenv("JARVIS_ANNOUNCE_CONFIDENCE", "borderline")

    @staticmethod
    def announce_latency() -> bool:
        """Whether to announce processing latency."""
        return os.getenv("JARVIS_ANNOUNCE_LATENCY", "false").lower() == "true"

    @staticmethod
    def announce_infrastructure() -> bool:
        """Whether to mention cloud infrastructure in announcements."""
        return os.getenv("JARVIS_ANNOUNCE_INFRASTRUCTURE", "false").lower() == "true"

    @staticmethod
    def max_trace_history() -> int:
        """Maximum number of traces to keep in memory."""
        return int(os.getenv("JARVIS_MAX_TRACE_HISTORY", "100"))

    @staticmethod
    def gcp_project_id() -> Optional[str]:
        """GCP project ID for cloud status checks."""
        return os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")

    @staticmethod
    def docker_enabled() -> bool:
        """Whether running in Docker."""
        return os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER", "false").lower() == "true"

    @staticmethod
    def cloud_run_ecapa_url() -> Optional[str]:
        """GCP Cloud Run ECAPA service URL."""
        return os.getenv("CLOUD_RUN_ECAPA_URL")

    @staticmethod
    def cloud_run_region() -> str:
        """GCP Cloud Run region."""
        return os.getenv("CLOUD_RUN_REGION", "us-central1")

    @staticmethod
    def vm_spot_gpu_url() -> Optional[str]:
        """GCP VM Spot GPU instance URL."""
        return os.getenv("GCP_VM_SPOT_GPU_URL")

    @staticmethod
    def vm_spot_zone() -> str:
        """GCP VM Spot instance zone."""
        return os.getenv("GCP_VM_SPOT_ZONE", "us-central1-a")

    @staticmethod
    def local_ml_service_url() -> str:
        """Local ML service URL."""
        return os.getenv("LOCAL_ML_SERVICE_URL", "http://localhost:8001/health")

    @staticmethod
    def jarvis_backend_url() -> str:
        """JARVIS backend API URL."""
        return os.getenv("JARVIS_BACKEND_URL", "http://localhost:8000")


# =============================================================================
# ENUMS AND MODELS
# =============================================================================

class DecisionOutcome(str, Enum):
    """Possible authentication decision outcomes."""
    AUTHENTICATED = "authenticated"
    CHALLENGED = "challenged"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    ERROR = "error"


class ReasoningPhase(str, Enum):
    """Phases of the reasoning process."""
    PERCEPTION = "perception"
    AUDIO_ANALYSIS = "audio_analysis"
    ML_VERIFICATION = "ml_verification"
    PHYSICS_ANALYSIS = "physics_analysis"
    BEHAVIORAL_ANALYSIS = "behavioral_analysis"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    REASONING = "reasoning"
    BAYESIAN_FUSION = "bayesian_fusion"
    DECISION = "decision"
    RESPONSE = "response"


class InfrastructureStatus(str, Enum):
    """Cloud infrastructure status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass
class PhaseTrace:
    """Trace of a single reasoning phase."""
    phase: ReasoningPhase
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: float = 0.0
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    confidence_delta: float = 0.0
    notes: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def complete(self, outputs: Dict[str, Any], notes: List[str] = None):
        """Mark phase as complete."""
        self.completed_at = datetime.now()
        self.duration_ms = (self.completed_at - self.started_at).total_seconds() * 1000
        self.outputs = outputs
        if notes:
            self.notes.extend(notes)


@dataclass
class HypothesisTrace:
    """Trace of a hypothesis evaluation."""
    hypothesis_id: str
    category: str
    description: str
    prior_probability: float
    posterior_probability: float
    evidence_for: List[str] = field(default_factory=list)
    evidence_against: List[str] = field(default_factory=list)
    is_best: bool = False
    is_security_threat: bool = False


@dataclass
class InfrastructureTrace:
    """Trace of infrastructure status."""
    component: str
    status: InfrastructureStatus
    latency_ms: Optional[float] = None
    location: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionTrace:
    """
    Complete trace of an authentication decision.

    This provides full transparency into WHY a decision was made,
    including all phases, hypotheses, and reasoning.
    """
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    session_id: Optional[str] = None
    user_id: str = "owner"

    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    total_duration_ms: float = 0.0

    # Outcome
    outcome: DecisionOutcome = DecisionOutcome.ERROR
    final_confidence: float = 0.0
    speaker_name: Optional[str] = None

    # Phase traces
    phases: List[PhaseTrace] = field(default_factory=list)

    # Hypotheses evaluated
    hypotheses: List[HypothesisTrace] = field(default_factory=list)
    best_hypothesis: Optional[str] = None

    # Infrastructure
    infrastructure: List[InfrastructureTrace] = field(default_factory=list)

    # Confidence breakdown
    ml_confidence: float = 0.0
    physics_confidence: float = 0.0
    behavioral_confidence: float = 0.0
    contextual_confidence: float = 0.0
    bayesian_authentic_prob: float = 0.0

    # Security
    spoofing_detected: bool = False
    spoofing_type: Optional[str] = None
    security_flags: List[str] = field(default_factory=list)

    # Reasoning chain (for borderline cases)
    reasoning_chain: List[str] = field(default_factory=list)
    reasoning_conclusion: Optional[str] = None

    # Decision factors
    decision_factors: Dict[str, float] = field(default_factory=dict)
    threshold_used: float = 0.85
    margin_from_threshold: float = 0.0

    # Announcements generated
    primary_announcement: Optional[str] = None
    debug_announcement: Optional[str] = None
    retry_guidance: Optional[str] = None

    # Errors and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_phase(self, phase: ReasoningPhase, inputs: Dict[str, Any] = None) -> PhaseTrace:
        """Start tracing a new phase."""
        trace = PhaseTrace(
            phase=phase,
            started_at=datetime.now(),
            inputs=inputs or {}
        )
        self.phases.append(trace)
        return trace

    def add_hypothesis(
        self,
        category: str,
        description: str,
        prior: float,
        posterior: float,
        evidence_for: List[str] = None,
        evidence_against: List[str] = None,
        is_security_threat: bool = False
    ) -> HypothesisTrace:
        """Add a hypothesis evaluation trace."""
        trace = HypothesisTrace(
            hypothesis_id=str(uuid4()),
            category=category,
            description=description,
            prior_probability=prior,
            posterior_probability=posterior,
            evidence_for=evidence_for or [],
            evidence_against=evidence_against or [],
            is_security_threat=is_security_threat
        )
        self.hypotheses.append(trace)
        return trace

    def add_reasoning(self, thought: str):
        """Add a reasoning step to the chain."""
        self.reasoning_chain.append(thought)

    def complete(
        self,
        outcome: DecisionOutcome,
        confidence: float,
        speaker_name: Optional[str] = None
    ):
        """Complete the decision trace."""
        self.completed_at = datetime.now()
        self.total_duration_ms = (self.completed_at - self.started_at).total_seconds() * 1000
        self.outcome = outcome
        self.final_confidence = confidence
        self.speaker_name = speaker_name

        # Mark best hypothesis
        if self.hypotheses:
            best = max(self.hypotheses, key=lambda h: h.posterior_probability)
            best.is_best = True
            self.best_hypothesis = best.category

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary for API response."""
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "timing": {
                "started_at": self.started_at.isoformat(),
                "completed_at": self.completed_at.isoformat() if self.completed_at else None,
                "total_duration_ms": self.total_duration_ms,
            },
            "outcome": {
                "decision": self.outcome.value,
                "final_confidence": self.final_confidence,
                "speaker_name": self.speaker_name,
                "margin_from_threshold": self.margin_from_threshold,
            },
            "confidence_breakdown": {
                "ml_verification": self.ml_confidence,
                "physics_analysis": self.physics_confidence,
                "behavioral_analysis": self.behavioral_confidence,
                "contextual_analysis": self.contextual_confidence,
                "bayesian_fusion": self.bayesian_authentic_prob,
            },
            "phases": [
                {
                    "phase": p.phase.value,
                    "duration_ms": p.duration_ms,
                    "confidence_delta": p.confidence_delta,
                    "notes": p.notes,
                    "error": p.error,
                }
                for p in self.phases
            ],
            "hypotheses": [
                {
                    "category": h.category,
                    "description": h.description,
                    "prior": h.prior_probability,
                    "posterior": h.posterior_probability,
                    "evidence_for": h.evidence_for,
                    "evidence_against": h.evidence_against,
                    "is_best": h.is_best,
                    "is_threat": h.is_security_threat,
                }
                for h in self.hypotheses
            ],
            "best_hypothesis": self.best_hypothesis,
            "reasoning_chain": self.reasoning_chain,
            "reasoning_conclusion": self.reasoning_conclusion,
            "decision_factors": self.decision_factors,
            "security": {
                "spoofing_detected": self.spoofing_detected,
                "spoofing_type": self.spoofing_type,
                "flags": self.security_flags,
            },
            "infrastructure": [
                {
                    "component": i.component,
                    "status": i.status.value,
                    "latency_ms": i.latency_ms,
                    "location": i.location,
                }
                for i in self.infrastructure
            ],
            "announcements": {
                "primary": self.primary_announcement,
                "debug": self.debug_announcement,
                "retry_guidance": self.retry_guidance,
            },
            "errors": self.errors,
            "warnings": self.warnings,
        }

    def generate_summary(self) -> str:
        """Generate human-readable summary of the decision."""
        lines = [
            f"━━━ Authentication Decision Trace [{self.trace_id[:8]}] ━━━",
            f"Outcome: {self.outcome.value.upper()}",
            f"Confidence: {self.final_confidence:.1%} (threshold: {self.threshold_used:.1%})",
            f"Duration: {self.total_duration_ms:.0f}ms",
            "",
            "Confidence Breakdown:",
            f"  ML Verification: {self.ml_confidence:.1%}",
            f"  Physics Analysis: {self.physics_confidence:.1%}",
            f"  Behavioral Analysis: {self.behavioral_confidence:.1%}",
            f"  Bayesian Fusion: {self.bayesian_authentic_prob:.1%}",
        ]

        if self.best_hypothesis:
            lines.extend([
                "",
                f"Best Hypothesis: {self.best_hypothesis}",
            ])

        if self.reasoning_chain:
            lines.extend([
                "",
                "Reasoning Chain:",
            ])
            for i, thought in enumerate(self.reasoning_chain, 1):
                lines.append(f"  {i}. {thought}")

        if self.spoofing_detected:
            lines.extend([
                "",
                f"⚠️ SECURITY ALERT: {self.spoofing_type}",
            ])

        lines.append("━" * 50)
        return "\n".join(lines)


# =============================================================================
# VERBOSE ANNOUNCEMENT GENERATOR
# =============================================================================

class VerboseAnnouncementGenerator:
    """
    Generates verbose, transparent announcements for voice authentication.

    Provides detailed spoken feedback about:
    - What's happening during authentication
    - Why decisions were made
    - Confidence breakdown
    - Hypothesis explanations
    - Infrastructure status
    - Debugging information
    """

    def __init__(self):
        self._stats = {
            "announcements_generated": 0,
            "verbose_announcements": 0,
            "debug_announcements": 0,
        }

    def generate_progress_announcement(
        self,
        phase: ReasoningPhase,
        details: Dict[str, Any] = None
    ) -> Optional[str]:
        """Generate announcement for authentication progress (debug mode)."""
        if not TransparencyConfig.debug_voice():
            return None

        self._stats["debug_announcements"] += 1

        announcements = {
            ReasoningPhase.PERCEPTION: "Capturing audio...",
            ReasoningPhase.AUDIO_ANALYSIS: "Analyzing audio quality...",
            ReasoningPhase.ML_VERIFICATION: "Running voice verification...",
            ReasoningPhase.PHYSICS_ANALYSIS: "Performing physics analysis...",
            ReasoningPhase.BEHAVIORAL_ANALYSIS: "Checking behavioral patterns...",
            ReasoningPhase.HYPOTHESIS_GENERATION: "Evaluating possibilities...",
            ReasoningPhase.REASONING: "Reasoning through borderline case...",
            ReasoningPhase.BAYESIAN_FUSION: "Fusing confidence scores...",
            ReasoningPhase.DECISION: "Making decision...",
        }

        return announcements.get(phase)

    def generate_result_announcement(
        self,
        trace: DecisionTrace,
        verbose: bool = None
    ) -> str:
        """
        Generate result announcement with configurable verbosity.

        Args:
            trace: Complete decision trace
            verbose: Override verbose mode (uses config if None)

        Returns:
            Announcement string
        """
        self._stats["announcements_generated"] += 1

        if verbose is None:
            verbose = TransparencyConfig.verbose_mode()

        if verbose:
            self._stats["verbose_announcements"] += 1
            return self._generate_verbose_announcement(trace)
        else:
            return self._generate_standard_announcement(trace)

    def _generate_standard_announcement(self, trace: DecisionTrace) -> str:
        """Generate standard (non-verbose) announcement."""
        name = trace.speaker_name or "there"
        confidence = trace.final_confidence
        outcome = trace.outcome

        if outcome == DecisionOutcome.AUTHENTICATED:
            if confidence >= 0.92:
                return f"Voice verified, {name}. Unlocking now."
            elif confidence >= 0.85:
                return f"Voice verified, {name}. {int(confidence * 100)}% confidence. Unlocking."
            else:
                # Borderline - explain why we're proceeding
                if trace.best_hypothesis:
                    return self._explain_borderline_success(trace, name)
                return f"Voice match at {int(confidence * 100)}%, {name}. Proceeding with unlock."

        elif outcome == DecisionOutcome.CHALLENGED:
            return self._generate_challenge_announcement(trace)

        elif outcome == DecisionOutcome.REJECTED:
            if trace.spoofing_detected:
                return self._generate_spoofing_announcement(trace)
            return self._generate_rejection_announcement(trace)

        elif outcome == DecisionOutcome.ESCALATED:
            return "Unusual pattern detected. Please use an alternative authentication method."

        else:
            return "Authentication error. Please try again."

    def _generate_verbose_announcement(self, trace: DecisionTrace) -> str:
        """Generate verbose announcement with full transparency."""
        name = trace.speaker_name or "there"
        parts = []

        # 1. Primary outcome
        if trace.outcome == DecisionOutcome.AUTHENTICATED:
            parts.append(f"Voice verified, {name}.")
        elif trace.outcome == DecisionOutcome.CHALLENGED:
            parts.append(f"Voice verification is borderline, {name}.")
        elif trace.outcome == DecisionOutcome.REJECTED:
            parts.append("Voice verification failed.")

        # 2. Confidence breakdown (if configured)
        announce_confidence = TransparencyConfig.announce_confidence()
        should_announce_confidence = (
            announce_confidence == "always" or
            (announce_confidence == "borderline" and trace.final_confidence < 0.90)
        )

        if should_announce_confidence:
            parts.append(
                f"Overall confidence {int(trace.final_confidence * 100)}%, "
                f"with ML at {int(trace.ml_confidence * 100)}%, "
                f"physics at {int(trace.physics_confidence * 100)}%, "
                f"and behavioral at {int(trace.behavioral_confidence * 100)}%."
            )

        # 3. Explain decision if borderline
        if TransparencyConfig.explain_decisions() and trace.best_hypothesis:
            explanation = self._get_hypothesis_explanation(trace.best_hypothesis)
            if explanation:
                parts.append(explanation)

        # 4. Reasoning conclusion
        if trace.reasoning_conclusion:
            parts.append(trace.reasoning_conclusion)

        # 5. Latency (if configured)
        if TransparencyConfig.announce_latency():
            parts.append(f"Processing took {int(trace.total_duration_ms)} milliseconds.")

        # 6. Infrastructure status (if configured)
        if TransparencyConfig.announce_infrastructure() and trace.infrastructure:
            infra_status = self._summarize_infrastructure(trace.infrastructure)
            if infra_status:
                parts.append(infra_status)

        # 7. Security warnings
        if trace.spoofing_detected:
            parts.append(f"Security alert: {trace.spoofing_type} detected. This attempt has been logged.")

        # 8. Final action
        if trace.outcome == DecisionOutcome.AUTHENTICATED:
            parts.append("Unlocking now.")
        elif trace.retry_guidance:
            parts.append(trace.retry_guidance)

        return " ".join(parts)

    def _explain_borderline_success(self, trace: DecisionTrace, name: str) -> str:
        """Generate explanation for borderline authentication success."""
        hypothesis = trace.best_hypothesis
        confidence = int(trace.final_confidence * 100)

        explanations = {
            "background_noise": (
                f"Voice confidence is {confidence}% due to background noise, {name}, "
                f"but your behavioral patterns match perfectly. Unlocking."
            ),
            "sick_voice": (
                f"Your voice sounds different today, {name} - hope you're feeling okay. "
                f"But your speech patterns match, so I'm confident it's you. Unlocking."
            ),
            "tired_voice": (
                f"You sound tired, {name}. Voice is at {confidence}%, "
                f"but your patterns are consistent. Unlocking for you."
            ),
            "different_microphone": (
                f"I notice you're using a different microphone, {name}. "
                f"Voice match is {confidence}%, but behavioral check passed. Unlocking."
            ),
            "different_environment": (
                f"The acoustics are different here, {name}. "
                f"Confidence is {confidence}%, but context confirms it's you. Unlocking."
            ),
        }

        return explanations.get(
            hypothesis,
            f"Voice match at {confidence}%, {name}. Context confirms identity. Unlocking."
        )

    def _get_hypothesis_explanation(self, hypothesis: str) -> Optional[str]:
        """Get spoken explanation for a hypothesis."""
        explanations = {
            "background_noise": "Background noise is affecting audio quality.",
            "sick_voice": "Voice characteristics suggest possible illness.",
            "tired_voice": "Voice patterns indicate fatigue.",
            "different_microphone": "Different microphone detected.",
            "different_environment": "Unfamiliar acoustic environment.",
            "replay_attack": "Possible recording playback detected.",
            "synthetic_voice": "Synthetic voice characteristics detected.",
        }
        return explanations.get(hypothesis)

    def _summarize_infrastructure(self, infra: List[InfrastructureTrace]) -> Optional[str]:
        """Summarize infrastructure status for announcement."""
        healthy = sum(1 for i in infra if i.status == InfrastructureStatus.HEALTHY)
        total = len(infra)

        if healthy == total:
            return "All cloud services are healthy."
        elif healthy > 0:
            return f"{healthy} of {total} cloud services healthy."
        else:
            return "Running in offline mode."

    def _generate_challenge_announcement(self, trace: DecisionTrace) -> str:
        """Generate challenge announcement."""
        confidence = int(trace.final_confidence * 100)

        if trace.best_hypothesis == "background_noise":
            return (
                f"I'm having trouble hearing clearly due to background noise. "
                f"Confidence is at {confidence}%. "
                "Could you speak a bit louder or move somewhere quieter?"
            )

        return (
            f"Voice verification is at {confidence}%, below the threshold. "
            "Please try again, speaking clearly into the microphone."
        )

    def _generate_rejection_announcement(self, trace: DecisionTrace) -> str:
        """Generate rejection announcement."""
        confidence = int(trace.final_confidence * 100)

        if confidence < 40:
            return "I don't recognize this voice. Access denied."

        if trace.warnings:
            return f"Voice verification failed at {confidence}%. {trace.warnings[0]}"

        return f"Voice verification failed at {confidence}%. Please try again."

    def _generate_spoofing_announcement(self, trace: DecisionTrace) -> str:
        """Generate spoofing detection announcement."""
        spoof_type = trace.spoofing_type or "suspicious audio characteristics"

        return (
            f"Security alert: I detected {spoof_type} consistent with a recording "
            "rather than a live voice. Access denied. This attempt has been logged."
        )

    def generate_debug_announcement(self, trace: DecisionTrace) -> str:
        """
        Generate detailed debug announcement for troubleshooting.

        This provides maximum transparency about what happened.
        """
        self._stats["debug_announcements"] += 1

        parts = [
            f"Debug report for authentication {trace.trace_id[:8]}.",
            f"Outcome: {trace.outcome.value}.",
            f"Total confidence: {int(trace.final_confidence * 100)}%.",
        ]

        # Phase timings
        if trace.phases:
            slowest = max(trace.phases, key=lambda p: p.duration_ms)
            parts.append(
                f"Slowest phase was {slowest.phase.value} at {int(slowest.duration_ms)} milliseconds."
            )

        # Confidence components
        parts.append(
            f"ML confidence: {int(trace.ml_confidence * 100)}%. "
            f"Physics: {int(trace.physics_confidence * 100)}%. "
            f"Behavioral: {int(trace.behavioral_confidence * 100)}%."
        )

        # Best hypothesis
        if trace.best_hypothesis:
            parts.append(f"Best hypothesis: {trace.best_hypothesis}.")

        # Reasoning summary
        if trace.reasoning_chain:
            parts.append(f"Reasoning involved {len(trace.reasoning_chain)} steps.")

        # Errors
        if trace.errors:
            parts.append(f"Errors encountered: {len(trace.errors)}.")

        return " ".join(parts)


# =============================================================================
# INFRASTRUCTURE STATUS CHECKER
# =============================================================================

class InfrastructureStatusChecker:
    """
    Checks status of cloud infrastructure components.

    Supports:
    - Docker container status
    - JARVIS Backend API
    - Local ML services
    - GCP Cloud Run services (ECAPA, etc.)
    - GCP VM Spot GPU instances

    Environment Variables:
    - DOCKER_CONTAINER: Set to "true" if running in Docker
    - LOCAL_ML_SERVICE_URL: Local ML service health endpoint
    - CLOUD_RUN_ECAPA_URL: GCP Cloud Run ECAPA service URL
    - CLOUD_RUN_REGION: GCP Cloud Run region (default: us-central1)
    - GCP_VM_SPOT_GPU_URL: GCP VM Spot GPU instance URL
    - GCP_VM_SPOT_ZONE: GCP VM Spot zone (default: us-central1-a)
    - JARVIS_BACKEND_URL: JARVIS backend API URL
    """

    def __init__(self):
        self._cache: Dict[str, Tuple[InfrastructureTrace, datetime]] = {}
        self._cache_ttl_seconds = 30

    async def check_all(self) -> List[InfrastructureTrace]:
        """Check all infrastructure components."""
        if not TransparencyConfig.cloud_status_enabled():
            return []

        traces = []

        # Run checks in parallel
        checks = [
            self._check_docker(),
            self._check_jarvis_backend(),
            self._check_local_ml_service(),
            self._check_gcp_cloud_run(),
            self._check_gcp_vm_spot(),
        ]

        results = await asyncio.gather(*checks, return_exceptions=True)

        for result in results:
            if isinstance(result, InfrastructureTrace):
                traces.append(result)
            elif isinstance(result, list):
                traces.extend(result)
            elif isinstance(result, Exception):
                logger.debug(f"Infrastructure check exception: {result}")

        return traces

    async def _check_docker(self) -> InfrastructureTrace:
        """Check if running in Docker container."""
        is_docker = TransparencyConfig.docker_enabled()

        details = {
            "running_in_container": is_docker,
            "dockerenv_exists": os.path.exists("/.dockerenv"),
            "env_var_set": os.getenv("DOCKER_CONTAINER", "").lower() == "true",
        }

        return InfrastructureTrace(
            component="docker",
            status=InfrastructureStatus.HEALTHY if is_docker else InfrastructureStatus.UNAVAILABLE,
            details=details
        )

    async def _check_jarvis_backend(self) -> InfrastructureTrace:
        """Check JARVIS backend API status."""
        backend_url = TransparencyConfig.jarvis_backend_url()

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                start = time.time()
                async with session.get(f"{backend_url}/health", timeout=3) as resp:
                    latency = (time.time() - start) * 1000

                    if resp.status == 200:
                        try:
                            data = await resp.json()
                            return InfrastructureTrace(
                                component="jarvis_backend",
                                status=InfrastructureStatus.HEALTHY,
                                latency_ms=latency,
                                location="local",
                                details={
                                    "url": backend_url,
                                    "status": data.get("status", "unknown"),
                                }
                            )
                        except:
                            return InfrastructureTrace(
                                component="jarvis_backend",
                                status=InfrastructureStatus.HEALTHY,
                                latency_ms=latency,
                                location="local",
                                details={"url": backend_url}
                            )
                    else:
                        return InfrastructureTrace(
                            component="jarvis_backend",
                            status=InfrastructureStatus.DEGRADED,
                            latency_ms=latency,
                            location="local",
                            details={"url": backend_url, "http_status": resp.status}
                        )
        except Exception as e:
            logger.debug(f"JARVIS backend check failed: {e}")

        return InfrastructureTrace(
            component="jarvis_backend",
            status=InfrastructureStatus.UNAVAILABLE,
            location="local",
            details={"url": backend_url, "error": "Connection failed"}
        )

    async def _check_local_ml_service(self) -> InfrastructureTrace:
        """Check local ML service status."""
        ml_url = TransparencyConfig.local_ml_service_url()

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                start = time.time()
                async with session.get(ml_url, timeout=2) as resp:
                    latency = (time.time() - start) * 1000

                    if resp.status == 200:
                        return InfrastructureTrace(
                            component="local_ml_service",
                            status=InfrastructureStatus.HEALTHY,
                            latency_ms=latency,
                            location="local",
                            details={"url": ml_url}
                        )
        except Exception as e:
            logger.debug(f"Local ML service check failed: {e}")

        return InfrastructureTrace(
            component="local_ml_service",
            status=InfrastructureStatus.UNAVAILABLE,
            location="local",
            details={"url": ml_url}
        )

    async def _check_gcp_cloud_run(self) -> List[InfrastructureTrace]:
        """Check GCP Cloud Run services."""
        traces = []

        cloud_run_url = TransparencyConfig.cloud_run_ecapa_url()
        cloud_run_region = TransparencyConfig.cloud_run_region()

        if cloud_run_url:
            try:
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    start = time.time()
                    # Try health endpoint first, then root
                    health_url = f"{cloud_run_url}/health"
                    try:
                        async with session.get(health_url, timeout=5) as resp:
                            latency = (time.time() - start) * 1000

                            status = (
                                InfrastructureStatus.HEALTHY if resp.status == 200
                                else InfrastructureStatus.DEGRADED
                            )

                            # Try to get response data
                            details = {
                                "url": cloud_run_url,
                                "region": cloud_run_region,
                                "http_status": resp.status,
                            }
                            try:
                                data = await resp.json()
                                details["response"] = data
                            except:
                                pass

                            traces.append(InfrastructureTrace(
                                component="gcp_cloud_run_ecapa",
                                status=status,
                                latency_ms=latency,
                                location=cloud_run_region,
                                details=details
                            ))
                    except asyncio.TimeoutError:
                        traces.append(InfrastructureTrace(
                            component="gcp_cloud_run_ecapa",
                            status=InfrastructureStatus.DEGRADED,
                            location=cloud_run_region,
                            details={"url": cloud_run_url, "error": "Timeout - cold start?"}
                        ))
            except Exception as e:
                logger.debug(f"GCP Cloud Run check failed: {e}")
                traces.append(InfrastructureTrace(
                    component="gcp_cloud_run_ecapa",
                    status=InfrastructureStatus.UNAVAILABLE,
                    location=cloud_run_region,
                    details={"url": cloud_run_url, "error": str(e)}
                ))
        else:
            # Not configured
            traces.append(InfrastructureTrace(
                component="gcp_cloud_run_ecapa",
                status=InfrastructureStatus.UNKNOWN,
                details={"configured": False, "env_var": "CLOUD_RUN_ECAPA_URL"}
            ))

        return traces

    async def _check_gcp_vm_spot(self) -> List[InfrastructureTrace]:
        """
        Check GCP VM Spot GPU instances.

        VM Spot instances can be preempted at any time, so we handle
        connection failures gracefully and indicate possible preemption.
        """
        traces = []

        vm_spot_url = TransparencyConfig.vm_spot_gpu_url()
        vm_spot_zone = TransparencyConfig.vm_spot_zone()

        if vm_spot_url:
            try:
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    start = time.time()
                    health_url = f"{vm_spot_url}/health"

                    try:
                        async with session.get(health_url, timeout=10) as resp:
                            latency = (time.time() - start) * 1000

                            status = (
                                InfrastructureStatus.HEALTHY if resp.status == 200
                                else InfrastructureStatus.DEGRADED
                            )

                            # Try to get detailed info
                            details = {
                                "url": vm_spot_url,
                                "zone": vm_spot_zone,
                                "http_status": resp.status,
                            }

                            if resp.status == 200:
                                try:
                                    data = await resp.json()
                                    details["gpu"] = data.get("gpu", {})
                                    details["instance_type"] = data.get("instance_type", "unknown")
                                    details["preemptible"] = data.get("preemptible", True)
                                except:
                                    pass

                            traces.append(InfrastructureTrace(
                                component="gcp_vm_spot_gpu",
                                status=status,
                                latency_ms=latency,
                                location=vm_spot_zone,
                                details=details
                            ))

                    except asyncio.TimeoutError:
                        # Timeout could mean instance is starting up or preempted
                        traces.append(InfrastructureTrace(
                            component="gcp_vm_spot_gpu",
                            status=InfrastructureStatus.DEGRADED,
                            location=vm_spot_zone,
                            details={
                                "url": vm_spot_url,
                                "error": "Timeout - instance may be starting or preempted"
                            }
                        ))

            except Exception as e:
                logger.debug(f"GCP VM Spot check failed: {e}")
                traces.append(InfrastructureTrace(
                    component="gcp_vm_spot_gpu",
                    status=InfrastructureStatus.UNAVAILABLE,
                    location=vm_spot_zone,
                    details={
                        "url": vm_spot_url,
                        "error": str(e),
                        "note": "Spot instance may be preempted - this is normal behavior"
                    }
                ))
        else:
            # Not configured
            traces.append(InfrastructureTrace(
                component="gcp_vm_spot_gpu",
                status=InfrastructureStatus.UNKNOWN,
                details={"configured": False, "env_var": "GCP_VM_SPOT_GPU_URL"}
            ))

        return traces


# =============================================================================
# VOICE TRANSPARENCY ENGINE
# =============================================================================

class VoiceTransparencyEngine:
    """
    Main engine for voice authentication transparency.

    Provides:
    - Full decision tracing
    - Verbose announcements
    - Infrastructure monitoring
    - Debug output
    - History retention
    """

    _instance: Optional['VoiceTransparencyEngine'] = None
    _initialized: bool = False

    def __init__(self):
        self._announcement_generator = VerboseAnnouncementGenerator()
        self._infra_checker = InfrastructureStatusChecker()
        self._trace_history: Deque[DecisionTrace] = deque(
            maxlen=TransparencyConfig.max_trace_history()
        )
        self._current_trace: Optional[DecisionTrace] = None
        self._speak_callback: Optional[Callable[[str], Any]] = None
        self._stats = {
            "traces_created": 0,
            "announcements_spoken": 0,
            "infrastructure_checks": 0,
        }
        self._initialized = True
        logger.info("✅ VoiceTransparencyEngine initialized")

    @classmethod
    async def get_instance(cls) -> 'VoiceTransparencyEngine':
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_speak_callback(self, callback: Callable[[str], Any]):
        """Set callback for speaking announcements."""
        self._speak_callback = callback

    # -------------------------------------------------------------------------
    # Trace Management
    # -------------------------------------------------------------------------

    def start_trace(
        self,
        user_id: str = "owner",
        session_id: Optional[str] = None
    ) -> DecisionTrace:
        """Start a new decision trace."""
        trace = DecisionTrace(
            user_id=user_id,
            session_id=session_id or str(uuid4())
        )
        self._current_trace = trace
        self._stats["traces_created"] += 1
        return trace

    def get_current_trace(self) -> Optional[DecisionTrace]:
        """Get the current active trace."""
        return self._current_trace

    def complete_trace(
        self,
        outcome: DecisionOutcome,
        confidence: float,
        speaker_name: Optional[str] = None
    ) -> DecisionTrace:
        """Complete the current trace."""
        if self._current_trace:
            self._current_trace.complete(outcome, confidence, speaker_name)
            self._trace_history.append(self._current_trace)
            trace = self._current_trace
            self._current_trace = None
            return trace
        raise ValueError("No active trace to complete")

    def get_trace_history(
        self,
        limit: int = 10,
        user_id: Optional[str] = None
    ) -> List[DecisionTrace]:
        """Get recent trace history."""
        traces = list(self._trace_history)
        if user_id:
            traces = [t for t in traces if t.user_id == user_id]
        return traces[-limit:]

    def get_trace_by_id(self, trace_id: str) -> Optional[DecisionTrace]:
        """Get a specific trace by ID."""
        for trace in self._trace_history:
            if trace.trace_id == trace_id:
                return trace
        return None

    # -------------------------------------------------------------------------
    # Phase Tracing
    # -------------------------------------------------------------------------

    async def trace_phase(
        self,
        phase: ReasoningPhase,
        inputs: Dict[str, Any] = None
    ) -> PhaseTrace:
        """Start tracing a phase in the current trace."""
        if not self._current_trace:
            self.start_trace()

        phase_trace = self._current_trace.add_phase(phase, inputs)

        # Optionally announce progress
        if TransparencyConfig.debug_voice():
            announcement = self._announcement_generator.generate_progress_announcement(
                phase, inputs
            )
            if announcement:
                await self._speak(announcement)

        return phase_trace

    def complete_phase(
        self,
        phase_trace: PhaseTrace,
        outputs: Dict[str, Any],
        notes: List[str] = None
    ):
        """Complete a phase trace."""
        phase_trace.complete(outputs, notes)

    # -------------------------------------------------------------------------
    # Hypothesis Tracing
    # -------------------------------------------------------------------------

    def trace_hypothesis(
        self,
        category: str,
        description: str,
        prior: float,
        posterior: float,
        evidence_for: List[str] = None,
        evidence_against: List[str] = None,
        is_security_threat: bool = False
    ):
        """Add hypothesis trace to current trace."""
        if self._current_trace:
            self._current_trace.add_hypothesis(
                category=category,
                description=description,
                prior=prior,
                posterior=posterior,
                evidence_for=evidence_for,
                evidence_against=evidence_against,
                is_security_threat=is_security_threat
            )

    # -------------------------------------------------------------------------
    # Reasoning Chain
    # -------------------------------------------------------------------------

    def add_reasoning_step(self, thought: str):
        """Add a reasoning step to the current trace."""
        if self._current_trace:
            self._current_trace.add_reasoning(thought)

    def set_reasoning_conclusion(self, conclusion: str):
        """Set the reasoning conclusion."""
        if self._current_trace:
            self._current_trace.reasoning_conclusion = conclusion

    # -------------------------------------------------------------------------
    # Infrastructure
    # -------------------------------------------------------------------------

    async def check_infrastructure(self) -> List[InfrastructureTrace]:
        """Check infrastructure status and add to current trace."""
        self._stats["infrastructure_checks"] += 1

        traces = await self._infra_checker.check_all()

        if self._current_trace:
            self._current_trace.infrastructure = traces

        return traces

    # -------------------------------------------------------------------------
    # Announcements
    # -------------------------------------------------------------------------

    async def generate_and_speak_announcement(
        self,
        trace: DecisionTrace,
        verbose: bool = None,
        speak: bool = True
    ) -> str:
        """Generate announcement and optionally speak it."""
        announcement = self._announcement_generator.generate_result_announcement(
            trace, verbose
        )

        trace.primary_announcement = announcement

        if speak:
            await self._speak(announcement)

        return announcement

    async def speak_debug_report(self, trace: DecisionTrace):
        """Speak a detailed debug report."""
        announcement = self._announcement_generator.generate_debug_announcement(trace)
        trace.debug_announcement = announcement
        await self._speak(announcement)

    async def _speak(self, text: str):
        """Speak text using configured callback."""
        if self._speak_callback:
            self._stats["announcements_spoken"] += 1
            try:
                result = self._speak_callback(text)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Failed to speak: {e}")
        else:
            logger.debug(f"No speak callback configured. Would say: {text}")

    # -------------------------------------------------------------------------
    # Confidence Recording
    # -------------------------------------------------------------------------

    def record_confidence_breakdown(
        self,
        ml_confidence: float,
        physics_confidence: float,
        behavioral_confidence: float,
        contextual_confidence: float = 0.0,
        bayesian_prob: float = 0.0
    ):
        """Record confidence breakdown in current trace."""
        if self._current_trace:
            self._current_trace.ml_confidence = ml_confidence
            self._current_trace.physics_confidence = physics_confidence
            self._current_trace.behavioral_confidence = behavioral_confidence
            self._current_trace.contextual_confidence = contextual_confidence
            self._current_trace.bayesian_authentic_prob = bayesian_prob

    def record_decision_factors(
        self,
        factors: Dict[str, float],
        threshold: float,
        final_confidence: float
    ):
        """Record decision factors."""
        if self._current_trace:
            self._current_trace.decision_factors = factors
            self._current_trace.threshold_used = threshold
            self._current_trace.margin_from_threshold = final_confidence - threshold

    # -------------------------------------------------------------------------
    # Security Recording
    # -------------------------------------------------------------------------

    def record_spoofing_detection(
        self,
        detected: bool,
        spoof_type: Optional[str] = None,
        flags: List[str] = None
    ):
        """Record spoofing detection result."""
        if self._current_trace:
            self._current_trace.spoofing_detected = detected
            self._current_trace.spoofing_type = spoof_type
            if flags:
                self._current_trace.security_flags.extend(flags)

    # -------------------------------------------------------------------------
    # Errors and Warnings
    # -------------------------------------------------------------------------

    def add_error(self, error: str):
        """Add error to current trace."""
        if self._current_trace:
            self._current_trace.errors.append(error)

    def add_warning(self, warning: str):
        """Add warning to current trace."""
        if self._current_trace:
            self._current_trace.warnings.append(warning)

    # -------------------------------------------------------------------------
    # Stats and Debugging
    # -------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            **self._stats,
            "announcement_generator_stats": self._announcement_generator._stats,
            "trace_history_count": len(self._trace_history),
            "config": {
                "transparency_enabled": TransparencyConfig.is_enabled(),
                "verbose_mode": TransparencyConfig.verbose_mode(),
                "debug_voice": TransparencyConfig.debug_voice(),
                "cloud_status_enabled": TransparencyConfig.cloud_status_enabled(),
                "explain_decisions": TransparencyConfig.explain_decisions(),
            }
        }


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

_transparency_engine: Optional[VoiceTransparencyEngine] = None


async def get_transparency_engine() -> VoiceTransparencyEngine:
    """Get the singleton VoiceTransparencyEngine instance."""
    global _transparency_engine

    if _transparency_engine is None:
        _transparency_engine = VoiceTransparencyEngine()

    return _transparency_engine
