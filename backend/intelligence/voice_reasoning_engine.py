"""
Voice Authentication Reasoning Engine
======================================

LangGraph-based multi-step authentication reasoning for intelligent decision-making.

This module implements adaptive authentication flow with:
- Partial match analysis and intelligent retry strategies
- Environmental noise detection and compensation
- Sick voice/voice change detection with pattern analysis
- Complete failure diagnosis with hypothesis generation
- Learning from failures to improve future authentications

Part of JARVIS v6.0 Clinical-Grade Intelligence Voice Authentication
Inspired by: LangGraph (multi-step reasoning), Open Interpreter (adaptive feedback)

Author: JARVIS AI System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class VoiceReasoningConfig:
    """Configuration for voice reasoning engine."""

    # Confidence thresholds for different scenarios
    partial_match_threshold: float = field(
        default_factory=lambda: float(os.getenv("VOICE_REASON_PARTIAL_THRESHOLD", "0.70"))
    )
    strong_match_threshold: float = field(
        default_factory=lambda: float(os.getenv("VOICE_REASON_STRONG_THRESHOLD", "0.85"))
    )
    excellent_match_threshold: float = field(
        default_factory=lambda: float(os.getenv("VOICE_REASON_EXCELLENT_THRESHOLD", "0.95"))
    )

    # Environmental detection
    enable_noise_detection: bool = field(
        default_factory=lambda: os.getenv("VOICE_REASON_NOISE_DETECTION", "true").lower() == "true"
    )
    low_snr_threshold: float = field(
        default_factory=lambda: float(os.getenv("VOICE_REASON_LOW_SNR", "12.0"))
    )
    normal_snr_baseline: float = field(
        default_factory=lambda: float(os.getenv("VOICE_REASON_NORMAL_SNR", "18.0"))
    )

    # Voice change detection
    enable_illness_detection: bool = field(
        default_factory=lambda: os.getenv("VOICE_REASON_ILLNESS_DETECT", "true").lower() == "true"
    )
    pitch_shift_threshold_hz: float = field(
        default_factory=lambda: float(os.getenv("VOICE_REASON_PITCH_SHIFT", "15.0"))
    )

    # Retry strategies
    max_retry_attempts: int = field(
        default_factory=lambda: int(os.getenv("VOICE_REASON_MAX_RETRIES", "3"))
    )
    retry_delay_seconds: float = field(
        default_factory=lambda: float(os.getenv("VOICE_REASON_RETRY_DELAY", "2.0"))
    )

    # Learning from failures
    enable_failure_learning: bool = field(
        default_factory=lambda: os.getenv("VOICE_REASON_FAILURE_LEARNING", "true").lower() == "true"
    )


# =============================================================================
# Enums
# =============================================================================

class ReasoningStep(str, Enum):
    """Steps in the reasoning chain."""
    INITIAL_ANALYSIS = "initial_analysis"
    ENVIRONMENTAL_CHECK = "environmental_check"
    VOICE_ANALYSIS = "voice_analysis"
    BEHAVIORAL_CHECK = "behavioral_check"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    RETRY_STRATEGY = "retry_strategy"
    FINAL_DECISION = "final_decision"


class FailureHypothesis(str, Enum):
    """Possible reasons for authentication failure."""
    WRONG_PERSON = "wrong_person"
    EQUIPMENT_FAILURE = "equipment_failure"
    ENVIRONMENTAL_NOISE = "environmental_noise"
    VOICE_ILLNESS = "voice_illness"
    VOICE_STRESS = "voice_stress"
    MICROPHONE_CHANGE = "microphone_change"
    LOCATION_CHANGE = "location_change"
    SPOOFING_ATTEMPT = "spoofing_attempt"


class RetryStrategy(str, Enum):
    """Intelligent retry strategies."""
    RETRY_WITH_GUIDANCE = "retry_with_guidance"  # Ask user to speak clearer/louder
    RETRY_WITH_FILTERING = "retry_with_filtering"  # Apply noise filtering
    RETRY_WITH_CALIBRATION = "retry_with_calibration"  # Recalibrate for new environment
    CHALLENGE_QUESTION = "challenge_question"  # Ask security question
    FALLBACK_PASSWORD = "fallback_password"  # Use password authentication
    DENY_ACCESS = "deny_access"  # No more retries


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EnvironmentalAnalysis:
    """Environmental condition analysis."""
    snr_db: float
    noise_level_db: float
    is_noisy: bool
    noise_type: Optional[str] = None  # "constant", "intermittent", "speech"
    recommendation: Optional[str] = None


@dataclass
class VoiceAnalysis:
    """Voice characteristics analysis."""
    fundamental_freq_hz: Optional[float] = None
    pitch_shift_hz: Optional[float] = None
    voice_quality: Optional[str] = None  # "clear", "hoarse", "stressed"
    speech_rhythm_match: float = 0.0
    illness_indicators: List[str] = field(default_factory=list)
    stress_indicators: List[str] = field(default_factory=list)


@dataclass
class ReasoningStep:
    """Single step in the reasoning chain."""
    step_name: str
    timestamp: float
    analysis: Dict[str, Any]
    decision: Optional[str] = None
    confidence: float = 0.0


@dataclass
class ReasoningResult:
    """Result of multi-step reasoning process."""
    final_decision: str  # "authenticate", "retry", "challenge", "deny"
    confidence: float
    reasoning_chain: List[ReasoningStep]
    retry_strategy: Optional[RetryStrategy] = None
    user_guidance: Optional[str] = None
    failure_hypotheses: List[Tuple[FailureHypothesis, float]] = field(default_factory=list)
    learned_insights: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "final_decision": self.final_decision,
            "confidence": self.confidence,
            "retry_strategy": self.retry_strategy.value if self.retry_strategy else None,
            "user_guidance": self.user_guidance,
            "failure_hypotheses": [
                {"hypothesis": h.value, "probability": p}
                for h, p in self.failure_hypotheses
            ],
            "learned_insights": self.learned_insights,
            "reasoning_steps": len(self.reasoning_chain),
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# Voice Reasoning Engine
# =============================================================================

class VoiceReasoningEngine:
    """
    LangGraph-inspired multi-step reasoning for voice authentication.

    Implements intelligent authentication decision-making through:
    1. Initial confidence analysis
    2. Environmental condition assessment
    3. Voice characteristics analysis
    4. Behavioral pattern matching
    5. Hypothesis generation for failures
    6. Adaptive retry strategy selection
    7. Learning from authentication outcomes

    Example Flow (Partial Match):
        Voice confidence: 72% (below 85% threshold)
        ↓
        Step 1: Detect partial match - confidence borderline
        Step 2: Analyze environment - high background noise (SNR: 12 dB vs normal 18 dB)
        Step 3: Voice analysis - slightly muffled
        Step 4: Generate hypothesis - environmental noise interference
        Step 5: Select retry strategy - ask user to speak louder/closer
        ↓
        Return: retry_with_guidance + "I'm having trouble hearing you clearly..."
    """

    def __init__(
        self,
        config: Optional[VoiceReasoningConfig] = None,
        tts_callback: Optional[Callable[[str], Any]] = None,
    ):
        self.config = config or VoiceReasoningConfig()
        self.tts_callback = tts_callback

        # Statistics
        self._total_reasonings = 0
        self._successful_authentications = 0
        self._successful_retries = 0
        self._learned_patterns: List[Dict[str, Any]] = []

        logger.info("[VoiceReasoning] Engine initialized")

    async def reason_about_authentication(
        self,
        voice_confidence: float,
        environmental_context: Optional[Dict[str, Any]] = None,
        voice_metadata: Optional[Dict[str, Any]] = None,
        behavioral_context: Optional[Dict[str, Any]] = None,
        retry_count: int = 0,
    ) -> ReasoningResult:
        """
        Perform multi-step reasoning about authentication result.

        Args:
            voice_confidence: Voice biometric confidence (0.0-1.0)
            environmental_context: Environmental data (SNR, noise, etc.)
            voice_metadata: Voice characteristics (pitch, quality, etc.)
            behavioral_context: Behavioral patterns (time, location, etc.)
            retry_count: Current retry attempt number

        Returns:
            ReasoningResult with decision and guidance
        """
        self._total_reasonings += 1
        reasoning_chain = []
        start_time = time.time()

        try:
            # ================================================================
            # STEP 1: Initial Confidence Analysis
            # ================================================================
            initial_step = await self._step_initial_analysis(
                voice_confidence, retry_count
            )
            reasoning_chain.append(initial_step)

            # If excellent match, skip detailed analysis
            if voice_confidence >= self.config.excellent_match_threshold:
                return ReasoningResult(
                    final_decision="authenticate",
                    confidence=voice_confidence,
                    reasoning_chain=reasoning_chain,
                    user_guidance="Of course, Derek. Unlocking for you.",
                )

            # If very low confidence, likely wrong person
            if voice_confidence < 0.50:
                return await self._handle_very_low_confidence(reasoning_chain)

            # ================================================================
            # STEP 2: Environmental Analysis
            # ================================================================
            if self.config.enable_noise_detection and environmental_context:
                env_step = await self._step_environmental_analysis(
                    environmental_context
                )
                reasoning_chain.append(env_step)

            # ================================================================
            # STEP 3: Voice Characteristics Analysis
            # ================================================================
            if self.config.enable_illness_detection and voice_metadata:
                voice_step = await self._step_voice_analysis(
                    voice_metadata, voice_confidence
                )
                reasoning_chain.append(voice_step)

            # ================================================================
            # STEP 4: Behavioral Pattern Check
            # ================================================================
            if behavioral_context:
                behavioral_step = await self._step_behavioral_analysis(
                    behavioral_context, voice_confidence
                )
                reasoning_chain.append(behavioral_step)

            # ================================================================
            # STEP 5: Hypothesis Generation (for failures/borderline)
            # ================================================================
            if voice_confidence < self.config.strong_match_threshold:
                hypotheses = await self._generate_failure_hypotheses(
                    voice_confidence,
                    environmental_context,
                    voice_metadata,
                    behavioral_context,
                )
            else:
                hypotheses = []

            # ================================================================
            # STEP 6: Select Retry Strategy
            # ================================================================
            retry_strategy, user_guidance = await self._select_retry_strategy(
                voice_confidence,
                hypotheses,
                retry_count,
                reasoning_chain,
            )

            # ================================================================
            # STEP 7: Final Decision
            # ================================================================
            final_decision = self._make_final_decision(
                voice_confidence, retry_strategy, retry_count
            )

            # ================================================================
            # Learning: Store insights from this authentication
            # ================================================================
            if self.config.enable_failure_learning:
                insights = await self._learn_from_authentication(
                    voice_confidence,
                    hypotheses,
                    retry_strategy,
                    final_decision,
                )
            else:
                insights = []

            result = ReasoningResult(
                final_decision=final_decision,
                confidence=voice_confidence,
                reasoning_chain=reasoning_chain,
                retry_strategy=retry_strategy,
                user_guidance=user_guidance,
                failure_hypotheses=hypotheses,
                learned_insights=insights,
            )

            # Track successful authentications
            if final_decision == "authenticate":
                self._successful_authentications += 1
                if retry_count > 0:
                    self._successful_retries += 1

            processing_time = (time.time() - start_time) * 1000
            logger.info(
                f"[VoiceReasoning] Decision: {final_decision}, "
                f"confidence={voice_confidence:.1%}, "
                f"steps={len(reasoning_chain)}, "
                f"time={processing_time:.0f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"[VoiceReasoning] Error in reasoning: {e}", exc_info=True)
            # Safe fallback
            return ReasoningResult(
                final_decision="deny",
                confidence=0.0,
                reasoning_chain=reasoning_chain,
                user_guidance="Authentication system error. Please use password.",
            )

    # =========================================================================
    # Reasoning Steps
    # =========================================================================

    async def _step_initial_analysis(
        self, confidence: float, retry_count: int
    ) -> Dict[str, Any]:
        """Step 1: Analyze initial confidence level."""
        if confidence >= self.config.excellent_match_threshold:
            analysis = "Excellent voice match - high confidence"
            decision = "proceed_fast"
        elif confidence >= self.config.strong_match_threshold:
            analysis = "Good voice match - acceptable confidence"
            decision = "authenticate"
        elif confidence >= self.config.partial_match_threshold:
            analysis = "Partial voice match - borderline confidence"
            decision = "analyze_further"
        else:
            analysis = "Low voice match - insufficient confidence"
            decision = "investigate_failure"

        return {
            "step_name": "initial_analysis",
            "timestamp": time.time(),
            "analysis": {
                "confidence": confidence,
                "retry_count": retry_count,
                "classification": analysis,
            },
            "decision": decision,
            "confidence": confidence,
        }

    async def _step_environmental_analysis(
        self, env_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Step 2: Analyze environmental conditions."""
        snr = env_context.get("snr_db", self.config.normal_snr_baseline)
        noise_level = env_context.get("noise_level_db", -42.0)

        is_noisy = snr < self.config.low_snr_threshold

        if is_noisy:
            analysis = f"High background noise detected (SNR: {snr:.1f} dB, normal: {self.config.normal_snr_baseline:.1f} dB)"
            recommendation = "Apply noise filtering and ask user to speak louder"
        else:
            analysis = f"Clean audio environment (SNR: {snr:.1f} dB)"
            recommendation = None

        return {
            "step_name": "environmental_analysis",
            "timestamp": time.time(),
            "analysis": {
                "snr_db": snr,
                "noise_level_db": noise_level,
                "is_noisy": is_noisy,
                "recommendation": recommendation,
            },
            "decision": "apply_filtering" if is_noisy else "environment_ok",
        }

    async def _step_voice_analysis(
        self, voice_metadata: Dict[str, Any], baseline_confidence: float
    ) -> Dict[str, Any]:
        """Step 3: Analyze voice characteristics for illness/stress."""
        pitch_shift = voice_metadata.get("pitch_shift_hz", 0.0)
        voice_quality = voice_metadata.get("voice_quality", "unknown")
        speech_rhythm_match = voice_metadata.get("speech_rhythm_match", 0.0)

        illness_indicators = []
        if abs(pitch_shift) > self.config.pitch_shift_threshold_hz:
            illness_indicators.append(f"Pitch shifted {pitch_shift:+.1f} Hz (illness indicator)")
        if voice_quality in ["hoarse", "rough"]:
            illness_indicators.append(f"Voice quality: {voice_quality}")

        # Check if speech patterns match despite voice changes
        patterns_match = speech_rhythm_match > 0.90

        analysis = {
            "pitch_shift_hz": pitch_shift,
            "voice_quality": voice_quality,
            "speech_rhythm_match": speech_rhythm_match,
            "illness_indicators": illness_indicators,
            "patterns_match_despite_voice_change": patterns_match,
        }

        # Decision: if patterns match well but voice doesn't, likely illness
        if patterns_match and baseline_confidence < 0.75:
            decision = "likely_illness"
            analysis["hypothesis"] = "Voice sounds different (illness) but speech patterns match"
        else:
            decision = "normal_variation"

        return {
            "step_name": "voice_analysis",
            "timestamp": time.time(),
            "analysis": analysis,
            "decision": decision,
        }

    async def _step_behavioral_analysis(
        self, behavioral_context: Dict[str, Any], voice_confidence: float
    ) -> Dict[str, Any]:
        """Step 4: Check behavioral patterns for context."""
        time_match = behavioral_context.get("is_typical_time", False)
        location_match = behavioral_context.get("location_trusted", False)
        device_stationary = behavioral_context.get("device_stationary", False)
        behavioral_confidence = behavioral_context.get("confidence", 0.5)

        # Strong behavioral match can boost borderline voice confidence
        strong_behavioral_match = (
            time_match and location_match and behavioral_confidence > 0.90
        )

        analysis = {
            "time_match": time_match,
            "location_match": location_match,
            "device_stationary": device_stationary,
            "behavioral_confidence": behavioral_confidence,
            "strong_match": strong_behavioral_match,
        }

        # If behavioral patterns are very strong, can compensate for lower voice confidence
        if strong_behavioral_match and voice_confidence > 0.68:
            decision = "behavioral_boost"
            analysis["recommendation"] = "Strong behavioral match compensates for lower voice confidence"
        else:
            decision = "behavioral_neutral"

        return {
            "step_name": "behavioral_analysis",
            "timestamp": time.time(),
            "analysis": analysis,
            "decision": decision,
            "confidence": behavioral_confidence,
        }

    # =========================================================================
    # Hypothesis Generation
    # =========================================================================

    async def _generate_failure_hypotheses(
        self,
        voice_confidence: float,
        env_context: Optional[Dict[str, Any]],
        voice_metadata: Optional[Dict[str, Any]],
        behavioral_context: Optional[Dict[str, Any]],
    ) -> List[Tuple[FailureHypothesis, float]]:
        """Generate ranked hypotheses for why authentication failed/borderline."""
        hypotheses = []

        # Hypothesis A: Wrong person
        if voice_confidence < 0.50:
            hypotheses.append((FailureHypothesis.WRONG_PERSON, 0.85))
        elif voice_confidence < 0.65:
            hypotheses.append((FailureHypothesis.WRONG_PERSON, 0.40))

        # Hypothesis B: Environmental noise
        if env_context:
            snr = env_context.get("snr_db", 18.0)
            if snr < self.config.low_snr_threshold:
                noise_probability = min(0.90, (self.config.normal_snr_baseline - snr) / 10.0)
                hypotheses.append((FailureHypothesis.ENVIRONMENTAL_NOISE, noise_probability))

        # Hypothesis C: Voice illness/stress
        if voice_metadata:
            pitch_shift = voice_metadata.get("pitch_shift_hz", 0.0)
            speech_match = voice_metadata.get("speech_rhythm_match", 0.0)
            if abs(pitch_shift) > self.config.pitch_shift_threshold_hz and speech_match > 0.85:
                hypotheses.append((FailureHypothesis.VOICE_ILLNESS, 0.75))
            elif voice_metadata.get("voice_quality") == "stressed":
                hypotheses.append((FailureHypothesis.VOICE_STRESS, 0.60))

        # Hypothesis D: Microphone/equipment change
        if env_context and env_context.get("microphone_changed"):
            hypotheses.append((FailureHypothesis.MICROPHONE_CHANGE, 0.80))

        # Hypothesis E: Location change
        if behavioral_context and not behavioral_context.get("location_trusted"):
            hypotheses.append((FailureHypothesis.LOCATION_CHANGE, 0.65))

        # Sort by probability
        hypotheses.sort(key=lambda x: x[1], reverse=True)
        return hypotheses[:3]  # Top 3 hypotheses

    # =========================================================================
    # Retry Strategy Selection
    # =========================================================================

    async def _select_retry_strategy(
        self,
        voice_confidence: float,
        hypotheses: List[Tuple[FailureHypothesis, float]],
        retry_count: int,
        reasoning_chain: List[Dict[str, Any]],
    ) -> Tuple[Optional[RetryStrategy], str]:
        """Select appropriate retry strategy based on analysis."""

        # Max retries reached
        if retry_count >= self.config.max_retry_attempts:
            return (
                RetryStrategy.FALLBACK_PASSWORD,
                "Maximum retry attempts reached. Please use your password to unlock.",
            )

        # Excellent match - no retry needed
        if voice_confidence >= self.config.strong_match_threshold:
            return (None, "Verified. Unlocking now, Derek.")

        # Check top hypothesis
        if not hypotheses:
            # No clear hypothesis - generic retry
            return (
                RetryStrategy.RETRY_WITH_GUIDANCE,
                "I'm having trouble verifying your voice. Please try again, speaking clearly.",
            )

        top_hypothesis, probability = hypotheses[0]

        # Environmental noise
        if top_hypothesis == FailureHypothesis.ENVIRONMENTAL_NOISE:
            return (
                RetryStrategy.RETRY_WITH_FILTERING,
                "I'm having trouble hearing you clearly - there's some background noise. "
                "Could you try again, maybe speak a bit louder and closer to the microphone?",
            )

        # Voice illness
        if top_hypothesis == FailureHypothesis.VOICE_ILLNESS:
            # Check if behavioral patterns are strong
            behavioral_strong = any(
                step.get("decision") == "behavioral_boost"
                for step in reasoning_chain
            )
            if behavioral_strong:
                return (
                    RetryStrategy.CHALLENGE_QUESTION,
                    "Your voice sounds different today, Derek - are you feeling alright? "
                    "For security, I'd like to ask a quick verification question.",
                )
            else:
                return (
                    RetryStrategy.RETRY_WITH_GUIDANCE,
                    "Your voice sounds a bit different. Let's try once more.",
                )

        # Microphone change
        if top_hypothesis == FailureHypothesis.MICROPHONE_CHANGE:
            return (
                RetryStrategy.RETRY_WITH_CALIBRATION,
                "It looks like you might be using a different microphone. "
                "Let me recalibrate - please say 'unlock my screen' once more.",
            )

        # Wrong person (high probability)
        if top_hypothesis == FailureHypothesis.WRONG_PERSON and probability > 0.70:
            return (
                RetryStrategy.DENY_ACCESS,
                "I don't recognize this voice. Access denied.",
            )

        # Default: generic retry with guidance
        return (
            RetryStrategy.RETRY_WITH_GUIDANCE,
            "Let me try that again. Please speak clearly into the microphone.",
        )

    # =========================================================================
    # Decision Making
    # =========================================================================

    def _make_final_decision(
        self,
        voice_confidence: float,
        retry_strategy: Optional[RetryStrategy],
        retry_count: int,
    ) -> str:
        """Make final authentication decision."""
        if retry_strategy == RetryStrategy.DENY_ACCESS:
            return "deny"
        elif retry_strategy == RetryStrategy.FALLBACK_PASSWORD:
            return "password"
        elif retry_strategy == RetryStrategy.CHALLENGE_QUESTION:
            return "challenge"
        elif retry_strategy in [
            RetryStrategy.RETRY_WITH_GUIDANCE,
            RetryStrategy.RETRY_WITH_FILTERING,
            RetryStrategy.RETRY_WITH_CALIBRATION,
        ]:
            return "retry"
        elif voice_confidence >= self.config.strong_match_threshold:
            return "authenticate"
        else:
            # Borderline - allow one retry
            return "retry" if retry_count == 0 else "challenge"

    async def _handle_very_low_confidence(
        self, reasoning_chain: List[Dict[str, Any]]
    ) -> ReasoningResult:
        """Handle very low confidence (likely wrong person)."""
        return ReasoningResult(
            final_decision="deny",
            confidence=0.0,
            reasoning_chain=reasoning_chain,
            user_guidance="I don't recognize this voice - this Mac is voice-locked to Derek only. "
                         "If you need access, please ask him directly or use the password.",
            failure_hypotheses=[(FailureHypothesis.WRONG_PERSON, 0.95)],
        )

    # =========================================================================
    # Learning
    # =========================================================================

    async def _learn_from_authentication(
        self,
        voice_confidence: float,
        hypotheses: List[Tuple[FailureHypothesis, float]],
        retry_strategy: Optional[RetryStrategy],
        final_decision: str,
    ) -> List[str]:
        """Learn insights from this authentication attempt."""
        insights = []

        # Learn from successful borderline authentications
        if final_decision == "authenticate" and voice_confidence < self.config.strong_match_threshold:
            insights.append(
                f"Successful authentication at {voice_confidence:.1%} confidence "
                f"(borderline) - learned pattern"
            )

        # Learn from environmental hypotheses
        if hypotheses and hypotheses[0][0] == FailureHypothesis.ENVIRONMENTAL_NOISE:
            insights.append("Environmental noise detected - learned filtering strategy")

        # Learn from illness detection
        if hypotheses and hypotheses[0][0] == FailureHypothesis.VOICE_ILLNESS:
            insights.append("Voice illness pattern detected - stored for future reference")

        # Store learned patterns
        if insights:
            self._learned_patterns.append({
                "timestamp": datetime.now().isoformat(),
                "confidence": voice_confidence,
                "decision": final_decision,
                "insights": insights,
            })

            # Keep only last 100 learned patterns
            if len(self._learned_patterns) > 100:
                self._learned_patterns = self._learned_patterns[-100:]

        return insights

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get reasoning engine statistics."""
        return {
            "total_reasonings": self._total_reasonings,
            "successful_authentications": self._successful_authentications,
            "successful_retries": self._successful_retries,
            "success_rate": (
                self._successful_authentications / self._total_reasonings
                if self._total_reasonings > 0
                else 0.0
            ),
            "retry_success_rate": (
                self._successful_retries / self._successful_authentications
                if self._successful_authentications > 0
                else 0.0
            ),
            "learned_patterns_count": len(self._learned_patterns),
            "config": {
                "partial_match_threshold": self.config.partial_match_threshold,
                "strong_match_threshold": self.config.strong_match_threshold,
                "excellent_match_threshold": self.config.excellent_match_threshold,
                "noise_detection": self.config.enable_noise_detection,
                "illness_detection": self.config.enable_illness_detection,
                "max_retries": self.config.max_retry_attempts,
            },
        }


# =============================================================================
# Singleton Access
# =============================================================================

_reasoning_engine: Optional[VoiceReasoningEngine] = None


async def get_voice_reasoning_engine(
    config: Optional[VoiceReasoningConfig] = None,
    tts_callback: Optional[Callable[[str], Any]] = None,
) -> VoiceReasoningEngine:
    """Get singleton voice reasoning engine instance."""
    global _reasoning_engine

    if _reasoning_engine is None:
        _reasoning_engine = VoiceReasoningEngine(config, tts_callback)
        logger.info("[VoiceReasoning] Singleton engine created")

    return _reasoning_engine
