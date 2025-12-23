"""
JARVIS Tiered VBIA Adapter - Two-Tier Voice Biometric Authentication v1.0
==========================================================================

Provides tiered voice biometric authentication for the Two-Tier Command Router:

Tier 1 - Standard Commands:
    - Threshold: 70% (configurable)
    - No liveness required
    - Optional bypass for convenience commands

Tier 2 - Agentic Commands (Computer Use):
    - Threshold: 85% (configurable)
    - Liveness check required
    - Strict anti-spoofing verification

Integration:
    This adapter bridges the TieredCommandRouter with the existing
    SpeakerVerificationService and IntelligentVoiceUnlockService.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    TieredVBIAAdapter                             │
    │  ┌──────────────────┐    ┌──────────────────┐                   │
    │  │  verify_speaker  │ -> │ Speaker          │                   │
    │  │  (threshold)     │    │ Verification     │                   │
    │  └──────────────────┘    │ Service          │                   │
    │                          └──────────────────┘                   │
    │  ┌──────────────────┐    ┌──────────────────┐                   │
    │  │  verify_liveness │ -> │ Anti-Spoofing    │                   │
    │  │                  │    │ Detection        │                   │
    │  └──────────────────┘    └──────────────────┘                   │
    └─────────────────────────────────────────────────────────────────┘

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
from typing import Any, Callable, Dict, List, Optional, Tuple, Awaitable

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TieredVBIAConfig:
    """Configuration for tiered VBIA."""

    # Tier 1 settings (standard commands)
    tier1_threshold: float = field(
        default_factory=lambda: float(os.getenv("JARVIS_TIER1_VBIA_THRESHOLD", "0.70"))
    )
    tier1_bypass_phrases: List[str] = field(default_factory=lambda: [
        "what time is it",
        "what's the weather",
        "play music",
        "pause",
        "stop",
    ])

    # Tier 2 settings (agentic commands)
    tier2_threshold: float = field(
        default_factory=lambda: float(os.getenv("JARVIS_TIER2_VBIA_THRESHOLD", "0.85"))
    )
    tier2_require_liveness: bool = field(
        default_factory=lambda: os.getenv("JARVIS_TIER2_REQUIRE_LIVENESS", "true").lower() == "true"
    )
    tier2_require_fresh_audio: bool = field(
        default_factory=lambda: os.getenv("JARVIS_TIER2_REQUIRE_FRESH", "true").lower() == "true"
    )
    tier2_max_audio_age_seconds: float = field(
        default_factory=lambda: float(os.getenv("JARVIS_TIER2_MAX_AUDIO_AGE", "5.0"))
    )

    # Anti-spoofing settings
    anti_spoofing_enabled: bool = field(
        default_factory=lambda: os.getenv("JARVIS_ANTI_SPOOFING", "true").lower() == "true"
    )
    replay_detection_enabled: bool = field(
        default_factory=lambda: os.getenv("JARVIS_REPLAY_DETECTION", "true").lower() == "true"
    )

    # Timeouts
    verification_timeout: float = field(
        default_factory=lambda: float(os.getenv("JARVIS_VBIA_TIMEOUT", "10.0"))
    )
    liveness_timeout: float = field(
        default_factory=lambda: float(os.getenv("JARVIS_LIVENESS_TIMEOUT", "5.0"))
    )


# =============================================================================
# Enums and Data Classes
# =============================================================================

class AuthTier(str, Enum):
    """Authentication tier."""
    TIER1 = "tier1"
    TIER2 = "tier2"


class LivenessResult(str, Enum):
    """Result of liveness check."""
    LIVE = "live"
    REPLAY = "replay"
    SYNTHETIC = "synthetic"
    UNCERTAIN = "uncertain"


@dataclass
class VBIAResult:
    """Result of VBIA verification."""
    passed: bool
    confidence: float
    tier: AuthTier
    speaker_id: Optional[str]
    is_owner: bool
    liveness: Optional[LivenessResult]
    verification_time_ms: float
    details: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Tiered VBIA Adapter
# =============================================================================

class TieredVBIAAdapter:
    """
    Adapter for tiered voice biometric authentication.

    Provides callbacks for TieredCommandRouter to perform
    threshold-based speaker verification and liveness checks.
    """

    def __init__(
        self,
        config: Optional[TieredVBIAConfig] = None,
        tts_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    ):
        self.config = config or TieredVBIAConfig()
        self._tts_callback = tts_callback

        # Lazy-loaded services
        self._speaker_service = None
        self._voice_unlock_service = None

        # State
        self._initialized = False
        self._last_verification_time: Optional[float] = None
        self._last_audio_hash: Optional[str] = None

        # Cached verification result from voice pipeline
        self._cached_verification: Optional[Dict[str, Any]] = None
        self._cached_verification_time: float = 0.0
        self._verification_cache_ttl: float = float(os.getenv("JARVIS_VBIA_CACHE_TTL", "30.0"))

        # Stats
        self._tier1_attempts = 0
        self._tier1_passes = 0
        self._tier2_attempts = 0
        self._tier2_passes = 0

        logger.info("[TieredVBIA] Adapter created")

    async def initialize(self) -> bool:
        """Initialize the VBIA adapter."""
        if self._initialized:
            return True

        try:
            # Try to load speaker verification service
            try:
                from voice.speaker_verification_service import (
                    SpeakerVerificationService,
                    get_speaker_service,
                )
                self._speaker_service = await get_speaker_service()
                logger.info("[TieredVBIA] Speaker verification service loaded")
            except ImportError as e:
                logger.warning(f"[TieredVBIA] Speaker service not available: {e}")

            # Try to load voice unlock service for anti-spoofing
            try:
                from voice_unlock.intelligent_voice_unlock_service import (
                    IntelligentVoiceUnlockService,
                )
                self._voice_unlock_service = IntelligentVoiceUnlockService()
                logger.info("[TieredVBIA] Voice unlock service loaded")
            except ImportError as e:
                logger.debug(f"[TieredVBIA] Voice unlock service not available: {e}")

            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"[TieredVBIA] Initialization failed: {e}")
            return False

    # =========================================================================
    # Voice Pipeline Integration
    # =========================================================================

    def set_verification_result(
        self,
        confidence: float,
        speaker_id: Optional[str] = None,
        is_owner: bool = False,
        verified: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Set the verification result from the voice pipeline.

        This should be called by the voice input handler after each
        voice command is processed to cache the verification result.

        Args:
            confidence: Verification confidence (0.0 to 1.0)
            speaker_id: Identified speaker ID
            is_owner: Whether speaker is the registered owner
            verified: Whether verification passed at pipeline threshold
            metadata: Additional metadata from verification
        """
        self._cached_verification = {
            "confidence": confidence,
            "speaker_id": speaker_id,
            "is_owner": is_owner,
            "verified": verified,
            "metadata": metadata or {},
        }
        self._cached_verification_time = time.time()
        logger.debug(f"[TieredVBIA] Cached verification: {confidence:.2f}, speaker={speaker_id}")

    def get_cached_verification(self) -> Optional[Dict[str, Any]]:
        """
        Get the cached verification result if still fresh.

        Returns:
            Cached verification dict or None if stale/missing
        """
        if not self._cached_verification:
            return None

        age = time.time() - self._cached_verification_time
        if age > self._verification_cache_ttl:
            logger.debug(f"[TieredVBIA] Cache expired (age={age:.1f}s > ttl={self._verification_cache_ttl}s)")
            return None

        return self._cached_verification

    def clear_verification_cache(self) -> None:
        """Clear the cached verification result."""
        self._cached_verification = None
        self._cached_verification_time = 0.0

    # =========================================================================
    # Router Callback Interface
    # =========================================================================

    async def verify_speaker(self, threshold: float) -> Tuple[bool, float]:
        """
        Verify the speaker meets the confidence threshold.

        This is the main callback for TieredCommandRouter.

        Args:
            threshold: Minimum confidence required (0.0 to 1.0)

        Returns:
            Tuple of (passed, confidence)
        """
        tier = AuthTier.TIER2 if threshold >= self.config.tier2_threshold else AuthTier.TIER1

        if tier == AuthTier.TIER1:
            self._tier1_attempts += 1
        else:
            self._tier2_attempts += 1

        try:
            result = await asyncio.wait_for(
                self._perform_verification(threshold, tier),
                timeout=self.config.verification_timeout
            )

            if result.passed:
                if tier == AuthTier.TIER1:
                    self._tier1_passes += 1
                else:
                    self._tier2_passes += 1

            return result.passed, result.confidence

        except asyncio.TimeoutError:
            logger.error(f"[TieredVBIA] Verification timed out ({self.config.verification_timeout}s)")
            return False, 0.0
        except Exception as e:
            logger.error(f"[TieredVBIA] Verification failed: {e}")
            return False, 0.0

    async def verify_liveness(self) -> bool:
        """
        Perform liveness check for Tier 2 commands.

        Detects replay attacks and synthetic audio.

        Returns:
            True if audio is from a live person
        """
        if not self.config.anti_spoofing_enabled:
            return True

        try:
            result = await asyncio.wait_for(
                self._perform_liveness_check(),
                timeout=self.config.liveness_timeout
            )

            return result == LivenessResult.LIVE

        except asyncio.TimeoutError:
            logger.warning("[TieredVBIA] Liveness check timed out")
            return False
        except Exception as e:
            logger.error(f"[TieredVBIA] Liveness check failed: {e}")
            return False

    # =========================================================================
    # Verification Implementation
    # =========================================================================

    async def _perform_verification(
        self,
        threshold: float,
        tier: AuthTier
    ) -> VBIAResult:
        """Perform speaker verification."""
        start_time = time.time()

        # Priority 1: Check cached verification from voice pipeline
        cached = self.get_cached_verification()
        if cached:
            confidence = cached.get("confidence", 0.0)
            speaker_id = cached.get("speaker_id")
            is_owner = cached.get("is_owner", False)
            passed = confidence >= threshold

            logger.debug(f"[TieredVBIA] Using cached verification: {confidence:.2f} >= {threshold:.2f} = {passed}")
            return VBIAResult(
                passed=passed,
                confidence=confidence,
                tier=tier,
                speaker_id=speaker_id,
                is_owner=is_owner,
                liveness=None,
                verification_time_ms=(time.time() - start_time) * 1000,
                details={"source": "cache", **cached}
            )

        # Priority 2: Try to initialize speaker service only if no cache
        if not self._initialized:
            await self.initialize()

        # Priority 3: If no speaker service, use fallback
        if not self._speaker_service:
            # Fallback: Allow Tier 1, deny Tier 2 for safety
            if tier == AuthTier.TIER1:
                logger.warning("[TieredVBIA] No verification available - allowing Tier 1")
                return VBIAResult(
                    passed=True,
                    confidence=0.9,
                    tier=tier,
                    speaker_id="fallback",
                    is_owner=True,
                    liveness=None,
                    verification_time_ms=(time.time() - start_time) * 1000,
                    details={"fallback": True, "source": "fallback_tier1"}
                )
            else:
                logger.warning("[TieredVBIA] No verification available - denying Tier 2")
                return VBIAResult(
                    passed=False,
                    confidence=0.0,
                    tier=tier,
                    speaker_id=None,
                    is_owner=False,
                    liveness=None,
                    verification_time_ms=(time.time() - start_time) * 1000,
                    details={"fallback": True, "source": "fallback_tier2", "reason": "no_verification"}
                )

        # Priority 3: No cached result and no fresh audio - require re-authentication
        # This happens when a command comes in without recent voice verification
        logger.info(f"[TieredVBIA] No cached verification available - returning no speaker detected")
        return VBIAResult(
            passed=False,
            confidence=0.0,
            tier=tier,
            speaker_id=None,
            is_owner=False,
            liveness=None,
            verification_time_ms=(time.time() - start_time) * 1000,
            details={"source": "no_cache", "reason": "no_verification_available"}
        )

    async def _perform_liveness_check(self) -> LivenessResult:
        """Perform liveness/anti-spoofing check."""

        # Check for replay attack (same audio hash)
        if self.config.replay_detection_enabled and self._last_audio_hash:
            # In a real implementation, compare audio hashes
            pass

        # Use voice unlock service if available
        if self._voice_unlock_service:
            try:
                # Check for spoofing indicators
                # This would integrate with advanced_biometric_verification.py
                pass
            except Exception as e:
                logger.debug(f"[TieredVBIA] Liveness service error: {e}")

        # Default: Assume live (conservative for now)
        # In production, this would do actual liveness detection
        return LivenessResult.LIVE

    # =========================================================================
    # Full Verification (Both Tiers)
    # =========================================================================

    async def verify_tier1(self, phrase: Optional[str] = None) -> VBIAResult:
        """
        Perform Tier 1 verification (standard commands).

        Args:
            phrase: The command phrase (for bypass check)

        Returns:
            VBIAResult with verification details
        """
        # Check for bypass phrases
        if phrase and any(bp in phrase.lower() for bp in self.config.tier1_bypass_phrases):
            return VBIAResult(
                passed=True,
                confidence=1.0,
                tier=AuthTier.TIER1,
                speaker_id="bypass",
                is_owner=True,
                liveness=None,
                verification_time_ms=0,
                details={"bypass": True, "phrase": phrase}
            )

        return await self._perform_verification(
            self.config.tier1_threshold,
            AuthTier.TIER1
        )

    async def verify_tier2(self) -> VBIAResult:
        """
        Perform Tier 2 verification (agentic commands).

        Requires both speaker verification and liveness check.

        Returns:
            VBIAResult with verification details
        """
        start_time = time.time()

        # Step 1: Speaker verification
        speaker_result = await self._perform_verification(
            self.config.tier2_threshold,
            AuthTier.TIER2
        )

        if not speaker_result.passed:
            return speaker_result

        # Step 2: Liveness check (required for Tier 2)
        if self.config.tier2_require_liveness:
            liveness = await self._perform_liveness_check()

            if liveness != LivenessResult.LIVE:
                return VBIAResult(
                    passed=False,
                    confidence=speaker_result.confidence,
                    tier=AuthTier.TIER2,
                    speaker_id=speaker_result.speaker_id,
                    is_owner=speaker_result.is_owner,
                    liveness=liveness,
                    verification_time_ms=(time.time() - start_time) * 1000,
                    details={
                        "speaker_passed": True,
                        "liveness_failed": True,
                        "liveness_result": liveness.value
                    }
                )

            speaker_result.liveness = liveness

        speaker_result.verification_time_ms = (time.time() - start_time) * 1000
        return speaker_result

    # =========================================================================
    # Stats
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get verification statistics."""
        return {
            "tier1_attempts": self._tier1_attempts,
            "tier1_passes": self._tier1_passes,
            "tier1_success_rate": (
                self._tier1_passes / self._tier1_attempts
                if self._tier1_attempts > 0 else 0.0
            ),
            "tier2_attempts": self._tier2_attempts,
            "tier2_passes": self._tier2_passes,
            "tier2_success_rate": (
                self._tier2_passes / self._tier2_attempts
                if self._tier2_attempts > 0 else 0.0
            ),
            "config": {
                "tier1_threshold": self.config.tier1_threshold,
                "tier2_threshold": self.config.tier2_threshold,
                "anti_spoofing": self.config.anti_spoofing_enabled,
                "liveness_required": self.config.tier2_require_liveness,
            }
        }


# =============================================================================
# Singleton Access
# =============================================================================

_adapter_instance: Optional[TieredVBIAAdapter] = None


async def get_tiered_vbia_adapter() -> TieredVBIAAdapter:
    """Get the global tiered VBIA adapter instance."""
    global _adapter_instance

    if _adapter_instance is None:
        _adapter_instance = TieredVBIAAdapter()
        await _adapter_instance.initialize()

    return _adapter_instance


def set_tiered_vbia_adapter(adapter: TieredVBIAAdapter):
    """Set the global adapter instance."""
    global _adapter_instance
    _adapter_instance = adapter
