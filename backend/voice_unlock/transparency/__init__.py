"""
Voice Transparency Module
=========================

Provides comprehensive transparency and debugging for voice authentication.

Components:
- VoiceTransparencyEngine: Main engine for tracing and announcements
- VerboseAnnouncementGenerator: Intelligent announcement generation
- InfrastructureStatusChecker: Cloud infrastructure monitoring
- DecisionTrace: Complete authentication decision trace

Usage:
    from voice_unlock.transparency import get_transparency_engine

    engine = await get_transparency_engine()
    trace = engine.start_trace(user_id="owner")
    # ... authentication process ...
    engine.complete_trace(outcome, confidence, speaker_name)
    await engine.generate_and_speak_announcement(trace, verbose=True)
"""

from .voice_transparency_engine import (
    # Main engine
    VoiceTransparencyEngine,
    get_transparency_engine,

    # Config
    TransparencyConfig,

    # Enums
    DecisionOutcome,
    ReasoningPhase,
    InfrastructureStatus,

    # Models
    DecisionTrace,
    PhaseTrace,
    HypothesisTrace,
    InfrastructureTrace,

    # Generators
    VerboseAnnouncementGenerator,
    InfrastructureStatusChecker,
)

__all__ = [
    # Main engine
    "VoiceTransparencyEngine",
    "get_transparency_engine",

    # Config
    "TransparencyConfig",

    # Enums
    "DecisionOutcome",
    "ReasoningPhase",
    "InfrastructureStatus",

    # Models
    "DecisionTrace",
    "PhaseTrace",
    "HypothesisTrace",
    "InfrastructureTrace",

    # Generators
    "VerboseAnnouncementGenerator",
    "InfrastructureStatusChecker",
]
