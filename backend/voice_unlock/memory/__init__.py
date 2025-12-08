"""
Voice Pattern Memory Module

ChromaDB-based persistent storage for voice authentication patterns,
behavioral biometrics, and attack signatures.

Features:
- Voice evolution tracking with drift detection
- Behavioral pattern storage and retrieval
- Attack signature learning and matching
- Environmental voice profile adaptation
- Temporal pattern queries
"""

from .schemas import (
    VoiceEvolutionRecord,
    BehavioralPatternRecord,
    AttackPatternRecord,
    EnvironmentalProfileRecord,
    SpeechBiometricsRecord,
    AuthenticationEventRecord,
    MemoryQueryResult,
    VoiceMemoryConfig,
)

from .voice_pattern_memory import (
    VoicePatternMemory,
    get_voice_pattern_memory,
    create_voice_pattern_memory,
)

from .drift_detector import (
    VoiceDriftDetector,
    DriftAnalysisResult,
    DriftType,
    get_drift_detector,
)

__all__ = [
    # Schemas
    "VoiceEvolutionRecord",
    "BehavioralPatternRecord",
    "AttackPatternRecord",
    "EnvironmentalProfileRecord",
    "SpeechBiometricsRecord",
    "AuthenticationEventRecord",
    "MemoryQueryResult",
    "VoiceMemoryConfig",
    # Memory
    "VoicePatternMemory",
    "get_voice_pattern_memory",
    "create_voice_pattern_memory",
    # Drift Detection
    "VoiceDriftDetector",
    "DriftAnalysisResult",
    "DriftType",
    "get_drift_detector",
]
