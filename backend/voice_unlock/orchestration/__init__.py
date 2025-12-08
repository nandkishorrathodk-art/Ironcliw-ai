"""
Voice Authentication Orchestration Module

LangChain-based multi-factor authentication orchestration
with dynamic fallback chains and tool-based verification.

Features:
- Registered voice authentication tools
- Dynamic fallback chain orchestration
- Multi-factor authentication fusion
- Graceful degradation on failures
"""

from .voice_auth_tools import (
    VoiceAuthToolRegistry,
    get_voice_auth_tools,
    voice_biometric_verify,
    behavioral_context_analyze,
    challenge_question_generate,
    challenge_response_verify,
    proximity_check_apple_watch,
    anti_spoofing_detect,
    bayesian_fusion_calculate,
)

from .voice_auth_orchestrator import (
    VoiceAuthOrchestrator,
    OrchestratorConfig,
    AuthenticationChainResult,
    FallbackLevel,
    get_voice_auth_orchestrator,
    create_voice_auth_orchestrator,
)

__all__ = [
    # Tools
    "VoiceAuthToolRegistry",
    "get_voice_auth_tools",
    "voice_biometric_verify",
    "behavioral_context_analyze",
    "challenge_question_generate",
    "challenge_response_verify",
    "proximity_check_apple_watch",
    "anti_spoofing_detect",
    "bayesian_fusion_calculate",
    # Orchestrator
    "VoiceAuthOrchestrator",
    "OrchestratorConfig",
    "AuthenticationChainResult",
    "FallbackLevel",
    "get_voice_auth_orchestrator",
    "create_voice_auth_orchestrator",
]
