"""
Ironcliw Voice Unlock Module
========================

Voice-based biometric authentication system for macOS unlocking.
Provides hands-free, secure access to Mac devices using voice recognition.

Features:
- Voice enrollment and profile management
- Real-time voice authentication
- Anti-spoofing protection
- System integration (screensaver, PAM)
- Multi-user support
- Apple Watch proximity detection
- ML optimization for 16GB RAM systems

Version: 2.0.0 - Clean Architecture (No Placeholders)
"""

__version__ = "2.0.0"
__author__ = "Ironcliw Team"

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# ============================================================================
# Lazy-loaded Service Instances
# ============================================================================
# Services are initialized on first access to reduce startup time and
# prevent import errors from affecting module loading.
# ============================================================================

_intelligent_unlock_service = None
_voice_unlock_system = None


async def get_intelligent_unlock_service():
    """
    Get the IntelligentVoiceUnlockService instance.

    This is the main service for voice-authenticated screen unlocking.
    Lazy-loaded to prevent blocking on import.

    FAST-PATH: Returns immediately if service is already initialized (no re-init).
    """
    global _intelligent_unlock_service

    # FAST-PATH: Return immediately if already initialized
    if _intelligent_unlock_service is not None and _intelligent_unlock_service.initialized:
        return _intelligent_unlock_service

    if _intelligent_unlock_service is None:
        try:
            from .intelligent_voice_unlock_service import (
                get_intelligent_unlock_service as _get_service
            )
            _intelligent_unlock_service = _get_service()
        except Exception as e:
            logger.error(f"Failed to create IntelligentVoiceUnlockService: {e}")
            raise

    # Only initialize if not already initialized
    if not _intelligent_unlock_service.initialized:
        try:
            await _intelligent_unlock_service.initialize()
            logger.info("✅ IntelligentVoiceUnlockService initialized")
        except Exception as e:
            logger.error(f"Failed to initialize IntelligentVoiceUnlockService: {e}")
            raise

    return _intelligent_unlock_service


def get_voice_unlock_system():
    """
    Get or create the voice unlock system instance (synchronous).

    For backwards compatibility with code expecting synchronous access.
    """
    global _voice_unlock_system

    if _voice_unlock_system is None:
        try:
            from .voice_unlock_integration import VoiceUnlockSystem
            _voice_unlock_system = VoiceUnlockSystem()
            logger.info("Voice Unlock System initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Voice Unlock System: {e}")
            return None

    return _voice_unlock_system


async def initialize_voice_unlock():
    """Initialize the voice unlock system asynchronously."""
    try:
        service = await get_intelligent_unlock_service()
        if service:
            logger.info("Voice Unlock System started successfully")
            return service
    except Exception as e:
        logger.error(f"Failed to start Voice Unlock System: {e}")
        return None


async def cleanup_voice_unlock():
    """Cleanup voice unlock system resources."""
    global _intelligent_unlock_service, _voice_unlock_system

    try:
        if _intelligent_unlock_service:
            # No explicit cleanup needed for intelligent service
            _intelligent_unlock_service = None

        if _voice_unlock_system and hasattr(_voice_unlock_system, 'stop'):
            await _voice_unlock_system.stop()
            _voice_unlock_system = None

        logger.info("Voice Unlock System cleaned up")
    except Exception as e:
        logger.error(f"Error during Voice Unlock cleanup: {e}")


def get_voice_unlock_status() -> Dict[str, Any]:
    """Get current voice unlock status."""
    try:
        if _intelligent_unlock_service:
            stats = _intelligent_unlock_service.get_stats()
            return {
                'available': True,
                'initialized': _intelligent_unlock_service.initialized,
                'stats': stats
            }
        else:
            return {
                'available': False,
                'error': 'Voice Unlock Service not initialized'
            }
    except Exception as e:
        logger.error(f"Failed to get Voice Unlock status: {e}")
        return {
            'available': False,
            'error': str(e)
        }


def check_dependencies() -> Dict[str, bool]:
    """Check if all required dependencies are available."""
    dependencies = {
        'numpy': False,
        'scipy': False,
        'scikit-learn': False,
        'torch': False,
        'torchaudio': False,
        'speechbrain': False,
        'sounddevice': False,
    }

    for dep in dependencies:
        try:
            if dep == 'scikit-learn':
                __import__('sklearn')
            else:
                __import__(dep.replace('-', '_'))
            dependencies[dep] = True
        except ImportError:
            pass

    return dependencies


# ============================================================================
# Enhanced Module Imports (v2.1)
# ============================================================================

# Lazy imports for new modules to prevent import errors
_reasoning_graph = None
_voice_pattern_memory = None
_voice_auth_orchestrator = None
_langfuse_tracer = None
_cost_tracker = None


async def get_voice_auth_reasoning_graph():
    """Get the LangGraph-based voice authentication reasoning graph."""
    global _reasoning_graph
    if _reasoning_graph is None:
        try:
            from .reasoning import get_voice_auth_reasoning_graph as _get_graph
            _reasoning_graph = await _get_graph()
            logger.info("✅ Voice Auth Reasoning Graph initialized")
        except Exception as e:
            logger.warning(f"Voice Auth Reasoning Graph not available: {e}")
    return _reasoning_graph


async def get_voice_pattern_memory():
    """Get the ChromaDB-based voice pattern memory."""
    global _voice_pattern_memory
    if _voice_pattern_memory is None:
        try:
            from .memory import get_voice_pattern_memory as _get_memory
            _voice_pattern_memory = await _get_memory()
            logger.info("✅ Voice Pattern Memory initialized")
        except Exception as e:
            logger.warning(f"Voice Pattern Memory not available: {e}")
    return _voice_pattern_memory


async def get_voice_auth_orchestrator():
    """Get the LangChain-based voice auth orchestrator with fallback chain."""
    global _voice_auth_orchestrator
    if _voice_auth_orchestrator is None:
        try:
            from .orchestration import get_voice_auth_orchestrator as _get_orch
            _voice_auth_orchestrator = await _get_orch()
            logger.info("✅ Voice Auth Orchestrator initialized")
        except Exception as e:
            logger.warning(f"Voice Auth Orchestrator not available: {e}")
    return _voice_auth_orchestrator


async def get_voice_auth_tracer():
    """Get the Langfuse-based audit trail tracer."""
    global _langfuse_tracer
    if _langfuse_tracer is None:
        try:
            from .observability import get_langfuse_tracer as _get_tracer
            _langfuse_tracer = await _get_tracer()
            logger.info("✅ Voice Auth Langfuse Tracer initialized")
        except Exception as e:
            logger.warning(f"Voice Auth Langfuse Tracer not available: {e}")
    return _langfuse_tracer


async def get_voice_auth_cost_tracker():
    """Get the Helicone-style cost tracker."""
    global _cost_tracker
    if _cost_tracker is None:
        try:
            from .observability import get_cost_tracker as _get_tracker
            _cost_tracker = await _get_tracker()
            logger.info("✅ Voice Auth Cost Tracker initialized")
        except Exception as e:
            logger.warning(f"Voice Auth Cost Tracker not available: {e}")
    return _cost_tracker


# ============================================================================
# Public API
# ============================================================================
__all__ = [
    # Async services
    'get_intelligent_unlock_service',
    'initialize_voice_unlock',
    'cleanup_voice_unlock',
    # Sync compatibility
    'get_voice_unlock_system',
    # Status & utilities
    'get_voice_unlock_status',
    'check_dependencies',
    # Enhanced modules (v2.1)
    'get_voice_auth_reasoning_graph',
    'get_voice_pattern_memory',
    'get_voice_auth_orchestrator',
    'get_voice_auth_tracer',
    'get_voice_auth_cost_tracker',
]
