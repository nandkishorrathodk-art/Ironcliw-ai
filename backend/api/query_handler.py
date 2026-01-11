"""
Query Handler for JARVIS v2.0
==============================

Handles natural language queries with Trinity integration.

ROUTING ARCHITECTURE:
    User Query
         ↓
    UAE Context (if available)
         ↓
    Prime Router (Trinity Integration)
         ↓
    ┌──────────────────────────────────┐
    │  Local JARVIS-Prime (preferred)   │
    │  Cloud Claude (fallback)          │
    └──────────────────────────────────┘
         ↓
    Response with source tracking
"""
import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# System prompt for JARVIS personality
JARVIS_SYSTEM_PROMPT = os.getenv(
    "JARVIS_SYSTEM_PROMPT",
    """You are JARVIS, an advanced AI assistant created by Derek Russell.
You are helpful, intelligent, and speak with a sophisticated yet approachable tone.
You have access to the user's screen, can control applications, and assist with various tasks.
When responding:
- Be concise but thorough
- Use natural, conversational language
- Address the user respectfully
- Provide actionable information when relevant"""
)


async def handle_query(command: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Handle a natural language query with Trinity-aware routing.

    This function routes queries through the Prime Router, which will:
    1. Try local JARVIS-Prime first (free, fast, private)
    2. Fall back to cloud Claude if Prime unavailable
    3. Use graceful degradation if both fail

    Args:
        command: The query text
        context: Optional context information

    Returns:
        Dict with success status, response, and routing metadata
    """
    try:
        logger.info(f"[QUERY] Processing query: {command}")

        # Try to get UAE context for enhanced understanding
        uae_context = None
        try:
            from main import app
            uae_engine = getattr(app.state, 'uae_engine', None)
            if uae_engine:
                # Get contextual information from UAE
                uae_context = await uae_engine.get_current_context()
                logger.debug("[QUERY] UAE context available")
        except Exception as e:
            logger.debug(f"[QUERY] UAE context not available: {e}")

        # Build conversation context
        conversation_context = []
        if context and "history" in context:
            conversation_context = context["history"]
        elif uae_context and "conversation_history" in uae_context:
            conversation_context = uae_context.get("conversation_history", [])

        # Route through Prime Router (Trinity integration)
        try:
            # Try relative import first (for backend/ working dir)
            try:
                from core.prime_router import get_prime_router
            except ImportError:
                from backend.core.prime_router import get_prime_router

            router = await get_prime_router()

            # Generate response through Trinity
            response = await router.generate(
                prompt=command,
                system_prompt=JARVIS_SYSTEM_PROMPT,
                context=conversation_context if conversation_context else None,
            )

            logger.info(f"[QUERY] Response from {response.source} (latency: {response.latency_ms:.1f}ms)")

            return {
                "success": True,
                "response": response.content,
                "source": response.source,
                "model": response.model,
                "latency_ms": response.latency_ms,
                "tokens_used": response.tokens_used,
                "fallback_used": response.fallback_used,
                "metadata": response.metadata,
            }

        except ImportError as e:
            logger.warning(f"[QUERY] Prime Router not available: {e}")
            # Fall back to direct UAE if Prime Router not available
            return await _fallback_to_uae(command, context)

    except Exception as e:
        logger.error(f"[QUERY] Error processing query: {e}")
        return {
            "success": False,
            "response": f"Sorry, I encountered an error: {str(e)}",
            "error": str(e),
            "source": "error",
        }


async def _fallback_to_uae(command: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Fallback to UAE engine when Prime Router is not available."""
    try:
        from main import app
        uae_engine = getattr(app.state, 'uae_engine', None)

        if uae_engine:
            logger.info("[QUERY] Using UAE engine fallback")
            response = await uae_engine.process_query(command)

            return {
                "success": True,
                "response": response.get("response", "Query processed successfully"),
                "analysis": response.get("analysis", {}),
                "suggestions": response.get("suggestions", []),
                "source": "uae_engine",
                "fallback_used": True,
            }
        else:
            # Final fallback - direct cloud call
            return await _fallback_to_cloud(command)

    except Exception as e:
        logger.warning(f"[QUERY] UAE fallback failed: {e}")
        return await _fallback_to_cloud(command)


async def _fallback_to_cloud(command: str) -> Dict[str, Any]:
    """
    Final fallback - routes through Prime Router instead of direct Claude API.

    v2.7 FIX: Previously called AsyncAnthropic directly, bypassing the entire
    Prime routing system. Now uses PrimeRouter which has its own fallback chain:
    LOCAL_PRIME → CLOUD_RUN → CLOUD_CLAUDE

    This ensures:
    - All API calls are tracked and monitored
    - Cost tracking works for all requests
    - Circuit breakers apply consistently
    - Local Prime gets priority when available
    """
    try:
        # v2.7: Use Prime Router instead of direct Anthropic client
        try:
            from core.prime_router import get_prime_router
        except ImportError:
            from backend.core.prime_router import get_prime_router

        router = await get_prime_router()

        # Generate through Prime Router (handles all fallback logic internally)
        response = await router.generate(
            prompt=command,
            system_prompt=JARVIS_SYSTEM_PROMPT,
            max_tokens=4096,
        )

        # Extract content from response
        content = ""
        if hasattr(response, 'content') and response.content:
            if isinstance(response.content, list):
                content = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content[0])
            else:
                content = str(response.content)
        elif isinstance(response, dict):
            content = response.get('content', response.get('response', ''))
        else:
            content = str(response)

        # Determine source from router
        source = getattr(response, 'source', 'prime_router_fallback')
        if isinstance(response, dict):
            source = response.get('source', 'prime_router_fallback')

        return {
            "success": True,
            "response": content,
            "source": source,
            "fallback_used": True,
            "routed_via_prime": True,  # v2.7: Track Prime routing
        }

    except Exception as e:
        logger.error(f"[QUERY] Prime router fallback failed: {e}")
        return {
            "success": False,
            "response": f"I apologize, but I'm experiencing technical difficulties: {str(e)}",
            "source": "degraded",
            "error": str(e),
        }


async def handle_query_stream(command: str, context: Optional[Dict[str, Any]] = None):
    """
    Handle a streaming query response.

    Yields response chunks as they arrive.
    """
    try:
        try:
            from core.prime_router import get_prime_router
        except ImportError:
            from backend.core.prime_router import get_prime_router

        router = await get_prime_router()

        async for chunk in router.generate_stream(
            prompt=command,
            system_prompt=JARVIS_SYSTEM_PROMPT,
        ):
            yield chunk

    except Exception as e:
        logger.error(f"[QUERY] Stream error: {e}")
        yield f"Error: {str(e)}"
