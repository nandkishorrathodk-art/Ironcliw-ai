"""
Context Intelligence Resolvers
==============================

Resolvers for ambiguous and contextual queries.

This module provides:
1. ContextualQueryResolver - resolves ambiguous contextual references
2. ImplicitReferenceResolver - resolves implicit pronouns and references

v2.0.0: Added ImplicitReferenceResolver bridging from core.nlp
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

from .contextual_query_resolver import (
    ContextualQueryResolver,
    get_contextual_resolver,
    QueryResolution,
    ResolutionStrategy,
    ContextualReference
)


# ============================================================================
# IMPLICIT REFERENCE RESOLVER - Bridge from core.nlp
# ============================================================================
# The ImplicitReferenceResolver is implemented in core.nlp but exposed here
# for easier access from context_intelligence handlers.
# ============================================================================

_implicit_resolver_available = False
_implicit_resolver_error: Optional[str] = None
ImplicitReferenceResolver = None

try:
    from core.nlp.implicit_reference_resolver import (
        ImplicitReferenceResolver,
        get_implicit_resolver as _get_implicit_resolver,
    )
    _implicit_resolver_available = True
except ImportError as e:
    _implicit_resolver_error = str(e)
    logger.debug(f"ImplicitReferenceResolver import deferred: {e}")


def get_implicit_reference_resolver():
    """
    Get the singleton ImplicitReferenceResolver instance.

    This is the primary entry point for handlers to access the resolver.
    Returns None if the resolver is not available (e.g., dependencies not loaded).

    Returns:
        ImplicitReferenceResolver or None if not available
    """
    if not _implicit_resolver_available:
        # Try lazy import in case dependencies are now available
        try:
            from core.nlp.implicit_reference_resolver import get_implicit_resolver
            return get_implicit_resolver()
        except ImportError:
            return None

    try:
        return _get_implicit_resolver()
    except Exception as e:
        logger.warning(f"Failed to get ImplicitReferenceResolver: {e}")
        return None


def is_implicit_resolver_available() -> bool:
    """Check if the ImplicitReferenceResolver is available."""
    if _implicit_resolver_available:
        return True
    # Try lazy check
    try:
        from core.nlp.implicit_reference_resolver import get_implicit_resolver
        return True
    except ImportError:
        return False


# Alias for backwards compatibility
get_implicit_resolver = get_implicit_reference_resolver


__all__ = [
    # Contextual resolver
    'ContextualQueryResolver',
    'get_contextual_resolver',
    'QueryResolution',
    'ResolutionStrategy',
    'ContextualReference',
    # Implicit reference resolver
    'ImplicitReferenceResolver',
    'get_implicit_reference_resolver',
    'get_implicit_resolver',
    'is_implicit_resolver_available',
]
