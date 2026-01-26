"""
Trinity Clients - Cross-Repo Integration for JARVIS Ecosystem
==============================================================

This package provides client modules for cross-repo communication between:
- JARVIS (main AI agent) 
- jarvis-prime (local LLM inference)
- reactor-core (training pipeline)

These clients enable:
1. Shared embedding service (single SentenceTransformer instance)
2. Unified cleanup coordination
3. Memory pressure signaling

Usage:
    # Get embeddings using JARVIS's centralized service
    from backend.core.trinity_clients import get_embeddings
    embeddings = await get_embeddings(["text1", "text2"])
    
    # Register for cross-repo cleanup
    from backend.core.trinity_clients import register_with_jarvis_cleanup
    register_with_jarvis_cleanup("my_resource", cleanup_callback)

To use in jarvis-prime or reactor-core:
    1. Copy the desired client module to your repo
    2. Update import paths as needed
    3. Or add JARVIS-AI-Agent to PYTHONPATH

Author: JARVIS System
Version: 1.0.0
"""

from __future__ import annotations

# Embedding client
from backend.core.trinity_clients.jarvis_embedding_client import (
    JARVISEmbeddingClient,
    get_embedding_client,
    get_embeddings,
    cleanup_embedding_client,
)

# Cross-repo cleanup (imported from parent)
try:
    from backend.core.cross_repo_cleanup import (
        register_resource,
        unregister_resource,
        register_cleanup_callback,
        cleanup_all_resources,
    )
except ImportError:
    # Fallback stubs for when running outside JARVIS
    def register_resource(*args, **kwargs):
        pass
    def unregister_resource(*args, **kwargs):
        pass
    def register_cleanup_callback(*args, **kwargs):
        pass
    async def cleanup_all_resources(*args, **kwargs):
        return {}

# Convenience aliases
register_with_jarvis_cleanup = register_cleanup_callback

# Cost client
try:
    from backend.core.trinity_clients.trinity_cost_client import (
        TrinityCostClient,
        get_cost_client,
        report_cost,
        check_budget,
        get_remaining_budget,
    )
except ImportError:
    # Fallback stubs for when running outside JARVIS
    async def get_cost_client(*args, **kwargs):
        return None
    async def report_cost(*args, **kwargs):
        return False
    async def check_budget(*args, **kwargs):
        return True
    async def get_remaining_budget(*args, **kwargs):
        return 1.0
    TrinityCostClient = None

__all__ = [
    # Embedding
    "JARVISEmbeddingClient",
    "get_embedding_client", 
    "get_embeddings",
    "cleanup_embedding_client",
    # Cleanup
    "register_resource",
    "unregister_resource",
    "register_cleanup_callback",
    "register_with_jarvis_cleanup",
    "cleanup_all_resources",
    # Cost tracking
    "TrinityCostClient",
    "get_cost_client",
    "report_cost",
    "check_budget",
    "get_remaining_budget",
]
