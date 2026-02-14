"""
JARVIS Neural Mesh — Browsing System Adapter

Wraps BrowsingAgent for Neural Mesh discovery and registration.
Follows the same adapter pattern as vision_adapter.py and voice_adapter.py.

v6.4: Initial implementation.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Callable, Dict, Optional

from ..base.base_neural_mesh_agent import BaseNeuralMeshAgent

logger = logging.getLogger(__name__)


class BrowsingComponentType(str, Enum):
    """Types of browsing components."""
    BROWSER = "browser"


class BrowsingSystemAdapter(BaseNeuralMeshAgent):
    """Neural Mesh adapter for BrowsingAgent.

    Delegates all task execution to the underlying BrowsingAgent singleton.
    Provides Neural Mesh lifecycle management (heartbeats, messaging, registry).
    """

    def __init__(self, agent_name: str = "browsing_agent"):
        super().__init__(
            agent_name=agent_name,
            agent_type="browsing",
            capabilities={"navigate", "search", "extract", "fill_form", "multi_tab", "summarize"},
            description="Structured web browsing via API search + Playwright",
        )
        self._browsing_agent = None
        self._task_handlers: Dict[str, Callable] = {}

    async def on_initialize(self, **kwargs) -> None:
        """Initialize by getting the BrowsingAgent singleton."""
        try:
            from browsing.browsing_agent import get_browsing_agent
            self._browsing_agent = await get_browsing_agent()
        except ImportError:
            logger.debug("[BROWSE-ADAPTER] browsing_agent module not available")
        except Exception as e:
            logger.warning(f"[BROWSE-ADAPTER] Failed to get BrowsingAgent: {e}")

        if self._browsing_agent:
            logger.debug(
                "[BROWSE-ADAPTER] Connected to BrowsingAgent "
                f"(playwright={self._browsing_agent._playwright_available})"
            )

    async def execute_task(self, payload: Dict[str, Any]) -> Any:
        """Delegate to BrowsingAgent."""
        if not self._browsing_agent:
            return {"success": False, "error": "BrowsingAgent not available"}
        return await self._browsing_agent.execute_task(payload)


# =============================================================================
# Factory function (for bridge discovery)
# =============================================================================

async def create_browsing_adapter(
    agent_name: str = "browsing_agent",
) -> Optional[BrowsingSystemAdapter]:
    """Create and initialize a Browsing System adapter.

    Args:
        agent_name: Name for the adapter (default: "browsing_agent")

    Returns:
        Initialized adapter or None if creation fails
    """
    try:
        adapter = BrowsingSystemAdapter(agent_name=agent_name)
        await adapter.initialize()
        return adapter

    except ImportError as e:
        logger.debug("Browsing system not available: %s", e)
        return None
    except Exception as e:
        logger.error("Failed to create Browsing adapter: %s", e)
        return None
