"""
Ironcliw Neural Mesh - Web Search Agent

Provides structured internet research capabilities:
- Search live web results
- Read pages safely
- Synthesize reports with source attribution
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from ..base.base_neural_mesh_agent import BaseNeuralMeshAgent
from ..data_models import AgentMessage, KnowledgeType, MessagePriority, MessageType

from backend.intelligence.web_research_service import get_web_research_service

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
        return max(value, minimum)
    except ValueError:
        return default


class WebSearchAgent(BaseNeuralMeshAgent):
    """Neural Mesh agent for internet research and synthesis."""

    def __init__(self) -> None:
        super().__init__(
            agent_name="web_search_agent",
            agent_type="intelligence",
            capabilities={
                "web_search",
                "web_research",
                "read_web_page",
                "source_synthesis",
                "report_generation",
                "current_information_lookup",
            },
            version="1.0.0",
            dependencies={"context_tracker_agent"},
            description="Performs live internet research and structured synthesis",
        )
        self._service = get_web_research_service()
        self._history: List[Dict[str, Any]] = []
        self._history_limit = _env_int("Ironcliw_WEBSEARCH_AGENT_HISTORY_LIMIT", 200, minimum=20)

    async def on_initialize(self, **kwargs) -> None:
        await self._service.initialize()

        # Optional subscription for cross-agent request/response flows.
        if self.message_bus is not None:
            await self.subscribe(MessageType.CUSTOM, self._handle_custom_request)

        logger.info("WebSearchAgent initialized")

    async def on_start(self) -> None:
        logger.info("WebSearchAgent started")

    async def on_stop(self) -> None:
        logger.info("WebSearchAgent stopping (history=%d)", len(self._history))

    async def execute_task(self, payload: Dict[str, Any]) -> Any:
        action = str(payload.get("action", "")).strip().lower()
        if not action:
            raise ValueError("Missing required action")

        if action in {"search", "web_search", "search_web"}:
            query = str(payload.get("query", payload.get("topic", ""))).strip()
            max_results = int(payload.get("max_results", 8))
            include_types = self._parse_type_filter(payload.get("include_types"))
            results = await self._service.search(query=query, max_results=max_results, include_types=include_types)
            response = {
                "status": "success",
                "action": "web_search",
                "query": query,
                "count": len(results),
                "results": results,
            }
            await self._record_history(response)
            await self._store_knowledge(query=query, report=response, knowledge_type=KnowledgeType.OBSERVATION)
            return response

        if action in {"read", "read_page", "web_read", "read_web_page"}:
            url = str(payload.get("url", "")).strip()
            max_chars = int(payload.get("max_chars", 12000))
            page = await self._service.read_page(url=url, max_chars=max_chars)
            response = {
                "status": "success",
                "action": "read_page",
                "result": page,
            }
            await self._record_history(response)
            return response

        if action in {"research", "web_research", "research_topic"}:
            query = str(payload.get("query", payload.get("topic", ""))).strip()
            max_results = int(payload.get("max_results", 8))
            max_sources = int(payload.get("max_sources", 5))
            include_types = self._parse_type_filter(payload.get("include_types"))
            report = await self._service.research(
                query=query,
                max_results=max_results,
                max_sources=max_sources,
                include_types=include_types,
            )
            response = {
                "status": "success",
                "action": "web_research",
                "report": report,
            }
            await self._record_history(response)
            await self._store_knowledge(query=query, report=report, knowledge_type=KnowledgeType.FACT)
            return response

        if action in {"report", "research_report", "research_markdown"}:
            query = str(payload.get("query", payload.get("topic", ""))).strip()
            report = await self._service.research(
                query=query,
                max_results=int(payload.get("max_results", 8)),
                max_sources=int(payload.get("max_sources", 5)),
                include_types=self._parse_type_filter(payload.get("include_types")),
            )
            return {
                "status": "success",
                "action": "research_report",
                "query": query,
                "markdown": report.get("markdown_report", ""),
                "report": report,
            }

        if action in {"service_health", "health"}:
            return {
                "status": "success",
                "action": "service_health",
                "health": self._service.get_health(),
            }

        if action in {"service_metrics", "metrics"}:
            return {
                "status": "success",
                "action": "service_metrics",
                "metrics": self._service.get_metrics(),
            }

        raise ValueError(f"Unknown web search action: {action}")

    async def _handle_custom_request(self, message: AgentMessage) -> None:
        msg_type = str(message.payload.get("type", "")).strip().lower()
        if msg_type not in {"web_search_request", "web_research_request"}:
            return

        try:
            result = await self.execute_task(message.payload)
        except Exception as exc:
            result = {"status": "error", "error": str(exc)}

        if not self.message_bus:
            return
        if not message.from_agent or message.from_agent == self.agent_name:
            return

        await self.publish(
            to_agent=message.from_agent,
            message_type=MessageType.CUSTOM,
            payload={
                "type": "web_research_response",
                "request_id": message.message_id,
                "result": result,
            },
            priority=MessagePriority.NORMAL,
        )

    async def _record_history(self, record: Dict[str, Any]) -> None:
        self._history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "record": record,
        })
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit:]

    async def _store_knowledge(
        self,
        query: str,
        report: Dict[str, Any],
        knowledge_type: KnowledgeType,
    ) -> None:
        if not self.knowledge_graph:
            return
        try:
            await self.add_knowledge(
                knowledge_type=knowledge_type,
                data={
                    "query": query,
                    "summary": report.get("summary") or report.get("report", {}).get("summary", ""),
                    "sources": report.get("sources") or report.get("report", {}).get("sources", []),
                },
                confidence=0.75,
                tags={"web_research", "live_information"},
            )
        except Exception as exc:
            logger.debug("WebSearchAgent knowledge write skipped: %s", exc)

    @staticmethod
    def _parse_type_filter(raw: Optional[Any]) -> Optional[Sequence[str]]:
        if raw is None:
            return None
        if isinstance(raw, (list, tuple, set)):
            values = [str(item).strip() for item in raw if str(item).strip()]
            return values or None
        values = [token.strip() for token in str(raw).split(",") if token.strip()]
        return values or None

