"""
Voice Authentication LangGraph State Machine

Enterprise-grade LangGraph implementation for intelligent voice authentication.
Provides adaptive multi-step verification with transparent reasoning and
automatic learning from authentication outcomes.

Features:
- Conditional routing based on confidence levels
- Parallel evidence collection (physics + behavioral)
- Early exit optimization for high-confidence cases
- Hypothesis-driven reasoning for borderline cases
- Comprehensive error recovery and timeout handling
- Real-time metrics and observability hooks

Architecture:
    PERCEIVING → ANALYZING → VERIFYING → COLLECTING_EVIDENCE
                                              ↓
    RESPONDING ← DECIDING ← REASONING ← HYPOTHESIZING
        ↓
    LEARNING (async, non-blocking)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from dataclasses import dataclass, field
from functools import lru_cache

try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.checkpoint.base import BaseCheckpointSaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = "END"
    CompiledStateGraph = None
    BaseCheckpointSaver = None

from pydantic import BaseModel, Field

from .voice_auth_state import (
    VoiceAuthConfig,
    VoiceAuthReasoningPhase,
    VoiceAuthReasoningState,
    ConfidenceLevel,
    DecisionType,
    HypothesisCategory,
)
from .voice_auth_nodes import (
    BaseVoiceAuthNode,
    PerceptionNode,
    AudioAnalysisNode,
    MLVerificationNode,
    EvidenceCollectionNode,
    HypothesisGeneratorNode,
    ReasoningNode,
    DecisionNode,
    ResponseGeneratorNode,
    LearningNode,
    create_voice_auth_nodes,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class GraphConfig:
    """Environment-driven configuration for the voice auth graph."""

    @staticmethod
    def get_max_iterations() -> int:
        """Maximum graph iterations before forced termination."""
        return int(os.getenv("VOICE_AUTH_GRAPH_MAX_ITERATIONS", "20"))

    @staticmethod
    def get_graph_timeout_ms() -> int:
        """Overall graph execution timeout in milliseconds."""
        return int(os.getenv("VOICE_AUTH_GRAPH_TIMEOUT_MS", "5000"))

    @staticmethod
    def get_enable_checkpointing() -> bool:
        """Whether to enable state checkpointing for recovery."""
        return os.getenv("VOICE_AUTH_GRAPH_CHECKPOINTING", "false").lower() == "true"

    @staticmethod
    def get_enable_parallel_evidence() -> bool:
        """Whether to collect physics and behavioral evidence in parallel."""
        return os.getenv("VOICE_AUTH_PARALLEL_EVIDENCE", "true").lower() == "true"

    @staticmethod
    def get_enable_early_exit() -> bool:
        """Whether to allow early exit for high-confidence cases."""
        return os.getenv("VOICE_AUTH_EARLY_EXIT", "true").lower() == "true"

    @staticmethod
    def get_enable_hypothesis_reasoning() -> bool:
        """Whether to enable hypothesis-driven reasoning for borderline cases."""
        return os.getenv("VOICE_AUTH_HYPOTHESIS_REASONING", "true").lower() == "true"

    @staticmethod
    def get_enable_async_learning() -> bool:
        """Whether learning node runs asynchronously."""
        return os.getenv("VOICE_AUTH_ASYNC_LEARNING", "true").lower() == "true"

    @staticmethod
    def get_retry_on_timeout() -> bool:
        """Whether to retry on node timeout."""
        return os.getenv("VOICE_AUTH_RETRY_ON_TIMEOUT", "true").lower() == "true"

    @staticmethod
    def get_max_retries() -> int:
        """Maximum retries per node."""
        return int(os.getenv("VOICE_AUTH_MAX_RETRIES", "2"))

    @staticmethod
    def get_enable_metrics() -> bool:
        """Whether to collect detailed metrics."""
        return os.getenv("VOICE_AUTH_METRICS_ENABLED", "true").lower() == "true"

    @staticmethod
    def get_log_level() -> str:
        """Logging level for graph operations."""
        return os.getenv("VOICE_AUTH_GRAPH_LOG_LEVEL", "INFO")


# =============================================================================
# GRAPH METRICS
# =============================================================================

@dataclass
class GraphExecutionMetrics:
    """Metrics for a single graph execution."""

    session_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    total_duration_ms: float = 0.0

    # Phase metrics
    phases_executed: List[str] = field(default_factory=list)
    phase_durations_ms: Dict[str, float] = field(default_factory=dict)

    # Decision metrics
    final_decision: Optional[str] = None
    final_confidence: float = 0.0
    early_exit_used: bool = False
    hypotheses_generated: int = 0
    reasoning_depth: int = 0

    # Error metrics
    errors_encountered: int = 0
    retries_performed: int = 0
    timeouts_occurred: int = 0

    # Resource metrics
    peak_memory_mb: float = 0.0
    api_calls_made: int = 0
    cache_hits: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging/observability."""
        return {
            "session_id": self.session_id,
            "total_duration_ms": self.total_duration_ms,
            "phases_executed": self.phases_executed,
            "phase_durations_ms": self.phase_durations_ms,
            "final_decision": self.final_decision,
            "final_confidence": self.final_confidence,
            "early_exit_used": self.early_exit_used,
            "hypotheses_generated": self.hypotheses_generated,
            "reasoning_depth": self.reasoning_depth,
            "errors_encountered": self.errors_encountered,
            "retries_performed": self.retries_performed,
            "timeouts_occurred": self.timeouts_occurred,
            "api_calls_made": self.api_calls_made,
            "cache_hits": self.cache_hits,
        }


class GraphMetricsCollector:
    """Collects and aggregates metrics across graph executions."""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._history: List[GraphExecutionMetrics] = []
        self._lock = asyncio.Lock()

        # Aggregated metrics
        self._total_executions = 0
        self._total_successes = 0
        self._total_failures = 0
        self._total_timeouts = 0
        self._avg_duration_ms = 0.0
        self._decision_counts: Dict[str, int] = {}

    async def record(self, metrics: GraphExecutionMetrics) -> None:
        """Record metrics from a graph execution."""
        async with self._lock:
            self._history.append(metrics)
            if len(self._history) > self.max_history:
                self._history.pop(0)

            # Update aggregates
            self._total_executions += 1
            if metrics.final_decision in ("AUTHENTICATE", "CHALLENGE"):
                self._total_successes += 1
            elif metrics.final_decision == "REJECT":
                self._total_failures += 1

            if metrics.timeouts_occurred > 0:
                self._total_timeouts += 1

            # Update decision counts
            if metrics.final_decision:
                self._decision_counts[metrics.final_decision] = \
                    self._decision_counts.get(metrics.final_decision, 0) + 1

            # Update rolling average
            n = len(self._history)
            self._avg_duration_ms = (
                self._avg_duration_ms * (n - 1) + metrics.total_duration_ms
            ) / n

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "total_executions": self._total_executions,
            "success_rate": self._total_successes / max(1, self._total_executions),
            "failure_rate": self._total_failures / max(1, self._total_executions),
            "timeout_rate": self._total_timeouts / max(1, self._total_executions),
            "avg_duration_ms": self._avg_duration_ms,
            "decision_distribution": self._decision_counts.copy(),
            "history_size": len(self._history),
        }


# Global metrics collector
_metrics_collector = GraphMetricsCollector()


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================

def route_after_perception(state: Dict[str, Any]) -> str:
    """Route after perception node completes."""
    # Check for errors
    if state.get("has_critical_error", False):
        return "error_handler"

    # Always proceed to audio analysis
    return VoiceAuthReasoningPhase.ANALYZING.value


def route_after_analysis(state: Dict[str, Any]) -> str:
    """Route after audio analysis completes."""
    if state.get("has_critical_error", False):
        return "error_handler"

    # Check if audio quality is too poor
    audio_result = state.get("audio_analysis_result")
    if audio_result:
        snr = audio_result.get("snr_db", 0)
        min_snr = float(os.getenv("VOICE_AUTH_MIN_SNR_DB", "5.0"))
        if snr < min_snr:
            # Audio too poor, need retry
            state["should_retry"] = True
            state["retry_reason"] = f"Audio SNR ({snr:.1f} dB) below minimum ({min_snr} dB)"
            return "response_generator"

    return VoiceAuthReasoningPhase.VERIFYING.value


def route_after_verification(state: Dict[str, Any]) -> str:
    """Route after ML verification - critical routing decision."""
    if state.get("has_critical_error", False):
        return "error_handler"

    ml_confidence = state.get("ml_confidence", 0.0)

    # Get thresholds
    instant_threshold = VoiceAuthConfig.get_instant_threshold()
    rejection_threshold = VoiceAuthConfig.get_rejection_threshold()

    # Early exit for very high confidence (fast path)
    if GraphConfig.get_enable_early_exit() and ml_confidence >= instant_threshold:
        logger.info(f"Early exit: ML confidence {ml_confidence:.3f} >= {instant_threshold}")
        state["early_exit"] = True
        return VoiceAuthReasoningPhase.DECIDING.value

    # Early rejection for very low confidence
    if ml_confidence < rejection_threshold:
        logger.info(f"Early rejection: ML confidence {ml_confidence:.3f} < {rejection_threshold}")
        state["early_rejection"] = True
        return VoiceAuthReasoningPhase.DECIDING.value

    # Need more evidence - proceed to evidence collection
    return VoiceAuthReasoningPhase.COLLECTING_EVIDENCE.value


def route_after_evidence(state: Dict[str, Any]) -> str:
    """Route after evidence collection."""
    if state.get("has_critical_error", False):
        return "error_handler"

    ml_confidence = state.get("ml_confidence", 0.0)
    confident_threshold = VoiceAuthConfig.get_confident_threshold()
    borderline_threshold = VoiceAuthConfig.get_borderline_threshold()

    # With evidence, check combined confidence
    physics_confidence = state.get("physics_confidence", 0.0)
    behavioral_confidence = state.get("behavioral_confidence", 0.0)

    # Quick fusion estimate
    combined = (
        ml_confidence * 0.5 +
        physics_confidence * 0.25 +
        behavioral_confidence * 0.25
    )

    # Clear case - proceed to decision
    if combined >= confident_threshold:
        return VoiceAuthReasoningPhase.DECIDING.value

    # Borderline case - need hypothesis reasoning
    if combined >= borderline_threshold and GraphConfig.get_enable_hypothesis_reasoning():
        return VoiceAuthReasoningPhase.HYPOTHESIZING.value

    # Below borderline - still try reasoning to understand why
    if GraphConfig.get_enable_hypothesis_reasoning():
        return VoiceAuthReasoningPhase.HYPOTHESIZING.value

    # No reasoning enabled - go straight to decision
    return VoiceAuthReasoningPhase.DECIDING.value


def route_after_hypotheses(state: Dict[str, Any]) -> str:
    """Route after hypothesis generation."""
    if state.get("has_critical_error", False):
        return "error_handler"

    hypotheses = state.get("hypotheses", [])

    # If we generated hypotheses that need evaluation, go to reasoning
    if hypotheses and len(hypotheses) > 0:
        return VoiceAuthReasoningPhase.REASONING.value

    # No hypotheses to reason about - go to decision
    return VoiceAuthReasoningPhase.DECIDING.value


def route_after_reasoning(state: Dict[str, Any]) -> str:
    """Route after reasoning node."""
    if state.get("has_critical_error", False):
        return "error_handler"

    # Always proceed to decision after reasoning
    return VoiceAuthReasoningPhase.DECIDING.value


def route_after_decision(state: Dict[str, Any]) -> str:
    """Route after decision - determines response type."""
    if state.get("has_critical_error", False):
        return "error_handler"

    # Always go to response generator
    return VoiceAuthReasoningPhase.RESPONDING.value


def route_after_response(state: Dict[str, Any]) -> str:
    """Route after response generation."""
    # Check if we should retry
    if state.get("should_retry", False):
        retry_count = state.get("retry_count", 0)
        max_retries = GraphConfig.get_max_retries()
        if retry_count < max_retries:
            state["retry_count"] = retry_count + 1
            return VoiceAuthReasoningPhase.PERCEIVING.value

    # Check if async learning is enabled
    if GraphConfig.get_enable_async_learning():
        return VoiceAuthReasoningPhase.LEARNING.value

    # End the graph
    return END


def route_after_learning(state: Dict[str, Any]) -> str:
    """Route after learning - always ends."""
    return END


def error_handler_route(state: Dict[str, Any]) -> str:
    """Handle errors and determine recovery path."""
    error_count = len(state.get("errors", []))
    max_errors = 3

    if error_count >= max_errors:
        # Too many errors - force decision
        state["forced_decision"] = True
        state["forced_decision_reason"] = f"Maximum errors ({max_errors}) exceeded"
        return VoiceAuthReasoningPhase.DECIDING.value

    # Try to recover based on current phase
    current_phase = state.get("current_phase", "")

    # If error in early phase, retry from beginning
    if current_phase in ("PERCEIVING", "ANALYZING"):
        return VoiceAuthReasoningPhase.PERCEIVING.value

    # If error in verification, skip to decision
    if current_phase == "VERIFYING":
        return VoiceAuthReasoningPhase.DECIDING.value

    # Default: go to decision
    return VoiceAuthReasoningPhase.DECIDING.value


# =============================================================================
# NODE WRAPPERS
# =============================================================================

class NodeWrapper:
    """Wraps a node to add metrics, error handling, and state conversion."""

    def __init__(
        self,
        node: BaseVoiceAuthNode,
        metrics_enabled: bool = True,
    ):
        self.node = node
        self.metrics_enabled = metrics_enabled
        self._execution_count = 0
        self._total_duration_ms = 0.0
        self._error_count = 0

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the wrapped node."""
        start_time = time.perf_counter()

        try:
            # Convert dict to state object if needed
            if isinstance(state, dict):
                state_obj = VoiceAuthReasoningState(**state)
            else:
                state_obj = state

            # Execute node
            result_state = await self.node.execute(state_obj)

            # Convert back to dict
            result_dict = result_state.model_dump()

            # Record metrics
            if self.metrics_enabled:
                duration_ms = (time.perf_counter() - start_time) * 1000
                self._execution_count += 1
                self._total_duration_ms += duration_ms

                # Add timing to result
                phase_timings = result_dict.get("phase_timings", {})
                phase_timings[self.node.node_name] = duration_ms
                result_dict["phase_timings"] = phase_timings

            return result_dict

        except Exception as e:
            self._error_count += 1
            logger.error(f"Node {self.node.node_name} error: {e}")

            # Return state with error recorded
            if isinstance(state, dict):
                state["errors"] = state.get("errors", []) + [{
                    "node": self.node.node_name,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }]
                state["has_critical_error"] = True
                return state
            else:
                state.errors.append({
                    "node": self.node.node_name,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                state.has_critical_error = True
                return state.model_dump()

    def get_stats(self) -> Dict[str, Any]:
        """Get node execution statistics."""
        return {
            "node_name": self.node.node_name,
            "execution_count": self._execution_count,
            "avg_duration_ms": (
                self._total_duration_ms / self._execution_count
                if self._execution_count > 0 else 0
            ),
            "error_count": self._error_count,
            "error_rate": (
                self._error_count / self._execution_count
                if self._execution_count > 0 else 0
            ),
        }


# =============================================================================
# ERROR HANDLER NODE
# =============================================================================

class ErrorHandlerNode:
    """Handles errors in the graph execution."""

    node_name = "error_handler"

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle errors and prepare for recovery or termination."""
        errors = state.get("errors", [])

        if errors:
            last_error = errors[-1]
            logger.warning(
                f"Error handler invoked. Last error in {last_error.get('node')}: "
                f"{last_error.get('error')}"
            )

        # Clear critical error flag to allow retry
        state["has_critical_error"] = False

        # Record that we handled an error
        state["errors_handled"] = state.get("errors_handled", 0) + 1

        return state


# =============================================================================
# VOICE AUTH GRAPH
# =============================================================================

class VoiceAuthenticationReasoningGraph:
    """
    LangGraph-based voice authentication reasoning system.

    Provides intelligent, adaptive authentication with:
    - Multi-phase verification pipeline
    - Hypothesis-driven reasoning for borderline cases
    - Early exit optimization for clear cases
    - Comprehensive error recovery
    - Real-time metrics and observability

    Usage:
        graph = VoiceAuthenticationReasoningGraph()
        result = await graph.authenticate(audio_data, user_id="derek")
    """

    def __init__(
        self,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        custom_nodes: Optional[Dict[str, BaseVoiceAuthNode]] = None,
        enable_metrics: bool = True,
    ):
        """
        Initialize the voice authentication graph.

        Args:
            checkpointer: Optional checkpoint saver for state recovery
            custom_nodes: Optional custom node implementations
            enable_metrics: Whether to collect execution metrics
        """
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph is required for VoiceAuthenticationReasoningGraph. "
                "Install with: pip install langgraph"
            )

        self.checkpointer = checkpointer
        self.enable_metrics = enable_metrics and GraphConfig.get_enable_metrics()

        # Initialize nodes
        self._nodes = custom_nodes or create_voice_auth_nodes()
        self._wrapped_nodes: Dict[str, NodeWrapper] = {}

        # Wrap nodes with metrics and error handling
        for name, node in self._nodes.items():
            self._wrapped_nodes[name] = NodeWrapper(node, self.enable_metrics)

        # Add error handler
        self._error_handler = ErrorHandlerNode()

        # Build the graph
        self._graph = self._build_graph()
        self._compiled_graph: Optional[CompiledStateGraph] = None

        # Execution tracking
        self._execution_count = 0
        self._last_execution_metrics: Optional[GraphExecutionMetrics] = None

        logger.info("VoiceAuthenticationReasoningGraph initialized")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        # Create state graph with dict state (will be converted to/from Pydantic)
        graph = StateGraph(dict)

        # Add all nodes
        graph.add_node(
            VoiceAuthReasoningPhase.PERCEIVING.value,
            self._wrapped_nodes["perception"]
        )
        graph.add_node(
            VoiceAuthReasoningPhase.ANALYZING.value,
            self._wrapped_nodes["audio_analysis"]
        )
        graph.add_node(
            VoiceAuthReasoningPhase.VERIFYING.value,
            self._wrapped_nodes["ml_verification"]
        )
        graph.add_node(
            VoiceAuthReasoningPhase.COLLECTING_EVIDENCE.value,
            self._wrapped_nodes["evidence_collection"]
        )
        graph.add_node(
            VoiceAuthReasoningPhase.HYPOTHESIZING.value,
            self._wrapped_nodes["hypothesis_generator"]
        )
        graph.add_node(
            VoiceAuthReasoningPhase.REASONING.value,
            self._wrapped_nodes["reasoning"]
        )
        graph.add_node(
            VoiceAuthReasoningPhase.DECIDING.value,
            self._wrapped_nodes["decision"]
        )
        graph.add_node(
            VoiceAuthReasoningPhase.RESPONDING.value,
            self._wrapped_nodes["response_generator"]
        )
        graph.add_node(
            VoiceAuthReasoningPhase.LEARNING.value,
            self._wrapped_nodes["learning"]
        )
        graph.add_node("error_handler", self._error_handler)

        # Set entry point
        graph.set_entry_point(VoiceAuthReasoningPhase.PERCEIVING.value)

        # Add conditional edges
        graph.add_conditional_edges(
            VoiceAuthReasoningPhase.PERCEIVING.value,
            route_after_perception,
            {
                VoiceAuthReasoningPhase.ANALYZING.value: VoiceAuthReasoningPhase.ANALYZING.value,
                "error_handler": "error_handler",
            }
        )

        graph.add_conditional_edges(
            VoiceAuthReasoningPhase.ANALYZING.value,
            route_after_analysis,
            {
                VoiceAuthReasoningPhase.VERIFYING.value: VoiceAuthReasoningPhase.VERIFYING.value,
                VoiceAuthReasoningPhase.RESPONDING.value: VoiceAuthReasoningPhase.RESPONDING.value,
                "error_handler": "error_handler",
            }
        )

        graph.add_conditional_edges(
            VoiceAuthReasoningPhase.VERIFYING.value,
            route_after_verification,
            {
                VoiceAuthReasoningPhase.DECIDING.value: VoiceAuthReasoningPhase.DECIDING.value,
                VoiceAuthReasoningPhase.COLLECTING_EVIDENCE.value: VoiceAuthReasoningPhase.COLLECTING_EVIDENCE.value,
                "error_handler": "error_handler",
            }
        )

        graph.add_conditional_edges(
            VoiceAuthReasoningPhase.COLLECTING_EVIDENCE.value,
            route_after_evidence,
            {
                VoiceAuthReasoningPhase.DECIDING.value: VoiceAuthReasoningPhase.DECIDING.value,
                VoiceAuthReasoningPhase.HYPOTHESIZING.value: VoiceAuthReasoningPhase.HYPOTHESIZING.value,
                "error_handler": "error_handler",
            }
        )

        graph.add_conditional_edges(
            VoiceAuthReasoningPhase.HYPOTHESIZING.value,
            route_after_hypotheses,
            {
                VoiceAuthReasoningPhase.REASONING.value: VoiceAuthReasoningPhase.REASONING.value,
                VoiceAuthReasoningPhase.DECIDING.value: VoiceAuthReasoningPhase.DECIDING.value,
                "error_handler": "error_handler",
            }
        )

        graph.add_conditional_edges(
            VoiceAuthReasoningPhase.REASONING.value,
            route_after_reasoning,
            {
                VoiceAuthReasoningPhase.DECIDING.value: VoiceAuthReasoningPhase.DECIDING.value,
                "error_handler": "error_handler",
            }
        )

        graph.add_conditional_edges(
            VoiceAuthReasoningPhase.DECIDING.value,
            route_after_decision,
            {
                VoiceAuthReasoningPhase.RESPONDING.value: VoiceAuthReasoningPhase.RESPONDING.value,
                "error_handler": "error_handler",
            }
        )

        graph.add_conditional_edges(
            VoiceAuthReasoningPhase.RESPONDING.value,
            route_after_response,
            {
                VoiceAuthReasoningPhase.PERCEIVING.value: VoiceAuthReasoningPhase.PERCEIVING.value,
                VoiceAuthReasoningPhase.LEARNING.value: VoiceAuthReasoningPhase.LEARNING.value,
                END: END,
            }
        )

        graph.add_conditional_edges(
            VoiceAuthReasoningPhase.LEARNING.value,
            route_after_learning,
            {END: END}
        )

        # Error handler can route to various recovery points
        graph.add_conditional_edges(
            "error_handler",
            error_handler_route,
            {
                VoiceAuthReasoningPhase.PERCEIVING.value: VoiceAuthReasoningPhase.PERCEIVING.value,
                VoiceAuthReasoningPhase.DECIDING.value: VoiceAuthReasoningPhase.DECIDING.value,
            }
        )

        return graph

    def compile(self) -> CompiledStateGraph:
        """Compile the graph for execution."""
        if self._compiled_graph is None:
            compile_kwargs = {}
            if self.checkpointer:
                compile_kwargs["checkpointer"] = self.checkpointer

            self._compiled_graph = self._graph.compile(**compile_kwargs)
            logger.info("Voice auth graph compiled successfully")

        return self._compiled_graph

    async def authenticate(
        self,
        audio_data: bytes,
        user_id: str = "unknown",
        sample_rate: int = 16000,
        config: Optional[Dict[str, Any]] = None,
        timeout_ms: Optional[int] = None,
    ) -> VoiceAuthReasoningState:
        """
        Authenticate a user via voice biometrics.

        Args:
            audio_data: Raw audio bytes
            user_id: Expected user ID
            sample_rate: Audio sample rate
            config: Optional configuration overrides
            timeout_ms: Optional timeout override

        Returns:
            VoiceAuthReasoningState with authentication result
        """
        start_time = time.perf_counter()
        timeout = timeout_ms or GraphConfig.get_graph_timeout_ms()

        # Create initial state
        initial_state = VoiceAuthReasoningState.create_initial_state(
            session_id=f"auth_{int(time.time() * 1000)}_{self._execution_count}",
            audio_data=audio_data,
            user_id=user_id,
            sample_rate=sample_rate,
        )

        # Initialize metrics
        metrics = GraphExecutionMetrics(
            session_id=initial_state.session_id,
            start_time=start_time,
        )

        try:
            # Compile graph if needed
            compiled = self.compile()

            # Execute with timeout
            result_dict = await asyncio.wait_for(
                self._execute_graph(compiled, initial_state.model_dump()),
                timeout=timeout / 1000.0,
            )

            # Convert result to state object
            result_state = VoiceAuthReasoningState(**result_dict)

            # Update metrics
            metrics.end_time = time.perf_counter()
            metrics.total_duration_ms = (metrics.end_time - metrics.start_time) * 1000
            metrics.phases_executed = result_state.execution_trace
            metrics.phase_durations_ms = result_state.phase_timings
            metrics.final_decision = result_state.decision.value if result_state.decision else None
            metrics.final_confidence = result_state.final_confidence
            metrics.early_exit_used = result_dict.get("early_exit", False)
            metrics.hypotheses_generated = len(result_state.hypotheses)
            metrics.reasoning_depth = len(result_state.reasoning_chain)
            metrics.errors_encountered = len(result_state.errors)

            # Record metrics
            if self.enable_metrics:
                await _metrics_collector.record(metrics)

            self._last_execution_metrics = metrics
            self._execution_count += 1

            logger.info(
                f"Authentication completed: {result_state.decision} "
                f"({result_state.final_confidence:.3f}) in {metrics.total_duration_ms:.1f}ms"
            )

            return result_state

        except asyncio.TimeoutError:
            metrics.end_time = time.perf_counter()
            metrics.total_duration_ms = (metrics.end_time - metrics.start_time) * 1000
            metrics.timeouts_occurred = 1
            metrics.final_decision = "TIMEOUT"

            if self.enable_metrics:
                await _metrics_collector.record(metrics)

            logger.error(f"Authentication timed out after {timeout}ms")

            # Return timeout state
            initial_state.decision = DecisionType.TIMEOUT
            initial_state.response_text = (
                "Authentication timed out. Please try again."
            )
            return initial_state

        except Exception as e:
            metrics.end_time = time.perf_counter()
            metrics.total_duration_ms = (metrics.end_time - metrics.start_time) * 1000
            metrics.errors_encountered = 1
            metrics.final_decision = "ERROR"

            if self.enable_metrics:
                await _metrics_collector.record(metrics)

            logger.exception(f"Authentication error: {e}")

            # Return error state
            initial_state.decision = DecisionType.REJECT
            initial_state.response_text = (
                f"Authentication error occurred. Please try again."
            )
            initial_state.errors.append({
                "node": "graph",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            return initial_state

    async def _execute_graph(
        self,
        compiled: CompiledStateGraph,
        initial_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute the compiled graph."""
        config = {"recursion_limit": GraphConfig.get_max_iterations()}

        # Run the graph
        result = await compiled.ainvoke(initial_state, config=config)

        return result

    def get_node_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all nodes."""
        return {
            name: wrapper.get_stats()
            for name, wrapper in self._wrapped_nodes.items()
        }

    def get_last_metrics(self) -> Optional[GraphExecutionMetrics]:
        """Get metrics from the last execution."""
        return self._last_execution_metrics

    @staticmethod
    def get_global_metrics() -> Dict[str, Any]:
        """Get global metrics across all graph instances."""
        return _metrics_collector.get_summary()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

# Singleton instance
_graph_instance: Optional[VoiceAuthenticationReasoningGraph] = None
_graph_lock = asyncio.Lock()


async def get_voice_auth_reasoning_graph(
    force_new: bool = False,
) -> VoiceAuthenticationReasoningGraph:
    """
    Get or create the voice authentication reasoning graph.

    Uses singleton pattern for efficiency but allows forcing new instance.

    Args:
        force_new: If True, creates a new instance

    Returns:
        VoiceAuthenticationReasoningGraph instance
    """
    global _graph_instance

    async with _graph_lock:
        if _graph_instance is None or force_new:
            _graph_instance = VoiceAuthenticationReasoningGraph()
        return _graph_instance


def create_voice_auth_graph(
    checkpointer: Optional[BaseCheckpointSaver] = None,
    custom_nodes: Optional[Dict[str, BaseVoiceAuthNode]] = None,
    enable_metrics: bool = True,
) -> VoiceAuthenticationReasoningGraph:
    """
    Create a new voice authentication graph instance.

    Use this when you need a custom configuration or isolated instance.

    Args:
        checkpointer: Optional checkpoint saver
        custom_nodes: Optional custom node implementations
        enable_metrics: Whether to collect metrics

    Returns:
        New VoiceAuthenticationReasoningGraph instance
    """
    return VoiceAuthenticationReasoningGraph(
        checkpointer=checkpointer,
        custom_nodes=custom_nodes,
        enable_metrics=enable_metrics,
    )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def authenticate_voice(
    audio_data: bytes,
    user_id: str = "unknown",
    sample_rate: int = 16000,
    timeout_ms: Optional[int] = None,
) -> Tuple[DecisionType, float, str]:
    """
    Convenience function for voice authentication.

    Args:
        audio_data: Raw audio bytes
        user_id: Expected user ID
        sample_rate: Audio sample rate
        timeout_ms: Optional timeout

    Returns:
        Tuple of (decision, confidence, response_text)
    """
    graph = await get_voice_auth_reasoning_graph()
    result = await graph.authenticate(
        audio_data=audio_data,
        user_id=user_id,
        sample_rate=sample_rate,
        timeout_ms=timeout_ms,
    )

    return (
        result.decision or DecisionType.REJECT,
        result.final_confidence,
        result.response_text or "Authentication failed.",
    )


__all__ = [
    # Main class
    "VoiceAuthenticationReasoningGraph",
    # Factory functions
    "get_voice_auth_reasoning_graph",
    "create_voice_auth_graph",
    # Convenience functions
    "authenticate_voice",
    # Configuration
    "GraphConfig",
    # Metrics
    "GraphExecutionMetrics",
    "GraphMetricsCollector",
    # Routing (for testing/customization)
    "route_after_perception",
    "route_after_analysis",
    "route_after_verification",
    "route_after_evidence",
    "route_after_hypotheses",
    "route_after_reasoning",
    "route_after_decision",
    "route_after_response",
    "route_after_learning",
]
