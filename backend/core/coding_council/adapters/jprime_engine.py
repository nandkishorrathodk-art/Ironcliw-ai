"""
v84.0: JARVIS Prime Local LLM Engine for Unified Coding Council
================================================================

Production-grade adapter for routing coding and reasoning tasks to JARVIS Prime's
local LLM inference engine. Supports CodeLlama, DeepSeek Coder, Qwen, and other
GGUF models with intelligent task-based model selection.

FEATURES:
    - OpenAI-compatible API integration with JARVIS Prime
    - Intelligent model routing (coding → CodeLlama, reasoning → Qwen)
    - Aider-style code editing with git awareness
    - Multi-agent planning simulation (MetaGPT-style)
    - Circuit breaker and retry with exponential backoff
    - Streaming support with backpressure handling
    - Automatic fallback to Claude on failure
    - Health monitoring and connection pooling
    - Cost tracking ($0 for local inference)

USAGE:
    from backend.core.coding_council.adapters.jprime_engine import JPrimeUnifiedEngine

    engine = JPrimeUnifiedEngine()
    await engine.initialize()

    # Aider-style editing
    result = await engine.edit_code_aider(
        description="Fix the authentication bug",
        target_files=["auth.py"],
    )

    # Multi-agent planning
    result = await engine.plan_multi_agent(
        task="Implement user dashboard",
        context={"requirements": "..."},
    )

Author: JARVIS v84.0
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration (Environment-Driven)
# =============================================================================

def _get_env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _get_env_bool(key: str, default: bool = False) -> bool:
    return _get_env(key, str(default)).lower() in ("true", "1", "yes", "on")


def _get_env_int(key: str, default: int) -> int:
    try:
        return int(_get_env(key, str(default)))
    except ValueError:
        return default


def _get_env_float(key: str, default: float) -> float:
    try:
        return float(_get_env(key, str(default)))
    except ValueError:
        return default


# =============================================================================
# Types and Enums
# =============================================================================

class ModelTaskType(str, Enum):
    """Task types for model routing."""
    CODING = "coding"
    REASONING = "reasoning"
    MATH = "math"
    CREATIVE = "creative"
    GENERAL = "general"
    CODE_REVIEW = "code_review"
    REFACTORING = "refactoring"
    DEBUGGING = "debugging"
    PLANNING = "planning"
    ANALYSIS = "analysis"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()  # Normal operation
    OPEN = auto()  # Failing, reject calls
    HALF_OPEN = auto()  # Testing recovery


@dataclass
class JPrimeConfig:
    """Configuration for JARVIS Prime engine."""
    # Connection
    base_url: str = field(
        default_factory=lambda: _get_env("JARVIS_PRIME_URL", "http://localhost:8000")
    )
    heartbeat_file: Path = field(
        default_factory=lambda: Path.home() / ".jarvis" / "trinity" / "components" / "jarvis_prime.json"
    )

    # Models
    coding_model: str = field(
        default_factory=lambda: _get_env("JPRIME_CODING_MODEL", "codellama-7b-instruct")
    )
    reasoning_model: str = field(
        default_factory=lambda: _get_env("JPRIME_REASONING_MODEL", "qwen2.5-7b-instruct")
    )
    general_model: str = field(
        default_factory=lambda: _get_env("JPRIME_GENERAL_MODEL", "llama-3-8b-instruct")
    )
    fast_model: str = field(
        default_factory=lambda: _get_env("JPRIME_FAST_MODEL", "phi-3.5-mini")
    )

    # Inference settings
    max_tokens: int = field(
        default_factory=lambda: _get_env_int("JPRIME_MAX_TOKENS", 4096)
    )
    temperature: float = field(
        default_factory=lambda: _get_env_float("JPRIME_TEMPERATURE", 0.2)
    )
    coding_temperature: float = field(
        default_factory=lambda: _get_env_float("JPRIME_CODING_TEMPERATURE", 0.1)
    )

    # Timeouts
    request_timeout: float = field(
        default_factory=lambda: _get_env_float("JPRIME_REQUEST_TIMEOUT", 120.0)
    )
    connect_timeout: float = field(
        default_factory=lambda: _get_env_float("JPRIME_CONNECT_TIMEOUT", 10.0)
    )

    # Retry settings
    max_retries: int = field(
        default_factory=lambda: _get_env_int("JPRIME_MAX_RETRIES", 3)
    )
    retry_base_delay: float = field(
        default_factory=lambda: _get_env_float("JPRIME_RETRY_DELAY", 1.0)
    )

    # Circuit breaker
    circuit_failure_threshold: int = field(
        default_factory=lambda: _get_env_int("JPRIME_CIRCUIT_THRESHOLD", 5)
    )
    circuit_recovery_timeout: float = field(
        default_factory=lambda: _get_env_float("JPRIME_CIRCUIT_RECOVERY", 30.0)
    )

    # Fallback
    fallback_to_claude: bool = field(
        default_factory=lambda: _get_env_bool("JPRIME_FALLBACK_CLAUDE", True)
    )

    # Git integration
    auto_commit: bool = field(
        default_factory=lambda: _get_env_bool("JPRIME_AUTO_COMMIT", False)
    )


@dataclass
class InferenceResult:
    """Result from LLM inference."""
    success: bool
    content: str
    model_used: str
    tokens_used: int = 0
    inference_time_ms: float = 0.0
    cost_usd: float = 0.0  # Always 0 for local
    error: Optional[str] = None
    fallback_used: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeEditResult:
    """Result from code editing operation."""
    success: bool
    files_modified: List[str] = field(default_factory=list)
    changes_made: List[str] = field(default_factory=list)
    diff: str = ""
    commit_sha: Optional[str] = None
    error: Optional[str] = None
    inference_result: Optional[InferenceResult] = None


# =============================================================================
# Circuit Breaker
# =============================================================================

class CircuitBreaker:
    """
    v84.0: Circuit breaker for protecting against cascading failures.

    Implements the circuit breaker pattern with:
    - CLOSED: Normal operation
    - OPEN: After threshold failures, reject all calls
    - HALF_OPEN: Allow one test call to check recovery
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        name: str = "jprime",
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._success_count = 0

        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def is_closed(self) -> bool:
        return self._state == CircuitState.CLOSED

    async def can_execute(self) -> bool:
        """Check if execution is allowed."""
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if time.time() - self._last_failure_time >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    logger.info(f"Circuit {self.name}: OPEN → HALF_OPEN")
                    return True
                return False

            if self._state == CircuitState.HALF_OPEN:
                return True

            return False

    async def record_success(self):
        """Record a successful call."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= 2:  # Require 2 successes to close
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info(f"Circuit {self.name}: HALF_OPEN → CLOSED")
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0  # Reset on success

    async def record_failure(self):
        """Record a failed call."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit {self.name}: HALF_OPEN → OPEN (test failed)")

            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(
                        f"Circuit {self.name}: CLOSED → OPEN "
                        f"(failures: {self._failure_count})"
                    )


# =============================================================================
# Task Classifier
# =============================================================================

class TaskClassifier:
    """
    v84.0: Intelligent task classification for model routing.

    Analyzes task descriptions to determine the best model category.
    """

    # Weighted keywords for each task type
    TASK_PATTERNS: Dict[ModelTaskType, Dict[str, float]] = {
        ModelTaskType.CODING: {
            "code": 1.0, "function": 1.0, "class": 1.0, "method": 0.8,
            "implement": 0.9, "write": 0.5, "create": 0.4, "add": 0.4,
            "python": 1.0, "javascript": 1.0, "typescript": 1.0, "java": 0.9,
            "rust": 0.9, "go": 0.9, "c++": 0.9, "swift": 0.9,
            "def ": 1.0, "async def": 1.0, "const ": 0.8, "let ": 0.8,
        },
        ModelTaskType.DEBUGGING: {
            "bug": 1.0, "fix": 0.9, "error": 0.9, "issue": 0.7,
            "crash": 1.0, "exception": 0.9, "traceback": 1.0,
            "doesn't work": 0.8, "not working": 0.8, "broken": 0.8,
            "debug": 1.0, "investigate": 0.6,
        },
        ModelTaskType.REFACTORING: {
            "refactor": 1.0, "restructure": 0.9, "reorganize": 0.8,
            "clean up": 0.8, "simplify": 0.7, "optimize": 0.7,
            "rename": 0.6, "extract": 0.7, "move": 0.5,
            "decouple": 0.8, "modularize": 0.8,
        },
        ModelTaskType.CODE_REVIEW: {
            "review": 1.0, "check": 0.5, "audit": 0.8,
            "quality": 0.7, "best practice": 0.8, "improve": 0.6,
            "suggestions": 0.6, "feedback": 0.6,
        },
        ModelTaskType.REASONING: {
            "analyze": 0.8, "explain": 0.8, "understand": 0.7,
            "why": 0.6, "how": 0.5, "reason": 0.9,
            "logic": 0.9, "think": 0.7, "consider": 0.6,
            "evaluate": 0.8, "compare": 0.7,
        },
        ModelTaskType.MATH: {
            "calculate": 1.0, "math": 1.0, "equation": 0.9,
            "formula": 0.9, "number": 0.6, "compute": 0.9,
            "algorithm": 0.8, "optimize": 0.5,
        },
        ModelTaskType.PLANNING: {
            "plan": 1.0, "design": 0.8, "architect": 0.9,
            "strategy": 0.8, "roadmap": 0.8, "steps": 0.6,
            "approach": 0.7, "structure": 0.6,
        },
        ModelTaskType.ANALYSIS: {
            "analyze": 1.0, "examine": 0.8, "investigate": 0.7,
            "explore": 0.6, "study": 0.6, "review": 0.5,
            "assess": 0.8, "evaluate": 0.8,
        },
    }

    # Code file extensions
    CODE_EXTENSIONS = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".kt",
        ".go", ".rs", ".c", ".cpp", ".h", ".hpp", ".swift",
        ".rb", ".php", ".cs", ".scala", ".clj", ".ex", ".exs",
    }

    @classmethod
    def classify(
        cls,
        description: str,
        target_files: Optional[List[str]] = None,
    ) -> Tuple[ModelTaskType, float]:
        """
        Classify a task based on description and target files.

        Args:
            description: Task description
            target_files: Optional list of target file paths

        Returns:
            Tuple of (task_type, confidence_score)
        """
        description_lower = description.lower()
        scores: Dict[ModelTaskType, float] = defaultdict(float)

        # Score based on keywords
        for task_type, patterns in cls.TASK_PATTERNS.items():
            for keyword, weight in patterns.items():
                if keyword in description_lower:
                    scores[task_type] += weight

        # Boost coding-related if target files are code files
        if target_files:
            code_file_count = sum(
                1 for f in target_files
                if any(f.endswith(ext) for ext in cls.CODE_EXTENSIONS)
            )
            if code_file_count > 0:
                code_boost = min(code_file_count * 0.5, 2.0)
                scores[ModelTaskType.CODING] += code_boost
                scores[ModelTaskType.DEBUGGING] += code_boost * 0.5
                scores[ModelTaskType.REFACTORING] += code_boost * 0.5

        # Detect code snippets in description
        code_indicators = ["```", "def ", "class ", "function ", "const ", "import "]
        for indicator in code_indicators:
            if indicator in description:
                scores[ModelTaskType.CODING] += 0.5

        if not scores:
            return ModelTaskType.GENERAL, 0.5

        # Get highest scoring type
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        # Normalize confidence (cap at 1.0)
        confidence = min(best_score / 3.0, 1.0)

        return best_type, confidence

    @classmethod
    def get_recommended_model(
        cls,
        task_type: ModelTaskType,
        config: JPrimeConfig,
    ) -> str:
        """Get recommended model for a task type."""
        model_map = {
            ModelTaskType.CODING: config.coding_model,
            ModelTaskType.DEBUGGING: config.coding_model,
            ModelTaskType.REFACTORING: config.coding_model,
            ModelTaskType.CODE_REVIEW: config.coding_model,
            ModelTaskType.REASONING: config.reasoning_model,
            ModelTaskType.MATH: config.reasoning_model,
            ModelTaskType.PLANNING: config.reasoning_model,
            ModelTaskType.ANALYSIS: config.reasoning_model,
            ModelTaskType.CREATIVE: config.general_model,
            ModelTaskType.GENERAL: config.general_model,
        }
        return model_map.get(task_type, config.general_model)


# =============================================================================
# JARVIS Prime Client
# =============================================================================

class JPrimeClient:
    """
    v84.0: Async HTTP client for JARVIS Prime OpenAI-compatible API.

    Features:
    - Connection pooling
    - Automatic retry with backoff
    - Circuit breaker integration
    - Streaming support
    - Health monitoring
    """

    def __init__(self, config: JPrimeConfig):
        self.config = config
        self._session: Optional[Any] = None
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=config.circuit_failure_threshold,
            recovery_timeout=config.circuit_recovery_timeout,
        )
        self._last_health_check = 0.0
        self._is_healthy = False

    async def initialize(self):
        """Initialize the client."""
        try:
            import aiohttp
            connector = aiohttp.TCPConnector(
                limit=10,
                limit_per_host=5,
                keepalive_timeout=30,
            )
            timeout = aiohttp.ClientTimeout(
                total=self.config.request_timeout,
                connect=self.config.connect_timeout,
            )
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
            )
            logger.info(f"JPrimeClient initialized: {self.config.base_url}")
        except ImportError:
            logger.error("aiohttp not installed - pip install aiohttp")
            raise

    async def close(self):
        """Close the client."""
        if self._session:
            await self._session.close()
            self._session = None

    async def check_health(self) -> bool:
        """Check if JARVIS Prime is healthy."""
        # Rate limit health checks
        if time.time() - self._last_health_check < 5.0:
            return self._is_healthy

        self._last_health_check = time.time()

        # First check heartbeat file
        if self.config.heartbeat_file.exists():
            try:
                with open(self.config.heartbeat_file) as f:
                    data = json.load(f)
                    heartbeat_age = time.time() - data.get("timestamp", 0)
                    if heartbeat_age < 30:
                        self._is_healthy = True
                        return True
            except Exception:
                pass

        # Then try HTTP health check
        try:
            url = f"{self.config.base_url}/health"
            async with self._session.get(url, timeout=5.0) as response:
                self._is_healthy = response.status == 200
                return self._is_healthy
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            self._is_healthy = False
            return False

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> InferenceResult:
        """
        Call the chat completion endpoint.

        Args:
            messages: List of message dicts with role and content
            model: Model to use (or auto-select)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Enable streaming

        Returns:
            InferenceResult with response
        """
        if not await self._circuit_breaker.can_execute():
            return InferenceResult(
                success=False,
                content="",
                model_used=model or "unknown",
                error="Circuit breaker open - JARVIS Prime unavailable",
            )

        url = f"{self.config.base_url}/v1/chat/completions"
        payload = {
            "messages": messages,
            "model": model or self.config.general_model,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "stream": stream,
        }

        start_time = time.time()

        for attempt in range(self.config.max_retries):
            try:
                async with self._session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        await self._circuit_breaker.record_success()

                        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        usage = data.get("usage", {})

                        return InferenceResult(
                            success=True,
                            content=content,
                            model_used=model or self.config.general_model,
                            tokens_used=usage.get("total_tokens", 0),
                            inference_time_ms=(time.time() - start_time) * 1000,
                            cost_usd=0.0,  # Local is free!
                        )
                    else:
                        error_text = await response.text()
                        raise Exception(f"HTTP {response.status}: {error_text}")

            except asyncio.TimeoutError:
                logger.warning(f"Request timeout (attempt {attempt + 1})")
            except Exception as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")

            if attempt < self.config.max_retries - 1:
                delay = self.config.retry_base_delay * (2 ** attempt)
                await asyncio.sleep(delay)

        await self._circuit_breaker.record_failure()

        return InferenceResult(
            success=False,
            content="",
            model_used=model or self.config.general_model,
            error=f"Failed after {self.config.max_retries} attempts",
        )

    async def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens."""
        url = f"{self.config.base_url}/v1/chat/completions"
        payload = {
            "messages": messages,
            "model": model or self.config.general_model,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "stream": True,
        }

        try:
            async with self._session.post(url, json=payload) as response:
                if response.status != 200:
                    return

                async for line in response.content:
                    line = line.decode().strip()
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"Streaming error: {e}")


# =============================================================================
# JARVIS Prime Unified Engine
# =============================================================================

class JPrimeUnifiedEngine:
    """
    v84.0: Unified engine for JARVIS Prime local LLM inference.

    Provides:
    - Aider-style code editing
    - MetaGPT-style multi-agent planning
    - Direct chat completion
    - Intelligent model routing
    - Git integration
    - Fallback to Claude
    """

    def __init__(self, config: Optional[JPrimeConfig] = None):
        self.config = config or JPrimeConfig()
        self._client: Optional[JPrimeClient] = None
        self._initialized = False
        self._classifier = TaskClassifier()

        # Statistics
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "fallback_requests": 0,
            "total_tokens": 0,
            "total_inference_time_ms": 0.0,
        }

    async def initialize(self):
        """Initialize the engine."""
        if self._initialized:
            return

        self._client = JPrimeClient(self.config)
        await self._client.initialize()
        self._initialized = True

        logger.info("JPrimeUnifiedEngine initialized")

    async def close(self):
        """Close the engine."""
        if self._client:
            await self._client.close()
        self._initialized = False

    async def is_available(self) -> bool:
        """Check if JARVIS Prime is available."""
        if not self._initialized:
            return False
        return await self._client.check_health()

    def _select_model(
        self,
        task_type: Optional[ModelTaskType] = None,
        description: Optional[str] = None,
        target_files: Optional[List[str]] = None,
    ) -> str:
        """Select the best model for a task."""
        if task_type:
            return self._classifier.get_recommended_model(task_type, self.config)

        if description:
            detected_type, _ = self._classifier.classify(description, target_files)
            return self._classifier.get_recommended_model(detected_type, self.config)

        return self.config.general_model

    # =========================================================================
    # Aider-Style Code Editing
    # =========================================================================

    async def edit_code_aider(
        self,
        description: str,
        target_files: List[str],
        repo_path: Optional[Path] = None,
        context_files: Optional[List[str]] = None,
        auto_commit: Optional[bool] = None,
    ) -> CodeEditResult:
        """
        Edit code using Aider-style prompting.

        Args:
            description: What changes to make
            target_files: Files to modify
            repo_path: Repository root path
            context_files: Additional context files (read-only)
            auto_commit: Whether to auto-commit changes

        Returns:
            CodeEditResult with changes
        """
        self._stats["total_requests"] += 1
        repo_path = repo_path or Path.cwd()
        auto_commit = auto_commit if auto_commit is not None else self.config.auto_commit

        # Select model
        model = self._select_model(
            description=description,
            target_files=target_files,
        )

        # Build context
        file_contents = {}
        for file_path in target_files:
            full_path = repo_path / file_path
            if full_path.exists():
                try:
                    file_contents[file_path] = full_path.read_text()
                except Exception as e:
                    logger.warning(f"Could not read {file_path}: {e}")

        if context_files:
            for file_path in context_files:
                full_path = repo_path / file_path
                if full_path.exists() and file_path not in file_contents:
                    try:
                        file_contents[file_path] = full_path.read_text()
                    except Exception:
                        pass

        # Build Aider-style prompt
        prompt = self._build_aider_prompt(description, target_files, file_contents)

        # Call LLM
        messages = [
            {"role": "system", "content": self._get_aider_system_prompt()},
            {"role": "user", "content": prompt},
        ]

        result = await self._client.chat_completion(
            messages=messages,
            model=model,
            temperature=self.config.coding_temperature,
        )

        if not result.success:
            self._stats["failed_requests"] += 1

            # Try fallback to Claude if enabled
            if self.config.fallback_to_claude:
                fallback_result = await self._fallback_to_claude(messages)
                if fallback_result.success:
                    result = fallback_result
                    result.fallback_used = True
                    self._stats["fallback_requests"] += 1

        if not result.success:
            return CodeEditResult(
                success=False,
                error=result.error,
                inference_result=result,
            )

        self._stats["successful_requests"] += 1
        self._stats["total_tokens"] += result.tokens_used
        self._stats["total_inference_time_ms"] += result.inference_time_ms

        # Parse response and apply changes
        changes = self._parse_aider_response(result.content)
        files_modified = []
        changes_made = []

        for file_path, new_content in changes.items():
            full_path = repo_path / file_path
            try:
                # Write file
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(new_content)
                files_modified.append(file_path)
                changes_made.append(f"Modified {file_path}")
            except Exception as e:
                logger.error(f"Failed to write {file_path}: {e}")

        # Auto-commit if enabled
        commit_sha = None
        if auto_commit and files_modified:
            commit_sha = await self._git_commit(
                repo_path,
                files_modified,
                f"[JPrime] {description[:50]}",
            )

        return CodeEditResult(
            success=len(files_modified) > 0,
            files_modified=files_modified,
            changes_made=changes_made,
            commit_sha=commit_sha,
            inference_result=result,
        )

    def _get_aider_system_prompt(self) -> str:
        """Get Aider-style system prompt."""
        return """You are an expert software engineer. Your task is to edit code files based on the user's request.

RULES:
1. Return the COMPLETE updated file content for each file you modify
2. Use the format: ```filename.ext ... ``` to indicate file content
3. Only modify files that need changes
4. Keep the existing code style and formatting
5. Add appropriate comments where helpful
6. Do not add unnecessary changes

RESPONSE FORMAT:
For each file you modify, output:
```path/to/file.py
<complete file content>
```

Only output file blocks and brief explanations. Do not include the original file content outside of file blocks."""

    def _build_aider_prompt(
        self,
        description: str,
        target_files: List[str],
        file_contents: Dict[str, str],
    ) -> str:
        """Build Aider-style edit prompt."""
        prompt_parts = [f"## Task\n{description}\n"]

        prompt_parts.append("## Files to Edit")
        for file_path in target_files:
            if file_path in file_contents:
                prompt_parts.append(f"\n### {file_path}\n```\n{file_contents[file_path]}\n```")

        return "\n".join(prompt_parts)

    def _parse_aider_response(self, response: str) -> Dict[str, str]:
        """Parse Aider-style response into file changes."""
        changes = {}

        # Match code blocks with filenames
        pattern = r"```(\S+)\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)

        for filename, content in matches:
            # Clean up filename (remove leading ./ or /)
            filename = filename.strip().lstrip("./").lstrip("/")
            if filename and not filename.startswith("```"):
                changes[filename] = content.strip()

        return changes

    # =========================================================================
    # MetaGPT-Style Planning
    # =========================================================================

    async def plan_multi_agent(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        agents: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a multi-agent plan (MetaGPT-style).

        Args:
            task: Task description
            context: Additional context
            agents: Agent roles to simulate

        Returns:
            Dict with plan from each agent role
        """
        self._stats["total_requests"] += 1
        agents = agents or ["ProductManager", "Architect", "Engineer"]
        model = self._select_model(task_type=ModelTaskType.PLANNING)

        plan = {}

        for agent in agents:
            prompt = self._build_agent_prompt(agent, task, context, plan)
            messages = [
                {"role": "system", "content": f"You are a {agent}. Provide your analysis and recommendations."},
                {"role": "user", "content": prompt},
            ]

            result = await self._client.chat_completion(
                messages=messages,
                model=model,
                temperature=0.3,
            )

            if result.success:
                plan[agent] = {
                    "output": result.content,
                    "tokens_used": result.tokens_used,
                }
                self._stats["total_tokens"] += result.tokens_used
            else:
                plan[agent] = {"error": result.error}

        self._stats["successful_requests"] += 1
        return plan

    def _build_agent_prompt(
        self,
        agent: str,
        task: str,
        context: Optional[Dict[str, Any]],
        previous_outputs: Dict[str, Any],
    ) -> str:
        """Build prompt for a specific agent role."""
        prompt_parts = [f"## Task\n{task}\n"]

        if context:
            prompt_parts.append(f"## Context\n{json.dumps(context, indent=2)}\n")

        if previous_outputs:
            prompt_parts.append("## Previous Analysis")
            for prev_agent, output in previous_outputs.items():
                if "output" in output:
                    prompt_parts.append(f"\n### {prev_agent}\n{output['output']}\n")

        agent_instructions = {
            "ProductManager": "Create a Product Requirements Document (PRD) with user stories, acceptance criteria, and success metrics.",
            "Architect": "Design the system architecture, including components, data flow, and technical decisions.",
            "Engineer": "Create a detailed implementation plan with specific code changes, files to modify, and step-by-step instructions.",
        }

        prompt_parts.append(f"## Your Role\n{agent_instructions.get(agent, 'Provide your analysis.')}")

        return "\n".join(prompt_parts)

    # =========================================================================
    # Direct Chat
    # =========================================================================

    async def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        task_type: Optional[ModelTaskType] = None,
    ) -> InferenceResult:
        """
        Direct chat completion.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            task_type: Task type for model selection

        Returns:
            InferenceResult
        """
        self._stats["total_requests"] += 1
        model = self._select_model(task_type=task_type, description=prompt)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        result = await self._client.chat_completion(
            messages=messages,
            model=model,
        )

        if result.success:
            self._stats["successful_requests"] += 1
            self._stats["total_tokens"] += result.tokens_used
            self._stats["total_inference_time_ms"] += result.inference_time_ms
        else:
            self._stats["failed_requests"] += 1

        return result

    # =========================================================================
    # Fallback
    # =========================================================================

    async def _fallback_to_claude(
        self,
        messages: List[Dict[str, str]],
    ) -> InferenceResult:
        """Fallback to Claude API when JARVIS Prime fails."""
        try:
            # Import Anthropic client
            from anthropic import AsyncAnthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                return InferenceResult(
                    success=False,
                    content="",
                    model_used="claude",
                    error="No ANTHROPIC_API_KEY set for fallback",
                )

            client = AsyncAnthropic(api_key=api_key)

            # Convert messages format
            system_msg = ""
            claude_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    claude_messages.append(msg)

            start_time = time.time()
            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=self.config.max_tokens,
                system=system_msg,
                messages=claude_messages,
            )

            return InferenceResult(
                success=True,
                content=response.content[0].text,
                model_used="claude-sonnet-4-20250514",
                tokens_used=response.usage.input_tokens + response.usage.output_tokens,
                inference_time_ms=(time.time() - start_time) * 1000,
                cost_usd=self._estimate_claude_cost(response.usage),
                fallback_used=True,
            )

        except Exception as e:
            logger.error(f"Claude fallback failed: {e}")
            return InferenceResult(
                success=False,
                content="",
                model_used="claude",
                error=str(e),
            )

    def _estimate_claude_cost(self, usage: Any) -> float:
        """Estimate Claude API cost."""
        # Claude Sonnet pricing (approximate)
        input_cost = usage.input_tokens * 0.003 / 1000
        output_cost = usage.output_tokens * 0.015 / 1000
        return input_cost + output_cost

    # =========================================================================
    # Git Integration
    # =========================================================================

    async def _git_commit(
        self,
        repo_path: Path,
        files: List[str],
        message: str,
    ) -> Optional[str]:
        """Commit changes to git."""
        try:
            # Add files
            for file_path in files:
                subprocess.run(
                    ["git", "add", file_path],
                    cwd=repo_path,
                    check=True,
                    capture_output=True,
                )

            # Commit
            result = subprocess.run(
                ["git", "commit", "-m", message],
                cwd=repo_path,
                check=True,
                capture_output=True,
            )

            # Get commit SHA
            sha_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
            )
            return sha_result.stdout.strip()[:8]

        except subprocess.CalledProcessError as e:
            logger.warning(f"Git commit failed: {e}")
            return None

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            **self._stats,
            "available": self._initialized,
            "circuit_breaker_state": self._client._circuit_breaker.state.name if self._client else "unknown",
        }


# =============================================================================
# Factory Function
# =============================================================================

_engine: Optional[JPrimeUnifiedEngine] = None


async def get_jprime_engine() -> JPrimeUnifiedEngine:
    """Get or create the global JPrime engine."""
    global _engine

    if _engine is None:
        _engine = JPrimeUnifiedEngine()
        await _engine.initialize()

    return _engine


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "JPrimeUnifiedEngine",
    "JPrimeConfig",
    "JPrimeClient",
    "InferenceResult",
    "CodeEditResult",
    "ModelTaskType",
    "TaskClassifier",
    "CircuitBreaker",
    "get_jprime_engine",
]
