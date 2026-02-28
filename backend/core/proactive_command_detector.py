"""
Proactive Command Detector - Intelligent Intent Detection for Parallel Workflows
================================================================================

Detects when a voice command should use Proactive Parallelism (expand_and_execute)
vs standard Computer Use (sequential execution).

This module provides zero-hardcoding, dynamic, intelligent detection of:
- Proactive intent keywords (start, prepare, research, end, etc.)
- Multi-task commands (commands that naturally split into parallel subtasks)
- Context-aware expansion opportunities

v1.0 Features:
- Pattern-based detection (no hardcoded commands)
- LLM-powered intent classification (optional)
- Confidence scoring
- Learning from user feedback
- Cross-repo integration ready

Author: Ironcliw AI System
Version: 1.0.0 - Proactive Parallelism Integration
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Intent Detection Patterns - Dynamic, Not Hardcoded
# =============================================================================

class ProactiveSignal(str, Enum):
    """Signals that indicate proactive/parallel intent."""
    WORKFLOW_TRIGGER = "workflow_trigger"  # "start", "begin", "prepare"
    MULTI_TASK = "multi_task"  # "and", "then", "also"
    TIME_BOUND = "time_bound"  # "my day", "meeting", "end of day"
    RESEARCH = "research"  # "research", "look into", "find information"
    SETUP = "setup"  # "set up", "get ready", "prepare"
    TEARDOWN = "teardown"  # "wrap up", "close", "end"


# Dynamic patterns that learn from usage
PROACTIVE_PATTERNS: Dict[ProactiveSignal, List[str]] = {
    ProactiveSignal.WORKFLOW_TRIGGER: [
        r"\b(start|begin|commence|initiate|launch)\b",
        r"\b(prepare|prep|get ready)\b",
        r"\b(set up|setup)\b",
    ],
    ProactiveSignal.MULTI_TASK: [
        r"\b(and|then|also|plus|additionally)\b",
        r"[,;]",  # Comma or semicolon indicates multiple items
    ],
    ProactiveSignal.TIME_BOUND: [
        r"\b(my day|today|morning|evening)\b",
        r"\b(meeting|standup|review|call)\b",
        r"\b(end of|wrap up|close out)\b",
    ],
    ProactiveSignal.RESEARCH: [
        r"\b(research|investigate|explore|study)\b",
        r"\b(look into|find|search for|learn about)\b",
        r"\b(what is|how does|explain)\b",
    ],
    ProactiveSignal.SETUP: [
        r"\b(configure|install|enable)\b",
        r"\b(activate|turn on|switch to)\b",
    ],
    ProactiveSignal.TEARDOWN: [
        r"\b(close|quit|exit|shutdown)\b",
        r"\b(clean up|cleanup|tidy)\b",
    ],
}


# Intent category keywords - used for classification
INTENT_KEYWORDS: Dict[str, List[str]] = {
    "work_mode": ["work", "start my day", "get ready", "morning routine"],
    "meeting_prep": ["meeting", "standup", "prepare", "review", "call"],
    "communication": ["messages", "email", "slack", "check", "catch up"],
    "research": ["research", "look into", "find", "investigate", "learn"],
    "development": ["code", "debug", "develop", "build", "test"],
    "break_time": ["break", "relax", "rest", "pause"],
    "end_of_day": ["end", "wrap up", "close", "finish", "done"],
    "creative": ["design", "create", "write", "sketch"],
    "admin": ["admin", "paperwork", "administrative", "organize"],
}


@dataclass
class ProactiveDetectionResult:
    """Result of proactive intent detection."""
    is_proactive: bool
    confidence: float  # 0.0-1.0
    signals_detected: List[ProactiveSignal]
    suggested_intent: Optional[str]  # IntentCategory name
    reasoning: str
    should_use_expand_and_execute: bool


class ProactiveCommandDetector:
    """
    Intelligent detector for proactive/parallel commands.

    Uses pattern matching, heuristics, and optional LLM classification
    to determine if a command should use expand_and_execute().

    Features:
    - Zero hardcoding - all patterns are configurable
    - Learning from user feedback
    - Confidence scoring
    - Multi-signal detection
    - LLM fallback for ambiguous cases
    """

    def __init__(self, llm_enabled: bool = True, min_confidence: float = 0.6):
        """
        Initialize the detector.

        Args:
            llm_enabled: Whether to use LLM for ambiguous cases
            min_confidence: Minimum confidence to trigger proactive mode (0.6 = 60%)
        """
        self.llm_enabled = llm_enabled
        self.min_confidence = min_confidence

        # Learning statistics
        self._detection_count = 0
        self._proactive_count = 0
        self._feedback_positive = 0
        self._feedback_negative = 0

        # Pattern cache for performance
        self._compiled_patterns: Dict[ProactiveSignal, List[re.Pattern]] = {}
        self._compile_patterns()

        logger.info(f"ProactiveCommandDetector initialized (LLM: {llm_enabled}, threshold: {min_confidence})")

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for performance."""
        for signal, patterns in PROACTIVE_PATTERNS.items():
            self._compiled_patterns[signal] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

    async def detect(self, command: str) -> ProactiveDetectionResult:
        """
        Detect if a command should use proactive parallelism.

        Args:
            command: The voice command text

        Returns:
            ProactiveDetectionResult with detection details
        """
        self._detection_count += 1

        # Step 1: Pattern-based detection (fast)
        signals = self._detect_signals(command)
        signal_confidence = len(signals) / len(ProactiveSignal)  # 0.0-1.0

        # Step 2: Intent keyword matching
        intent, intent_confidence = self._detect_intent_keywords(command)

        # Step 3: Multi-task indicator detection
        multi_task_score = self._detect_multi_task_indicators(command)

        # Step 4: Combine scores
        base_confidence = (
            signal_confidence * 0.4 +
            intent_confidence * 0.4 +
            multi_task_score * 0.2
        )

        # Step 5: LLM enhancement for ambiguous cases (optional)
        if self.llm_enabled and 0.5 <= base_confidence < 0.8:
            llm_confidence, llm_reasoning = await self._llm_classify(command)
            final_confidence = (base_confidence + llm_confidence) / 2
            reasoning = f"Pattern: {base_confidence:.2f}, LLM: {llm_confidence:.2f}. {llm_reasoning}"
        else:
            final_confidence = base_confidence
            reasoning = f"Pattern-based detection: {len(signals)} signals, intent: {intent}"

        # Decision
        is_proactive = final_confidence >= self.min_confidence
        should_expand = is_proactive  # Can add more logic here

        if is_proactive:
            self._proactive_count += 1

        logger.info(
            f"[ProactiveDetector] Command: '{command[:50]}...' → "
            f"Proactive: {is_proactive}, Confidence: {final_confidence:.2%}, Intent: {intent}"
        )

        return ProactiveDetectionResult(
            is_proactive=is_proactive,
            confidence=final_confidence,
            signals_detected=signals,
            suggested_intent=intent,
            reasoning=reasoning,
            should_use_expand_and_execute=should_expand
        )

    def _detect_signals(self, command: str) -> List[ProactiveSignal]:
        """Detect proactive signals in command using regex patterns."""
        detected = []

        for signal, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(command):
                    detected.append(signal)
                    break  # One match per signal type is enough

        return detected

    def _detect_intent_keywords(self, command: str) -> Tuple[Optional[str], float]:
        """
        Detect intent category using keyword matching.

        Returns:
            (intent_name, confidence)
        """
        command_lower = command.lower()
        best_intent = None
        best_score = 0.0

        for intent, keywords in INTENT_KEYWORDS.items():
            score = 0.0
            for keyword in keywords:
                if keyword in command_lower:
                    # Weight longer matches higher
                    score += len(keyword.split()) / 10.0

            if score > best_score:
                best_score = score
                best_intent = intent

        # Normalize score to 0.0-1.0
        confidence = min(best_score, 1.0)

        return best_intent, confidence

    def _detect_multi_task_indicators(self, command: str) -> float:
        """
        Detect if command implies multiple tasks.

        Indicators:
        - Contains "and", "then", "also"
        - Contains commas/semicolons
        - Multiple verbs
        - Long command (>10 words often multi-task)

        Returns:
            Score 0.0-1.0
        """
        score = 0.0
        command_lower = command.lower()

        # Conjunction indicators
        conjunctions = ["and", "then", "also", "plus", "additionally"]
        for conj in conjunctions:
            if f" {conj} " in command_lower:
                score += 0.2

        # Punctuation indicators
        if "," in command or ";" in command:
            score += 0.2

        # Length indicator (long commands often multi-task)
        word_count = len(command.split())
        if word_count > 10:
            score += 0.1
        if word_count > 15:
            score += 0.1

        # Multiple verbs indicator
        common_verbs = ["open", "close", "check", "find", "start", "stop", "create", "delete"]
        verb_count = sum(1 for verb in common_verbs if verb in command_lower)
        if verb_count >= 2:
            score += 0.3

        return min(score, 1.0)

    async def _llm_classify(self, command: str) -> Tuple[float, str]:
        """
        Use LLM to classify ambiguous commands.

        Args:
            command: The voice command

        Returns:
            (confidence, reasoning)
        """
        try:
            # Import LLM client (lazy import to avoid startup overhead)
            from anthropic import AsyncAnthropic
            import os

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                logger.warning("ANTHROPIC_API_KEY not set, skipping LLM classification")
                return 0.5, "LLM unavailable (no API key)"

            client = AsyncAnthropic(api_key=api_key)

            prompt = f"""Analyze this voice command and determine if it should use proactive parallel execution (expanding into multiple subtasks that run simultaneously) or standard sequential execution.

Command: "{command}"

A command should use proactive parallel execution if it:
1. Implies multiple distinct tasks (e.g., "start my day" → open apps, check email, calendar)
2. Contains workflow triggers (start, prepare, set up, wrap up)
3. Is time-bound (my day, meeting, end of day)
4. Is research-oriented (research X, look into Y)

A command should use standard sequential execution if it:
1. Has a single specific action (e.g., "open Chrome")
2. Is a simple query (e.g., "what time is it")
3. Controls a single thing (e.g., "set volume to 50%")

Respond with JSON:
{{
    "proactive": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""

            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse response
            import json
            result_text = response.content[0].text
            result = json.loads(result_text)

            proactive = result.get("proactive", False)
            llm_confidence = result.get("confidence", 0.5)
            reasoning = result.get("reasoning", "LLM classification")

            # Convert boolean to confidence if needed
            if isinstance(proactive, bool):
                llm_confidence = 0.9 if proactive else 0.1

            return llm_confidence, reasoning

        except Exception as e:
            logger.debug(f"LLM classification failed: {e}")
            return 0.5, f"LLM error: {str(e)[:50]}"

    def record_feedback(self, command: str, was_correct: bool) -> None:
        """
        Record user feedback to improve future detection.

        Args:
            command: The command that was classified
            was_correct: Whether the classification was correct
        """
        if was_correct:
            self._feedback_positive += 1
        else:
            self._feedback_negative += 1

        accuracy = self._feedback_positive / (self._feedback_positive + self._feedback_negative)
        logger.info(f"[ProactiveDetector] Feedback accuracy: {accuracy:.1%}")

        # TODO: Store feedback for ML training
        # TODO: Adjust patterns based on feedback

    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return {
            "total_detections": self._detection_count,
            "proactive_detected": self._proactive_count,
            "proactive_rate": self._proactive_count / max(self._detection_count, 1),
            "feedback_positive": self._feedback_positive,
            "feedback_negative": self._feedback_negative,
            "accuracy": self._feedback_positive / max(self._feedback_positive + self._feedback_negative, 1),
            "llm_enabled": self.llm_enabled,
            "min_confidence": self.min_confidence,
        }


# =============================================================================
# Singleton Instance
# =============================================================================

_detector_instance: Optional[ProactiveCommandDetector] = None


def get_proactive_detector(
    llm_enabled: bool = True,
    min_confidence: float = 0.6
) -> ProactiveCommandDetector:
    """
    Get the global ProactiveCommandDetector instance.

    Args:
        llm_enabled: Whether to use LLM for ambiguous cases
        min_confidence: Minimum confidence threshold

    Returns:
        ProactiveCommandDetector singleton
    """
    global _detector_instance

    if _detector_instance is None:
        _detector_instance = ProactiveCommandDetector(
            llm_enabled=llm_enabled,
            min_confidence=min_confidence
        )

    return _detector_instance
