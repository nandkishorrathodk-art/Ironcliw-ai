"""
Turn Detector (Layer 3)
========================

Detects when the user has finished speaking their conversational turn.
V1 uses adaptive silence timing based on question context.

The turn detector operates on VAD output (speech/silence events) and does
NOT depend on transcript content — avoiding the chicken-and-egg problem
where STT needs turn boundaries, but turn detection would need STT output.

Timing thresholds:
    - 300ms after yes/no questions (short expected response)
    - 600ms default
    - 900ms after open-ended questions (user may be thinking)

Architecture:
    VAD (speech/silence) ──▶ TurnDetector ──▶ "turn_end" event
                                  ▲
                                  │
                        set_question_context()
                        (from LLM pipeline)
"""

import logging
import os
import time
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class QuestionType(Enum):
    """Type of the last question Ironcliw asked, affects silence threshold."""
    YES_NO = "yes_no"
    OPEN_ENDED = "open_ended"
    DEFAULT = "default"
    COMMAND = "command"       # Wake word / command mode
    FOLLOW_UP = "follow_up"  # Follow-up question in conversation


# Silence thresholds (ms) — configurable via env vars
_THRESHOLDS = {
    QuestionType.YES_NO: int(os.getenv("Ironcliw_TURN_SILENCE_YESNO_MS", "300")),
    QuestionType.OPEN_ENDED: int(os.getenv("Ironcliw_TURN_SILENCE_OPEN_MS", "900")),
    QuestionType.DEFAULT: int(os.getenv("Ironcliw_TURN_SILENCE_DEFAULT_MS", "600")),
    QuestionType.COMMAND: int(os.getenv("Ironcliw_TURN_SILENCE_COMMAND_MS", "500")),
    QuestionType.FOLLOW_UP: int(os.getenv("Ironcliw_TURN_SILENCE_FOLLOWUP_MS", "700")),
}

# Minimum speech duration before a turn can end (prevent false triggers)
_MIN_SPEECH_MS = int(os.getenv("Ironcliw_TURN_MIN_SPEECH_MS", "200"))


class TurnDetector:
    """
    Detects conversational turn boundaries from VAD events.

    V1 Algorithm:
        1. VAD reports speech → start tracking
        2. VAD reports silence → start silence timer
        3. If speech resumes before timer expires → cancel timer
        4. If timer expires → emit "turn_end"

    The timer duration adapts based on the question context set by the
    conversation pipeline.
    """

    def __init__(self):
        self._question_type = QuestionType.DEFAULT
        self._speech_active = False
        self._speech_start_ms: Optional[float] = None
        self._silence_start_ms: Optional[float] = None
        self._total_speech_ms: float = 0.0
        self._last_turn_end_ms: float = 0.0

        # Stats
        self._turns_detected = 0
        self._false_starts = 0  # Speech too short to count

    def set_question_context(self, question_type: str) -> None:
        """
        Set the expected response type for adaptive silence thresholds.

        Args:
            question_type: One of "yes_no", "open_ended", "default",
                          "command", "follow_up"
        """
        try:
            self._question_type = QuestionType(question_type)
        except ValueError:
            self._question_type = QuestionType.DEFAULT
            logger.debug(
                f"[TurnDetector] Unknown question type '{question_type}', "
                f"using default"
            )

    @property
    def silence_threshold_ms(self) -> int:
        """Current silence threshold based on question context."""
        return _THRESHOLDS.get(self._question_type, _THRESHOLDS[QuestionType.DEFAULT])

    def on_vad_result(
        self, is_speech: bool, timestamp_ms: Optional[float] = None
    ) -> Optional[str]:
        """
        Process a VAD result and detect turn boundaries.

        Args:
            is_speech: True if VAD detected speech, False for silence
            timestamp_ms: Optional timestamp (ms). If None, uses current time.

        Returns:
            "turn_end" if a turn boundary was detected, None otherwise.
        """
        now_ms = timestamp_ms if timestamp_ms is not None else time.time() * 1000

        if is_speech:
            if not self._speech_active:
                # Speech just started
                self._speech_active = True
                self._speech_start_ms = now_ms
                self._silence_start_ms = None
            else:
                # Speech continues — cancel any silence timer
                self._silence_start_ms = None

            # Track cumulative speech duration
            if self._speech_start_ms is not None:
                self._total_speech_ms = now_ms - self._speech_start_ms

        else:
            # Silence detected
            if self._speech_active:
                if self._silence_start_ms is None:
                    # Silence just started
                    self._silence_start_ms = now_ms
                else:
                    # Check if silence has exceeded threshold
                    silence_duration_ms = now_ms - self._silence_start_ms

                    if silence_duration_ms >= self.silence_threshold_ms:
                        # Check minimum speech duration
                        if self._total_speech_ms >= _MIN_SPEECH_MS:
                            # Turn end detected!
                            self._turns_detected += 1
                            self._last_turn_end_ms = now_ms
                            self._reset_state()
                            return "turn_end"
                        else:
                            # Too short — false start
                            self._false_starts += 1
                            self._reset_state()
                            return None

        return None

    def reset(self) -> None:
        """Reset the turn detector state. Call after handling a turn."""
        self._reset_state()
        self._question_type = QuestionType.DEFAULT

    def _reset_state(self) -> None:
        """Internal state reset."""
        self._speech_active = False
        self._speech_start_ms = None
        self._silence_start_ms = None
        self._total_speech_ms = 0.0

    @property
    def is_speech_active(self) -> bool:
        return self._speech_active

    @property
    def current_speech_duration_ms(self) -> float:
        return self._total_speech_ms

    def get_status(self) -> dict:
        """Get detector status."""
        return {
            "speech_active": self._speech_active,
            "question_type": self._question_type.value,
            "silence_threshold_ms": self.silence_threshold_ms,
            "total_speech_ms": self._total_speech_ms,
            "turns_detected": self._turns_detected,
            "false_starts": self._false_starts,
        }
