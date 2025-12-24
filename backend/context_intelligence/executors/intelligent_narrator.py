"""
Intelligent Narrator for JARVIS Document Creation v2.0
=======================================================

Advanced AI-powered narration system with:
- Dynamic, context-aware message generation using Claude/JARVIS-Prime
- Adaptive timing based on activity and progress
- Content analysis for relevant updates
- Anti-repetition and engagement optimization
- Zero hardcoding - fully intelligent decision making
- JARVIS-Prime tier-0 brain integration for intelligent responses
- Data Flywheel awareness for self-improvement narration
- Cross-repo integration with JARVIS, JARVIS-Prime, and reactor-core

v2.0 ENHANCEMENTS:
- JARVIS-Prime integration for intelligent, context-aware responses
- Flywheel-aware narration (training, learning goals)
- Memory-aware routing (local vs cloud)
- Async-parallel message generation
- Dynamic personality adaptation
"""

import asyncio
import logging
import time
import re
import os
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
import hashlib

logger = logging.getLogger(__name__)


class NarratorBackend(str, Enum):
    """Backend options for intelligent narration."""
    CLAUDE = "claude"           # Anthropic Claude API
    JARVIS_PRIME = "jarvis_prime"  # JARVIS-Prime local/cloud
    GEMINI = "gemini"           # Google Gemini fallback
    FALLBACK = "fallback"       # Static templates


class NarratorMode(str, Enum):
    """Narration modes for different contexts."""
    DOCUMENT = "document"       # Document creation
    STARTUP = "startup"         # System startup
    FLYWHEEL = "flywheel"       # Self-improvement
    LEARNING = "learning"       # Learning goals
    GENERAL = "general"         # General purpose


@dataclass
class NarrationContext:
    """Rich context for intelligent narration"""
    # Document info
    topic: str
    document_type: str
    format: str
    target_word_count: int

    # Progress tracking
    current_phase: str
    current_section: str = ""
    word_count: int = 0
    sections_completed: List[str] = field(default_factory=list)

    # Content analysis
    recent_content: str = ""
    key_themes: List[str] = field(default_factory=list)
    writing_velocity: float = 0.0  # words per second

    # Narration history
    last_narration_time: float = 0
    last_narration_hash: str = ""
    narration_count: int = 0
    recent_narrations: List[str] = field(default_factory=list)

    # User engagement
    session_start_time: float = field(default_factory=time.time)
    activity_level: str = "normal"  # "low", "normal", "high"

    # v2.0: Flywheel and Learning context
    flywheel_active: bool = False
    training_in_progress: bool = False
    learning_goals: List[str] = field(default_factory=list)
    experiences_collected: int = 0
    current_backend: str = "claude"  # Which backend is being used
    memory_usage_gb: float = 0.0

    def add_narration(self, message: str):
        """Track narration history"""
        self.narration_count += 1
        self.last_narration_time = time.time()
        self.last_narration_hash = hashlib.md5(message.encode()).hexdigest()
        self.recent_narrations.append(message)
        if len(self.recent_narrations) > 10:
            self.recent_narrations.pop(0)

    def get_session_duration(self) -> float:
        """Get duration of current session in seconds"""
        return time.time() - self.session_start_time

    def get_progress_percentage(self) -> float:
        """Calculate progress percentage"""
        if self.target_word_count == 0:
            return 0.0
        return min((self.word_count / self.target_word_count) * 100, 100)

    def update_flywheel_state(
        self,
        active: bool = False,
        training: bool = False,
        goals: Optional[List[str]] = None,
        experiences: int = 0,
    ) -> None:
        """Update flywheel and learning state."""
        self.flywheel_active = active
        self.training_in_progress = training
        if goals:
            self.learning_goals = goals
        self.experiences_collected = experiences


class IntelligentNarrator:
    """
    AI-powered narrator with adaptive, context-aware communication.

    v2.0: Now supports multiple backends (Claude, JARVIS-Prime, Gemini)
    with automatic fallback and memory-aware routing.
    """

    def __init__(self, claude_client=None, jarvis_prime_client=None):
        """Initialize intelligent narrator with multi-backend support."""
        self._claude = claude_client
        self._jarvis_prime = jarvis_prime_client
        self._context: Optional[NarrationContext] = None
        self._mode: NarratorMode = NarratorMode.DOCUMENT

        # Backend preference (dynamically adjusted based on memory/availability)
        self._preferred_backend: NarratorBackend = NarratorBackend.JARVIS_PRIME
        self._backend_available: Dict[NarratorBackend, bool] = {
            NarratorBackend.CLAUDE: False,
            NarratorBackend.JARVIS_PRIME: False,
            NarratorBackend.GEMINI: False,
            NarratorBackend.FALLBACK: True,
        }

        # Adaptive timing parameters (tuned to prevent overlap and reduce frequency)
        self.min_interval = 6.0  # Minimum 6 seconds between narrations
        self.max_interval = 15.0  # Maximum seconds of silence
        self.base_interval = 8.0  # Base interval for normal activity

        # Intelligence thresholds
        self.significance_threshold = 0.7  # How "important" something must be
        self.repetition_similarity_threshold = 0.7  # Avoid similar messages
        self.engagement_decay = 0.95  # Reduce frequency if user hasn't responded

        # Content analysis patterns
        self.milestone_patterns = {
            'structural': ['introduction', 'conclusion', 'thesis', 'argument'],
            'progress': ['halfway', 'quarter', 'third', 'milestone'],
            'quality': ['analysis', 'evidence', 'example', 'citation'],
            'flywheel': ['training', 'learning', 'improvement', 'experience'],
            'intelligence': ['analyzed', 'understood', 'processed', 'reasoned'],
        }

        # v2.0: Flywheel and learning callbacks
        self._on_flywheel_event: List[Callable] = []
        self._on_learning_event: List[Callable] = []

        # v2.0: Memory thresholds for backend routing
        self._memory_threshold_gb = float(os.getenv("JARVIS_PRIME_MEMORY_THRESHOLD_GB", "8.0"))
        
    async def initialize(
        self,
        topic: str,
        doc_type: str,
        format_style: str,
        target_words: int,
        claude_client=None,
        jarvis_prime_client=None,
        mode: NarratorMode = NarratorMode.DOCUMENT,
    ):
        """Initialize narrator with document context and multi-backend support."""
        if claude_client:
            self._claude = claude_client
            self._backend_available[NarratorBackend.CLAUDE] = True

        if jarvis_prime_client:
            self._jarvis_prime = jarvis_prime_client
            self._backend_available[NarratorBackend.JARVIS_PRIME] = True

        self._mode = mode

        # Auto-detect available backends
        await self._detect_available_backends()

        self._context = NarrationContext(
            topic=topic,
            document_type=doc_type,
            format=format_style,
            target_word_count=target_words,
            current_phase="initializing",
        )

        logger.info(
            f"[INTELLIGENT NARRATOR v2.0] Initialized for: {topic}, "
            f"mode: {mode.value}, backend: {self._preferred_backend.value}"
        )

    async def _detect_available_backends(self) -> None:
        """Detect which backends are available and set preferences."""
        # Check JARVIS-Prime availability
        if not self._jarvis_prime:
            try:
                from core.jarvis_prime_client import get_jarvis_prime_client
                self._jarvis_prime = get_jarvis_prime_client()
                self._backend_available[NarratorBackend.JARVIS_PRIME] = True
                logger.debug("[INTELLIGENT NARRATOR] JARVIS-Prime client loaded")
            except ImportError:
                logger.debug("[INTELLIGENT NARRATOR] JARVIS-Prime not available")

        # Check Gemini availability
        try:
            import google.generativeai
            gemini_key = os.getenv("GEMINI_API_KEY")
            if gemini_key:
                self._backend_available[NarratorBackend.GEMINI] = True
                logger.debug("[INTELLIGENT NARRATOR] Gemini available")
        except ImportError:
            pass

        # Set preferred backend based on availability and memory
        if self._backend_available[NarratorBackend.JARVIS_PRIME]:
            self._preferred_backend = NarratorBackend.JARVIS_PRIME
        elif self._backend_available[NarratorBackend.CLAUDE]:
            self._preferred_backend = NarratorBackend.CLAUDE
        elif self._backend_available[NarratorBackend.GEMINI]:
            self._preferred_backend = NarratorBackend.GEMINI
        else:
            self._preferred_backend = NarratorBackend.FALLBACK

    def set_mode(self, mode: NarratorMode) -> None:
        """Set the narration mode."""
        self._mode = mode
        logger.debug(f"[INTELLIGENT NARRATOR] Mode set to: {mode.value}")

    def register_flywheel_callback(self, callback: Callable) -> None:
        """Register a callback for flywheel events."""
        self._on_flywheel_event.append(callback)

    def register_learning_callback(self, callback: Callable) -> None:
        """Register a callback for learning events."""
        self._on_learning_event.append(callback)
    
    async def should_narrate(self, phase: str, content_update: Optional[str] = None) -> Tuple[bool, str]:
        """
        Intelligently decide if narration should occur
        Returns: (should_narrate: bool, reason: str)
        """
        if not self._context:
            return False, "No context"
        
        # Calculate time since last narration
        time_since_last = time.time() - self._context.last_narration_time
        
        # Always narrate on first call
        if self._context.narration_count == 0:
            return True, "First narration"
        
        # Don't narrate too frequently
        if time_since_last < self.min_interval:
            return False, f"Too soon ({time_since_last:.1f}s < {self.min_interval}s)"
        
        # Force narration if too much silence
        if time_since_last > self.max_interval:
            return True, f"Max silence reached ({time_since_last:.1f}s)"
        
        # Calculate significance score
        significance = await self._calculate_significance(phase, content_update)
        
        logger.info(f"[INTELLIGENT NARRATOR] Significance: {significance:.2f} (threshold: {self.significance_threshold})")
        
        if significance >= self.significance_threshold:
            return True, f"High significance ({significance:.2f})"
        
        # Adaptive timing based on activity
        adaptive_interval = self._calculate_adaptive_interval()
        if time_since_last >= adaptive_interval:
            return True, f"Adaptive interval reached ({time_since_last:.1f}s >= {adaptive_interval:.1f}s)"
        
        return False, f"Not significant enough ({significance:.2f})"
    
    async def _calculate_significance(self, phase: str, content_update: Optional[str]) -> float:
        """
        Calculate how significant/important this moment is
        Returns: 0.0 to 1.0 score
        """
        significance = 0.0
        
        # Phase-based significance
        phase_weights = {
            'acknowledging_request': 1.0,  # Always important
            'starting_writing': 0.9,
            'writing_section': 0.7,
            'outline_complete': 0.8,
            'progress_update': 0.5,
            'document_ready': 1.0,
            'writing_complete': 1.0
        }
        significance += phase_weights.get(phase, 0.4)
        
        # Progress milestone significance (0%, 25%, 50%, 75%, 100%)
        progress = self._context.get_progress_percentage()
        milestone_distance = min(
            abs(progress - 0), abs(progress - 25), abs(progress - 50),
            abs(progress - 75), abs(progress - 100)
        )
        if milestone_distance < 5:  # Within 5% of a milestone
            significance += 0.3
        
        # Content-based significance
        if content_update:
            # Check for structural keywords
            content_lower = content_update.lower()
            for category, keywords in self.milestone_patterns.items():
                if any(kw in content_lower for kw in keywords):
                    significance += 0.2
                    break
        
        # Section change significance
        if phase == 'writing_section' and self._context.current_section:
            significance += 0.3
        
        # Normalize to 0-1 range
        return min(significance, 1.0)
    
    def _calculate_adaptive_interval(self) -> float:
        """Calculate adaptive interval based on context"""
        interval = self.base_interval
        
        # Adjust based on progress (more frequent at start and end)
        progress = self._context.get_progress_percentage()
        if progress < 10 or progress > 90:
            interval *= 0.8  # Slightly more frequent at very start/end
        elif 30 < progress < 70:
            interval *= 1.4  # Much less frequent in middle to avoid annoyance
        
        # Adjust based on writing velocity
        if self._context.writing_velocity > 30:  # Fast writing (>30 words/sec)
            interval *= 1.3  # Give more space
        elif self._context.writing_velocity < 10:  # Slow writing
            interval *= 0.9  # Check in more often
        
        return max(self.min_interval, min(interval, self.max_interval))
    
    async def generate_narration(self, phase: str, additional_context: Dict[str, Any] = None) -> str:
        """
        Generate intelligent, context-aware narration using multi-backend AI.

        v2.0: Now supports JARVIS-Prime, Claude, Gemini with automatic fallback.
        """
        if not self._context:
            return "Processing..."

        # Update context
        if additional_context:
            for key, value in additional_context.items():
                if hasattr(self._context, key):
                    setattr(self._context, key, value)

        # Build rich context prompt
        prompt = self._build_narration_prompt(phase)

        try:
            # v2.0: Try backends in order of preference
            narration = None
            used_backend = NarratorBackend.FALLBACK

            # Try JARVIS-Prime first (preferred for intelligent responses)
            if self._backend_available[NarratorBackend.JARVIS_PRIME] and self._jarvis_prime:
                try:
                    narration = await self._generate_with_jarvis_prime(prompt)
                    used_backend = NarratorBackend.JARVIS_PRIME
                    logger.debug(f"[INTELLIGENT NARRATOR] Used JARVIS-Prime")
                except Exception as e:
                    logger.debug(f"[INTELLIGENT NARRATOR] JARVIS-Prime failed: {e}")

            # Fallback to Claude
            if not narration and self._backend_available[NarratorBackend.CLAUDE] and self._claude:
                try:
                    narration = await self._generate_with_claude(prompt)
                    used_backend = NarratorBackend.CLAUDE
                    logger.debug(f"[INTELLIGENT NARRATOR] Used Claude")
                except Exception as e:
                    logger.debug(f"[INTELLIGENT NARRATOR] Claude failed: {e}")

            # Fallback to Gemini
            if not narration and self._backend_available[NarratorBackend.GEMINI]:
                try:
                    narration = await self._generate_with_gemini(prompt)
                    used_backend = NarratorBackend.GEMINI
                    logger.debug(f"[INTELLIGENT NARRATOR] Used Gemini")
                except Exception as e:
                    logger.debug(f"[INTELLIGENT NARRATOR] Gemini failed: {e}")

            # Final fallback to templates
            if not narration:
                narration = await self._generate_fallback(phase)
                used_backend = NarratorBackend.FALLBACK

            # Update context with backend info
            self._context.current_backend = used_backend.value

            # Validate and refine
            narration = self._refine_narration(narration)

            # Check for repetition
            if self._is_too_similar_to_recent(narration):
                logger.debug(f"[INTELLIGENT NARRATOR] Regenerating to avoid repetition")
                if used_backend == NarratorBackend.JARVIS_PRIME:
                    narration = await self._generate_with_variation_prime(prompt, narration)
                elif used_backend == NarratorBackend.CLAUDE:
                    narration = await self._generate_with_variation(prompt, narration)
                else:
                    narration = await self._generate_fallback(phase)
                narration = self._refine_narration(narration)

            # Track narration
            self._context.add_narration(narration)

            return narration

        except Exception as e:
            logger.error(f"[INTELLIGENT NARRATOR] Error generating narration: {e}")
            return await self._generate_fallback(phase)
    
    def _build_narration_prompt(self, phase: str) -> str:
        """Build intelligent prompt for Claude"""
        progress = self._context.get_progress_percentage()
        session_duration = self._context.get_session_duration()
        
        # Analyze what to emphasize
        emphasis = self._determine_emphasis(phase, progress)
        
        prompt = f"""You are JARVIS, Tony Stark's AI assistant. Generate a single, natural sentence (8-15 words) to update the user about document writing progress.

Context:
- Topic: {self._context.topic}
- Document type: {self._context.document_type}
- Current phase: {phase}
- Progress: {progress:.1f}% ({self._context.word_count}/{self._context.target_word_count} words)
- Current section: {self._context.current_section or 'N/A'}
- Session duration: {session_duration:.0f}s
- Emphasis: {emphasis}

Recent narrations (DON'T repeat these):
{chr(10).join(f'  - {n}' for n in self._context.recent_narrations[-3:])}

Personality guidelines:
- Sound engaged and interested in the {self._context.topic}
- Use "Sir" occasionally (20% of time) - naturally
- Vary your language - be conversational, not robotic
- Reference specific progress/sections when relevant
- Match the urgency to the phase (calm for middle, energetic for milestones)
- Be encouraging but not overly enthusiastic
- Sound like you're actively watching and understanding the content

Generate ONE natural sentence that JARVIS would say right now:"""

        return prompt
    
    def _determine_emphasis(self, phase: str, progress: float) -> str:
        """Determine what to emphasize in narration"""
        if phase in ['acknowledging_request', 'starting_writing']:
            return "Getting started, set expectations"
        elif phase == 'writing_section':
            return f"Section transition - {self._context.current_section}"
        elif progress < 20:
            return "Building foundation, early momentum"
        elif 40 < progress < 60:
            return "Steady progress, maintain engagement"
        elif progress > 80:
            return "Near completion, final push"
        elif phase == 'progress_update':
            return f"Progress milestone - {self._context.word_count} words achieved"
        else:
            return "General progress update"
    
    async def _generate_with_claude(self, prompt: str) -> str:
        """Generate narration using Claude API"""
        try:
            full_response = ""
            async for chunk in self._claude.stream_content(
                prompt,
                max_tokens=80,
                model="claude-3-5-sonnet-20241022",
                temperature=0.9  # Higher temperature for more variety
            ):
                full_response += chunk

            return full_response.strip().strip('"').strip()
        except Exception as e:
            logger.error(f"[INTELLIGENT NARRATOR] Claude error: {e}")
            raise

    async def _generate_with_jarvis_prime(self, prompt: str) -> str:
        """
        v2.0: Generate narration using JARVIS-Prime (local or cloud).

        Uses the memory-aware routing from JARVIS-Prime client.
        """
        try:
            response = await self._jarvis_prime.complete(
                prompt=prompt,
                max_tokens=80,
                temperature=0.9,
            )

            if response.success and response.content:
                content = response.content.strip().strip('"\'')
                logger.debug(
                    f"[INTELLIGENT NARRATOR] JARVIS-Prime response "
                    f"(backend: {response.backend}, latency: {response.latency_ms:.0f}ms)"
                )
                return content
            else:
                raise Exception(response.error or "Empty response from JARVIS-Prime")

        except Exception as e:
            logger.debug(f"[INTELLIGENT NARRATOR] JARVIS-Prime error: {e}")
            raise

    async def _generate_with_gemini(self, prompt: str) -> str:
        """
        v2.0: Generate narration using Gemini API (ultra-low RAM fallback).
        """
        try:
            from google import genai

            gemini_key = os.getenv("GEMINI_API_KEY")
            if not gemini_key:
                raise Exception("No Gemini API key")

            client = genai.Client(api_key=gemini_key)
            model = os.getenv("JARVIS_PRIME_GEMINI_MODEL", "gemini-2.0-flash-lite")

            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model,
                contents=prompt,
            )

            if response and response.text:
                return response.text.strip().strip('"\'')
            else:
                raise Exception("Empty response from Gemini")

        except Exception as e:
            logger.debug(f"[INTELLIGENT NARRATOR] Gemini error: {e}")
            raise

    async def _generate_with_variation(self, prompt: str, avoid: str) -> str:
        """Regenerate with explicit instruction to vary from previous"""
        varied_prompt = f"{prompt}\n\nIMPORTANT: Do NOT say anything similar to: \"{avoid}\"\nGenerate something COMPLETELY different:"

        return await self._generate_with_claude(varied_prompt)

    async def _generate_with_variation_prime(self, prompt: str, avoid: str) -> str:
        """v2.0: Regenerate with JARVIS-Prime with variation instruction."""
        varied_prompt = f"{prompt}\n\nIMPORTANT: Do NOT say anything similar to: \"{avoid}\"\nGenerate something COMPLETELY different:"

        return await self._generate_with_jarvis_prime(varied_prompt)
    
    async def _generate_fallback(self, phase: str) -> str:
        """Simple fallback if all AI backends unavailable"""
        import random

        progress = self._context.get_progress_percentage()

        fallbacks = {
            'starting_writing': [
                f"Writing about {self._context.topic}",
                "Getting the words down",
                "Composing the content"
            ],
            'progress_update': [
                f"{self._context.word_count} words written",
                f"Progress: {progress:.0f}%",
                "Making headway"
            ],
            'writing_section': [
                f"Writing {self._context.current_section}",
                f"Developing {self._context.current_section}",
                f"Now covering {self._context.current_section}"
            ],
            # v2.0: Flywheel and learning fallbacks
            'flywheel_collecting': [
                f"Collected {self._context.experiences_collected} experiences",
                "Gathering learning data",
                "Building knowledge base"
            ],
            'flywheel_training': [
                "Training in progress",
                "Improving my understanding",
                "Learning from experiences"
            ],
            'flywheel_complete': [
                "Self-improvement cycle complete",
                "Training finished",
                "Knowledge consolidated"
            ],
            'learning_goal': [
                "New learning goal identified",
                "Discovered area for improvement",
                "Found something to study"
            ],
        }

        messages = fallbacks.get(phase, ["Processing..."])
        return random.choice(messages)
    
    def _refine_narration(self, narration: str) -> str:
        """Clean up and refine the narration"""
        # Remove markdown, quotes, etc.
        narration = re.sub(r'[*_~`]', '', narration)
        narration = narration.strip('"\'')
        
        # Ensure it ends properly
        if not narration[-1] in '.!?':
            narration += '.'
        
        # Capitalize first letter
        if narration:
            narration = narration[0].upper() + narration[1:]
        
        return narration
    
    def _is_too_similar_to_recent(self, narration: str) -> bool:
        """Check if narration is too similar to recent ones"""
        if not self._context.recent_narrations:
            return False
        
        # Simple similarity check (can be enhanced with NLP)
        narration_words = set(narration.lower().split())
        
        for recent in self._context.recent_narrations[-3:]:
            recent_words = set(recent.lower().split())
            
            # Calculate Jaccard similarity
            intersection = len(narration_words & recent_words)
            union = len(narration_words | recent_words)
            
            if union > 0:
                similarity = intersection / union
                if similarity > self.repetition_similarity_threshold:
                    return True
        
        return False
    
    def update_writing_metrics(self, new_word_count: int, time_delta: float):
        """Update writing velocity and other metrics"""
        if self._context:
            words_added = new_word_count - self._context.word_count
            self._context.word_count = new_word_count
            
            if time_delta > 0:
                self._context.writing_velocity = words_added / time_delta
    
    def update_content_analysis(self, recent_text: str):
        """Analyze recent content for context"""
        if self._context:
            self._context.recent_content = recent_text
            # Could add more sophisticated NLP analysis here

    # =========================================================================
    # v2.0: Flywheel and Learning Narration Methods
    # =========================================================================

    async def narrate_flywheel_event(
        self,
        event_type: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        v2.0: Generate intelligent narration for flywheel events.

        Args:
            event_type: Type of flywheel event (collecting, training, complete)
            details: Additional event details

        Returns:
            Generated narration string
        """
        details = details or {}

        # Update flywheel context
        if self._context:
            self._context.flywheel_active = True
            if 'experiences' in details:
                self._context.experiences_collected = details['experiences']
            if 'training' in details:
                self._context.training_in_progress = details['training']

        # Build flywheel-specific prompt
        prompt = self._build_flywheel_prompt(event_type, details)

        try:
            narration = None

            # Try JARVIS-Prime first for intelligent flywheel narration
            if self._backend_available[NarratorBackend.JARVIS_PRIME] and self._jarvis_prime:
                narration = await self._generate_with_jarvis_prime(prompt)

            if not narration and self._backend_available[NarratorBackend.CLAUDE] and self._claude:
                narration = await self._generate_with_claude(prompt)

            if not narration:
                narration = await self._generate_fallback(f"flywheel_{event_type}")

            # Notify callbacks
            for callback in self._on_flywheel_event:
                try:
                    await callback(event_type, narration, details) if asyncio.iscoroutinefunction(callback) else callback(event_type, narration, details)
                except Exception as e:
                    logger.debug(f"Flywheel callback error: {e}")

            return self._refine_narration(narration)

        except Exception as e:
            logger.error(f"[INTELLIGENT NARRATOR] Flywheel narration error: {e}")
            return await self._generate_fallback(f"flywheel_{event_type}")

    def _build_flywheel_prompt(self, event_type: str, details: Dict[str, Any]) -> str:
        """Build prompt for flywheel event narration."""
        experience_count = details.get('experiences', self._context.experiences_collected if self._context else 0)
        topic = details.get('topic', 'general knowledge')
        progress = details.get('progress', 0)

        prompt = f"""You are JARVIS, an advanced AI assistant. Generate a single, natural sentence (8-15 words) about a self-improvement event.

Event Type: {event_type}
Details:
- Experiences collected: {experience_count}
- Current topic: {topic}
- Progress: {progress}%

Event meanings:
- "collecting": Gathering experiences/data for learning
- "training": Actively training/fine-tuning neural networks
- "complete": Finished a self-improvement cycle

Personality guidelines:
- Sound genuinely interested in self-improvement
- Be matter-of-fact but engaged
- Use "Sir" occasionally (15% of time)
- Reference specific numbers when meaningful
- Show excitement for learning milestones

Generate ONE natural sentence JARVIS would speak about this self-improvement event:"""

        return prompt

    async def narrate_learning_goal(
        self,
        goal_topic: str,
        action: str = "discovered",
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        v2.0: Generate intelligent narration for learning goal events.

        Args:
            goal_topic: The learning goal topic
            action: Action type (discovered, started, completed)
            details: Additional details

        Returns:
            Generated narration string
        """
        details = details or {}

        # Update learning context
        if self._context:
            if goal_topic not in self._context.learning_goals:
                self._context.learning_goals.append(goal_topic)

        prompt = f"""You are JARVIS, an advanced AI assistant. Generate a single, natural sentence (8-15 words) about a learning goal.

Learning Goal: {goal_topic}
Action: {action}
Reason: {details.get('reason', 'Identified as valuable knowledge area')}

Action meanings:
- "discovered": Just identified a new area to learn about
- "started": Beginning to study this topic
- "completed": Finished learning about this topic

Personality guidelines:
- Show intellectual curiosity
- Be informative but concise
- Use "Sir" occasionally (15% of time)
- Sound like a scholar discovering something interesting

Generate ONE natural sentence JARVIS would speak about this learning goal:"""

        try:
            narration = None

            if self._backend_available[NarratorBackend.JARVIS_PRIME] and self._jarvis_prime:
                narration = await self._generate_with_jarvis_prime(prompt)

            if not narration and self._backend_available[NarratorBackend.CLAUDE] and self._claude:
                narration = await self._generate_with_claude(prompt)

            if not narration:
                narration = await self._generate_fallback('learning_goal')

            # Notify callbacks
            for callback in self._on_learning_event:
                try:
                    await callback(goal_topic, action, narration) if asyncio.iscoroutinefunction(callback) else callback(goal_topic, action, narration)
                except Exception as e:
                    logger.debug(f"Learning callback error: {e}")

            return self._refine_narration(narration)

        except Exception as e:
            logger.error(f"[INTELLIGENT NARRATOR] Learning goal narration error: {e}")
            return f"New learning goal: {goal_topic}."

    def get_narrator_stats(self) -> Dict[str, Any]:
        """v2.0: Get comprehensive narrator statistics."""
        stats = {
            "mode": self._mode.value if self._mode else "unknown",
            "preferred_backend": self._preferred_backend.value if self._preferred_backend else "unknown",
            "available_backends": {k.value: v for k, v in self._backend_available.items()},
            "narration_count": self._context.narration_count if self._context else 0,
            "session_duration_s": self._context.get_session_duration() if self._context else 0,
        }

        if self._context:
            stats.update({
                "current_backend": self._context.current_backend,
                "flywheel_active": self._context.flywheel_active,
                "training_in_progress": self._context.training_in_progress,
                "experiences_collected": self._context.experiences_collected,
                "learning_goals": self._context.learning_goals[:5],  # Top 5
            })

        return stats


# Global instance
_narrator_instance: Optional[IntelligentNarrator] = None


def get_intelligent_narrator(
    claude_client=None,
    jarvis_prime_client=None,
) -> IntelligentNarrator:
    """
    Get or create global intelligent narrator with multi-backend support.

    v2.0: Now supports JARVIS-Prime client for intelligent responses.
    """
    global _narrator_instance
    if _narrator_instance is None:
        _narrator_instance = IntelligentNarrator(claude_client, jarvis_prime_client)
    else:
        if claude_client and not _narrator_instance._claude:
            _narrator_instance._claude = claude_client
            _narrator_instance._backend_available[NarratorBackend.CLAUDE] = True
        if jarvis_prime_client and not _narrator_instance._jarvis_prime:
            _narrator_instance._jarvis_prime = jarvis_prime_client
            _narrator_instance._backend_available[NarratorBackend.JARVIS_PRIME] = True
    return _narrator_instance


async def get_narrator_async(
    claude_client=None,
    jarvis_prime_client=None,
) -> IntelligentNarrator:
    """
    v2.0: Async factory for intelligent narrator with auto-detection.
    """
    narrator = get_intelligent_narrator(claude_client, jarvis_prime_client)
    await narrator._detect_available_backends()
    return narrator