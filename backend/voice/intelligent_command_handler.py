#!/usr/bin/env python3
"""
Intelligent Command Handler for JARVIS
Uses Swift classifier for intelligent command routing without hardcoding
"""

import os
import asyncio
import logging
import re
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

# Import Swift bridge
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'swift_bridge'))
from python_bridge import IntelligentCommandRouter

# Import existing components
from system_control import ClaudeCommandInterpreter, CommandCategory
from chatbots.claude_vision_chatbot import ClaudeVisionChatbot

# Import VisualMonitorAgent for God Mode surveillance
logger = logging.getLogger(__name__)

try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from neural_mesh.agents.visual_monitor_agent import VisualMonitorAgent
    VISUAL_MONITOR_AVAILABLE = True
except ImportError as e:
    VISUAL_MONITOR_AVAILABLE = False
    logger.warning(f"VisualMonitorAgent not available: {e}")

class IntelligentCommandHandler:
    """
    Handles commands using Swift-based intelligent classification
    No hardcoding - learns and adapts dynamically
    """
    
    def __init__(self, user_name: str = "Sir", vision_analyzer: Optional[Any] = None):
        self.user_name = user_name
        self.router = IntelligentCommandRouter()

        # Initialize handlers
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if self.api_key:
            self.command_interpreter = ClaudeCommandInterpreter(self.api_key)
            # Pass the vision analyzer to the chatbot
            self.claude_chatbot = ClaudeVisionChatbot(self.api_key, vision_analyzer=vision_analyzer)
            self.enabled = True
        else:
            self.enabled = False
            logger.warning("Intelligent command handling disabled - no API key")

        # Initialize VisualMonitorAgent for God Mode surveillance (lazy loading)
        self._visual_monitor_agent: Optional[VisualMonitorAgent] = None
        self._visual_monitor_initialized = False

        # Track command history for learning
        self.command_history = []
        self.max_history = 100

        # Track surveillance operations for learning and milestones
        self.surveillance_history = []
        self.surveillance_stats = {
            'total_operations': 0,
            'successful_detections': 0,
            'god_mode_operations': 0,
            'apps_monitored': set(),
            'total_windows_watched': 0,
            'fastest_detection_time': float('inf'),
            'average_confidence': 0.0,
        }
        self.last_milestone_announced = 0

    async def _get_visual_monitor_agent(self) -> Optional[VisualMonitorAgent]:
        """
        Lazy initialization of VisualMonitorAgent for God Mode surveillance.
        Returns None if agent cannot be initialized.
        """
        if not VISUAL_MONITOR_AVAILABLE:
            logger.warning("VisualMonitorAgent not available - God Mode disabled")
            return None

        if self._visual_monitor_initialized:
            return self._visual_monitor_agent

        try:
            logger.info("Initializing VisualMonitorAgent for God Mode surveillance...")
            agent = VisualMonitorAgent()
            await agent.on_initialize()
            await agent.on_start()
            self._visual_monitor_agent = agent
            self._visual_monitor_initialized = True
            logger.info("✅ VisualMonitorAgent initialized - God Mode active")
            return agent
        except Exception as e:
            logger.error(f"Failed to initialize VisualMonitorAgent: {e}")
            self._visual_monitor_initialized = True  # Don't retry
            return None

    def _parse_watch_command(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Parse voice commands for God Mode surveillance.

        Patterns detected:
        - "watch [app] for [trigger]"
        - "monitor [app] for [trigger]"
        - "watch all [app] windows for [trigger]"
        - "monitor [app] across all spaces for [trigger]"
        - "notify me when [app] says [trigger]"
        - "alert me when [app] shows [trigger]"

        Returns:
            Dict with:
                - app_name: Application to watch (e.g., "Terminal", "Chrome")
                - trigger_text: Text to detect (e.g., "Build Complete")
                - all_spaces: True if should watch across all spaces
                - max_duration: Optional timeout in seconds
            Or None if not a watch command
        """
        text_lower = text.lower().strip()

        # Watch/monitor keywords (required)
        watch_keywords = [
            r'\bwatch\b', r'\bmonitor\b', r'\btrack\b', r'\bobserve\b',
            r'\bnotify\s+me\s+when\b', r'\balert\s+me\s+when\b',
            r'\btell\s+me\s+when\b', r'\blet\s+me\s+know\s+when\b'
        ]

        # Check if this is a watch command
        has_watch_keyword = any(re.search(pattern, text_lower) for pattern in watch_keywords)
        if not has_watch_keyword:
            return None

        # For/when keywords (trigger separator)
        trigger_separators = [r'\bfor\b', r'\bwhen\b', r'\bsays\b', r'\bshows\b', r'\bdisplays\b']

        # Multi-space keywords
        all_spaces_keywords = [
            r'\ball\s+spaces\b', r'\bevery\s+space\b', r'\bacross\s+all\b',
            r'\ball\s+.*\s+windows\b', r'\bevery\s+.*\s+window\b'
        ]

        # Detect if all_spaces mode
        all_spaces = any(re.search(pattern, text_lower) for pattern in all_spaces_keywords)

        # Extract app name and trigger text using regex patterns
        app_name = None
        trigger_text = None

        # Pattern 1: "watch/monitor [app] for/when [trigger]"
        # Enhanced to handle "across all spaces", "on all spaces" etc.
        pattern1 = re.compile(
            r'(?:watch|monitor|track|observe)\s+(?:all\s+)?(?:the\s+)?(\w+(?:\s+\w+)?)\s+'
            r'(?:windows?\s+)?(?:across\s+all\s+spaces?\s+)?(?:on\s+all\s+spaces?\s+)?(?:for|when)\s+(.+)',
            re.IGNORECASE
        )
        match1 = pattern1.search(text)
        if match1:
            app_name = match1.group(1).strip()
            trigger_text = match1.group(2).strip()

            # Clean up trigger text if duration pattern is present
            # Remove "X minutes/seconds/hours when it says" prefix from trigger
            duration_prefix_pattern = re.compile(
                r'^\d+\s+(?:second|minute|hour|min|sec|hr)s?\s+(?:when\s+it\s+says|when)\s+',
                re.IGNORECASE
            )
            trigger_text = duration_prefix_pattern.sub('', trigger_text)

        # Pattern 2: "notify/alert me when [app] says/shows [trigger]"
        if not app_name:
            pattern2 = re.compile(
                r'(?:notify|alert|tell|let)\s+me\s+when\s+(?:the\s+)?(\w+(?:\s+\w+)?)\s+(?:says|shows|displays)\s+(.+)',
                re.IGNORECASE
            )
            match2 = pattern2.search(text)
            if match2:
                app_name = match2.group(1).strip()
                trigger_text = match2.group(2).strip()

        # Pattern 3: "watch for [trigger] in [app]"
        if not app_name:
            pattern3 = re.compile(
                r'(?:watch|monitor|track)\s+(?:for|when)\s+(.+?)\s+(?:in|on)\s+(?:the\s+)?(\w+(?:\s+\w+)?)',
                re.IGNORECASE
            )
            match3 = pattern3.search(text)
            if match3:
                trigger_text = match3.group(1).strip()
                app_name = match3.group(2).strip()

        if not app_name or not trigger_text:
            logger.debug(f"Could not parse watch command: '{text}' (app={app_name}, trigger={trigger_text})")
            return None

        # Clean up trigger text (remove quotes, extra words)
        trigger_text = trigger_text.strip('"\'').strip()

        # Remove common filler words from trigger
        filler_words = ['please', 'jarvis', 'the', 'a', 'an']
        trigger_words = trigger_text.split()
        trigger_words = [w for w in trigger_words if w.lower() not in filler_words]
        trigger_text = ' '.join(trigger_words)

        # Capitalize app name (Terminal, Chrome, etc.)
        app_name = app_name.title()

        # Extract duration if mentioned (e.g., "for 5 minutes", "for 2 hours")
        max_duration = None
        duration_pattern = re.compile(r'for\s+(\d+)\s+(second|minute|hour|min|sec|hr)s?', re.IGNORECASE)
        duration_match = duration_pattern.search(text_lower)
        if duration_match:
            amount = int(duration_match.group(1))
            unit = duration_match.group(2).lower()

            # Convert to seconds
            if unit.startswith('sec'):
                max_duration = amount
            elif unit.startswith('min'):
                max_duration = amount * 60
            elif unit.startswith('hour') or unit.startswith('hr'):
                max_duration = amount * 3600

        result = {
            'app_name': app_name,
            'trigger_text': trigger_text,
            'all_spaces': all_spaces,
            'max_duration': max_duration,
            'original_command': text
        }

        logger.info(f"📡 Parsed watch command: {result}")
        return result

    def _build_surveillance_start_message(
        self,
        app_name: str,
        trigger_text: str,
        all_spaces: bool,
        max_duration: Optional[int]
    ) -> str:
        """
        Build context-aware, natural surveillance start message.

        Varies based on:
        - God Mode vs single window
        - Duration specified vs indefinite
        - Time of day
        - Complexity of trigger
        """
        import random
        from datetime import datetime

        hour = datetime.now().hour

        # God Mode (all spaces) messages
        if all_spaces:
            god_mode_variations = [
                f"Got it, {self.user_name}. I'll scan every {app_name} window across all your desktop spaces for '{trigger_text}'.",
                f"On it. Activating parallel surveillance - I'll watch all {app_name} windows for '{trigger_text}'.",
                f"Understood. Spawning watchers for every {app_name} window. Looking for '{trigger_text}'.",
                f"Sure thing, {self.user_name}. I'll keep eyes on all {app_name} windows for '{trigger_text}'.",
            ]

            # Add duration context if specified
            if max_duration:
                minutes = max_duration / 60
                if minutes < 2:
                    duration_msg = f" for the next {int(max_duration)} seconds"
                else:
                    duration_msg = f" for the next {int(minutes)} minutes"
            else:
                duration_msg = " until I find it"

            base = random.choice(god_mode_variations)
            return base.replace("'.", f"'{duration_msg}.")

        # Single window messages
        else:
            single_variations = [
                f"On it, {self.user_name}. Watching {app_name} for '{trigger_text}'.",
                f"Got it. I'll let you know when {app_name} shows '{trigger_text}'.",
                f"Sure. Monitoring {app_name} for '{trigger_text}'.",
                f"Watching {app_name} now, {self.user_name}. Looking for '{trigger_text}'.",
            ]

            # Early morning/late night - more subdued
            if hour < 7 or hour >= 22:
                single_variations.append(f"Quietly monitoring {app_name} for '{trigger_text}'.")

            return random.choice(single_variations)

    def _format_error_response(
        self,
        error_type: str,
        app_name: str,
        trigger_text: str
    ) -> str:
        """Format error responses in natural, helpful language."""
        import random

        if error_type == "initialization_failed":
            return random.choice([
                f"I'm sorry, {self.user_name}. My visual surveillance system isn't responding right now. Try again in a moment?",
                f"The monitoring system failed to start, {self.user_name}. This usually resolves itself - want to try once more?",
                f"Having trouble initializing the watchers, {self.user_name}. Give me a second and try again?",
            ])
        elif error_type == "no_windows":
            return random.choice([
                f"I don't see any {app_name} windows open right now, {self.user_name}. Could you open {app_name} first?",
                f"No {app_name} windows detected. Make sure {app_name} is running, then I can watch it for you.",
                f"I'm not finding any {app_name} windows to monitor. Is {app_name} open?",
            ])
        elif error_type == "runtime_error":
            return random.choice([
                f"I hit a snag while monitoring {app_name}, {self.user_name}. Want to try again?",
                f"Something went wrong with the surveillance on {app_name}. Let's give it another shot?",
                f"Had an issue watching {app_name}. This is unusual - shall we try once more?",
            ])
        else:
            return f"Sorry, {self.user_name}. I ran into an issue while trying to watch {app_name}."

    def _format_timeout_response(
        self,
        app_name: str,
        trigger_text: str,
        max_duration: Optional[int]
    ) -> str:
        """Format timeout responses naturally."""
        import random

        if max_duration:
            minutes = max_duration / 60
            if minutes < 2:
                time_str = f"{int(max_duration)} seconds"
            else:
                time_str = f"{int(minutes)} minutes"

            return random.choice([
                f"I watched {app_name} for {time_str}, {self.user_name}, but didn't see '{trigger_text}'. Want me to keep looking?",
                f"No '{trigger_text}' showed up in {app_name} after {time_str}. Everything else okay?",
                f"Surveillance wrapped up after {time_str}. '{trigger_text}' never appeared in {app_name}, {self.user_name}.",
            ])
        else:
            return random.choice([
                f"Monitoring timed out, {self.user_name}. I didn't spot '{trigger_text}' in {app_name}.",
                f"Surveillance completed but '{trigger_text}' never showed in {app_name}.",
                f"I watched {app_name} until timeout, but no '{trigger_text}' detected.",
            ])

    def _build_success_response(
        self,
        result: Dict[str, Any],
        watch_params: Dict[str, Any],
        initial_msg: str
    ) -> str:
        """
        Build sophisticated, context-aware success response for surveillance detection.

        Varies based on:
        - Confidence level (high/medium/low)
        - God Mode vs single window
        - Time of day
        - Multi-window statistics
        - Space detection details
        - Learning acknowledgments
        - Milestone celebrations
        """
        import random
        from datetime import datetime

        app_name = watch_params['app_name']
        trigger_text = watch_params['trigger_text']
        all_spaces = watch_params['all_spaces']

        # Extract detection details
        confidence = result.get('confidence', 0.0)
        space_id = result.get('space_id')
        window_count = result.get('window_count', 1)
        detection_time = result.get('detection_time', 0.0)

        hour = datetime.now().hour

        # Record this operation for learning (BEFORE checking milestones)
        self._record_surveillance_operation(watch_params, result, success=True)

        # Check for milestone celebration
        milestone_msg = self._check_milestone()

        # Check for learning acknowledgment
        learning_msg = self._generate_learning_acknowledgment(watch_params, result)

        # ===================================================================
        # HIGH CONFIDENCE (>90%) - Natural, confident, minimal explanation
        # ===================================================================
        if confidence > 0.90:
            if all_spaces and window_count > 1:
                # God Mode with multiple windows
                base_messages = [
                    f"Found it, {self.user_name}! I detected '{trigger_text}' in {app_name}",
                    f"Got it! '{trigger_text}' just showed up in {app_name}",
                    f"There it is - '{trigger_text}' appeared in {app_name}",
                    f"Success! I spotted '{trigger_text}' in {app_name}",
                ]

                # Add space context
                if space_id:
                    space_context = f" on Space {space_id}"
                else:
                    space_context = ""

                # Add statistics for God Mode
                stats_variations = [
                    f" I was watching {window_count} {app_name} windows in parallel - this one triggered first.",
                    f" Out of {window_count} {app_name} windows I was monitoring, this was the winner.",
                    f" I had {window_count} {app_name} watchers running - this window detected it first.",
                ]

                base = random.choice(base_messages)
                stats = random.choice(stats_variations)

                # Build base response
                # Early morning/late night - more subdued
                if hour < 7 or hour >= 22:
                    response = f"{base}{space_context}. {stats}"
                else:
                    response = f"{base}{space_context}. Confidence: {int(confidence * 100)}%. {stats}"

                # Append learning/milestone messages
                if learning_msg:
                    response += f"\n\n{learning_msg}"
                if milestone_msg:
                    response += f"\n\n{milestone_msg}"

                return response

            else:
                # Single window - keep it simple and confident
                simple_messages = [
                    f"Found it, {self.user_name}! '{trigger_text}' just appeared in {app_name}.",
                    f"There it is - I detected '{trigger_text}' in {app_name}.",
                    f"Got it! '{trigger_text}' showed up in {app_name}.",
                    f"Success! I spotted '{trigger_text}' in {app_name}.",
                ]
                response = random.choice(simple_messages)

                # Append learning/milestone messages
                if learning_msg:
                    response += f"\n\n{learning_msg}"
                if milestone_msg:
                    response += f"\n\n{milestone_msg}"

                return response

        # ===================================================================
        # MEDIUM CONFIDENCE (85-90%) - Brief acknowledgment, show confidence
        # ===================================================================
        elif confidence > 0.85:
            if all_spaces and window_count > 1:
                messages = [
                    f"I believe I found it, {self.user_name}. '{trigger_text}' detected in {app_name}",
                    f"Got it with good confidence - '{trigger_text}' in {app_name}",
                    f"Detected '{trigger_text}' in {app_name}",
                ]

                base = random.choice(messages)

                if space_id:
                    space_context = f" on Space {space_id}"
                else:
                    space_context = ""

                stats = f" Confidence: {int(confidence * 100)}%. Monitoring {window_count} windows - this one matched."
                response = f"{base}{space_context}.{stats}"

                # Append learning/milestone messages
                if learning_msg:
                    response += f"\n\n{learning_msg}"
                if milestone_msg:
                    response += f"\n\n{milestone_msg}"

                return response

            else:
                messages = [
                    f"I detected '{trigger_text}' in {app_name}, {self.user_name}. Confidence: {int(confidence * 100)}%.",
                    f"Found it! '{trigger_text}' appeared in {app_name}. Confidence: {int(confidence * 100)}%.",
                    f"Got it - '{trigger_text}' in {app_name}. Looks good ({int(confidence * 100)}% confidence).",
                ]
                response = random.choice(messages)

                # Append learning/milestone messages
                if learning_msg:
                    response += f"\n\n{learning_msg}"
                if milestone_msg:
                    response += f"\n\n{milestone_msg}"

                return response

        # ===================================================================
        # BORDERLINE CONFIDENCE (80-85%) - Show thought process
        # ===================================================================
        elif confidence > 0.80:
            explanations = [
                f"I detected what appears to be '{trigger_text}' in {app_name}, {self.user_name}. "
                f"Confidence is {int(confidence * 100)}% - slightly lower than usual, possibly due to "
                f"partial occlusion or font rendering. Want me to keep watching to confirm?",

                f"Found a match for '{trigger_text}' in {app_name} with {int(confidence * 100)}% confidence. "
                f"The text is there, though OCR clarity was borderline. Should be the right one.",

                f"Detected '{trigger_text}' in {app_name} - confidence is {int(confidence * 100)}%. "
                f"A bit lower than ideal, possibly screen positioning or text overlap. "
                f"But I'm reasonably confident this is it.",
            ]

            base = random.choice(explanations)

            if all_spaces and window_count > 1:
                stats = f" I was monitoring {window_count} windows when this triggered."
                response = base + stats
            else:
                response = base

            # Append learning/milestone messages
            if learning_msg:
                response += f"\n\n{learning_msg}"
            if milestone_msg:
                response += f"\n\n{milestone_msg}"

            return response

        # ===================================================================
        # LOW CONFIDENCE (<80%) - Multi-factor explanation
        # ===================================================================
        else:
            multi_factor_explanations = [
                f"I detected something that might be '{trigger_text}' in {app_name}, {self.user_name}. "
                f"OCR confidence is only {int(confidence * 100)}%, which is below my usual threshold. "
                f"This could be due to small font size, partial visibility, or text overlap. "
                f"Want me to try again with enhanced OCR?",

                f"Borderline detection: '{trigger_text}' in {app_name} at {int(confidence * 100)}% confidence. "
                f"I saw text that closely matches, but couldn't get a definitive read. "
                f"Could be legitimate or could be similar text. Should I keep monitoring?",

                f"I found text resembling '{trigger_text}' in {app_name}, but confidence is low ({int(confidence * 100)}%). "
                f"The pattern matches, but clarity is poor. This might be it, or might be a false positive. "
                f"Your call - accept this detection or wait for clearer signal?",
            ]

            response = random.choice(multi_factor_explanations)

            # Append learning/milestone messages (even for low confidence)
            if learning_msg:
                response += f"\n\n{learning_msg}"
            if milestone_msg:
                response += f"\n\n{milestone_msg}"

            return response

    def _record_surveillance_operation(
        self,
        watch_params: Dict[str, Any],
        result: Dict[str, Any],
        success: bool
    ):
        """
        Record surveillance operation for learning and milestone tracking.

        This enables:
        - Milestone celebrations (10th, 25th, 50th, 100th operation)
        - Pattern learning (common apps, trigger patterns)
        - Performance tracking (detection times, confidence trends)
        """
        app_name = watch_params['app_name']
        all_spaces = watch_params['all_spaces']

        # Update stats
        self.surveillance_stats['total_operations'] += 1

        if success:
            self.surveillance_stats['successful_detections'] += 1

            # Track confidence trend
            confidence = result.get('confidence', 0.0)
            total_detections = self.surveillance_stats['successful_detections']
            current_avg = self.surveillance_stats['average_confidence']
            self.surveillance_stats['average_confidence'] = (
                (current_avg * (total_detections - 1) + confidence) / total_detections
            )

            # Track detection time
            detection_time = result.get('detection_time', 0.0)
            if detection_time > 0 and detection_time < self.surveillance_stats['fastest_detection_time']:
                self.surveillance_stats['fastest_detection_time'] = detection_time

            # Track window count for God Mode
            window_count = result.get('window_count', 1)
            self.surveillance_stats['total_windows_watched'] += window_count

        if all_spaces:
            self.surveillance_stats['god_mode_operations'] += 1

        # Track apps monitored
        self.surveillance_stats['apps_monitored'].add(app_name)

        # Record in history
        record = {
            'timestamp': datetime.now().isoformat(),
            'app_name': app_name,
            'trigger_text': watch_params['trigger_text'],
            'all_spaces': all_spaces,
            'success': success,
            'confidence': result.get('confidence', 0.0) if success else 0.0,
        }

        self.surveillance_history.append(record)

        # Limit history size
        if len(self.surveillance_history) > self.max_history:
            self.surveillance_history.pop(0)

    def _check_milestone(self) -> Optional[str]:
        """
        Check if we've reached a surveillance milestone worth celebrating.

        Milestones:
        - 10th, 25th, 50th, 100th, 250th, 500th, 1000th operation

        Returns:
            Celebration message if milestone reached, None otherwise
        """
        import random

        total_ops = self.surveillance_stats['total_operations']
        milestones = [10, 25, 50, 100, 250, 500, 1000]

        # Check if we hit a milestone and haven't announced it yet
        for milestone in milestones:
            if total_ops == milestone and self.last_milestone_announced < milestone:
                self.last_milestone_announced = milestone

                successful = self.surveillance_stats['successful_detections']
                god_mode_count = self.surveillance_stats['god_mode_operations']
                apps_count = len(self.surveillance_stats['apps_monitored'])
                avg_confidence = self.surveillance_stats['average_confidence']

                # Different celebration messages based on milestone
                if milestone == 10:
                    celebrations = [
                        f"By the way, {self.user_name} - that was your 10th surveillance operation! "
                        f"You've successfully detected {successful} out of {total_ops} triggers. "
                        f"We're just getting started!",

                        f"Fun milestone: That's 10 surveillance operations, {self.user_name}! "
                        f"{successful} successful detections so far. I'm learning your patterns.",
                    ]

                elif milestone == 25:
                    celebrations = [
                        f"Milestone achieved! 25 surveillance operations completed, {self.user_name}. "
                        f"{successful} successful detections ({int(successful/total_ops*100)}% success rate). "
                        f"You've used God Mode {god_mode_count} times across {apps_count} different apps.",

                        f"Nice - that's your 25th surveillance operation! {successful} successful detections "
                        f"with an average confidence of {int(avg_confidence*100)}%. "
                        f"The system is getting sharper.",
                    ]

                elif milestone >= 50:
                    fastest = self.surveillance_stats['fastest_detection_time']
                    total_windows = self.surveillance_stats['total_windows_watched']

                    celebrations = [
                        f"🎯 Major milestone: {milestone} surveillance operations completed, {self.user_name}! "
                        f"Stats: {successful}/{total_ops} successful ({int(successful/total_ops*100)}%), "
                        f"God Mode used {god_mode_count} times, {total_windows} total windows monitored, "
                        f"average confidence {int(avg_confidence*100)}%. "
                        f"Fastest detection: {fastest:.1f}s. You're a surveillance pro!",

                        f"Impressive! {milestone} operations and counting. "
                        f"Success rate: {int(successful/total_ops*100)}%, "
                        f"average confidence: {int(avg_confidence*100)}%, "
                        f"{apps_count} apps monitored. "
                        f"The surveillance system is finely tuned to your needs now, {self.user_name}.",
                    ]

                return random.choice(celebrations)

        return None

    def _generate_learning_acknowledgment(
        self,
        watch_params: Dict[str, Any],
        result: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate acknowledgment when the system learns something new.

        Examples:
        - First time monitoring a new app
        - First God Mode operation
        - Improved confidence trend
        - New detection pattern

        Returns:
            Learning acknowledgment message if applicable, None otherwise
        """
        import random

        app_name = watch_params['app_name']
        all_spaces = watch_params['all_spaces']

        # First time monitoring this app
        if app_name not in self.surveillance_stats['apps_monitored']:
            first_app_messages = [
                f"First time monitoring {app_name}, {self.user_name}. I've learned its visual characteristics now.",
                f"New app learned: {app_name}. Future surveillance on this app will be faster.",
                f"That's my first {app_name} surveillance. I've got its signature now.",
            ]
            return random.choice(first_app_messages)

        # First God Mode operation
        if all_spaces and self.surveillance_stats['god_mode_operations'] == 0:
            god_mode_first = [
                f"First God Mode operation activated, {self.user_name}. "
                f"Parallel surveillance is now part of my skill set.",

                f"That was my first multi-space surveillance. "
                f"I can now watch across your entire workspace simultaneously.",
            ]
            return random.choice(god_mode_first)

        # Confidence improvement
        confidence = result.get('confidence', 0.0)
        avg_confidence = self.surveillance_stats['average_confidence']

        if confidence > 0.95 and avg_confidence < 0.90 and self.surveillance_stats['successful_detections'] > 5:
            confidence_improvement = [
                f"Detection confidence is improving, {self.user_name}. "
                f"This one was {int(confidence*100)}% - well above my average of {int(avg_confidence*100)}%.",

                f"That's my highest confidence detection yet at {int(confidence*100)}%. "
                f"My OCR accuracy is getting sharper.",
            ]
            return random.choice(confidence_improvement)

        return None

    async def _execute_surveillance_command(self, watch_params: Dict[str, Any]) -> str:
        """
        Execute God Mode surveillance command by routing to VisualMonitorAgent.

        Args:
            watch_params: Dict with app_name, trigger_text, all_spaces, max_duration

        Returns:
            Voice-friendly response string
        """
        app_name = watch_params['app_name']
        trigger_text = watch_params['trigger_text']
        all_spaces = watch_params['all_spaces']
        max_duration = watch_params.get('max_duration')

        # Get VisualMonitorAgent (lazy init)
        agent = await self._get_visual_monitor_agent()
        if not agent:
            return self._format_error_response("initialization_failed", app_name, trigger_text)

        try:
            # Voice acknowledgment before starting - context-aware and natural
            initial_msg = self._build_surveillance_start_message(
                app_name, trigger_text, all_spaces, max_duration
            )

            logger.info(f"🏎️  Activating God Mode: app={app_name}, trigger='{trigger_text}', "
                       f"all_spaces={all_spaces}, duration={max_duration}")

            # Prepare action config for when trigger is detected
            action_config = {
                'voice_announcement': True,
                'user_name': self.user_name,
                'app_name': app_name,
                'trigger_text': trigger_text
            }

            # Call VisualMonitorAgent's unified watch interface
            # This routes to either single-window or God Mode based on all_spaces parameter
            result = await agent.watch(
                app_name=app_name,
                trigger_text=trigger_text,
                all_spaces=all_spaces,
                action_config=action_config,
                max_duration=max_duration
            )

            # Process results and generate voice-friendly response
            return self._format_surveillance_response(result, watch_params, initial_msg)

        except asyncio.TimeoutError:
            return self._format_timeout_response(app_name, trigger_text, max_duration)

        except Exception as e:
            logger.error(f"Surveillance command execution error: {e}", exc_info=True)
            return self._format_error_response("runtime_error", app_name, trigger_text)

    def _format_surveillance_response(
        self,
        result: Dict[str, Any],
        watch_params: Dict[str, Any],
        initial_msg: str
    ) -> str:
        """
        Format surveillance results into voice-friendly response.

        Args:
            result: Result dict from VisualMonitorAgent
            watch_params: Original watch parameters
            initial_msg: Initial acknowledgment message

        Returns:
            Human-friendly response string
        """
        app_name = watch_params['app_name']
        trigger_text = watch_params['trigger_text']
        all_spaces = watch_params['all_spaces']

        # Check result status
        status = result.get('status', 'unknown')
        trigger_detected = result.get('trigger_detected', False)

        if status == 'error':
            # Record failed operation
            self._record_surveillance_operation(watch_params, result, success=False)
            error_msg = result.get('error', 'unknown error')
            return (f"{initial_msg}\n\n"
                   f"However, I encountered an error: {error_msg}")

        if trigger_detected:
            # SUCCESS - Trigger was found!
            # Note: _build_success_response handles recording internally
            return self._build_success_response(
                result, watch_params, initial_msg
            )

        else:
            # NO TRIGGER DETECTED - Record as failed operation
            self._record_surveillance_operation(watch_params, result, success=False)

            if status == 'timeout':
                duration = watch_params.get('max_duration', 300)
                return (f"{initial_msg}\n\n"
                       f"Surveillance completed after {duration} seconds. "
                       f"I didn't detect '{trigger_text}' in {app_name}, {self.user_name}.")
            elif status == 'no_windows':
                return (f"{initial_msg}\n\n"
                       f"However, I couldn't find any {app_name} windows to monitor. "
                       f"Please make sure {app_name} is open.")
            else:
                return (f"{initial_msg}\n\n"
                       f"Monitoring complete, but I didn't detect '{trigger_text}' in {app_name}.")

    async def handle_command(self, text: str, context: Optional[Dict] = None) -> Tuple[str, str]:
        """
        Intelligently handle command using Swift classification
        
        Returns:
            Tuple of (response, handler_used)
        """
        if not self.enabled:
            return "I need my API key to handle commands intelligently.", "error"
        
        try:
            # Check for weather-related queries FIRST - route to system for Weather app workflow
            if any(word in text.lower() for word in ['weather', 'temperature', 'forecast', 'rain', 'snow', 'sunny', 'cloudy', 'hot', 'cold', 'humid', 'windy', 'storm']):
                logger.info(f"Detected weather query, routing to system handler for Weather app workflow")
                # Create classification for weather
                classification = {'type': 'system', 'confidence': 0.9, 'intent': 'weather'}
                response = await self._handle_system_command(text, classification)
                return response, 'system'
            
            # Get intelligent classification from Swift
            try:
                result = await self.router.route_command(text, context)
                if isinstance(result, tuple) and len(result) == 2:
                    handler_type, classification = result
                else:
                    # Fallback for incorrect return format
                    logger.warning(f"Unexpected router response format: {type(result)}")
                    handler_type = 'conversation'
                    classification = {'confidence': 0.5, 'type': 'conversation'}
            except (ValueError, TypeError) as e:
                # Handle unpacking errors
                logger.warning(f"Router unpacking error: {e}")
                handler_type = 'conversation'
                classification = {'confidence': 0.5, 'type': 'conversation'}
            except Exception as e:
                # Catch all other errors
                logger.warning(f"Router error: {e}")
                handler_type = 'conversation'
                classification = {'confidence': 0.5, 'type': 'conversation'}
            
            logger.info(f"Intelligent routing: '{text}' → {handler_type} "
                       f"(confidence: {classification.get('confidence', 0):.2f})")
            
            
            if handler_type == 'system':
                response = await self._handle_system_command(text, classification)
            elif handler_type == 'vision':
                response = await self._handle_vision_command(text, classification)
            elif handler_type == 'conversation':
                response = await self._handle_conversation(text, classification)
            else:
                # Fallback
                response = await self._handle_fallback(text, classification)
                handler_type = 'fallback'
            
            # Record for learning
            self._record_command(text, handler_type, classification, response)
            
            return response, handler_type
            
        except Exception as e:
            logger.error(f"Error in intelligent command handling: {e}")
            return f"I encountered an error processing your command, {self.user_name}.", "error"
    
    async def _handle_system_command(self, text: str, classification: Dict[str, Any]) -> str:
        """Handle system control commands"""
        try:
            # Build context with classification insights
            context = {
                "classification": classification,
                "entities": classification.get('entities', {}),
                "intent": classification.get('intent', 'unknown'),
                "user": self.user_name,
                "timestamp": datetime.now().isoformat()
            }
            
            # Use command interpreter
            intent = await self.command_interpreter.interpret_command(text, context)
            
            # Execute if confident
            if intent.confidence >= 0.6:
                result = await self.command_interpreter.execute_intent(intent)
                
                if result.success:
                    # Learn from successful execution
                    await self.router.provide_feedback(text, 'system', True)
                    return self._format_success_response(intent, result)
                else:
                    return f"I couldn't complete that action: {result.message}"
            else:
                return f"I'm not confident about that command, {self.user_name}. Could you rephrase?"
                
        except Exception as e:
            logger.error(f"System command error: {e}")
            return f"I encountered an error with that system command, {self.user_name}."
    
    async def _handle_vision_command(self, text: str, classification: Dict[str, Any]) -> str:
        """Handle vision analysis commands"""
        try:
            # Build context with classification insights
            context = {
                "classification": classification,
                "entities": classification.get('entities', {}),
                "intent": classification.get('intent', 'unknown'),
                "user": self.user_name,
                "timestamp": datetime.now().isoformat()
            }

            # ===================================================================
            # PRIORITY 1: Check for God Mode surveillance commands FIRST
            # ===================================================================
            watch_params = self._parse_watch_command(text)
            if watch_params:
                logger.info(f"🎯 God Mode surveillance command detected: {watch_params}")
                return await self._execute_surveillance_command(watch_params)

            # ===================================================================
            # PRIORITY 2: Check for screen monitoring commands
            # ===================================================================
            monitoring_keywords = [
                'monitor', 'monitoring', 'watch', 'watching', 'track', 'tracking',
                'continuous', 'continuously', 'real-time', 'realtime', 'actively',
                'surveillance', 'observe', 'observing', 'stream', 'streaming'
            ]

            screen_keywords = ['screen', 'display', 'desktop', 'workspace', 'monitor']

            # Check if this is a monitoring command
            text_lower = text.lower()
            has_monitoring = any(keyword in text_lower for keyword in monitoring_keywords)
            has_screen = any(keyword in text_lower for keyword in screen_keywords)
            
            if has_monitoring and has_screen:
                # This is a screen monitoring command - delegate to chatbot
                if self.claude_chatbot:
                    try:
                        response = await self.claude_chatbot.generate_response(text)
                        # Learn from successful monitoring command
                        await self.router.provide_feedback(text, 'vision', True)
                        return response
                    except Exception as e:
                        logger.error(f"Error handling monitoring command: {e}")
                        return f"I encountered an error setting up monitoring, {self.user_name}."
                else:
                    return f"I need my vision capabilities to monitor your screen, {self.user_name}."
            
            # Use command interpreter for other vision commands
            intent = await self.command_interpreter.interpret_command(text, context)
            
            # For "can you see my screen?" type questions, ensure we get a proper response
            if any(phrase in text.lower() for phrase in ['can you see', 'do you see', 'are you able to see']):
                # This is a yes/no question about vision capability
                # Add confirmation flag to context
                context['is_vision_confirmation'] = True
                
                # Re-interpret with the updated context
                intent = await self.command_interpreter.interpret_command(text, context)
                
                # Execute the vision command to get screen content with timeout
                try:
                    result = await asyncio.wait_for(
                        self.command_interpreter.execute_intent(intent),
                        timeout=30.0
                    )
                except asyncio.TimeoutError:
                    logger.error("Vision command timed out after 30 seconds")
                    return f"I'm sorry {self.user_name}, the vision analysis is taking too long. Please try again."
                
                if result.success and result.message:
                    # Check if we got the unwanted "options" response
                    if "I'm not quite sure what you'd like me to do" in result.message:
                        # Force a direct screen analysis instead
                        direct_intent = await self.command_interpreter.interpret_command("describe my screen", context)
                        direct_result = await self.command_interpreter.execute_intent(direct_intent)
                        if direct_result.success and direct_result.message:
                            return f"Yes {self.user_name}, I can see your screen. {direct_result.message}"
                    else:
                        # Format as a confirmation with description
                        return f"Yes {self.user_name}, I can see your screen. {result.message}"
                else:
                    return f"I'm having trouble accessing the screen right now, {self.user_name}."
            
            # For other vision commands, execute normally
            if intent.confidence >= 0.6:
                try:
                    result = await asyncio.wait_for(
                        self.command_interpreter.execute_intent(intent),
                        timeout=30.0
                    )
                except asyncio.TimeoutError:
                    logger.error("Vision command timed out after 30 seconds")
                    return f"I'm sorry {self.user_name}, the vision analysis is taking too long. Please try again."
                
                if result.success:
                    # Learn from successful execution
                    await self.router.provide_feedback(text, 'vision', True)
                    return result.message  # Return the actual vision analysis
                else:
                    return f"I couldn't analyze the screen: {result.message}"
            else:
                return f"I'm not confident about that vision command, {self.user_name}. Could you rephrase?"
                
        except Exception as e:
            logger.error(f"Vision command error: {e}")
            error_msg = str(e).lower()
            
            # Provide specific error messages based on the error type
            if "503" in error_msg or "service unavailable" in error_msg:
                return f"The vision system is temporarily unavailable, {self.user_name}. Please try again in a moment."
            elif "timeout" in error_msg:
                return f"The vision analysis is taking too long, {self.user_name}. Please try again."
            elif "connection" in error_msg or "connect" in error_msg:
                return f"I'm having trouble connecting to the vision system, {self.user_name}. Please check if the service is running."
            elif "permission" in error_msg or "denied" in error_msg:
                return f"I don't have permission to access the screen, {self.user_name}. Please check the system permissions."
            else:
                return f"I encountered an error analyzing the screen: {str(e)[:100]}. Please try again, {self.user_name}."
    
    async def _handle_conversation(self, text: str, classification: Dict[str, Any]) -> str:
        """Handle conversational queries"""
        if self.claude_chatbot:
            try:
                response = await self.claude_chatbot.generate_response(text)
                return response
            except Exception as e:
                logger.error(f"Conversation error: {e}")
                return f"I'm having trouble with that conversation, {self.user_name}."
        else:
            return f"I need my Claude API to have conversations, {self.user_name}."
    
    async def _handle_fallback(self, text: str, classification: Dict[str, Any]) -> str:
        """Fallback handler for uncertain classifications"""
        confidence = classification.get('confidence', 0)
        
        if confidence < 0.3:
            return f"I'm not sure how to interpret that command, {self.user_name}. Could you rephrase?"
        else:
            # Try conversation as fallback
            return await self._handle_conversation(text, classification)
    
    def _format_success_response(self, intent: Any, result: Any) -> str:
        """Format successful command execution response"""
        if intent.action == "close_app":
            return f"{intent.target} has been closed, {self.user_name}."
        elif intent.action == "open_app":
            return f"I've opened {intent.target} for you, {self.user_name}."
        elif intent.action == "switch_app":
            return f"Switched to {intent.target}, {self.user_name}."
        elif intent.action in ["describe_screen", "analyze_window", "check_screen"]:
            # For vision commands, return the actual analysis result
            if hasattr(result, 'message') and result.message:
                return result.message
            else:
                return f"I've analyzed your screen, {self.user_name}."
        elif intent.action == "check_weather_app" or (hasattr(intent, 'raw_command') and 'weather' in intent.raw_command.lower()):
            # For weather commands, return the actual weather information
            if hasattr(result, 'message') and result.message:
                return result.message
            else:
                return f"I've checked the weather for you, {self.user_name}."
        else:
            # For any other command with a message, return it
            if hasattr(result, 'message') and result.message and not result.message.startswith("Command executed"):
                return result.message
            else:
                return f"Command executed successfully, {self.user_name}."
    
    def _record_command(self, text: str, handler: str, 
                       classification: Dict[str, Any], response: str):
        """Record command for learning and analysis"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'command': text,
            'handler': handler,
            'classification': classification,
            'response_preview': response[:100] + '...' if len(response) > 100 else response
        }
        
        self.command_history.append(record)
        
        # Limit history size
        if len(self.command_history) > self.max_history:
            self.command_history.pop(0)
    
    async def get_classification_stats(self) -> Dict[str, Any]:
        """Get statistics about command classification"""
        stats = await self.router.classifier.get_stats()
        
        # Add local stats
        handler_counts = {}
        for record in self.command_history:
            handler = record['handler']
            handler_counts[handler] = handler_counts.get(handler, 0) + 1
        
        stats['recent_commands'] = len(self.command_history)
        stats['handler_distribution'] = handler_counts
        
        return stats
    
    async def improve_from_feedback(self, command: str, 
                                  correct_handler: str, 
                                  was_successful: bool):
        """Improve classification based on user feedback"""
        await self.router.provide_feedback(command, correct_handler, was_successful)
        logger.info(f"Learned: '{command}' should use {correct_handler} handler")
