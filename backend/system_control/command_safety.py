"""
Command Safety Tier Classification System
Classifies shell commands by risk level and determines execution requirements.

Philosophy:
- Never auto-execute destructive commands
- Require confirmation for file system changes
- Allow safe read-only commands freely
- Track commands that can be reverted
- Learn from user overrides

Safety Tiers:
- GREEN (Safe): Read-only, no side effects, auto-executable
- YELLOW (Caution): Modifies state, requires confirmation once
- RED (Dangerous): Irreversible/destructive, always confirm

Cross-Repo Integration (v10.3):
- Emits safety events to Reactor Core for training
- Writes classification state to shared file for Ironcliw Prime
- Supports async classification with event emission
"""
from __future__ import annotations

import asyncio
import json
import re
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Callable, Awaitable, Dict, List, Optional, Set, Tuple, Any
from pathlib import Path
import shlex

if TYPE_CHECKING:
    from clients.reactor_core_client import ReactorCoreClient

logger = logging.getLogger(__name__)

# Cross-repo state directory (v10.3)
COMMAND_SAFETY_STATE_DIR = Path.home() / ".jarvis" / "cross_repo" / "command_safety"


class SafetyTier(str, Enum):
    """Command safety classification tiers."""
    GREEN = "green"      # Safe, auto-executable
    YELLOW = "yellow"    # Caution, confirm once
    RED = "red"          # Dangerous, always confirm
    UNKNOWN = "unknown"  # Unclassified, default to caution


class RiskCategory(str, Enum):
    """Categories of command risks."""
    DATA_LOSS = "data_loss"                      # rm, dd, format
    SYSTEM_MODIFICATION = "system_modification"  # chmod, chown, sudo
    NETWORK_EXPOSURE = "network_exposure"        # curl, wget with pipes
    PROCESS_CONTROL = "process_control"          # kill, pkill
    FILE_MODIFICATION = "file_modification"      # mv, cp, write operations
    PACKAGE_MANAGEMENT = "package_management"    # npm, pip, brew install
    VERSION_CONTROL = "version_control"          # git push, git reset
    DATABASE_OPERATION = "database_operation"    # DROP, DELETE, TRUNCATE
    SAFE_READ = "safe_read"                      # ls, cat, grep
    SAFE_NAVIGATION = "safe_navigation"          # cd, pwd


@dataclass
class CommandClassification:
    """Result of command safety classification."""
    command: str
    tier: SafetyTier
    risk_categories: List[RiskCategory]
    requires_confirmation: bool
    is_reversible: bool
    confidence: float  # 0.0-1.0
    reasoning: str
    suggested_alternative: Optional[str] = None
    dry_run_available: bool = False
    # Cross-repo metadata (v10.3)
    classification_id: str = field(default_factory=lambda: f"cmd-{int(datetime.now().timestamp() * 1000)}")
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: Optional[str] = None

    @property
    def is_safe(self) -> bool:
        """Quick check if command is safe to execute."""
        return self.tier == SafetyTier.GREEN

    @property
    def is_destructive(self) -> bool:
        """Check if command is potentially destructive."""
        return (
            self.tier == SafetyTier.RED
            or RiskCategory.DATA_LOSS in self.risk_categories
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization (v10.3)."""
        return {
            "classification_id": self.classification_id,
            "command": self.command,
            "tier": self.tier.value,
            "risk_categories": [r.value for r in self.risk_categories],
            "requires_confirmation": self.requires_confirmation,
            "is_reversible": self.is_reversible,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "suggested_alternative": self.suggested_alternative,
            "dry_run_available": self.dry_run_available,
            "is_safe": self.is_safe,
            "is_destructive": self.is_destructive,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
        }

    def to_prime_context(self) -> str:
        """Generate context for Ironcliw Prime routing (v10.3)."""
        if self.tier == SafetyTier.GREEN:
            return ""  # No context needed for safe commands

        lines = ["[COMMAND SAFETY]"]
        lines.append(f"- Command: {self.command[:50]}{'...' if len(self.command) > 50 else ''}")
        lines.append(f"- Tier: {self.tier.value.upper()}")
        lines.append(f"- Risks: {', '.join(r.value for r in self.risk_categories)}")

        if self.suggested_alternative:
            lines.append(f"- Safer alternative: {self.suggested_alternative}")

        lines.append("[/COMMAND SAFETY]")
        return "\n".join(lines)


class CommandSafetyClassifier:
    """
    Classifies shell commands by safety tier.

    Uses pattern matching, command parsing, and heuristics to determine
    if a command is safe to execute automatically or requires user confirmation.

    Cross-Repo Integration (v10.3):
    - Emits classification events to Reactor Core for training
    - Writes state to shared files for Ironcliw Prime
    - Tracks classification statistics per session
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        enable_cross_repo: bool = True,
        session_id: Optional[str] = None,
    ):
        """
        Initialize command safety classifier.

        Args:
            config_path: Optional path to custom safety rules JSON
            enable_cross_repo: Enable cross-repo event emission (v10.3)
            session_id: Session ID for cross-repo tracking
        """
        # Cross-repo integration (v10.3)
        self.enable_cross_repo = enable_cross_repo
        self.session_id = session_id or f"cmd-{int(datetime.now().timestamp())}"
        self._reactor_client: Optional["ReactorCoreClient"] = None
        self._event_callbacks: List[Callable[[Dict[str, Any]], Awaitable[None]]] = []

        # Classification statistics (v10.3)
        self._stats = {
            "total_classifications": 0,
            "green_count": 0,
            "yellow_count": 0,
            "red_count": 0,
            "blocked_count": 0,
            "session_start": datetime.now().isoformat(),
        }

        # Initialize cross-repo state directory
        if self.enable_cross_repo:
            self._init_cross_repo_state()

        # GREEN tier: Safe, read-only commands
        self.green_commands: Set[str] = {
            # File viewing
            'ls', 'cat', 'less', 'more', 'head', 'tail', 'file', 'stat',
            'wc', 'du', 'df', 'tree',

            # Text processing
            'grep', 'egrep', 'fgrep', 'sed', 'awk', 'cut', 'sort', 'uniq',
            'tr', 'diff', 'comm', 'cmp',

            # Navigation
            'cd', 'pwd', 'pushd', 'popd', 'dirs',

            # Process inspection
            'ps', 'top', 'htop', 'pgrep', 'jobs', 'fg', 'bg',

            # System info
            'uname', 'hostname', 'whoami', 'id', 'groups', 'uptime', 'date',
            'cal', 'which', 'whereis', 'whatis', 'man', 'info', 'env',

            # Git read-only
            'git status', 'git log', 'git diff', 'git show', 'git branch',
            'git remote', 'git config --get', 'git ls-files', 'git blame',

            # Network inspection
            'ping', 'netstat', 'ifconfig', 'ip addr', 'nslookup', 'dig',
            'traceroute', 'mtr',

            # Python/Node inspection
            'python --version', 'python -m pip list', 'node --version',
            'npm list', 'pip list', 'pip show',

            # Docker inspection
            'docker ps', 'docker images', 'docker logs', 'docker inspect',

            # Other safe utilities
            'echo', 'printf', 'sleep', 'true', 'false', 'yes', 'seq',
            'basename', 'dirname', 'realpath', 'readlink',
        }

        # YELLOW tier: Modify state but generally safe with confirmation
        self.yellow_commands: Set[str] = {
            # Package management
            'npm install', 'npm update', 'npm ci',
            'pip install', 'pip install --upgrade',
            'brew install', 'brew upgrade', 'brew update',
            'apt install', 'apt update', 'apt upgrade',
            'yum install', 'yum update',

            # Git operations
            'git add', 'git commit', 'git pull', 'git fetch', 'git merge',
            'git checkout', 'git switch', 'git stash', 'git cherry-pick',

            # File operations (non-destructive)
            'cp', 'mv', 'mkdir', 'touch', 'ln',

            # Build operations
            'make', 'cmake', 'cargo build', 'go build', 'npm run build',
            'python setup.py install',

            # Testing
            'pytest', 'npm test', 'cargo test', 'go test', 'jest',

            # Docker operations
            'docker build', 'docker run', 'docker start', 'docker stop',

            # Process control (non-critical)
            'kill', 'killall', 'pkill',
        }

        # RED tier: Destructive/dangerous commands
        self.red_commands: Set[str] = {
            # Data deletion
            'rm', 'rm -f', 'rm -rf', 'rmdir', 'unlink', 'shred',

            # Disk operations
            'dd', 'fdisk', 'mkfs', 'parted', 'gparted', 'format',

            # System modifications
            'chmod', 'chown', 'chgrp', 'usermod', 'groupmod',
            'systemctl', 'service', 'launchctl',

            # Network operations with risk
            'curl | sh', 'wget | sh', 'curl | bash', 'wget | bash',
            'scp', 'sftp', 'rsync --delete',

            # Git destructive
            'git push --force', 'git push -f', 'git reset --hard',
            'git clean -fd', 'git rebase -i',

            # Database operations
            'DROP TABLE', 'DROP DATABASE', 'TRUNCATE', 'DELETE FROM',
            'ALTER TABLE', 'UPDATE', 'mysql', 'psql',

            # System commands
            'sudo', 'su', 'doas', 'shutdown', 'reboot', 'init',
            'halt', 'poweroff',

            # Package removal
            'npm uninstall', 'pip uninstall', 'brew uninstall',
            'apt remove', 'apt purge', 'yum remove',

            # Docker destructive
            'docker rm', 'docker rmi', 'docker system prune',
        }

        # Destructive patterns (regex)
        self.destructive_patterns: List[Tuple[re.Pattern, str]] = [
            (re.compile(r'\brm\s+-[rf]+'), "rm with -rf flags"),
            (re.compile(r'\|.*\b(sh|bash|zsh|fish)\b'), "pipe to shell"),
            (re.compile(r'>\s*/dev/(sd[a-z]|disk\d+)'), "write to disk device"),
            (re.compile(r'sudo\s+rm'), "sudo rm"),
            (re.compile(r'dd\s+.*of='), "dd output to file/device"),
            (re.compile(r'mkfs\.\w+'), "filesystem creation"),
            (re.compile(r'chmod\s+777'), "chmod 777"),
            (re.compile(r'--force\b'), "force flag"),
            (re.compile(r'DROP\s+(TABLE|DATABASE)\b', re.IGNORECASE), "SQL DROP"),
            (re.compile(r'DELETE\s+FROM\b', re.IGNORECASE), "SQL DELETE"),
            (re.compile(r'TRUNCATE\b', re.IGNORECASE), "SQL TRUNCATE"),
            (re.compile(r'git\s+push.*--force'), "git force push"),
            (re.compile(r'git\s+reset\s+--hard'), "git hard reset"),
            (re.compile(r'npm\s+install\s+-g'), "npm global install"),
            (re.compile(r':\(\)\{.*:\|:&\};:'), "fork bomb pattern"),
        ]

        # Commands with dry-run support
        self.dry_run_supported: Dict[str, str] = {
            'rm': 'rm -i',  # Interactive mode
            'rsync': 'rsync --dry-run',
            'apt': 'apt --dry-run',
            'npm': 'npm --dry-run',
            'pip': 'pip install --dry-run',
            'ansible-playbook': 'ansible-playbook --check',
            'terraform': 'terraform plan',
        }

        # Reversible operations (have undo mechanisms)
        self.reversible_commands: Set[str] = {
            'git add', 'git commit', 'git stash', 'git checkout',
            'mv', 'cp', 'mkdir', 'touch',
            'npm install', 'pip install',
        }

        logger.info(
            f"[COMMAND-SAFETY] Initialized with "
            f"{len(self.green_commands)} green, "
            f"{len(self.yellow_commands)} yellow, "
            f"{len(self.red_commands)} red commands"
        )

    def classify(self, command: str) -> CommandClassification:
        """
        Classify a command by safety tier.

        Args:
            command: Shell command to classify

        Returns:
            CommandClassification with tier, risks, and recommendations
        """
        command = command.strip()
        if not command:
            return CommandClassification(
                command="",
                tier=SafetyTier.UNKNOWN,
                risk_categories=[],
                requires_confirmation=True,
                is_reversible=False,
                confidence=1.0,
                reasoning="Empty command",
            )

        # Parse command to extract base command
        base_cmd = self._extract_base_command(command)
        full_cmd = self._extract_full_command(command)

        # Check for destructive patterns first
        destructive_match = self._check_destructive_patterns(command)
        if destructive_match:
            return CommandClassification(
                command=command,
                tier=SafetyTier.RED,
                risk_categories=[RiskCategory.DATA_LOSS],
                requires_confirmation=True,
                is_reversible=False,
                confidence=0.95,
                reasoning=f"Destructive pattern detected: {destructive_match}",
                dry_run_available=self._supports_dry_run(base_cmd),
            )

        # Check tier classifications
        if self._matches_command_set(full_cmd, self.green_commands):
            return CommandClassification(
                command=command,
                tier=SafetyTier.GREEN,
                risk_categories=[RiskCategory.SAFE_READ],
                requires_confirmation=False,
                is_reversible=True,
                confidence=0.9,
                reasoning="Safe read-only command",
            )

        if self._matches_command_set(full_cmd, self.red_commands):
            risk_cats = self._determine_risk_categories(command, base_cmd)
            return CommandClassification(
                command=command,
                tier=SafetyTier.RED,
                risk_categories=risk_cats,
                requires_confirmation=True,
                is_reversible=False,
                confidence=0.9,
                reasoning="Dangerous command requiring confirmation",
                suggested_alternative=self._suggest_safer_alternative(command),
                dry_run_available=self._supports_dry_run(base_cmd),
            )

        if self._matches_command_set(full_cmd, self.yellow_commands):
            is_reversible = full_cmd in self.reversible_commands
            risk_cats = self._determine_risk_categories(command, base_cmd)
            return CommandClassification(
                command=command,
                tier=SafetyTier.YELLOW,
                risk_categories=risk_cats,
                requires_confirmation=True,
                is_reversible=is_reversible,
                confidence=0.85,
                reasoning="Modifies state, requires user confirmation",
                dry_run_available=self._supports_dry_run(base_cmd),
            )

        # Unknown command - default to YELLOW (cautious)
        return CommandClassification(
            command=command,
            tier=SafetyTier.YELLOW,
            risk_categories=[RiskCategory.SYSTEM_MODIFICATION],
            requires_confirmation=True,
            is_reversible=False,
            confidence=0.5,
            reasoning="Unknown command, defaulting to caution",
        )

    def classify_batch(self, commands: List[str]) -> List[CommandClassification]:
        """
        Classify multiple commands.

        Args:
            commands: List of commands to classify

        Returns:
            List of classifications
        """
        return [self.classify(cmd) for cmd in commands]

    def _extract_base_command(self, command: str) -> str:
        """Extract base command name (e.g., 'git' from 'git push')."""
        try:
            # Split by pipes, semicolons, etc. and take first command
            first_cmd = re.split(r'[|;&]', command)[0].strip()

            # Parse with shlex to handle quotes
            tokens = shlex.split(first_cmd)
            if not tokens:
                return ""

            # Skip environment variables (VAR=value cmd)
            for token in tokens:
                if '=' not in token or token.startswith('-'):
                    return token.split('/')[-1]  # Handle /usr/bin/cmd

            return tokens[0].split('/')[-1]

        except Exception as e:
            logger.warning(f"[COMMAND-SAFETY] Failed to parse command '{command}': {e}")
            return command.split()[0] if command.split() else ""

    def _extract_full_command(self, command: str) -> str:
        """Extract full command with subcommands (e.g., 'git push' from 'git push origin')."""
        try:
            tokens = shlex.split(command.split('|')[0].split(';')[0].strip())
            if not tokens:
                return ""

            # For commands like 'git push', return both parts
            base = self._extract_base_command(command)

            # Common multi-part commands
            if base in ['git', 'docker', 'npm', 'pip', 'brew', 'apt', 'yum', 'cargo', 'go']:
                # Find first non-flag token after base command
                for i, token in enumerate(tokens):
                    if token == base and i + 1 < len(tokens):
                        next_token = tokens[i + 1]
                        if not next_token.startswith('-'):
                            return f"{base} {next_token}"

            return base

        except Exception:
            return self._extract_base_command(command)

    def _matches_command_set(self, full_cmd: str, command_set: Set[str]) -> bool:
        """Check if command matches any command in set."""
        # Exact match
        if full_cmd in command_set:
            return True

        # Check if any command in set starts with our full_cmd
        for known_cmd in command_set:
            if known_cmd.startswith(full_cmd):
                return True

        return False

    def _check_destructive_patterns(self, command: str) -> Optional[str]:
        """Check if command matches destructive patterns."""
        for pattern, description in self.destructive_patterns:
            if pattern.search(command):
                return description
        return None

    def _determine_risk_categories(self, command: str, base_cmd: str) -> List[RiskCategory]:
        """Determine risk categories for command."""
        risks = []

        # Data loss risks
        if any(word in command for word in ['rm', 'dd', 'shred', 'DELETE', 'DROP', 'TRUNCATE']):
            risks.append(RiskCategory.DATA_LOSS)

        # System modification
        if any(word in command for word in ['chmod', 'chown', 'sudo', 'systemctl']):
            risks.append(RiskCategory.SYSTEM_MODIFICATION)

        # Network exposure
        if re.search(r'\|\s*(sh|bash)', command) and any(cmd in command for cmd in ['curl', 'wget']):
            risks.append(RiskCategory.NETWORK_EXPOSURE)

        # Process control
        if base_cmd in ['kill', 'killall', 'pkill']:
            risks.append(RiskCategory.PROCESS_CONTROL)

        # File modification
        if base_cmd in ['mv', 'cp', 'touch', 'mkdir']:
            risks.append(RiskCategory.FILE_MODIFICATION)

        # Package management
        if 'install' in command or 'uninstall' in command or 'upgrade' in command:
            risks.append(RiskCategory.PACKAGE_MANAGEMENT)

        # Version control
        if command.startswith('git'):
            risks.append(RiskCategory.VERSION_CONTROL)

        # Database
        if any(op in command.upper() for op in ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'UPDATE']):
            risks.append(RiskCategory.DATABASE_OPERATION)

        # Default to safe read if no risks identified
        if not risks:
            risks.append(RiskCategory.SAFE_READ)

        return risks

    def _supports_dry_run(self, base_cmd: str) -> bool:
        """Check if command supports dry-run mode."""
        return base_cmd in self.dry_run_supported

    def _suggest_safer_alternative(self, command: str) -> Optional[str]:
        """Suggest safer alternative for dangerous command."""
        base_cmd = self._extract_base_command(command)

        # Suggest dry-run if available
        if base_cmd in self.dry_run_supported:
            return self.dry_run_supported[base_cmd]

        # Specific suggestions
        if 'rm -rf' in command:
            return command.replace('rm -rf', 'rm -i')

        if 'git push --force' in command:
            return command.replace('--force', '--force-with-lease')

        if 'chmod 777' in command:
            return command.replace('777', '755')

        return None

    def add_custom_rule(
        self,
        command_pattern: str,
        tier: SafetyTier,
        is_reversible: bool = False,
    ) -> None:
        """
        Add custom safety rule (for user-specific workflows).

        Args:
            command_pattern: Command or pattern to classify
            tier: Safety tier to assign
            is_reversible: Whether operation can be undone
        """
        if tier == SafetyTier.GREEN:
            self.green_commands.add(command_pattern)
        elif tier == SafetyTier.YELLOW:
            self.yellow_commands.add(command_pattern)
        elif tier == SafetyTier.RED:
            self.red_commands.add(command_pattern)

        if is_reversible:
            self.reversible_commands.add(command_pattern)

        logger.info(f"[COMMAND-SAFETY] Added custom rule: '{command_pattern}' -> {tier.value}")

    # ========================================================================
    # CROSS-REPO INTEGRATION (v10.3)
    # ========================================================================

    def _init_cross_repo_state(self) -> None:
        """Initialize cross-repo state directory."""
        try:
            COMMAND_SAFETY_STATE_DIR.mkdir(parents=True, exist_ok=True)
            self._write_classification_state()
        except Exception as e:
            logger.warning(f"[COMMAND-SAFETY] Failed to init cross-repo state: {e}")

    def _write_classification_state(self) -> None:
        """Write classification state to disk for Ironcliw Prime."""
        if not self.enable_cross_repo:
            return

        try:
            state_file = COMMAND_SAFETY_STATE_DIR / "classifier_state.json"
            state = {
                "session_id": self.session_id,
                "stats": self._stats,
                "last_update": datetime.now().isoformat(),
            }
            state_file.write_text(json.dumps(state, indent=2))
        except Exception as e:
            logger.warning(f"[COMMAND-SAFETY] Failed to write state: {e}")

    async def _emit_classification_event(
        self,
        event_type: str,
        classification: CommandClassification,
    ) -> None:
        """Emit classification event to Reactor Core and callbacks (v10.3)."""
        if not self.enable_cross_repo:
            return

        event = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "classification": classification.to_dict(),
        }

        # Write event to file for Reactor Core
        try:
            events_dir = COMMAND_SAFETY_STATE_DIR / "events"
            events_dir.mkdir(parents=True, exist_ok=True)
            event_file = events_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{event_type}.json"
            event_file.write_text(json.dumps(event, indent=2))
        except Exception as e:
            logger.warning(f"[COMMAND-SAFETY] Failed to write event: {e}")

        # Call registered callbacks
        for callback in self._event_callbacks:
            try:
                await callback(event)
            except Exception as e:
                logger.warning(f"[COMMAND-SAFETY] Event callback error: {e}")

        # Send to Reactor Core client if connected
        if self._reactor_client:
            try:
                await self._reactor_client.emit_safety_event(event_type, event)
            except Exception as e:
                logger.warning(f"[COMMAND-SAFETY] Failed to send to Reactor Core: {e}")

    async def classify_async(
        self,
        command: str,
        emit_events: bool = True,
    ) -> CommandClassification:
        """
        Classify command asynchronously with event emission (v10.3).

        Args:
            command: Shell command to classify
            emit_events: Whether to emit classification events

        Returns:
            CommandClassification with tier, risks, and recommendations
        """
        # Use sync classify first
        classification = self.classify(command)
        classification.session_id = self.session_id

        # Update statistics
        self._stats["total_classifications"] += 1
        if classification.tier == SafetyTier.GREEN:
            self._stats["green_count"] += 1
        elif classification.tier == SafetyTier.YELLOW:
            self._stats["yellow_count"] += 1
        elif classification.tier == SafetyTier.RED:
            self._stats["red_count"] += 1
            if classification.is_destructive:
                self._stats["blocked_count"] += 1

        # Write updated state
        self._write_classification_state()

        # Emit events for non-GREEN classifications
        if emit_events and classification.tier != SafetyTier.GREEN:
            await self._emit_classification_event(
                "command_classified",
                classification,
            )

            # Emit special event for RED tier
            if classification.tier == SafetyTier.RED:
                await self._emit_classification_event(
                    "dangerous_command_detected",
                    classification,
                )

        return classification

    async def classify_batch_async(
        self,
        commands: List[str],
        emit_events: bool = True,
    ) -> List[CommandClassification]:
        """
        Classify multiple commands asynchronously (v10.3).

        Args:
            commands: List of commands to classify
            emit_events: Whether to emit classification events

        Returns:
            List of classifications
        """
        tasks = [self.classify_async(cmd, emit_events) for cmd in commands]
        return await asyncio.gather(*tasks)

    def set_reactor_client(self, client: "ReactorCoreClient") -> None:
        """
        Set the Reactor Core client for cross-repo event emission.

        Args:
            client: ReactorCoreClient instance
        """
        self._reactor_client = client
        logger.info("[COMMAND-SAFETY] Reactor Core client connected")

    def register_event_callback(
        self,
        callback: Callable[[Dict[str, Any]], Awaitable[None]],
    ) -> None:
        """
        Register a callback for classification events.

        Args:
            callback: Async function(event_dict) to call on events
        """
        self._event_callbacks.append(callback)
        logger.info(f"[COMMAND-SAFETY] Event callback registered (total: {len(self._event_callbacks)})")

    def get_stats(self) -> Dict[str, Any]:
        """Get classification statistics."""
        return {
            **self._stats,
            "session_id": self.session_id,
            "last_update": datetime.now().isoformat(),
        }

    def get_classification_summary_for_prime(self) -> str:
        """
        Get classification summary for Ironcliw Prime context (v10.3).

        Returns:
            Formatted string for prompt injection
        """
        if self._stats["total_classifications"] == 0:
            return ""

        lines = ["[COMMAND SAFETY SUMMARY]"]
        lines.append(f"- Total commands: {self._stats['total_classifications']}")

        if self._stats["red_count"] > 0:
            lines.append(f"- Dangerous commands blocked: {self._stats['red_count']}")

        if self._stats["yellow_count"] > 0:
            lines.append(f"- Commands requiring confirmation: {self._stats['yellow_count']}")

        lines.append("[/COMMAND SAFETY SUMMARY]")
        return "\n".join(lines)

    def reset_stats(self) -> None:
        """Reset classification statistics (call at session start)."""
        self._stats = {
            "total_classifications": 0,
            "green_count": 0,
            "yellow_count": 0,
            "red_count": 0,
            "blocked_count": 0,
            "session_start": datetime.now().isoformat(),
        }
        self._write_classification_state()
        logger.info("[COMMAND-SAFETY] Statistics reset")


# Global instance
_global_classifier: Optional[CommandSafetyClassifier] = None


def get_command_classifier() -> CommandSafetyClassifier:
    """Get or create global command safety classifier."""
    global _global_classifier

    if _global_classifier is None:
        _global_classifier = CommandSafetyClassifier()

    return _global_classifier
