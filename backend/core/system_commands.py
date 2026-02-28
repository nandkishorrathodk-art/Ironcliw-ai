"""
Platform Abstraction Layer - System Commands Module

This module provides cross-platform system command execution abstraction,
enabling Ironcliw to run platform-specific commands in a unified interface.

Created: 2026-02-22
Purpose: Windows/Linux porting - Phase 1 (PAL)
"""

import subprocess
import shutil
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Dict, Any
import logging

from backend.core.platform_abstraction import PlatformDetector, SupportedPlatform


logger = logging.getLogger(__name__)


class SystemCommandInterface(ABC):
    """
    Abstract base class for platform-specific system commands.
    
    All platform implementations must inherit from this class and
    implement the required abstract methods.
    """
    
    @abstractmethod
    def get_shell_executable(self) -> str:
        """
        Get the path to the platform's default shell executable.
        
        Returns:
            str: Path to shell executable (e.g., /bin/bash, cmd.exe)
        """
        pass
    
    @abstractmethod
    def get_open_command(self) -> str:
        """
        Get the command to open files/URLs.
        
        Returns:
            str: Command name (e.g., "open", "start", "xdg-open")
        """
        pass
    
    @abstractmethod
    def get_say_command(self) -> Optional[str]:
        """
        Get the command for text-to-speech (if available).
        
        Returns:
            str | None: Command name or None if not available
        """
        pass
    
    @abstractmethod
    def get_clipboard_copy_command(self) -> Optional[str]:
        """
        Get the command to copy to clipboard.
        
        Returns:
            str | None: Command name or None if not available
        """
        pass
    
    @abstractmethod
    def get_clipboard_paste_command(self) -> Optional[str]:
        """
        Get the command to paste from clipboard.
        
        Returns:
            str | None: Command name or None if not available
        """
        pass
    
    @abstractmethod
    def execute_command(
        self,
        command: List[str],
        shell: bool = False,
        capture_output: bool = True,
        timeout: Optional[int] = None,
    ) -> Tuple[int, str, str]:
        """
        Execute a system command.
        
        Args:
            command: Command and arguments as list
            shell: Whether to execute through shell
            capture_output: Whether to capture stdout/stderr
            timeout: Timeout in seconds
        
        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        pass
    
    @abstractmethod
    def get_process_list_command(self) -> List[str]:
        """
        Get the command to list running processes.
        
        Returns:
            list: Command and arguments
        """
        pass
    
    @abstractmethod
    def get_kill_process_command(self, pid: int) -> List[str]:
        """
        Get the command to kill a process by PID.
        
        Args:
            pid: Process ID
        
        Returns:
            list: Command and arguments
        """
        pass
    
    @abstractmethod
    def get_environment_variable(self, name: str) -> Optional[str]:
        """
        Get an environment variable value.
        
        Args:
            name: Variable name
        
        Returns:
            str | None: Variable value or None if not set
        """
        pass
    
    @abstractmethod
    def set_environment_variable(self, name: str, value: str) -> bool:
        """
        Set an environment variable.
        
        Args:
            name: Variable name
            value: Variable value
        
        Returns:
            bool: True if successful, False otherwise
        """
        pass


class MacOSCommands(SystemCommandInterface):
    """macOS-specific system commands implementation."""
    
    def get_shell_executable(self) -> str:
        return "/bin/zsh"  # macOS default shell since Catalina
    
    def get_open_command(self) -> str:
        return "open"
    
    def get_say_command(self) -> Optional[str]:
        # macOS has built-in 'say' command
        return "say" if shutil.which("say") else None
    
    def get_clipboard_copy_command(self) -> Optional[str]:
        return "pbcopy" if shutil.which("pbcopy") else None
    
    def get_clipboard_paste_command(self) -> Optional[str]:
        return "pbpaste" if shutil.which("pbpaste") else None
    
    def execute_command(
        self,
        command: List[str],
        shell: bool = False,
        capture_output: bool = True,
        timeout: Optional[int] = None,
    ) -> Tuple[int, str, str]:
        """Execute command on macOS."""
        try:
            result = subprocess.run(
                command,
                shell=shell,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
            )
            return result.returncode, result.stdout or "", result.stderr or ""
        except subprocess.TimeoutExpired as e:
            logger.error(f"Command timeout: {command}")
            return -1, "", f"Command timeout after {timeout}s"
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return -1, "", str(e)
    
    def get_process_list_command(self) -> List[str]:
        return ["ps", "aux"]
    
    def get_kill_process_command(self, pid: int) -> List[str]:
        return ["kill", str(pid)]
    
    def get_environment_variable(self, name: str) -> Optional[str]:
        import os
        return os.environ.get(name)
    
    def set_environment_variable(self, name: str, value: str) -> bool:
        import os
        try:
            os.environ[name] = value
            return True
        except Exception as e:
            logger.error(f"Failed to set env var {name}: {e}")
            return False


class WindowsCommands(SystemCommandInterface):
    """Windows-specific system commands implementation."""
    
    def get_shell_executable(self) -> str:
        return "cmd.exe"
    
    def get_open_command(self) -> str:
        return "start"
    
    def get_say_command(self) -> Optional[str]:
        # Windows doesn't have built-in say command
        # Will use pyttsx3 library instead (handled separately)
        return None
    
    def get_clipboard_copy_command(self) -> Optional[str]:
        # Windows doesn't have built-in clipboard command
        # Will use pyperclip library instead (handled separately)
        return None
    
    def get_clipboard_paste_command(self) -> Optional[str]:
        # Windows doesn't have built-in clipboard command
        # Will use pyperclip library instead (handled separately)
        return None
    
    def execute_command(
        self,
        command: List[str],
        shell: bool = False,
        capture_output: bool = True,
        timeout: Optional[int] = None,
    ) -> Tuple[int, str, str]:
        """Execute command on Windows."""
        try:
            # Windows-specific: use CREATE_NEW_PROCESS_GROUP for proper handling
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if not shell else 0
            
            result = subprocess.run(
                command,
                shell=shell,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
                creationflags=creationflags,
            )
            return result.returncode, result.stdout or "", result.stderr or ""
        except subprocess.TimeoutExpired as e:
            logger.error(f"Command timeout: {command}")
            return -1, "", f"Command timeout after {timeout}s"
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return -1, "", str(e)
    
    def get_process_list_command(self) -> List[str]:
        return ["tasklist"]
    
    def get_kill_process_command(self, pid: int) -> List[str]:
        return ["taskkill", "/F", "/PID", str(pid)]
    
    def get_environment_variable(self, name: str) -> Optional[str]:
        import os
        return os.environ.get(name)
    
    def set_environment_variable(self, name: str, value: str) -> bool:
        import os
        try:
            os.environ[name] = value
            return True
        except Exception as e:
            logger.error(f"Failed to set env var {name}: {e}")
            return False


class LinuxCommands(SystemCommandInterface):
    """Linux-specific system commands implementation."""
    
    def get_shell_executable(self) -> str:
        return "/bin/bash"
    
    def get_open_command(self) -> str:
        # Try to find the best available command
        if shutil.which("xdg-open"):
            return "xdg-open"
        elif shutil.which("gnome-open"):
            return "gnome-open"
        elif shutil.which("kde-open"):
            return "kde-open"
        else:
            return "xdg-open"  # Default, might not work
    
    def get_say_command(self) -> Optional[str]:
        # Linux might have espeak or festival
        if shutil.which("espeak"):
            return "espeak"
        elif shutil.which("festival"):
            return "festival"
        else:
            return None
    
    def get_clipboard_copy_command(self) -> Optional[str]:
        # Try to find clipboard command
        if shutil.which("xclip"):
            return "xclip -selection clipboard"
        elif shutil.which("xsel"):
            return "xsel --clipboard --input"
        else:
            return None
    
    def get_clipboard_paste_command(self) -> Optional[str]:
        # Try to find clipboard command
        if shutil.which("xclip"):
            return "xclip -selection clipboard -o"
        elif shutil.which("xsel"):
            return "xsel --clipboard --output"
        else:
            return None
    
    def execute_command(
        self,
        command: List[str],
        shell: bool = False,
        capture_output: bool = True,
        timeout: Optional[int] = None,
    ) -> Tuple[int, str, str]:
        """Execute command on Linux."""
        try:
            result = subprocess.run(
                command,
                shell=shell,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
            )
            return result.returncode, result.stdout or "", result.stderr or ""
        except subprocess.TimeoutExpired as e:
            logger.error(f"Command timeout: {command}")
            return -1, "", f"Command timeout after {timeout}s"
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return -1, "", str(e)
    
    def get_process_list_command(self) -> List[str]:
        return ["ps", "aux"]
    
    def get_kill_process_command(self, pid: int) -> List[str]:
        return ["kill", str(pid)]
    
    def get_environment_variable(self, name: str) -> Optional[str]:
        import os
        return os.environ.get(name)
    
    def set_environment_variable(self, name: str, value: str) -> bool:
        import os
        try:
            os.environ[name] = value
            return True
        except Exception as e:
            logger.error(f"Failed to set env var {name}: {e}")
            return False


class SystemCommandFactory:
    """
    Factory class to create platform-specific system command instances.
    
    This factory automatically detects the current platform and returns
    the appropriate SystemCommandInterface implementation.
    """
    
    _instance: Optional[SystemCommandInterface] = None
    
    @classmethod
    def get_instance(cls) -> SystemCommandInterface:
        """
        Get the platform-specific system command instance (singleton).
        
        Returns:
            SystemCommandInterface: Platform-specific implementation
        """
        if cls._instance is None:
            detector = PlatformDetector()
            platform = detector.get_platform()
            
            if platform == SupportedPlatform.MACOS:
                cls._instance = MacOSCommands()
                logger.info("System commands: macOS implementation")
            elif platform == SupportedPlatform.WINDOWS:
                cls._instance = WindowsCommands()
                logger.info("System commands: Windows implementation")
            elif platform == SupportedPlatform.LINUX:
                cls._instance = LinuxCommands()
                logger.info("System commands: Linux implementation")
            else:
                # Fallback to Unix-like commands
                logger.warning("Unknown platform, using Unix-like commands")
                cls._instance = MacOSCommands()
        
        return cls._instance


# Convenience function to get system commands
def get_system_commands() -> SystemCommandInterface:
    """Get the platform-specific system command interface."""
    return SystemCommandFactory.get_instance()
