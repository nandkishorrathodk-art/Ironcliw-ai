"""
Cross-Platform Path Management
===============================

Handles platform-specific path conventions and directory structures.
Ensures Ironcliw data is stored in appropriate locations on each OS.
"""

import os
import platform
from pathlib import Path
from typing import Optional


class PathManager:
    """
    Cross-platform path management for Ironcliw data directories.
    
    Follows platform conventions:
    - macOS: ~/.jarvis/
    - Windows: %LOCALAPPDATA%\\Ironcliw\\
    - Linux: ~/.local/share/jarvis/ (XDG compliant)
    """
    
    def __init__(self):
        self.os = platform.system()
        self.home = Path.home()
        
        # Detect environment override
        env_home = os.getenv("Ironcliw_HOME")
        if env_home:
            self._jarvis_home = Path(env_home)
        else:
            self._jarvis_home = self._get_default_jarvis_home()
        
        # Ensure base directory exists
        self._jarvis_home.mkdir(parents=True, exist_ok=True)
    
    def _get_default_jarvis_home(self) -> Path:
        """Get default Ironcliw home directory for current platform."""
        if self.os == "Windows":
            # Windows: C:\Users\<user>\AppData\Local\Ironcliw
            local_app_data = os.getenv("LOCALAPPDATA")
            if local_app_data:
                return Path(local_app_data) / "Ironcliw"
            return self.home / "AppData" / "Local" / "Ironcliw"
        
        elif self.os == "Darwin":
            # macOS: ~/.jarvis
            return self.home / ".jarvis"
        
        elif self.os == "Linux":
            # Linux: ~/.local/share/jarvis (XDG Base Directory spec)
            xdg_data_home = os.getenv("XDG_DATA_HOME")
            if xdg_data_home:
                return Path(xdg_data_home) / "jarvis"
            return self.home / ".local" / "share" / "jarvis"
        
        else:
            # Fallback for unknown platforms
            return self.home / ".jarvis"
    
    def get_jarvis_home(self) -> Path:
        """
        Get Ironcliw home directory.
        
        Returns:
            Path: Platform-specific Ironcliw data directory
        """
        return self._jarvis_home
    
    def get_cache_dir(self) -> Path:
        """
        Get cache directory for temporary/cached data.
        
        Returns:
            Path: Cache directory
        """
        if self.os == "Windows":
            return self._jarvis_home / "Cache"
        elif self.os == "Darwin":
            return self._jarvis_home / "cache"
        elif self.os == "Linux":
            xdg_cache = os.getenv("XDG_CACHE_HOME")
            if xdg_cache:
                cache_dir = Path(xdg_cache) / "jarvis"
            else:
                cache_dir = self.home / ".cache" / "jarvis"
            cache_dir.mkdir(parents=True, exist_ok=True)
            return cache_dir
        else:
            return self._jarvis_home / "cache"
    
    def get_models_dir(self) -> Path:
        """
        Get ML models directory.
        
        Returns:
            Path: Models directory
        """
        models_dir = self._jarvis_home / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir
    
    def get_logs_dir(self) -> Path:
        """
        Get logs directory.
        
        Returns:
            Path: Logs directory
        """
        if self.os == "Windows":
            logs_dir = self._jarvis_home / "Logs"
        elif self.os == "Linux":
            # Linux: ~/.local/state/jarvis/logs (XDG State spec)
            xdg_state = os.getenv("XDG_STATE_HOME")
            if xdg_state:
                logs_dir = Path(xdg_state) / "jarvis" / "logs"
            else:
                logs_dir = self.home / ".local" / "state" / "jarvis" / "logs"
        else:
            logs_dir = self._jarvis_home / "logs"
        
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir
    
    def get_config_dir(self) -> Path:
        """
        Get configuration directory.
        
        Returns:
            Path: Config directory
        """
        if self.os == "Linux":
            # Linux: ~/.config/jarvis (XDG Config spec)
            xdg_config = os.getenv("XDG_CONFIG_HOME")
            if xdg_config:
                config_dir = Path(xdg_config) / "jarvis"
            else:
                config_dir = self.home / ".config" / "jarvis"
        else:
            config_dir = self._jarvis_home / "config"
        
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir
    
    def get_data_dir(self) -> Path:
        """
        Get user data directory (databases, persistent state).
        
        Returns:
            Path: Data directory
        """
        data_dir = self._jarvis_home / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir
    
    def get_temp_dir(self) -> Path:
        """
        Get temporary directory for transient files.
        
        Returns:
            Path: Temp directory
        """
        if self.os == "Windows":
            temp_dir = Path(os.getenv("TEMP", "")) / "Ironcliw"
        elif self.os == "Linux":
            temp_dir = Path("/tmp") / f"jarvis-{os.getuid()}"
        else:
            temp_dir = self._jarvis_home / "tmp"
        
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir
    
    def get_trinity_state_dir(self) -> Path:
        """
        Get Trinity cross-repo state directory.
        
        Returns:
            Path: Trinity state directory
        """
        trinity_dir = self._jarvis_home / "trinity" / "state"
        trinity_dir.mkdir(parents=True, exist_ok=True)
        return trinity_dir
    
    def get_cross_repo_dir(self) -> Path:
        """
        Get cross-repo communication directory.
        
        Returns:
            Path: Cross-repo directory
        """
        cross_repo_dir = self._jarvis_home / "cross_repo"
        cross_repo_dir.mkdir(parents=True, exist_ok=True)
        return cross_repo_dir
    
    def get_signals_dir(self) -> Path:
        """
        Get signals directory for inter-component communication.
        
        Returns:
            Path: Signals directory
        """
        signals_dir = self._jarvis_home / "signals"
        signals_dir.mkdir(parents=True, exist_ok=True)
        return signals_dir
    
    def get_voice_profiles_dir(self) -> Path:
        """
        Get voice biometric profiles directory.
        
        Returns:
            Path: Voice profiles directory
        """
        profiles_dir = self._jarvis_home / "voice_profiles"
        profiles_dir.mkdir(parents=True, exist_ok=True)
        return profiles_dir
    
    def get_screenshots_dir(self) -> Path:
        """
        Get screenshots directory.
        
        Returns:
            Path: Screenshots directory
        """
        screenshots_dir = self._jarvis_home / "screenshots"
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        return screenshots_dir
    
    def get_workspace_file(self, filename: str) -> Path:
        """
        Get path to file in Ironcliw workspace.
        
        Args:
            filename: Name of file
        
        Returns:
            Path: Full path to file
        """
        return self._jarvis_home / filename
    
    def get_log_file(self, filename: str) -> Path:
        """
        Get path to log file.
        
        Args:
            filename: Log file name
        
        Returns:
            Path: Full path to log file
        """
        return self.get_logs_dir() / filename
    
    def get_model_file(self, model_name: str) -> Path:
        """
        Get path to ML model file.
        
        Args:
            model_name: Model filename
        
        Returns:
            Path: Full path to model
        """
        return self.get_models_dir() / model_name
    
    def ensure_dir(self, path: Path) -> Path:
        """
        Ensure directory exists, creating if necessary.
        
        Args:
            path: Directory path
        
        Returns:
            Path: The directory path (for chaining)
        """
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def to_dict(self) -> dict:
        """
        Export all paths as dictionary.
        
        Returns:
            dict: All managed paths
        """
        return {
            "os": self.os,
            "home": str(self.home),
            "jarvis_home": str(self.get_jarvis_home()),
            "cache": str(self.get_cache_dir()),
            "models": str(self.get_models_dir()),
            "logs": str(self.get_logs_dir()),
            "config": str(self.get_config_dir()),
            "data": str(self.get_data_dir()),
            "temp": str(self.get_temp_dir()),
            "trinity_state": str(self.get_trinity_state_dir()),
            "cross_repo": str(self.get_cross_repo_dir()),
            "signals": str(self.get_signals_dir()),
            "voice_profiles": str(self.get_voice_profiles_dir()),
            "screenshots": str(self.get_screenshots_dir()),
        }
    
    def __repr__(self) -> str:
        """String representation of path manager."""
        return (
            f"PathManager(\n"
            f"  OS: {self.os}\n"
            f"  Home: {self.get_jarvis_home()}\n"
            f"  Cache: {self.get_cache_dir()}\n"
            f"  Models: {self.get_models_dir()}\n"
            f"  Logs: {self.get_logs_dir()}\n"
            f")"
        )


# Global singleton instance
_path_manager: Optional[PathManager] = None


def get_path_manager() -> PathManager:
    """
    Get global path manager instance (singleton).
    
    Returns:
        PathManager: Cached path manager instance
    """
    global _path_manager
    if _path_manager is None:
        _path_manager = PathManager()
    return _path_manager
