"""
JARVIS Windows File Watcher
═══════════════════════════════════════════════════════════════════════════════

Windows file system monitoring using ReadDirectoryChangesW via watchdog.

Features:
    - Directory monitoring
    - Recursive watching
    - File pattern filtering
    - Change type detection (created, modified, deleted)

Windows File System Monitoring:
    - Uses watchdog library (cross-platform)
    - Internally uses ReadDirectoryChangesW on Windows
    - Replaces macOS FSEvents

Event Types:
    - created: New file or directory created
    - modified: File content or metadata changed
    - deleted: File or directory deleted
    - moved: File or directory moved (reported as deleted + created)

Author: JARVIS System
Version: 1.0.0 (Windows Port)
"""
from __future__ import annotations

import time
import fnmatch
from typing import Callable, List, Optional, Dict
from pathlib import Path

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
except ImportError:
    Observer = None
    FileSystemEventHandler = None
    FileSystemEvent = None
    print("Warning: watchdog not installed. File watching unavailable.")
    print("Install with: pip install watchdog")

from ..base import BaseFileWatcher


class _WatchHandler(FileSystemEventHandler):
    """Internal event handler for watchdog"""
    
    def __init__(self, callback: Callable[[str, str], None], patterns: Optional[List[str]] = None):
        super().__init__()
        self.callback = callback
        self.patterns = patterns or []
    
    def _should_process(self, path: str) -> bool:
        """Check if file matches patterns"""
        if not self.patterns:
            return True
        
        for pattern in self.patterns:
            if fnmatch.fnmatch(path, pattern):
                return True
        
        return False
    
    def on_created(self, event: FileSystemEvent):
        """File or directory created"""
        if not event.is_directory and self._should_process(event.src_path):
            try:
                self.callback('created', event.src_path)
            except Exception as e:
                print(f"Warning: File watcher callback error: {e}")
    
    def on_modified(self, event: FileSystemEvent):
        """File or directory modified"""
        if not event.is_directory and self._should_process(event.src_path):
            try:
                self.callback('modified', event.src_path)
            except Exception as e:
                print(f"Warning: File watcher callback error: {e}")
    
    def on_deleted(self, event: FileSystemEvent):
        """File or directory deleted"""
        if not event.is_directory and self._should_process(event.src_path):
            try:
                self.callback('deleted', event.src_path)
            except Exception as e:
                print(f"Warning: File watcher callback error: {e}")
    
    def on_moved(self, event: FileSystemEvent):
        """File or directory moved"""
        if not event.is_directory:
            if self._should_process(event.src_path):
                try:
                    self.callback('deleted', event.src_path)
                except Exception as e:
                    print(f"Warning: File watcher callback error: {e}")
            
            if self._should_process(event.dest_path):
                try:
                    self.callback('created', event.dest_path)
                except Exception as e:
                    print(f"Warning: File watcher callback error: {e}")


class WindowsFileWatcher(BaseFileWatcher):
    """Windows implementation of file system monitoring"""
    
    def __init__(self):
        """Initialize Windows file watcher"""
        if Observer is None:
            raise RuntimeError(
                "watchdog library is required for file watching.\n"
                "Install with: pip install watchdog"
            )
        
        self._observer = Observer()
        self._watches: Dict[str, any] = {}
        self._observer.start()
    
    def watch_directory(self, 
                       path: Path, 
                       callback: Callable[[str, str], None],
                       recursive: bool = True,
                       patterns: Optional[List[str]] = None) -> str:
        """
        Start watching a directory for changes
        Returns watch_id for later removal
        callback(event_type, file_path) where event_type is 'created', 'modified', 'deleted'
        """
        try:
            if not path.exists():
                raise FileNotFoundError(f"Directory does not exist: {path}")
            
            if not path.is_dir():
                raise ValueError(f"Path is not a directory: {path}")
            
            watch_id = f"watch_{id(path)}_{time.time()}"
            
            handler = _WatchHandler(callback, patterns)
            
            watch = self._observer.schedule(
                handler,
                str(path),
                recursive=recursive
            )
            
            self._watches[watch_id] = {
                'watch': watch,
                'path': path,
                'handler': handler,
            }
            
            return watch_id
        except Exception as e:
            print(f"Warning: Failed to watch directory {path}: {e}")
            return ""
    
    def unwatch_directory(self, watch_id: str) -> bool:
        """Stop watching a directory"""
        try:
            if watch_id not in self._watches:
                return False
            
            watch_info = self._watches[watch_id]
            self._observer.unschedule(watch_info['watch'])
            
            del self._watches[watch_id]
            
            return True
        except Exception as e:
            print(f"Warning: Failed to unwatch directory: {e}")
            return False
    
    def stop_all_watches(self) -> bool:
        """Stop all active watches"""
        try:
            self._observer.unschedule_all()
            self._watches.clear()
            return True
        except Exception as e:
            print(f"Warning: Failed to stop all watches: {e}")
            return False
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            if hasattr(self, '_observer') and self._observer:
                self._observer.stop()
                self._observer.join(timeout=2.0)
        except:
            pass
