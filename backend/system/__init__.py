"""
JARVIS System Module

Contains system-level utilities and integrations:
- app_library.py: v67.0 CEREBRO PROTOCOL - Dynamic App Resolution via macOS Spotlight
"""

from .app_library import AppLibrary, get_app_library, resolve_app_name, is_app_installed, is_app_running

__all__ = [
    'AppLibrary',
    'get_app_library',
    'resolve_app_name',
    'is_app_installed',
    'is_app_running',
]
