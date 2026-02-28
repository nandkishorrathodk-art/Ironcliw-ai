"""
Ironcliw Windows Permissions Manager
═══════════════════════════════════════════════════════════════════════════════

Windows permission management using UAC and Windows Privacy Settings.

Features:
    - Check permission status
    - Request permissions via Settings app
    - UAC elevation detection
    - Admin privilege checking

Windows Permission Model:
    Unlike macOS TCC (Transparency, Consent, and Control), Windows uses:
    - UAC (User Account Control) for admin elevation
    - Windows Privacy Settings for app permissions
    - Registry-based permission storage

Permissions Managed:
    - Microphone (Privacy Settings)
    - Camera (Privacy Settings)
    - Screen Recording (no explicit permission on Windows)
    - Accessibility (no explicit permission on Windows)
    - Automation (no explicit permission on Windows)
    - Notifications (Windows 10+ notification settings)

Author: Ironcliw System
Version: 1.0.0 (Windows Port)
"""
from __future__ import annotations

import os
import sys
import ctypes
import subprocess
from typing import List, Optional
from pathlib import Path

from ..base import (
    BasePermissions,
    PermissionType,
)


class WindowsPermissions(BasePermissions):
    """Windows implementation of permission management"""
    
    def __init__(self):
        """Initialize Windows permissions manager"""
        self._is_admin = self._check_admin_privileges()
    
    def _check_admin_privileges(self) -> bool:
        """Check if running with administrator privileges"""
        try:
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except:
            return False
    
    def check_permission(self, permission: PermissionType) -> bool:
        """Check if permission is granted"""
        if permission == PermissionType.MICROPHONE:
            return self._check_microphone_permission()
        elif permission == PermissionType.CAMERA:
            return self._check_camera_permission()
        elif permission == PermissionType.SCREEN_RECORDING:
            return True
        elif permission == PermissionType.ACCESSIBILITY:
            return True
        elif permission == PermissionType.AUTOMATION:
            return True
        elif permission == PermissionType.NOTIFICATIONS:
            return True
        else:
            return False
    
    def _check_microphone_permission(self) -> bool:
        """Check microphone permission via registry"""
        try:
            import winreg
            
            key_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone"
            
            try:
                key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_READ)
                value, _ = winreg.QueryValueEx(key, "Value")
                winreg.CloseKey(key)
                
                return value == "Allow"
            except FileNotFoundError:
                return True
        except Exception as e:
            print(f"Warning: Failed to check microphone permission: {e}")
            return True
    
    def _check_camera_permission(self) -> bool:
        """Check camera permission via registry"""
        try:
            import winreg
            
            key_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam"
            
            try:
                key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_READ)
                value, _ = winreg.QueryValueEx(key, "Value")
                winreg.CloseKey(key)
                
                return value == "Allow"
            except FileNotFoundError:
                return True
        except Exception as e:
            print(f"Warning: Failed to check camera permission: {e}")
            return True
    
    def request_permission(self, permission: PermissionType) -> bool:
        """Request permission from user"""
        if permission == PermissionType.MICROPHONE:
            return self._request_microphone_permission()
        elif permission == PermissionType.CAMERA:
            return self._request_camera_permission()
        elif permission == PermissionType.SCREEN_RECORDING:
            print("Screen recording does not require explicit permission on Windows")
            return True
        elif permission == PermissionType.ACCESSIBILITY:
            print("Accessibility features do not require explicit permission on Windows")
            return True
        elif permission == PermissionType.AUTOMATION:
            print("Automation does not require explicit permission on Windows")
            return True
        elif permission == PermissionType.NOTIFICATIONS:
            return self._request_notification_permission()
        else:
            return False
    
    def _request_microphone_permission(self) -> bool:
        """Open Windows Privacy Settings for microphone"""
        try:
            subprocess.Popen([
                'start',
                'ms-settings:privacy-microphone'
            ], shell=True)
            
            print("\nPlease grant microphone permission in Windows Settings")
            print("Settings > Privacy > Microphone")
            print("Enable: 'Allow apps to access your microphone'")
            
            return True
        except Exception as e:
            print(f"Warning: Failed to open microphone settings: {e}")
            return False
    
    def _request_camera_permission(self) -> bool:
        """Open Windows Privacy Settings for camera"""
        try:
            subprocess.Popen([
                'start',
                'ms-settings:privacy-webcam'
            ], shell=True)
            
            print("\nPlease grant camera permission in Windows Settings")
            print("Settings > Privacy > Camera")
            print("Enable: 'Allow apps to access your camera'")
            
            return True
        except Exception as e:
            print(f"Warning: Failed to open camera settings: {e}")
            return False
    
    def _request_notification_permission(self) -> bool:
        """Open Windows notification settings"""
        try:
            subprocess.Popen([
                'start',
                'ms-settings:notifications'
            ], shell=True)
            
            print("\nPlease grant notification permission in Windows Settings")
            print("Settings > System > Notifications")
            
            return True
        except Exception as e:
            print(f"Warning: Failed to open notification settings: {e}")
            return False
    
    def has_all_required_permissions(self) -> bool:
        """Check if all required permissions are granted"""
        required = [
            PermissionType.MICROPHONE,
            PermissionType.SCREEN_RECORDING,
        ]
        
        for perm in required:
            if not self.check_permission(perm):
                return False
        
        return True
    
    def get_missing_permissions(self) -> List[PermissionType]:
        """Get list of missing permissions"""
        required = [
            PermissionType.MICROPHONE,
            PermissionType.CAMERA,
            PermissionType.SCREEN_RECORDING,
            PermissionType.NOTIFICATIONS,
        ]
        
        missing = []
        for perm in required:
            if not self.check_permission(perm):
                missing.append(perm)
        
        return missing
    
    def open_permission_settings(self, permission: Optional[PermissionType] = None) -> bool:
        """Open system permission settings"""
        if permission:
            return self.request_permission(permission)
        else:
            try:
                subprocess.Popen([
                    'start',
                    'ms-settings:privacy'
                ], shell=True)
                return True
            except Exception as e:
                print(f"Warning: Failed to open privacy settings: {e}")
                return False
    
    def request_uac_elevation(self) -> bool:
        """Request UAC elevation for admin privileges"""
        if self._is_admin:
            return True
        
        try:
            script = sys.argv[0]
            params = ' '.join(sys.argv[1:])
            
            ctypes.windll.shell32.ShellExecuteW(
                None,
                "runas",
                sys.executable,
                f'"{script}" {params}',
                None,
                1
            )
            
            return True
        except Exception as e:
            print(f"Warning: Failed to request UAC elevation: {e}")
            return False
    
    def is_admin(self) -> bool:
        """Check if running with admin privileges"""
        return self._is_admin
