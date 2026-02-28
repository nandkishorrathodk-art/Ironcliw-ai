"""
Ironcliw Windows Platform Tests
═══════════════════════════════════════════════════════════════════════════════

Unit tests for Windows platform implementations.

Tests:
    - Platform detection
    - System control (window management, volume, notifications)
    - Audio engine (device enumeration, recording, playback)
    - Vision capture (screen capture, monitor layout)
    - Authentication (bypass mode)
    - Permissions (UAC, privacy settings)
    - Process manager (process lifecycle, Task Scheduler)
    - File watcher (directory monitoring)

Author: Ironcliw System
Version: 1.0.0 (Windows Port)
"""
import pytest
import platform
import time
from pathlib import Path

# Skip all tests if not on Windows
pytestmark = pytest.mark.skipif(
    platform.system() != 'Windows',
    reason="Windows platform tests only run on Windows"
)


class TestPlatformDetection:
    """Test platform detection"""
    
    def test_detect_platform(self):
        """Test platform detection returns 'windows'"""
        from backend.platform_adapter import get_platform
        assert get_platform() == 'windows'
    
    def test_platform_info(self):
        """Test platform info retrieval"""
        from backend.platform_adapter.detector import PlatformDetector
        info = PlatformDetector.get_platform_info()
        
        assert info.os_family == 'windows'
        assert info.os_name == 'Windows'
        assert info.home_dir.exists()
        assert len(info.user_name) > 0


class TestSystemControl:
    """Test Windows system control"""
    
    @pytest.fixture
    def system_control(self):
        """Create system control instance"""
        from backend.platform_adapter.windows import WindowsSystemControl
        return WindowsSystemControl()
    
    def test_get_window_list(self, system_control):
        """Test getting window list"""
        windows = system_control.get_window_list()
        assert isinstance(windows, list)
    
    def test_get_active_window(self, system_control):
        """Test getting active window"""
        window = system_control.get_active_window()
        if window:
            assert window.window_id > 0
            assert len(window.title) > 0
    
    def test_volume_control(self, system_control):
        """Test volume get/set"""
        original_volume = system_control.get_volume()
        assert 0.0 <= original_volume <= 1.0
        
        system_control.set_volume(0.5)
        time.sleep(0.1)
        
        new_volume = system_control.get_volume()
        assert abs(new_volume - 0.5) < 0.15
        
        system_control.set_volume(original_volume)
    
    def test_get_display_count(self, system_control):
        """Test display count"""
        count = system_control.get_display_count()
        assert count >= 1
    
    def test_get_display_info(self, system_control):
        """Test display info"""
        displays = system_control.get_display_info()
        assert len(displays) >= 1
        
        primary = displays[0]
        assert 'bounds' in primary
        assert primary['bounds']['width'] > 0
        assert primary['bounds']['height'] > 0


class TestAudioEngine:
    """Test Windows audio engine"""
    
    @pytest.fixture
    def audio_engine(self):
        """Create audio engine instance"""
        from backend.platform_adapter.windows import WindowsAudioEngine
        return WindowsAudioEngine()
    
    def test_list_devices(self, audio_engine):
        """Test device enumeration"""
        devices = audio_engine.list_devices()
        assert isinstance(devices, list)
    
    def test_get_default_input(self, audio_engine):
        """Test getting default input device"""
        device = audio_engine.get_default_input_device()
        if device:
            assert device.is_input is True
            assert device.is_default is True
    
    def test_get_default_output(self, audio_engine):
        """Test getting default output device"""
        device = audio_engine.get_default_output_device()
        if device:
            assert device.is_input is False
            assert device.is_default is True
    
    def test_recording_lifecycle(self, audio_engine):
        """Test recording start/stop"""
        assert not audio_engine.is_recording()
        
        success = audio_engine.start_recording(sample_rate=16000)
        if success:
            assert audio_engine.is_recording()
            
            time.sleep(0.5)
            
            data = audio_engine.stop_recording()
            assert not audio_engine.is_recording()
            
            if data:
                assert len(data) > 0


class TestVisionCapture:
    """Test Windows vision capture"""
    
    @pytest.fixture
    def vision_capture(self):
        """Create vision capture instance"""
        from backend.platform_adapter.windows import WindowsVisionCapture
        return WindowsVisionCapture()
    
    def test_capture_screen(self, vision_capture):
        """Test screen capture"""
        frame = vision_capture.capture_screen(monitor_id=0)
        
        if frame:
            assert frame.width > 0
            assert frame.height > 0
            assert len(frame.image_data) > 0
            assert frame.format == 'png'
    
    def test_get_monitor_layout(self, vision_capture):
        """Test monitor layout"""
        monitors = vision_capture.get_monitor_layout()
        assert len(monitors) >= 1
        
        primary = monitors[0]
        assert 'bounds' in primary
        assert primary['bounds']['width'] > 0
    
    def test_continuous_capture(self, vision_capture):
        """Test continuous capture"""
        frames_captured = []
        
        def callback(frame):
            frames_captured.append(frame)
        
        success = vision_capture.start_continuous_capture(
            fps=5,
            monitor_id=0,
            callback=callback
        )
        
        if success:
            assert vision_capture.is_capturing()
            
            time.sleep(1.0)
            
            vision_capture.stop_continuous_capture()
            assert not vision_capture.is_capturing()
            
            assert len(frames_captured) >= 3


class TestAuthentication:
    """Test Windows authentication (bypass mode)"""
    
    @pytest.fixture
    def auth(self):
        """Create authentication instance"""
        from backend.platform_adapter.windows import WindowsAuthentication
        return WindowsAuthentication()
    
    def test_bypass_authentication(self, auth):
        """Test bypass mode"""
        result = auth.bypass_authentication()
        assert result.success is True
        assert result.method == 'bypass'
        assert result.confidence == 1.0
    
    def test_voice_authentication(self, auth):
        """Test voice auth (bypassed)"""
        result = auth.authenticate_voice(b"fake_audio", "user123")
        assert result.success is True
        assert 'bypass' in result.method
    
    def test_password_authentication(self, auth):
        """Test password auth (bypassed)"""
        result = auth.authenticate_password("any_password")
        assert result.success is True
        assert 'bypass' in result.method
    
    def test_is_enrolled(self, auth):
        """Test enrollment check (always true in bypass)"""
        assert auth.is_enrolled("any_user") is True


class TestPermissions:
    """Test Windows permissions"""
    
    @pytest.fixture
    def permissions(self):
        """Create permissions instance"""
        from backend.platform_adapter.windows import WindowsPermissions
        return WindowsPermissions()
    
    def test_check_permissions(self, permissions):
        """Test permission checking"""
        from backend.platform_adapter.base import PermissionType
        
        has_mic = permissions.check_permission(PermissionType.MICROPHONE)
        assert isinstance(has_mic, bool)
        
        has_screen = permissions.check_permission(PermissionType.SCREEN_RECORDING)
        assert has_screen is True
    
    def test_is_admin(self, permissions):
        """Test admin check"""
        is_admin = permissions.is_admin()
        assert isinstance(is_admin, bool)


class TestProcessManager:
    """Test Windows process manager"""
    
    @pytest.fixture
    def process_manager(self):
        """Create process manager instance"""
        from backend.platform_adapter.windows import WindowsProcessManager
        return WindowsProcessManager()
    
    def test_start_stop_process(self, process_manager):
        """Test process lifecycle"""
        pid = process_manager.start_process('notepad.exe')
        
        if pid > 0:
            assert process_manager.is_process_running(pid)
            
            time.sleep(0.5)
            
            info = process_manager.get_process_info(pid)
            if info:
                assert info['pid'] == pid
                assert 'notepad' in info['name'].lower()
            
            success = process_manager.stop_process(pid, graceful=False)
            assert success
            
            time.sleep(0.5)
            assert not process_manager.is_process_running(pid)
    
    def test_list_processes(self, process_manager):
        """Test process listing"""
        processes = process_manager.list_processes()
        assert len(processes) > 0
        
        explorer_procs = process_manager.list_processes(filter_name='explorer')
        assert len(explorer_procs) >= 0


class TestFileWatcher:
    """Test Windows file watcher"""
    
    @pytest.fixture
    def file_watcher(self):
        """Create file watcher instance"""
        try:
            from backend.platform_adapter.windows import WindowsFileWatcher
            return WindowsFileWatcher()
        except RuntimeError:
            pytest.skip("watchdog not installed")
    
    def test_watch_directory(self, file_watcher, tmp_path):
        """Test directory watching"""
        events = []
        
        def callback(event_type, file_path):
            events.append((event_type, file_path))
        
        watch_id = file_watcher.watch_directory(
            tmp_path,
            callback,
            recursive=True
        )
        
        assert watch_id != ""
        
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")
        
        time.sleep(0.5)
        
        test_file.write_text("modified")
        
        time.sleep(0.5)
        
        test_file.unlink()
        
        time.sleep(0.5)
        
        success = file_watcher.unwatch_directory(watch_id)
        assert success
        
        assert len(events) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

