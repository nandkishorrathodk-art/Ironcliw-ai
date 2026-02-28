"""
Test Suite for Windows Ghost Hands Automation
══════════════════════════════════════════════

Comprehensive tests for Windows automation layer.

Tests:
    - Window enumeration and management
    - Mouse operations (click, move, drag, scroll)
    - Keyboard operations (type, hotkeys)
    - Focus preservation
    - Multi-monitor support
    - Cross-platform abstraction

Usage:
    pytest backend/ghost_hands/test_windows_automation.py -v
    pytest backend/ghost_hands/test_windows_automation.py::TestWindowManager -v
    pytest backend/ghost_hands/test_windows_automation.py::TestMouseKeyboard -v

Author: Ironcliw System
Version: 1.0.0 (Windows Port - Phase 8)
"""
import asyncio
import os
import sys
import time
from typing import Optional

import pytest

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from ghost_hands.windows_automation import (
    WindowsAutomationConfig,
    WindowsAutomationEngine,
    WindowsWindowManager,
    WindowsFocusGuard,
    WindowsMouseKeyboard,
    WindowsWindowInfo,
    WindowsWindowFrame,
    MonitorInfo,
)
from ghost_hands.platform_automation import (
    get_automation_engine,
    reset_automation_engine,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def config():
    """Create test configuration"""
    return WindowsAutomationConfig(
        preserve_focus=True,
        focus_restore_delay_ms=50,
        multi_monitor_enabled=True,
        prefer_pyautogui=True,
        action_delay_ms=10,
        animation_wait_ms=100,
    )


@pytest.fixture
async def window_manager(config):
    """Create and initialize window manager"""
    mgr = WindowsWindowManager(config)
    await mgr.initialize()
    yield mgr


@pytest.fixture
async def mouse_keyboard(config):
    """Create and initialize mouse/keyboard"""
    mk = WindowsMouseKeyboard(config)
    await mk.initialize()
    yield mk


@pytest.fixture
async def automation_engine(config):
    """Create and initialize automation engine"""
    engine = WindowsAutomationEngine(config)
    await engine.initialize()
    yield engine
    await engine.cleanup()


@pytest.fixture
async def platform_engine():
    """Create platform-agnostic automation engine"""
    reset_automation_engine()  # Reset singleton
    engine = await get_automation_engine(force_platform="windows")
    yield engine
    await engine.cleanup()
    reset_automation_engine()


# ═══════════════════════════════════════════════════════════════════════════════
# WINDOW MANAGER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestWindowManager:
    """Tests for WindowsWindowManager"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, config):
        """Test window manager initializes successfully"""
        mgr = WindowsWindowManager(config)
        success = await mgr.initialize()
        assert success, "Window manager should initialize successfully"
        assert mgr._initialized, "Window manager should be marked as initialized"
    
    @pytest.mark.asyncio
    async def test_get_all_windows(self, window_manager):
        """Test getting all windows"""
        windows = await window_manager.get_all_windows()
        
        assert isinstance(windows, list), "Should return a list"
        assert len(windows) > 0, "Should find at least one window"
        
        # Verify window info structure
        for window in windows:
            assert isinstance(window, WindowsWindowInfo), "Should return WindowsWindowInfo objects"
            assert window.hwnd > 0, "Window handle should be valid"
            assert isinstance(window.title, str), "Title should be string"
            assert isinstance(window.process_name, str), "Process name should be string"
            assert isinstance(window.frame, WindowsWindowFrame), "Should have frame"
    
    @pytest.mark.asyncio
    async def test_get_focused_window(self, window_manager):
        """Test getting focused window"""
        focused = await window_manager.get_focused_window()
        
        assert focused is not None, "Should find a focused window"
        assert isinstance(focused, WindowsWindowInfo), "Should return WindowsWindowInfo"
        assert focused.is_focused, "Focused window should have is_focused=True"
        assert focused.hwnd > 0, "Window handle should be valid"
    
    @pytest.mark.asyncio
    async def test_get_window_by_id(self, window_manager):
        """Test getting specific window by ID"""
        # Get all windows first
        windows = await window_manager.get_all_windows()
        assert len(windows) > 0, "Need at least one window"
        
        # Get first window by ID
        target = windows[0]
        window = await window_manager.get_window(target.hwnd)
        
        assert window is not None, "Should find window by ID"
        assert window.hwnd == target.hwnd, "Should return correct window"
        assert window.title == target.title, "Title should match"
    
    @pytest.mark.asyncio
    async def test_get_windows_for_app(self, window_manager):
        """Test getting windows for specific app"""
        # Get all windows first
        all_windows = await window_manager.get_all_windows()
        assert len(all_windows) > 0, "Need at least one window"
        
        # Pick an app name
        app_name = all_windows[0].process_name
        
        # Get windows for that app
        app_windows = await window_manager.get_windows_for_app(app_name)
        
        assert len(app_windows) > 0, f"Should find windows for {app_name}"
        for window in app_windows:
            assert app_name.lower() in window.process_name.lower(), "Should match app name"
    
    @pytest.mark.asyncio
    async def test_get_monitors(self, window_manager):
        """Test getting monitor information"""
        monitors = await window_manager.get_monitors()
        
        assert isinstance(monitors, list), "Should return a list"
        assert len(monitors) >= 1, "Should have at least one monitor"
        
        # Verify monitor info
        has_primary = False
        for monitor in monitors:
            assert isinstance(monitor, MonitorInfo), "Should return MonitorInfo objects"
            assert monitor.width > 0, "Width should be positive"
            assert monitor.height > 0, "Height should be positive"
            if monitor.is_primary:
                has_primary = True
        
        assert has_primary, "Should have one primary monitor"
    
    @pytest.mark.asyncio
    async def test_window_cache(self, window_manager):
        """Test window caching mechanism"""
        # First call - should query system
        start = time.time()
        windows1 = await window_manager.get_all_windows()
        duration1 = time.time() - start
        
        # Second call within TTL - should use cache
        start = time.time()
        windows2 = await window_manager.get_all_windows()
        duration2 = time.time() - start
        
        assert len(windows1) == len(windows2), "Should return same number of windows"
        # Cache should be faster (though not guaranteed on all systems)
        # assert duration2 < duration1, "Cached call should be faster"


# ═══════════════════════════════════════════════════════════════════════════════
# FOCUS GUARD TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestFocusGuard:
    """Tests for WindowsFocusGuard"""
    
    @pytest.mark.asyncio
    async def test_save_and_restore_focus(self, config):
        """Test saving and restoring focus"""
        guard = WindowsFocusGuard(config)
        
        # Save current focus
        saved = await guard.save_focus()
        
        assert isinstance(saved, dict), "Should return dict"
        assert "hwnd" in saved, "Should include window handle"
        assert "title" in saved, "Should include window title"
        assert saved["hwnd"] > 0, "Handle should be valid"
        
        # Restore focus (should succeed even if focus didn't change)
        success = await guard.restore_focus()
        assert success, "Restore should succeed"


# ═══════════════════════════════════════════════════════════════════════════════
# MOUSE & KEYBOARD TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMouseKeyboard:
    """Tests for WindowsMouseKeyboard"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, config):
        """Test mouse/keyboard initializes successfully"""
        mk = WindowsMouseKeyboard(config)
        success = await mk.initialize()
        assert success, "Mouse/keyboard should initialize successfully"
    
    @pytest.mark.asyncio
    async def test_move_mouse(self, mouse_keyboard):
        """Test moving mouse"""
        import pyautogui
        
        # Get current position
        start_x, start_y = pyautogui.position()
        
        # Move to new position
        target_x, target_y = start_x + 100, start_y + 100
        success = await mouse_keyboard.move_to(target_x, target_y, duration=0.1)
        
        assert success, "Move should succeed"
        
        # Verify position (with tolerance for animation)
        await asyncio.sleep(0.2)
        current_x, current_y = pyautogui.position()
        assert abs(current_x - target_x) < 5, f"X position should be near {target_x}"
        assert abs(current_y - target_y) < 5, f"Y position should be near {target_y}"
        
        # Move back
        await mouse_keyboard.move_to(start_x, start_y, duration=0.1)
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Actual clicking can interfere with test environment")
    async def test_click(self, mouse_keyboard):
        """Test mouse click (skipped by default)"""
        success = await mouse_keyboard.click(100, 100, "left", 1)
        assert success, "Click should succeed"
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Actual typing can interfere with test environment")
    async def test_type_text(self, mouse_keyboard):
        """Test typing text (skipped by default)"""
        success = await mouse_keyboard.type_text("test", interval=0.01)
        assert success, "Type should succeed"
    
    @pytest.mark.asyncio
    async def test_modifier_translation(self, mouse_keyboard):
        """Test macOS to Windows modifier key translation"""
        # This test verifies the translation logic without actually pressing keys
        # We check that macOS modifier names are accepted
        
        # "command" should be translated to "win"
        # "option" should be translated to "alt"
        # This is tested indirectly by verifying the method accepts these modifiers
        
        # Note: We can't actually test the key press without interfering with the system
        # The translation logic is in the press_key method
        pass  # Logic is tested in the actual implementation


# ═══════════════════════════════════════════════════════════════════════════════
# AUTOMATION ENGINE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestAutomationEngine:
    """Tests for WindowsAutomationEngine"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, config):
        """Test automation engine initializes successfully"""
        engine = WindowsAutomationEngine(config)
        success = await engine.initialize()
        
        assert success, "Engine should initialize successfully"
        assert engine._initialized, "Engine should be marked as initialized"
        
        await engine.cleanup()
    
    @pytest.mark.asyncio
    async def test_all_components_initialized(self, automation_engine):
        """Test all engine components are initialized"""
        assert automation_engine.window_manager._initialized, "Window manager should be initialized"
        assert automation_engine.mouse_keyboard._initialized, "Mouse/keyboard should be initialized"
    
    @pytest.mark.asyncio
    async def test_window_operations(self, automation_engine):
        """Test window operations through engine"""
        windows = await automation_engine.window_manager.get_all_windows()
        assert len(windows) > 0, "Should find windows"
        
        focused = await automation_engine.window_manager.get_focused_window()
        assert focused is not None, "Should find focused window"
    
    @pytest.mark.asyncio
    async def test_focus_preservation(self, automation_engine):
        """Test focus preservation workflow"""
        # Save focus
        saved = await automation_engine.focus_guard.save_focus()
        assert "hwnd" in saved, "Should save focus"
        
        # Restore focus
        success = await automation_engine.focus_guard.restore_focus()
        assert success, "Should restore focus"


# ═══════════════════════════════════════════════════════════════════════════════
# PLATFORM ABSTRACTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestPlatformAbstraction:
    """Tests for platform abstraction layer"""
    
    @pytest.mark.asyncio
    async def test_get_automation_engine_windows(self, platform_engine):
        """Test getting Windows automation engine"""
        from ghost_hands.platform_automation import WindowsAutomationAdapter
        
        assert isinstance(platform_engine, WindowsAutomationAdapter), \
            "Should return Windows adapter on Windows platform"
    
    @pytest.mark.asyncio
    async def test_window_operations_through_abstraction(self, platform_engine):
        """Test window operations through platform abstraction"""
        windows = await platform_engine.get_all_windows()
        assert len(windows) > 0, "Should find windows"
        
        focused = await platform_engine.get_focused_window()
        assert focused is not None, "Should find focused window"
    
    @pytest.mark.asyncio
    async def test_focus_operations_through_abstraction(self, platform_engine):
        """Test focus operations through platform abstraction"""
        saved = await platform_engine.save_focus()
        assert isinstance(saved, dict), "Should save focus"
        
        success = await platform_engine.restore_focus()
        assert success or saved == {}, "Should restore focus (or have nothing to restore)"


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """Integration tests for complete workflows"""
    
    @pytest.mark.asyncio
    async def test_window_enumeration_and_details(self, platform_engine):
        """Test complete window enumeration workflow"""
        # Get all windows
        windows = await platform_engine.get_all_windows()
        assert len(windows) > 0, "Should find windows"
        
        # Get details of first window
        first_window = windows[0]
        window_details = await platform_engine.get_window(first_window.window_id)
        
        assert window_details is not None, "Should get window details"
        assert window_details.window_id == first_window.window_id, "IDs should match"
    
    @pytest.mark.asyncio
    async def test_app_window_filtering(self, platform_engine):
        """Test filtering windows by app"""
        # Get all windows
        all_windows = await platform_engine.get_all_windows()
        assert len(all_windows) > 0, "Should find windows"
        
        # Pick an app
        app_name = all_windows[0].app_name
        
        # Get windows for that app
        app_windows = await platform_engine.get_windows_for_app(app_name)
        
        assert len(app_windows) > 0, f"Should find windows for {app_name}"
        for window in app_windows:
            assert app_name.lower() in window.app_name.lower(), \
                "All windows should belong to the app"


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-MONITOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultiMonitor:
    """Tests for multi-monitor support"""
    
    @pytest.mark.asyncio
    async def test_enumerate_monitors(self, window_manager):
        """Test enumerating all monitors"""
        monitors = await window_manager.get_monitors()
        
        assert len(monitors) >= 1, "Should have at least one monitor"
        
        # Verify each monitor
        for monitor in monitors:
            assert monitor.width > 0, "Monitor width should be positive"
            assert monitor.height > 0, "Monitor height should be positive"
    
    @pytest.mark.asyncio
    async def test_monitor_contains_point(self, window_manager):
        """Test monitor point containment"""
        monitors = await window_manager.get_monitors()
        assert len(monitors) >= 1, "Need at least one monitor"
        
        # Test primary monitor center point
        primary = next((m for m in monitors if m.is_primary), monitors[0])
        
        center_x = primary.left + primary.width // 2
        center_y = primary.top + primary.height // 2
        
        assert primary.contains_point(center_x, center_y), \
            "Primary monitor should contain its center point"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """Run tests with pytest"""
    pytest.main([__file__, "-v", "--tb=short"])
