"""
Unit tests for Platform Abstraction Layer (PAL)

Tests all three PAL modules:
- platform_abstraction.py (platform detection)
- system_commands.py (system command abstraction)
- platform_display.py (display abstraction)

Created: 2026-02-22
Purpose: Windows/Linux porting - Phase 1 (PAL) verification
"""

import unittest
import platform
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.core.platform_abstraction import (
    PlatformDetector,
    SupportedPlatform,
    get_platform,
    is_macos,
    is_windows,
    is_linux,
    is_unix_like,
    is_supported,
)

from backend.core.system_commands import (
    SystemCommandFactory,
    MacOSCommands,
    WindowsCommands,
    LinuxCommands,
    get_system_commands,
)

from backend.display.platform_display import (
    DisplayFactory,
    MacOSDisplay,
    WindowsDisplay,
    LinuxDisplay,
    DisplayInfo,
    get_display_interface,
)


class TestPlatformDetector(unittest.TestCase):
    """Test platform detection functionality."""
    
    def setUp(self):
        """Reset singleton before each test."""
        PlatformDetector._instance = None
        PlatformDetector._platform = None
        PlatformDetector._platform_info = None
    
    def test_singleton_pattern(self):
        """Test that PlatformDetector is a singleton."""
        detector1 = PlatformDetector()
        detector2 = PlatformDetector()
        self.assertIs(detector1, detector2)
    
    def test_platform_detection(self):
        """Test that platform is detected correctly."""
        detector = PlatformDetector()
        platform_result = detector.get_platform()
        self.assertIsInstance(platform_result, SupportedPlatform)
        self.assertIn(platform_result, [
            SupportedPlatform.MACOS,
            SupportedPlatform.WINDOWS,
            SupportedPlatform.LINUX,
            SupportedPlatform.UNKNOWN,
        ])
    
    def test_platform_name(self):
        """Test platform name string."""
        detector = PlatformDetector()
        name = detector.get_platform_name()
        self.assertIsInstance(name, str)
        self.assertIn(name, ["macos", "windows", "linux", "unknown"])
    
    def test_platform_info(self):
        """Test platform info dictionary."""
        detector = PlatformDetector()
        info = detector.get_platform_info()
        self.assertIsInstance(info, dict)
        self.assertIn("system", info)
        self.assertIn("release", info)
        self.assertIn("architecture", info)
    
    def test_is_supported(self):
        """Test platform support check."""
        detector = PlatformDetector()
        # Current platform should be supported
        self.assertTrue(detector.is_supported())
    
    def test_platform_query_methods(self):
        """Test platform query methods return booleans."""
        detector = PlatformDetector()
        self.assertIsInstance(detector.is_macos(), bool)
        self.assertIsInstance(detector.is_windows(), bool)
        self.assertIsInstance(detector.is_linux(), bool)
        self.assertIsInstance(detector.is_unix_like(), bool)
    
    def test_exactly_one_platform_true(self):
        """Test that exactly one platform detection method returns True."""
        detector = PlatformDetector()
        platforms = [
            detector.is_macos(),
            detector.is_windows(),
            detector.is_linux(),
        ]
        # Exactly one should be True (unless unknown platform)
        if detector.is_supported():
            self.assertEqual(sum(platforms), 1)
    
    def test_unix_like_platforms(self):
        """Test Unix-like detection."""
        detector = PlatformDetector()
        if detector.is_macos() or detector.is_linux():
            self.assertTrue(detector.is_unix_like())
        else:
            self.assertFalse(detector.is_unix_like())
    
    def test_config_dir_path(self):
        """Test configuration directory path."""
        detector = PlatformDetector()
        config_dir = detector.get_config_dir()
        self.assertIsInstance(config_dir, str)
        self.assertTrue(len(config_dir) > 0)
        # Should contain 'jarvis' or 'Ironcliw'
        self.assertTrue('jarvis' in config_dir.lower())
    
    def test_log_dir_path(self):
        """Test log directory path."""
        detector = PlatformDetector()
        log_dir = detector.get_log_dir()
        self.assertIsInstance(log_dir, str)
        self.assertTrue(len(log_dir) > 0)
        # Should contain 'logs'
        self.assertTrue('logs' in log_dir.lower())
    
    def test_data_dir_path(self):
        """Test data directory path."""
        detector = PlatformDetector()
        data_dir = detector.get_data_dir()
        self.assertIsInstance(data_dir, str)
        self.assertTrue(len(data_dir) > 0)
    
    def test_cache_dir_path(self):
        """Test cache directory path."""
        detector = PlatformDetector()
        cache_dir = detector.get_cache_dir()
        self.assertIsInstance(cache_dir, str)
        self.assertTrue(len(cache_dir) > 0)
    
    def test_convenience_functions(self):
        """Test module-level convenience functions."""
        self.assertIsInstance(get_platform(), SupportedPlatform)
        self.assertIsInstance(is_macos(), bool)
        self.assertIsInstance(is_windows(), bool)
        self.assertIsInstance(is_linux(), bool)
        self.assertIsInstance(is_unix_like(), bool)
        self.assertIsInstance(is_supported(), bool)
    
    def test_repr_and_str(self):
        """Test string representations."""
        detector = PlatformDetector()
        repr_str = repr(detector)
        str_str = str(detector)
        self.assertIsInstance(repr_str, str)
        self.assertIsInstance(str_str, str)
        self.assertTrue(len(repr_str) > 0)
        self.assertTrue(len(str_str) > 0)


class TestSystemCommands(unittest.TestCase):
    """Test system command abstraction."""
    
    def setUp(self):
        """Reset singleton before each test."""
        SystemCommandFactory._instance = None
    
    def test_factory_singleton(self):
        """Test that factory returns singleton."""
        cmd1 = SystemCommandFactory.get_instance()
        cmd2 = SystemCommandFactory.get_instance()
        self.assertIs(cmd1, cmd2)
    
    def test_correct_platform_implementation(self):
        """Test that correct platform implementation is returned."""
        detector = PlatformDetector()
        commands = SystemCommandFactory.get_instance()
        
        if detector.is_macos():
            self.assertIsInstance(commands, MacOSCommands)
        elif detector.is_windows():
            self.assertIsInstance(commands, WindowsCommands)
        elif detector.is_linux():
            self.assertIsInstance(commands, LinuxCommands)
    
    def test_get_shell_executable(self):
        """Test shell executable retrieval."""
        commands = get_system_commands()
        shell = commands.get_shell_executable()
        self.assertIsInstance(shell, str)
        self.assertTrue(len(shell) > 0)
    
    def test_get_open_command(self):
        """Test open command retrieval."""
        commands = get_system_commands()
        open_cmd = commands.get_open_command()
        self.assertIsInstance(open_cmd, str)
        self.assertTrue(len(open_cmd) > 0)
    
    def test_process_commands(self):
        """Test process list and kill commands."""
        commands = get_system_commands()
        
        ps_cmd = commands.get_process_list_command()
        self.assertIsInstance(ps_cmd, list)
        self.assertTrue(len(ps_cmd) > 0)
        
        kill_cmd = commands.get_kill_process_command(12345)
        self.assertIsInstance(kill_cmd, list)
        self.assertTrue(len(kill_cmd) > 0)
    
    def test_environment_variables(self):
        """Test environment variable operations."""
        commands = get_system_commands()
        
        # Set a test variable
        test_var = "Ironcliw_TEST_VAR_12345"
        test_value = "test_value"
        
        result = commands.set_environment_variable(test_var, test_value)
        self.assertTrue(result)
        
        # Get the variable
        retrieved = commands.get_environment_variable(test_var)
        self.assertEqual(retrieved, test_value)
        
        # Clean up
        import os
        if test_var in os.environ:
            del os.environ[test_var]
    
    def test_execute_command_simple(self):
        """Test simple command execution."""
        commands = get_system_commands()
        detector = PlatformDetector()
        
        if detector.is_windows():
            # Windows: echo command
            returncode, stdout, stderr = commands.execute_command(
                ["cmd", "/c", "echo", "test"],
                timeout=5,
            )
        else:
            # Unix-like: echo command
            returncode, stdout, stderr = commands.execute_command(
                ["echo", "test"],
                timeout=5,
            )
        
        self.assertEqual(returncode, 0)
        self.assertIn("test", stdout.lower())
    
    def test_convenience_function(self):
        """Test convenience function."""
        commands = get_system_commands()
        self.assertIsNotNone(commands)


class TestDisplayAbstraction(unittest.TestCase):
    """Test display abstraction functionality."""
    
    def setUp(self):
        """Reset singleton before each test."""
        DisplayFactory._instance = None
    
    def test_factory_singleton(self):
        """Test that factory returns singleton."""
        display1 = DisplayFactory.get_instance()
        display2 = DisplayFactory.get_instance()
        self.assertIs(display1, display2)
    
    def test_correct_platform_implementation(self):
        """Test that correct platform implementation is returned."""
        detector = PlatformDetector()
        display = DisplayFactory.get_instance()
        
        if detector.is_macos():
            self.assertIsInstance(display, MacOSDisplay)
        elif detector.is_windows():
            self.assertIsInstance(display, WindowsDisplay)
        elif detector.is_linux():
            self.assertIsInstance(display, LinuxDisplay)
    
    def test_get_displays(self):
        """Test display enumeration."""
        display = get_display_interface()
        displays = display.get_displays()
        
        self.assertIsInstance(displays, list)
        self.assertTrue(len(displays) > 0)
        
        # Check first display structure
        first_display = displays[0]
        self.assertIsInstance(first_display, DisplayInfo)
        self.assertIsInstance(first_display.id, str)
        self.assertIsInstance(first_display.name, str)
        self.assertIsInstance(first_display.width, int)
        self.assertIsInstance(first_display.height, int)
        self.assertIsInstance(first_display.is_primary, bool)
    
    def test_get_primary_display(self):
        """Test primary display retrieval."""
        display = get_display_interface()
        primary = display.get_primary_display()
        
        self.assertIsNotNone(primary)
        self.assertIsInstance(primary, DisplayInfo)
        self.assertTrue(primary.is_primary)
    
    def test_get_total_screen_size(self):
        """Test total screen size calculation."""
        display = get_display_interface()
        width, height = display.get_total_screen_size()
        
        self.assertIsInstance(width, int)
        self.assertIsInstance(height, int)
        self.assertTrue(width > 0)
        self.assertTrue(height > 0)
    
    def test_multi_monitor_support(self):
        """Test multi-monitor support flag."""
        display = get_display_interface()
        supports = display.supports_multi_monitor()
        self.assertIsInstance(supports, bool)
    
    def test_virtual_desktop_support(self):
        """Test virtual desktop support flag."""
        display = get_display_interface()
        supports = display.supports_virtual_desktops()
        self.assertIsInstance(supports, bool)
    
    def test_get_virtual_desktops(self):
        """Test virtual desktop enumeration."""
        display = get_display_interface()
        desktops = display.get_virtual_desktops()
        self.assertIsInstance(desktops, list)
        # May be empty if not implemented yet
    
    def test_convenience_function(self):
        """Test convenience function."""
        display = get_display_interface()
        self.assertIsNotNone(display)


class TestCrossPlatformCompatibility(unittest.TestCase):
    """Test cross-platform compatibility and edge cases."""
    
    def test_all_modules_import(self):
        """Test that all PAL modules import successfully."""
        try:
            from backend.core import platform_abstraction
            from backend.core import system_commands
            from backend.display import platform_display
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import PAL module: {e}")
    
    def test_no_hardcoded_paths(self):
        """Test that no hardcoded platform-specific paths leak."""
        detector = PlatformDetector()
        
        config_dir = detector.get_config_dir()
        log_dir = detector.get_log_dir()
        data_dir = detector.get_data_dir()
        cache_dir = detector.get_cache_dir()
        
        # On Windows, paths should not contain Unix-style home
        if detector.is_windows():
            self.assertNotIn("~/.jarvis", config_dir)
            self.assertNotIn("~/.jarvis", log_dir)
        
        # On Unix, paths should not contain Windows-style AppData
        if detector.is_unix_like():
            self.assertNotIn("APPDATA", config_dir)
            self.assertNotIn("LOCALAPPDATA", cache_dir)
    
    def test_platform_consistency(self):
        """Test that all modules report the same platform."""
        from backend.core.platform_abstraction import PlatformDetector
        
        # Reset all singletons
        PlatformDetector._instance = None
        PlatformDetector._platform = None
        SystemCommandFactory._instance = None
        DisplayFactory._instance = None
        
        detector = PlatformDetector()
        detected_platform = detector.get_platform()
        
        # All modules should agree on platform
        commands = SystemCommandFactory.get_instance()
        display = DisplayFactory.get_instance()
        
        # Verify correct implementations are chosen
        if detected_platform == SupportedPlatform.MACOS:
            self.assertIsInstance(commands, MacOSCommands)
            self.assertIsInstance(display, MacOSDisplay)
        elif detected_platform == SupportedPlatform.WINDOWS:
            self.assertIsInstance(commands, WindowsCommands)
            self.assertIsInstance(display, WindowsDisplay)
        elif detected_platform == SupportedPlatform.LINUX:
            self.assertIsInstance(commands, LinuxCommands)
            self.assertIsInstance(display, LinuxDisplay)


def run_tests():
    """Run all PAL tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPlatformDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestSystemCommands))
    suite.addTests(loader.loadTestsFromTestCase(TestDisplayAbstraction))
    suite.addTests(loader.loadTestsFromTestCase(TestCrossPlatformCompatibility))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success/failure
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
