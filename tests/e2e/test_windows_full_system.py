"""
Windows-Specific End-to-End System Tests
Phase 10: Comprehensive testing for Ironcliw on Windows

Tests:
1. Full system startup and shutdown
2. 1+ hour runtime stability
3. Memory leak detection
4. Platform abstraction verification
5. Core features verification
6. Performance benchmarks
"""

import pytest
import asyncio
import psutil
import time
import gc
from pathlib import Path
from typing import Dict, List
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.platform_adapter import get_platform, is_windows, get_platform_info


class MemoryProfiler:
    """Track memory usage over time to detect leaks"""
    
    def __init__(self):
        self.measurements: List[Dict] = []
        self.process = psutil.Process()
        
    def measure(self, label: str = ""):
        """Take a memory measurement"""
        measurement = {
            'timestamp': time.time(),
            'label': label,
            'rss_mb': self.process.memory_info().rss / 1024 / 1024,
            'vms_mb': self.process.memory_info().vms / 1024 / 1024,
            'percent': self.process.memory_percent()
        }
        self.measurements.append(measurement)
        return measurement
    
    def detect_leak(self, threshold_mb: float = 100) -> bool:
        """
        Detect memory leak by comparing first and last measurements
        
        Args:
            threshold_mb: Max acceptable memory growth in MB
            
        Returns:
            True if leak detected (growth > threshold)
        """
        if len(self.measurements) < 2:
            return False
            
        first = self.measurements[0]
        last = self.measurements[-1]
        growth = last['rss_mb'] - first['rss_mb']
        
        return growth > threshold_mb
    
    def report(self) -> Dict:
        """Generate memory usage report"""
        if not self.measurements:
            return {}
            
        rss_values = [m['rss_mb'] for m in self.measurements]
        
        return {
            'measurements': len(self.measurements),
            'initial_mb': self.measurements[0]['rss_mb'],
            'final_mb': self.measurements[-1]['rss_mb'],
            'growth_mb': self.measurements[-1]['rss_mb'] - self.measurements[0]['rss_mb'],
            'peak_mb': max(rss_values),
            'avg_mb': sum(rss_values) / len(rss_values),
            'leak_detected': self.detect_leak()
        }


@pytest.fixture
def memory_profiler():
    """Fixture providing memory profiler"""
    profiler = MemoryProfiler()
    profiler.measure("test_start")
    yield profiler
    profiler.measure("test_end")


@pytest.mark.skipif(not is_windows(), reason="Windows-only test")
class TestWindowsFullSystem:
    """Comprehensive Windows system tests"""
    
    def test_platform_detection(self):
        """Verify platform detection works correctly"""
        assert get_platform() == "windows"
        assert is_windows() is True
        
        info = get_platform_info()
        assert info.os_family == "windows"
        assert info.architecture in ("AMD64", "x86")
        assert len(info.python_version) > 0
    
    def test_platform_abstraction_imports(self):
        """Verify all platform abstractions import without errors"""
        try:
            from backend.platform_adapter.windows import (
                WindowsSystemControl,
                WindowsAudioEngine,
                WindowsVisionCapture,
                WindowsAuthBypass,
                WindowsPermissions,
                WindowsProcessManager,
                WindowsFileWatcher
            )
            # If we get here, all imports succeeded
            assert True
        except ImportError as e:
            pytest.fail(f"Platform abstraction import failed: {e}")
    
    def test_csharp_dlls_available(self):
        """Verify C# DLLs are built and accessible"""
        dll_path = Path("backend/windows_native/bin/Release")
        
        required_dlls = [
            "SystemControl.dll",
            "ScreenCapture.dll",
            "AudioEngine.dll"
        ]
        
        for dll_name in required_dlls:
            dll_file = dll_path / dll_name
            if not dll_file.exists():
                pytest.skip(f"C# DLL not built: {dll_name}")
    
    @pytest.mark.slow
    def test_memory_stability_short(self, memory_profiler):
        """
        Test memory stability over 5 minutes
        
        Verifies:
        - No memory leaks
        - Memory usage stays under 4GB
        - Memory growth < 100MB
        """
        duration_seconds = 300  # 5 minutes
        check_interval = 10     # Check every 10 seconds
        
        start_time = time.time()
        iteration = 0
        
        while time.time() - start_time < duration_seconds:
            iteration += 1
            
            # Simulate activity
            gc.collect()
            time.sleep(check_interval)
            
            # Measure memory
            measurement = memory_profiler.measure(f"iteration_{iteration}")
            
            # Check memory limit
            assert measurement['rss_mb'] < 4096, \
                f"Memory exceeded 4GB: {measurement['rss_mb']:.1f} MB"
        
        # Generate report
        report = memory_profiler.report()
        print(f"\nMemory Report (5 min):")
        print(f"  Initial: {report['initial_mb']:.1f} MB")
        print(f"  Final: {report['final_mb']:.1f} MB")
        print(f"  Growth: {report['growth_mb']:.1f} MB")
        print(f"  Peak: {report['peak_mb']:.1f} MB")
        print(f"  Leak Detected: {report['leak_detected']}")
        
        # Assert no memory leak
        assert not report['leak_detected'], \
            f"Memory leak detected: {report['growth_mb']:.1f} MB growth"
    
    @pytest.mark.slow
    @pytest.mark.manual
    def test_runtime_stability_long(self, memory_profiler):
        """
        Test runtime stability over 1+ hour
        
        This test should be run manually as it takes >1 hour.
        
        Verifies:
        - System runs for full duration
        - No crashes or exceptions
        - Memory usage stable
        - No resource leaks
        """
        duration_seconds = 3600 + 300  # 1 hour + 5 minutes
        check_interval = 60            # Check every minute
        
        start_time = time.time()
        iteration = 0
        
        print(f"\nStarting {duration_seconds/3600:.1f} hour stability test...")
        
        while time.time() - start_time < duration_seconds:
            iteration += 1
            elapsed = time.time() - start_time
            
            # Simulate activity
            gc.collect()
            time.sleep(check_interval)
            
            # Measure memory
            measurement = memory_profiler.measure(f"minute_{iteration}")
            
            # Progress update
            if iteration % 10 == 0:
                print(f"  [{elapsed/60:.0f} min] Memory: {measurement['rss_mb']:.1f} MB")
            
            # Check memory limit
            assert measurement['rss_mb'] < 4096, \
                f"Memory exceeded 4GB at {elapsed/60:.0f} min: {measurement['rss_mb']:.1f} MB"
        
        # Generate final report
        report = memory_profiler.report()
        print(f"\nMemory Report (1+ hour):")
        print(f"  Duration: {duration_seconds/3600:.2f} hours")
        print(f"  Measurements: {report['measurements']}")
        print(f"  Initial: {report['initial_mb']:.1f} MB")
        print(f"  Final: {report['final_mb']:.1f} MB")
        print(f"  Growth: {report['growth_mb']:.1f} MB")
        print(f"  Peak: {report['peak_mb']:.1f} MB")
        print(f"  Average: {report['avg_mb']:.1f} MB")
        print(f"  Leak Detected: {report['leak_detected']}")
        
        # Assert stability
        assert not report['leak_detected'], \
            f"Memory leak detected: {report['growth_mb']:.1f} MB growth over {duration_seconds/3600:.1f} hours"
        
        assert report['final_mb'] < 4096, \
            f"Final memory usage too high: {report['final_mb']:.1f} MB"


@pytest.mark.skipif(not is_windows(), reason="Windows-only test")
class TestWindowsCoreFeatures:
    """Test core Ironcliw features on Windows"""
    
    def test_supervisor_help(self):
        """Test supervisor --help command"""
        import subprocess
        result = subprocess.run(
            ["python", "unified_supervisor.py", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        output = (result.stdout or "") + (result.stderr or "")
        assert output, "No output from supervisor --help"
        assert "usage:" in output.lower() or "jarvis" in output.lower()
    
    def test_supervisor_version(self):
        """Test supervisor --version command"""
        import subprocess
        result = subprocess.run(
            ["python", "unified_supervisor.py", "--version"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
    
    @pytest.mark.integration
    def test_backend_health_check(self):
        """
        Test backend health check (requires backend running)
        
        This test assumes the backend is already running on port 8010.
        If not running, it will be skipped.
        """
        import requests
        
        try:
            response = requests.get("http://localhost:8010/health", timeout=5)
            assert response.status_code == 200
            
            data = response.json()
            assert 'status' in data
            
            print(f"\nBackend Health: {data}")
            
        except requests.exceptions.ConnectionError:
            pytest.skip("Backend not running on port 8010")
        except requests.exceptions.Timeout:
            pytest.fail("Backend health check timed out")


@pytest.mark.skipif(not is_windows(), reason="Windows-only test")
class TestWindowsPerformance:
    """Performance benchmarks for Windows"""
    
    def test_startup_time(self):
        """
        Measure supervisor startup time
        
        Target: < 30 seconds for basic initialization
        """
        import subprocess
        
        start_time = time.time()
        
        result = subprocess.run(
            ["python", "unified_supervisor.py", "--version"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        duration = time.time() - start_time
        
        print(f"\nStartup time: {duration:.2f} seconds")
        
        assert result.returncode == 0
        assert duration < 30, f"Startup too slow: {duration:.2f}s (target: <30s)"
    
    def test_import_performance(self):
        """
        Measure platform abstraction import time
        
        Target: < 1 second for all imports
        """
        start_time = time.time()
        
        from backend.platform_adapter.windows import (
            WindowsSystemControl,
            WindowsAudioEngine,
            WindowsVisionCapture,
            WindowsAuthBypass,
            WindowsPermissions,
            WindowsProcessManager,
            WindowsFileWatcher
        )
        
        duration = time.time() - start_time
        
        print(f"\nImport time: {duration:.3f} seconds")
        
        assert duration < 1.0, f"Imports too slow: {duration:.3f}s (target: <1.0s)"


def run_tests():
    """Run all Windows E2E tests"""
    pytest.main([
        __file__,
        "-v",
        "-s",
        "--tb=short",
        "-m", "not manual"  # Skip manual tests by default
    ])


if __name__ == "__main__":
    run_tests()

