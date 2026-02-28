"""
Ironcliw Windows Performance Benchmark Suite
Phase 10: Comprehensive performance testing

Benchmarks:
1. System startup time
2. Memory usage baseline and peak
3. Platform abstraction overhead
4. Import performance
5. API response times
6. Resource cleanup efficiency

Target Performance (from spec.md section 7.3):
- Startup time: < 30 seconds
- Memory usage: < 4GB sustained
- API latency: < 100ms (95th percentile)
- Memory growth: < 100MB per hour
"""

import time
import sys
import gc
import statistics
from pathlib import Path
from typing import Dict, List
import psutil
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.platform_adapter import get_platform, get_platform_info


class BenchmarkRunner:
    """Runs and reports performance benchmarks"""
    
    def __init__(self):
        self.results: Dict[str, any] = {}
        self.process = psutil.Process()
        
    def benchmark_imports(self, iterations: int = 10) -> Dict:
        """
        Benchmark platform abstraction import performance
        
        Measures:
        - Cold import time
        - Warm import time (from cache)
        - Memory impact
        """
        print("\n" + "="*60)
        print("BENCHMARK: Platform Imports")
        print("="*60)
        
        # Cold import (first time)
        gc.collect()
        mem_before = self.process.memory_info().rss / 1024 / 1024
        
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
        cold_time = time.time() - start_time
        
        mem_after = self.process.memory_info().rss / 1024 / 1024
        mem_impact = mem_after - mem_before
        
        # Warm imports (cached)
        warm_times = []
        for _ in range(iterations):
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
            warm_times.append(time.time() - start_time)
        
        avg_warm = statistics.mean(warm_times)
        
        result = {
            'cold_import_ms': cold_time * 1000,
            'warm_import_ms': avg_warm * 1000,
            'memory_impact_mb': mem_impact,
            'target_ms': 1000,  # < 1 second target
            'passes': cold_time < 1.0
        }
        
        print(f"  Cold Import:   {result['cold_import_ms']:.1f} ms")
        print(f"  Warm Import:   {result['warm_import_ms']:.3f} ms (avg of {iterations})")
        print(f"  Memory Impact: {result['memory_impact_mb']:.1f} MB")
        print(f"  Target:        < {result['target_ms']} ms")
        print(f"  Status:        {'✓ PASS' if result['passes'] else '✗ FAIL'}")
        
        return result
    
    def benchmark_platform_detection(self, iterations: int = 1000) -> Dict:
        """
        Benchmark platform detection overhead
        
        Measures how fast we can detect the current platform.
        This is called frequently so it needs to be fast.
        """
        print("\n" + "="*60)
        print("BENCHMARK: Platform Detection")
        print("="*60)
        
        from backend.platform_adapter import get_platform, is_windows
        
        # Time get_platform()
        times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            platform = get_platform()
            times.append(time.perf_counter() - start_time)
        
        avg_time_us = statistics.mean(times) * 1_000_000  # microseconds
        p95_time_us = statistics.quantiles(times, n=20)[18] * 1_000_000
        
        result = {
            'avg_time_us': avg_time_us,
            'p95_time_us': p95_time_us,
            'iterations': iterations,
            'target_us': 10,  # < 10 microseconds target
            'passes': avg_time_us < 10
        }
        
        print(f"  Iterations:    {iterations}")
        print(f"  Avg Time:      {avg_time_us:.2f} μs")
        print(f"  P95 Time:      {p95_time_us:.2f} μs")
        print(f"  Target:        < {result['target_us']} μs")
        print(f"  Status:        {'✓ PASS' if result['passes'] else '✗ FAIL'}")
        
        return result
    
    def benchmark_memory_baseline(self) -> Dict:
        """
        Measure baseline memory usage
        
        Records:
        - RSS (Resident Set Size)
        - VMS (Virtual Memory Size)
        - Memory percentage
        """
        print("\n" + "="*60)
        print("BENCHMARK: Memory Baseline")
        print("="*60)
        
        gc.collect()  # Force garbage collection first
        time.sleep(0.1)
        
        mem_info = self.process.memory_info()
        
        result = {
            'rss_mb': mem_info.rss / 1024 / 1024,
            'vms_mb': mem_info.vms / 1024 / 1024,
            'percent': self.process.memory_percent(),
            'target_mb': 4096,  # < 4GB target
            'passes': (mem_info.rss / 1024 / 1024) < 4096
        }
        
        print(f"  RSS:           {result['rss_mb']:.1f} MB")
        print(f"  VMS:           {result['vms_mb']:.1f} MB")
        print(f"  Percent:       {result['percent']:.1f}%")
        print(f"  Target:        < {result['target_mb']} MB")
        print(f"  Status:        {'✓ PASS' if result['passes'] else '✗ FAIL'}")
        
        return result
    
    def benchmark_supervisor_startup(self) -> Dict:
        """
        Measure supervisor startup time
        
        Runs `python unified_supervisor.py --version` and times it.
        This gives us a baseline for how long initialization takes.
        """
        print("\n" + "="*60)
        print("BENCHMARK: Supervisor Startup")
        print("="*60)
        
        import subprocess
        
        # Warm up (first run may include Python startup overhead)
        subprocess.run(
            ["python", "unified_supervisor.py", "--version"],
            capture_output=True,
            timeout=60
        )
        
        # Actual benchmark
        start_time = time.time()
        result_process = subprocess.run(
            ["python", "unified_supervisor.py", "--version"],
            capture_output=True,
            text=True,
            timeout=60
        )
        duration = time.time() - start_time
        
        success = result_process.returncode == 0
        
        result = {
            'duration_seconds': duration,
            'return_code': result_process.returncode,
            'target_seconds': 30,  # < 30 seconds target
            'passes': success and duration < 30
        }
        
        print(f"  Duration:      {result['duration_seconds']:.2f} seconds")
        print(f"  Return Code:   {result['return_code']}")
        print(f"  Target:        < {result['target_seconds']} seconds")
        print(f"  Status:        {'✓ PASS' if result['passes'] else '✗ FAIL'}")
        
        return result
    
    def run_all_benchmarks(self) -> Dict:
        """Run all benchmarks and return results"""
        print("\n" + "="*80)
        print(" Ironcliw Windows Performance Benchmark Suite".center(80))
        print("="*80)
        
        # System info
        platform_info = get_platform_info()
        cpu_count = psutil.cpu_count(logical=True)
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        print(f"\nPlatform: {platform_info.os_family}")
        print(f"Architecture: {platform_info.architecture}")
        print(f"Python: {platform_info.python_version}")
        print(f"CPU Cores: {cpu_count}")
        print(f"Total RAM: {total_memory_gb:.1f} GB")
        
        # Run benchmarks
        results = {
            'platform_info': {
                'platform': platform_info.os_family,
                'architecture': platform_info.architecture,
                'python_version': platform_info.python_version,
                'cpu_count': cpu_count,
                'total_memory_gb': total_memory_gb
            },
            'benchmarks': {}
        }
        
        try:
            results['benchmarks']['imports'] = self.benchmark_imports()
        except Exception as e:
            print(f"  Import benchmark failed: {e}")
            results['benchmarks']['imports'] = {'error': str(e)}
        
        try:
            results['benchmarks']['platform_detection'] = self.benchmark_platform_detection()
        except Exception as e:
            print(f"  Platform detection benchmark failed: {e}")
            results['benchmarks']['platform_detection'] = {'error': str(e)}
        
        try:
            results['benchmarks']['memory_baseline'] = self.benchmark_memory_baseline()
        except Exception as e:
            print(f"  Memory baseline failed: {e}")
            results['benchmarks']['memory_baseline'] = {'error': str(e)}
        
        try:
            results['benchmarks']['supervisor_startup'] = self.benchmark_supervisor_startup()
        except Exception as e:
            print(f"  Supervisor startup benchmark failed: {e}")
            results['benchmarks']['supervisor_startup'] = {'error': str(e)}
        
        # Summary
        print("\n" + "="*80)
        print(" SUMMARY".center(80))
        print("="*80)
        
        passed = sum(1 for b in results['benchmarks'].values() 
                    if isinstance(b, dict) and b.get('passes', False))
        total = len([b for b in results['benchmarks'].values() 
                    if isinstance(b, dict) and 'passes' in b])
        
        print(f"\nBenchmarks Passed: {passed}/{total}")
        
        for name, result in results['benchmarks'].items():
            if isinstance(result, dict) and 'passes' in result:
                status = "✓ PASS" if result['passes'] else "✗ FAIL"
                print(f"  {name:30s} {status}")
        
        print("\n" + "="*80)
        
        return results
    
    def save_results(self, results: Dict, filename: str = "benchmark_results.json"):
        """Save benchmark results to JSON file"""
        output_path = Path(__file__).parent.parent / "tests" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


def main():
    """Run benchmarks and save results"""
    runner = BenchmarkRunner()
    results = runner.run_all_benchmarks()
    runner.save_results(results)
    
    # Exit with error code if any benchmarks failed
    passed = sum(1 for b in results['benchmarks'].values() 
                if isinstance(b, dict) and b.get('passes', False))
    total = len([b for b in results['benchmarks'].values() 
                if isinstance(b, dict) and 'passes' in b])
    
    if passed < total:
        sys.exit(1)  # Some benchmarks failed
    else:
        sys.exit(0)  # All passed


if __name__ == "__main__":
    main()

