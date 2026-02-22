"""
Phase 8 Integration Test Runner
================================

Runs all Phase 8 integration tests and generates a comprehensive report.

Created: 2026-02-23
Purpose: Windows/Linux porting - Phase 8 (Integration Testing)
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import platform

def run_test_file(test_file: Path) -> dict:
    """Run a single test file and return results."""
    print(f"\n{'='*70}")
    print(f"Running: {test_file.name}")
    print(f"{'='*70}")
    
    try:
        # Run test file directly with Python
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return {
            "file": test_file.name,
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        
    except subprocess.TimeoutExpired:
        print(f"⚠️  Test timed out after 5 minutes")
        return {
            "file": test_file.name,
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": "Test timed out",
        }
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return {
            "file": test_file.name,
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
        }


def main():
    """Run all Phase 8 integration tests."""
    print("\n" + "="*70)
    print("JARVIS PHASE 8 INTEGRATION TEST SUITE")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version}")
    
    project_root = Path(__file__).parent
    
    # Test files to run
    test_files = []
    
    # Integration tests
    integration_dir = project_root / "backend" / "tests" / "integration"
    if integration_dir.exists():
        test_files.extend([
            integration_dir / "test_trinity_cross_platform.py",
            integration_dir / "test_frontend_websocket.py",
            integration_dir / "test_computer_use_e2e.py",
            integration_dir / "test_performance_benchmarks.py",
        ])
    
    # Platform-specific tests
    tests_dir = project_root / "backend" / "tests"
    detector_platform = platform.system().lower()
    
    if detector_platform == "windows":
        test_files.append(tests_dir / "test_windows_platform.py")
    elif detector_platform == "linux":
        test_files.append(tests_dir / "test_linux_platform.py")
    
    # Run tests
    results = []
    for test_file in test_files:
        if not test_file.exists():
            print(f"\n⚠️  Test file not found: {test_file.name}")
            results.append({
                "file": test_file.name,
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": "File not found",
            })
            continue
        
        result = run_test_file(test_file)
        results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUITE SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in results if r["success"])
    failed = len(results) - passed
    
    print(f"\nTotal: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {passed/len(results)*100:.1f}%")
    
    print(f"\nDetailed Results:")
    for result in results:
        status = "PASSED" if result["success"] else "FAILED"
        print(f"  [{status}] {result['file']}")
        if not result["success"] and result["stderr"]:
            print(f"    Error: {result['stderr'][:100]}")
    
    # Exit code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
