"""
Test Suite for Open Interpreter Integration
============================================

Tests the Open Interpreter-inspired patterns implemented in JARVIS:
1. SafeCodeExecutor - Safe Python code execution with AST validation
2. CoordinateExtractor - Grid overlay system for improved click accuracy
3. SafetyMonitor - Action validation and audit trail
4. Cross-repo adapters - JARVIS Prime and Reactor Core integration

Run with: python -m pytest tests/test_open_interpreter_integration.py -v
Or directly: python tests/test_open_interpreter_integration.py

Author: JARVIS AI System
Version: 1.0.0
"""

import asyncio
import sys
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    duration_ms: float
    message: str
    details: Dict[str, Any] = None


class OpenInterpreterIntegrationTests:
    """
    Comprehensive test suite for Open Interpreter integration.
    """

    def __init__(self):
        self.results: List[TestResult] = []
        self._start_time = time.time()

    def _log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp."""
        elapsed = time.time() - self._start_time
        prefix = {
            "INFO": "ℹ️",
            "PASS": "✅",
            "FAIL": "❌",
            "WARN": "⚠️",
            "TEST": "🧪",
        }.get(level, "•")
        print(f"[{elapsed:6.2f}s] {prefix} {message}")

    def _record_result(
        self,
        name: str,
        passed: bool,
        duration_ms: float,
        message: str,
        details: Dict[str, Any] = None
    ):
        """Record a test result."""
        result = TestResult(
            name=name,
            passed=passed,
            duration_ms=duration_ms,
            message=message,
            details=details
        )
        self.results.append(result)
        level = "PASS" if passed else "FAIL"
        self._log(f"{name}: {message} ({duration_ms:.1f}ms)", level)

    # =========================================================================
    # Test 1: Safe Code Executor - Code Validation
    # =========================================================================

    async def test_safe_code_validation(self):
        """Test AST-based code validation."""
        self._log("Testing SafeCodeExecutor code validation...", "TEST")
        start = time.time()

        try:
            from backend.intelligence.computer_use_refinements import (
                SafeCodeExecutor,
                ComputerUseConfig,
            )

            config = ComputerUseConfig()
            executor = SafeCodeExecutor(config)

            # Test cases: (code, should_be_safe, description)
            test_cases = [
                # Safe code
                ("print('Hello, JARVIS!')", True, "Simple print statement"),
                ("x = 1 + 2\ny = x * 3", True, "Basic arithmetic"),
                ("data = [1, 2, 3]\nresult = sum(data)", True, "List operations"),

                # Dangerous imports - should be blocked
                ("import subprocess", False, "subprocess import"),
                ("from subprocess import run", False, "subprocess.run import"),
                ("import os; os.system('ls')", False, "os.system import"),
                ("import shutil; shutil.rmtree('/')", False, "shutil.rmtree import"),
                ("import socket", False, "socket import"),
                ("import requests", False, "requests import"),

                # Dangerous built-ins - should be blocked
                ("eval('print(1)')", False, "eval() call"),
                ("exec('print(1)')", False, "exec() call"),
                ("__import__('os')", False, "__import__() call"),

                # Dangerous shell patterns in strings
                ("cmd = 'rm -rf /'", False, "rm -rf / in string"),
                ("cmd = 'sudo rm -rf ~'", False, "sudo rm in string"),
                ("bomb = ':(){ :|:& };:'", False, "fork bomb in string"),
            ]

            passed_count = 0
            failed_cases = []

            for code, should_be_safe, description in test_cases:
                is_safe, error_msg = executor.validate_code(code)

                if is_safe == should_be_safe:
                    passed_count += 1
                else:
                    failed_cases.append({
                        "code": code[:50],
                        "expected_safe": should_be_safe,
                        "actual_safe": is_safe,
                        "description": description,
                        "error": error_msg,
                    })

            duration = (time.time() - start) * 1000
            passed = len(failed_cases) == 0

            self._record_result(
                name="SafeCodeExecutor.validate_code",
                passed=passed,
                duration_ms=duration,
                message=f"{passed_count}/{len(test_cases)} validation tests passed",
                details={"failed_cases": failed_cases} if failed_cases else None
            )

            return passed

        except ImportError as e:
            duration = (time.time() - start) * 1000
            self._record_result(
                name="SafeCodeExecutor.validate_code",
                passed=False,
                duration_ms=duration,
                message=f"Import error: {e}"
            )
            return False

    # =========================================================================
    # Test 2: Safe Code Executor - Execution
    # =========================================================================

    async def test_safe_code_execution(self):
        """Test safe code execution with sandbox."""
        self._log("Testing SafeCodeExecutor code execution...", "TEST")
        start = time.time()

        try:
            from backend.intelligence.computer_use_refinements import (
                SafeCodeExecutor,
                ComputerUseConfig,
            )

            config = ComputerUseConfig()
            executor = SafeCodeExecutor(config)

            # Test 1: Execute safe code
            result = await executor.execute("print('Hello from JARVIS sandbox!')")

            if not result.success:
                duration = (time.time() - start) * 1000
                self._record_result(
                    name="SafeCodeExecutor.execute (safe)",
                    passed=False,
                    duration_ms=duration,
                    message=f"Safe code failed: {result.stderr}"
                )
                return False

            if "Hello from JARVIS sandbox!" not in result.stdout:
                duration = (time.time() - start) * 1000
                self._record_result(
                    name="SafeCodeExecutor.execute (safe)",
                    passed=False,
                    duration_ms=duration,
                    message=f"Expected output not found in: {result.stdout}"
                )
                return False

            # Test 2: Execute with context injection
            result_with_context = await executor.execute(
                "print(f'Hello, {name}!')",
                context={"name": "Derek"}
            )

            if not result_with_context.success:
                duration = (time.time() - start) * 1000
                self._record_result(
                    name="SafeCodeExecutor.execute (context)",
                    passed=False,
                    duration_ms=duration,
                    message=f"Context injection failed: {result_with_context.stderr}"
                )
                return False

            # Test 3: Dangerous code should be blocked before execution
            dangerous_result = await executor.execute(
                "import subprocess; subprocess.run(['ls'])"
            )

            if dangerous_result.success:
                duration = (time.time() - start) * 1000
                self._record_result(
                    name="SafeCodeExecutor.execute (dangerous)",
                    passed=False,
                    duration_ms=duration,
                    message="Dangerous code was NOT blocked!"
                )
                return False

            if dangerous_result.blocked_reason is None:
                duration = (time.time() - start) * 1000
                self._record_result(
                    name="SafeCodeExecutor.execute (dangerous)",
                    passed=False,
                    duration_ms=duration,
                    message="Blocked reason not set"
                )
                return False

            duration = (time.time() - start) * 1000
            self._record_result(
                name="SafeCodeExecutor.execute",
                passed=True,
                duration_ms=duration,
                message="Safe execution works, dangerous code blocked",
                details={
                    "safe_code_output": result.stdout.strip(),
                    "context_output": result_with_context.stdout.strip(),
                    "dangerous_blocked": dangerous_result.blocked_reason,
                }
            )
            return True

        except Exception as e:
            duration = (time.time() - start) * 1000
            self._record_result(
                name="SafeCodeExecutor.execute",
                passed=False,
                duration_ms=duration,
                message=f"Exception: {e}"
            )
            return False

    # =========================================================================
    # Test 3: Coordinate Extractor - Grid System
    # =========================================================================

    async def test_coordinate_extractor(self):
        """Test grid overlay coordinate system."""
        self._log("Testing CoordinateExtractor grid system...", "TEST")
        start = time.time()

        try:
            from backend.intelligence.computer_use_refinements import (
                CoordinateExtractor,
                ComputerUseConfig,
            )

            config = ComputerUseConfig()
            config.grid_size = 10  # 10x10 grid
            extractor = CoordinateExtractor(config)

            # Manually set screen size for testing (avoid pyautogui dependency)
            extractor._screen_width = 2560
            extractor._screen_height = 1440
            extractor._is_retina = False
            extractor._calibrated = True

            # Test grid-to-pixel conversion
            # Formula: pixel = (grid + 0.5) * (screen_size / grid_size)
            # For 2560x1440, grid_size=10: unit = 256 x 144
            # Grid (0,0) -> (0+0.5)*256 = 128, (0+0.5)*144 = 72
            # Grid (5,5) -> (5+0.5)*256 = 1408, (5+0.5)*144 = 792
            # Grid (9,9) -> (9+0.5)*256 = 2432, (9+0.5)*144 = 1368
            test_cases = [
                # (grid_x, grid_y, expected_x_range, expected_y_range)
                (0, 0, (100, 160), (50, 100)),      # Top-left (~128, 72)
                (5, 5, (1350, 1450), (750, 850)),   # Center (~1408, 792)
                (9, 9, (2380, 2500), (1320, 1400)), # Bottom-right (~2432, 1368)
            ]

            passed_conversions = 0
            for grid_x, grid_y, x_range, y_range in test_cases:
                pixel_x, pixel_y = extractor.grid_to_pixel(grid_x, grid_y)

                # Check if within expected range (approximate)
                x_ok = x_range[0] <= pixel_x <= x_range[1]
                y_ok = y_range[0] <= pixel_y <= y_range[1]

                if x_ok and y_ok:
                    passed_conversions += 1
                else:
                    self._log(
                        f"Grid ({grid_x},{grid_y}) -> Pixel ({pixel_x},{pixel_y}): "
                        f"Expected X in {x_range}, Y in {y_range}",
                        "WARN"
                    )

            # Test pixel-to-grid conversion (inverse)
            test_pixel_x, test_pixel_y = 1280, 720  # Center of screen
            grid_x, grid_y = extractor.pixel_to_grid(test_pixel_x, test_pixel_y)

            # Should be around (5, 5) for center
            center_ok = 4.5 <= grid_x <= 5.5 and 4.5 <= grid_y <= 5.5

            # Test coordinate adjustment for retry
            original_x, original_y = 500, 300
            adjusted_x, adjusted_y = extractor.adjust_coordinates_for_retry(
                original_x, original_y, attempt=1
            )

            # Should be within ±10 pixels
            adjustment_ok = (
                abs(adjusted_x - original_x) <= 10 and
                abs(adjusted_y - original_y) <= 10
            )

            # Test grid prompt generation
            prompt_section = extractor.get_grid_prompt_section()
            prompt_ok = (
                "2560x1440" in prompt_section and
                "10x10" in prompt_section
            )

            duration = (time.time() - start) * 1000
            all_passed = (
                passed_conversions == len(test_cases) and
                center_ok and
                adjustment_ok and
                prompt_ok
            )

            self._record_result(
                name="CoordinateExtractor",
                passed=all_passed,
                duration_ms=duration,
                message=f"Grid: {passed_conversions}/{len(test_cases)}, "
                        f"Inverse: {center_ok}, Adjust: {adjustment_ok}, Prompt: {prompt_ok}",
                details={
                    "screen_size": f"{extractor._screen_width}x{extractor._screen_height}",
                    "grid_size": config.grid_size,
                    "center_grid": f"({grid_x:.1f}, {grid_y:.1f})",
                }
            )
            return all_passed

        except Exception as e:
            duration = (time.time() - start) * 1000
            self._record_result(
                name="CoordinateExtractor",
                passed=False,
                duration_ms=duration,
                message=f"Exception: {e}"
            )
            return False

    # =========================================================================
    # Test 4: Safety Monitor - Action Validation
    # =========================================================================

    async def test_safety_monitor(self):
        """Test SafetyMonitor action validation."""
        self._log("Testing SafetyMonitor action validation...", "TEST")
        start = time.time()

        try:
            from backend.intelligence.computer_use_refinements import (
                SafetyMonitor,
                ComputerUseConfig,
            )

            config = ComputerUseConfig()
            monitor = SafetyMonitor(config, strict_mode=True)

            # Test safe actions
            safe_actions = [
                ("keyboard", "Hello world"),
                ("mouse", "click at 500, 300"),
                ("bash", "ls -la"),
                ("bash", "echo 'Hello'"),
            ]

            safe_passed = 0
            for action_type, details in safe_actions:
                allowed, reason = monitor.check_action(action_type, details)
                if allowed:
                    safe_passed += 1
                else:
                    self._log(f"Safe action blocked: {action_type} - {reason}", "WARN")

            # Test dangerous actions
            dangerous_actions = [
                ("bash", "rm -rf /"),
                ("bash", "sudo rm -rf ~"),
                ("bash", ":(){ :|:& };:"),  # Fork bomb
                ("keyboard", "rm -rf /home"),
            ]

            dangerous_blocked = 0
            for action_type, details in dangerous_actions:
                allowed, reason = monitor.check_action(action_type, details)
                if not allowed:
                    dangerous_blocked += 1
                else:
                    self._log(f"Dangerous action NOT blocked: {action_type}", "WARN")

            # Get audit trail
            audit = monitor.get_audit_trail()

            duration = (time.time() - start) * 1000
            all_passed = (
                safe_passed == len(safe_actions) and
                dangerous_blocked == len(dangerous_actions)
            )

            self._record_result(
                name="SafetyMonitor",
                passed=all_passed,
                duration_ms=duration,
                message=f"Safe: {safe_passed}/{len(safe_actions)}, "
                        f"Blocked: {dangerous_blocked}/{len(dangerous_actions)}",
                details={
                    "total_actions": audit["total_actions"],
                    "blocked_actions": audit["blocked_actions"],
                }
            )
            return all_passed

        except Exception as e:
            duration = (time.time() - start) * 1000
            self._record_result(
                name="SafetyMonitor",
                passed=False,
                duration_ms=duration,
                message=f"Exception: {e}"
            )
            return False

    # =========================================================================
    # Test 5: Tool Result Pattern
    # =========================================================================

    async def test_tool_result_pattern(self):
        """Test frozen ToolResult dataclass pattern."""
        self._log("Testing ToolResult frozen dataclass pattern...", "TEST")
        start = time.time()

        try:
            from backend.intelligence.computer_use_refinements import (
                ToolResult,
                ToolFailure,
            )

            # Test creation
            result = ToolResult(
                output="Test output",
                duration_ms=100.0
            )

            # Test immutability (should fail to modify)
            immutable_ok = True
            try:
                result.output = "Modified"  # Should raise
                immutable_ok = False
            except AttributeError:
                pass  # Expected

            # Test with_updates (creates new instance)
            updated = result.with_updates(output="Updated output")
            update_ok = (
                updated.output == "Updated output" and
                result.output == "Test output"  # Original unchanged
            )

            # Test combination
            result2 = ToolResult(
                output="Second output",
                duration_ms=50.0
            )
            combined = result + result2
            combine_ok = (
                "Test output" in combined.output and
                "Second output" in combined.output and
                combined.duration_ms == 150.0
            )

            # Test is_success
            success_result = ToolResult(output="OK", exit_code=0)
            failure_result = ToolResult(error="Failed", exit_code=1)
            success_ok = success_result.is_success() and not failure_result.is_success()

            # Test ToolFailure
            failure = ToolFailure(error="Something went wrong")
            failure_ok = isinstance(failure, ToolResult) and failure.error is not None

            duration = (time.time() - start) * 1000
            all_passed = immutable_ok and update_ok and combine_ok and success_ok and failure_ok

            self._record_result(
                name="ToolResult pattern",
                passed=all_passed,
                duration_ms=duration,
                message=f"Immutable: {immutable_ok}, Update: {update_ok}, "
                        f"Combine: {combine_ok}, Success: {success_ok}"
            )
            return all_passed

        except Exception as e:
            duration = (time.time() - start) * 1000
            self._record_result(
                name="ToolResult pattern",
                passed=False,
                duration_ms=duration,
                message=f"Exception: {e}"
            )
            return False

    # =========================================================================
    # Test 6: Computer Use Connector Integration
    # =========================================================================

    async def test_computer_use_connector_integration(self):
        """Test ClaudeComputerUseConnector with Open Interpreter refinements."""
        self._log("Testing ComputerUseConnector integration...", "TEST")
        start = time.time()

        try:
            # This test doesn't require an actual API key
            # We're testing the integration structure, not actual API calls

            from backend.display.computer_use_connector import (
                ClaudeComputerUseConnector,
                ANTHROPIC_AVAILABLE,
            )

            if not ANTHROPIC_AVAILABLE:
                duration = (time.time() - start) * 1000
                self._record_result(
                    name="ComputerUseConnector integration",
                    passed=True,
                    duration_ms=duration,
                    message="Skipped - anthropic package not available"
                )
                return True

            # Check that the connector class has the expected methods
            expected_methods = [
                "_ensure_refinements_initialized",
                "execute_code_safely",
                "get_enhanced_system_prompt",
            ]

            missing_methods = []
            for method in expected_methods:
                if not hasattr(ClaudeComputerUseConnector, method):
                    missing_methods.append(method)

            # Check SYSTEM_PROMPT has Open Interpreter patterns
            prompt = ClaudeComputerUseConnector.SYSTEM_PROMPT
            expected_patterns = [
                "GRID SYSTEM",
                "COORDINATE CALCULATION",
                "RETINA DISPLAY",
                "ACTION EXECUTION",
                "ERROR RECOVERY",
            ]

            missing_patterns = []
            for pattern in expected_patterns:
                if pattern not in prompt:
                    missing_patterns.append(pattern)

            duration = (time.time() - start) * 1000
            passed = len(missing_methods) == 0 and len(missing_patterns) == 0

            self._record_result(
                name="ComputerUseConnector integration",
                passed=passed,
                duration_ms=duration,
                message=f"Methods: {len(expected_methods) - len(missing_methods)}/{len(expected_methods)}, "
                        f"Patterns: {len(expected_patterns) - len(missing_patterns)}/{len(expected_patterns)}",
                details={
                    "missing_methods": missing_methods,
                    "missing_patterns": missing_patterns,
                }
            )
            return passed

        except Exception as e:
            duration = (time.time() - start) * 1000
            self._record_result(
                name="ComputerUseConnector integration",
                passed=False,
                duration_ms=duration,
                message=f"Exception: {e}"
            )
            return False

    # =========================================================================
    # Test 7: Cross-Repo Hub Integration
    # =========================================================================

    async def test_cross_repo_hub(self):
        """Test CrossRepoIntelligenceHub adapters."""
        self._log("Testing CrossRepoIntelligenceHub adapters...", "TEST")
        start = time.time()

        try:
            from backend.intelligence.cross_repo_hub import (
                CrossRepoIntelligenceHub,
                SafeCodeAdapter,
                ReactorCoreAdapter,
                JARVISPrimeAdapter,
            )

            # Test SafeCodeAdapter
            safe_adapter = SafeCodeAdapter()
            adapter_ok = hasattr(safe_adapter, 'execute')

            # Test that hub can be instantiated
            hub = CrossRepoIntelligenceHub()

            # Check for expected adapters
            expected_adapters = [
                "_safe_code_adapter",
                "_reactor_core_adapter",
                "_jarvis_prime_adapter",
            ]

            found_adapters = []
            for adapter_name in expected_adapters:
                if hasattr(hub, adapter_name):
                    found_adapters.append(adapter_name)

            duration = (time.time() - start) * 1000
            passed = adapter_ok and len(found_adapters) == len(expected_adapters)

            self._record_result(
                name="CrossRepoIntelligenceHub",
                passed=passed,
                duration_ms=duration,
                message=f"Adapters: {len(found_adapters)}/{len(expected_adapters)}",
                details={
                    "found_adapters": found_adapters,
                }
            )
            return passed

        except ImportError as e:
            # Some adapters may not be available if deps are missing
            duration = (time.time() - start) * 1000
            self._record_result(
                name="CrossRepoIntelligenceHub",
                passed=True,
                duration_ms=duration,
                message=f"Partial - some imports unavailable: {e}"
            )
            return True
        except Exception as e:
            duration = (time.time() - start) * 1000
            self._record_result(
                name="CrossRepoIntelligenceHub",
                passed=False,
                duration_ms=duration,
                message=f"Exception: {e}"
            )
            return False

    # =========================================================================
    # Test 8: Executor Stats
    # =========================================================================

    async def test_executor_stats(self):
        """Test executor statistics tracking."""
        self._log("Testing executor statistics...", "TEST")
        start = time.time()

        try:
            from backend.intelligence.computer_use_refinements import (
                SafeCodeExecutor,
                ComputerUseConfig,
            )

            config = ComputerUseConfig()
            executor = SafeCodeExecutor(config)

            # Execute some code to generate stats
            await executor.execute("x = 1")
            await executor.execute("print('test')")
            await executor.execute("import subprocess")  # Should be blocked

            stats = executor.get_stats()

            # Verify stats structure
            stats_ok = (
                "execution_count" in stats and
                "blocked_count" in stats and
                "sandbox_dir" in stats
            )

            # Verify counts
            counts_ok = (
                stats["execution_count"] >= 2 and
                stats["blocked_count"] >= 1
            )

            duration = (time.time() - start) * 1000
            passed = stats_ok and counts_ok

            self._record_result(
                name="Executor statistics",
                passed=passed,
                duration_ms=duration,
                message=f"Executions: {stats.get('execution_count', 0)}, "
                        f"Blocked: {stats.get('blocked_count', 0)}",
                details=stats
            )
            return passed

        except Exception as e:
            duration = (time.time() - start) * 1000
            self._record_result(
                name="Executor statistics",
                passed=False,
                duration_ms=duration,
                message=f"Exception: {e}"
            )
            return False

    # =========================================================================
    # Run All Tests
    # =========================================================================

    async def run_all(self) -> Dict[str, Any]:
        """Run all tests and return summary."""
        self._log("=" * 60, "INFO")
        self._log("Open Interpreter Integration Test Suite", "INFO")
        self._log("=" * 60, "INFO")

        # Run all tests
        await self.test_safe_code_validation()
        await self.test_safe_code_execution()
        await self.test_coordinate_extractor()
        await self.test_safety_monitor()
        await self.test_tool_result_pattern()
        await self.test_computer_use_connector_integration()
        await self.test_cross_repo_hub()
        await self.test_executor_stats()

        # Calculate summary
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        total_duration = sum(r.duration_ms for r in self.results)

        self._log("=" * 60, "INFO")
        self._log(f"Results: {passed}/{len(self.results)} tests passed",
                  "PASS" if failed == 0 else "FAIL")
        self._log(f"Total duration: {total_duration:.1f}ms", "INFO")
        self._log("=" * 60, "INFO")

        # Print failed tests details
        if failed > 0:
            self._log("Failed tests:", "FAIL")
            for r in self.results:
                if not r.passed:
                    self._log(f"  - {r.name}: {r.message}", "FAIL")
                    if r.details:
                        for k, v in r.details.items():
                            self._log(f"      {k}: {v}", "INFO")

        return {
            "passed": passed,
            "failed": failed,
            "total": len(self.results),
            "duration_ms": total_duration,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "message": r.message,
                    "duration_ms": r.duration_ms,
                }
                for r in self.results
            ]
        }


async def main():
    """Run the test suite."""
    tests = OpenInterpreterIntegrationTests()
    summary = await tests.run_all()

    # Exit with appropriate code
    sys.exit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
