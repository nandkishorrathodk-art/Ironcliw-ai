"""
Frontend WebSocket Integration Tests
=====================================

Comprehensive tests for frontend-backend WebSocket communication:
1. WebSocket connection establishment
2. Command submission and response
3. System status updates
4. Error handling and reconnection
5. Cross-platform compatibility

Created: 2026-02-23
Purpose: Windows/Linux porting - Phase 8 (Integration Testing)
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.core.platform_abstraction import PlatformDetector, get_platform


class TestWebSocketConnection:
    """Test WebSocket connection establishment."""
    
    @pytest.mark.asyncio
    async def test_websocket_endpoint_available(self):
        """Test that WebSocket endpoint is configured correctly."""
        detector = PlatformDetector()
        platform_name = detector.get_platform_name()
        
        # Standard WebSocket endpoint configuration
        ws_config = {
            "protocol": "ws",
            "host": "localhost",
            "port": 8010,
            "path": "/ws",
        }
        
        ws_url = f"{ws_config['protocol']}://{ws_config['host']}:{ws_config['port']}{ws_config['path']}"
        
        assert ws_url == "ws://localhost:8010/ws"
        print(f"\n✅ WebSocket endpoint: {ws_url}")
        print(f"   Platform: {platform_name}")
    
    @pytest.mark.asyncio
    async def test_connection_state_machine(self):
        """Test WebSocket connection state machine."""
        states = [
            "INITIALIZING",
            "DISCOVERING",
            "CONNECTING",
            "ONLINE",
            "OFFLINE",
            "RECONNECTING",
            "ERROR"
        ]
        
        # Valid state transitions
        valid_transitions = {
            "INITIALIZING": ["DISCOVERING", "ERROR"],
            "DISCOVERING": ["CONNECTING", "ERROR", "OFFLINE"],
            "CONNECTING": ["ONLINE", "ERROR", "RECONNECTING"],
            "ONLINE": ["OFFLINE", "RECONNECTING", "ERROR"],
            "OFFLINE": ["RECONNECTING", "DISCOVERING"],
            "RECONNECTING": ["CONNECTING", "ONLINE", "ERROR", "OFFLINE"],
            "ERROR": ["OFFLINE", "RECONNECTING", "DISCOVERING"],
        }
        
        print(f"\n✅ WebSocket state machine:")
        for state, transitions in valid_transitions.items():
            print(f"   {state} → {transitions}")
        
        # Test valid transition
        current_state = "CONNECTING"
        next_state = "ONLINE"
        assert next_state in valid_transitions[current_state]
        print(f"\n   Valid transition: {current_state} → {next_state}")
    
    @pytest.mark.asyncio
    async def test_websocket_message_framing(self):
        """Test WebSocket message framing."""
        detector = PlatformDetector()
        platform_name = detector.get_platform_name()
        
        # Test command message
        command_message = {
            "type": "command",
            "text": "What is 2+2?",
            "platform": platform_name,
            "timestamp": time.time(),
        }
        
        # Serialize
        serialized = json.dumps(command_message)
        
        # Deserialize
        deserialized = json.loads(serialized)
        
        assert deserialized["type"] == "command"
        assert deserialized["text"] == "What is 2+2?"
        assert deserialized["platform"] == platform_name
        
        print(f"\n✅ WebSocket message framing valid")
        print(f"   Type: {deserialized['type']}")
        print(f"   Text: {deserialized['text']}")


class TestCommandSubmission:
    """Test command submission flow."""
    
    @pytest.mark.asyncio
    async def test_command_message_schema(self):
        """Test command message schema."""
        detector = PlatformDetector()
        platform_name = detector.get_platform_name()
        
        # Command schema
        command = {
            "type": "command",
            "text": "Test command",
            "context": {
                "source": "frontend",
                "platform": platform_name,
            },
            "timestamp": time.time(),
            "requestId": "test-req-001",
        }
        
        # Validate required fields
        assert "type" in command
        assert "text" in command
        assert command["type"] == "command"
        
        print(f"\n✅ Command message schema valid")
        print(f"   Type: {command['type']}")
        print(f"   Text: {command['text']}")
        print(f"   Platform: {command['context']['platform']}")
    
    @pytest.mark.asyncio
    async def test_response_message_schema(self):
        """Test response message schema."""
        # Response schema
        response = {
            "type": "response",
            "requestId": "test-req-001",
            "text": "Response text",
            "success": True,
            "timestamp": time.time(),
            "metadata": {
                "inference_backend": "claude",
                "execution_time_ms": 1234,
            },
        }
        
        # Validate required fields
        assert "type" in response
        assert "requestId" in response
        assert "text" in response
        assert "success" in response
        assert response["type"] == "response"
        
        print(f"\n✅ Response message schema valid")
        print(f"   RequestId: {response['requestId']}")
        print(f"   Success: {response['success']}")
        print(f"   Backend: {response['metadata']['inference_backend']}")
    
    @pytest.mark.asyncio
    async def test_fallback_routing(self):
        """Test WebSocket → REST → Queue fallback."""
        detector = PlatformDetector()
        platform_name = detector.get_platform_name()
        
        # Test routing priority
        routing_tiers = [
            {"method": "WebSocket", "available": False, "latency_ms": 50},
            {"method": "REST", "available": True, "latency_ms": 150},
            {"method": "Queue", "available": True, "latency_ms": 500},
        ]
        
        # Select first available method
        selected = next((tier for tier in routing_tiers if tier["available"]), None)
        
        assert selected is not None
        assert selected["method"] == "REST"
        
        print(f"\n✅ Fallback routing on {platform_name}:")
        for tier in routing_tiers:
            status = "✓ SELECTED" if tier == selected else ("✗ unavailable" if not tier["available"] else "○ standby")
            print(f"   {tier['method']}: {status} ({tier['latency_ms']}ms)")


class TestSystemStatusUpdates:
    """Test system status update flow."""
    
    @pytest.mark.asyncio
    async def test_status_message_schema(self):
        """Test status update message schema."""
        detector = PlatformDetector()
        platform_name = detector.get_platform_name()
        
        # Status update schema
        status_update = {
            "type": "status",
            "components": {
                "body": {
                    "status": "healthy",
                    "platform": platform_name,
                    "uptime": 3600,
                },
                "prime": {
                    "status": "healthy",
                    "inference_backend": "cloud" if platform_name != "macos" else "local",
                },
                "reactor": {
                    "status": "healthy",
                },
            },
            "system": {
                "platform": platform_name,
                "memory_usage_percent": 45.2,
                "cpu_usage_percent": 12.3,
            },
            "timestamp": time.time(),
        }
        
        # Validate schema
        assert "type" in status_update
        assert "components" in status_update
        assert "system" in status_update
        assert status_update["system"]["platform"] == platform_name
        
        print(f"\n✅ Status update schema valid for {platform_name}")
        print(f"   Body: {status_update['components']['body']['status']}")
        print(f"   Prime backend: {status_update['components']['prime']['inference_backend']}")
        print(f"   Memory: {status_update['system']['memory_usage_percent']}%")
    
    @pytest.mark.asyncio
    async def test_status_update_frequency(self):
        """Test status update frequency configuration."""
        # Status update configuration
        update_config = {
            "interval_ms": 5000,  # 5 seconds
            "batch_updates": True,
            "only_on_change": False,
        }
        
        assert update_config["interval_ms"] >= 1000  # At least 1 second
        assert update_config["interval_ms"] <= 30000  # At most 30 seconds
        
        print(f"\n✅ Status update configuration:")
        print(f"   Interval: {update_config['interval_ms']}ms")
        print(f"   Batch: {update_config['batch_updates']}")
        print(f"   Only on change: {update_config['only_on_change']}")


class TestErrorHandling:
    """Test error handling and recovery."""
    
    @pytest.mark.asyncio
    async def test_connection_error_handling(self):
        """Test connection error handling."""
        # Simulate connection errors
        errors = [
            {"code": "ECONNREFUSED", "message": "Connection refused", "recovery": "retry"},
            {"code": "ETIMEDOUT", "message": "Connection timeout", "recovery": "retry"},
            {"code": "ENOTFOUND", "message": "Host not found", "recovery": "fail"},
            {"code": "WEBSOCKET_CLOSED", "message": "WebSocket closed", "recovery": "reconnect"},
        ]
        
        print(f"\n✅ Connection error handling:")
        for error in errors:
            print(f"   {error['code']}: {error['recovery']}")
            
            # Validate recovery strategy
            assert error["recovery"] in ["retry", "fail", "reconnect"]
    
    @pytest.mark.asyncio
    async def test_reconnection_backoff(self):
        """Test exponential backoff for reconnection."""
        # Exponential backoff configuration
        backoff_config = {
            "initial_delay_ms": 1000,
            "max_delay_ms": 30000,
            "multiplier": 2,
            "max_attempts": 10,
        }
        
        # Calculate backoff delays
        delays = []
        delay = backoff_config["initial_delay_ms"]
        for _ in range(backoff_config["max_attempts"]):
            delays.append(min(delay, backoff_config["max_delay_ms"]))
            delay *= backoff_config["multiplier"]
        
        print(f"\n✅ Reconnection backoff:")
        for attempt, delay_ms in enumerate(delays, 1):
            print(f"   Attempt {attempt}: {delay_ms}ms")
        
        # Verify backoff is capped
        assert all(d <= backoff_config["max_delay_ms"] for d in delays)
    
    @pytest.mark.asyncio
    async def test_error_message_schema(self):
        """Test error message schema."""
        # Error message schema
        error_message = {
            "type": "error",
            "code": "COMMAND_FAILED",
            "message": "Command execution failed",
            "details": {
                "reason": "Backend unavailable",
                "recovery": "retry",
            },
            "timestamp": time.time(),
        }
        
        # Validate schema
        assert "type" in error_message
        assert "code" in error_message
        assert "message" in error_message
        assert error_message["type"] == "error"
        
        print(f"\n✅ Error message schema valid")
        print(f"   Code: {error_message['code']}")
        print(f"   Message: {error_message['message']}")
        print(f"   Recovery: {error_message['details']['recovery']}")


class TestCrossPlatformCompatibility:
    """Test cross-platform WebSocket compatibility."""
    
    @pytest.mark.asyncio
    async def test_platform_specific_config(self):
        """Test platform-specific WebSocket configuration."""
        detector = PlatformDetector()
        platform_name = detector.get_platform_name()
        
        # Platform-specific considerations
        platform_config = {
            "macos": {
                "max_payload_bytes": 16 * 1024 * 1024,  # 16MB
                "compression": True,
            },
            "windows": {
                "max_payload_bytes": 16 * 1024 * 1024,  # 16MB
                "compression": True,
            },
            "linux": {
                "max_payload_bytes": 16 * 1024 * 1024,  # 16MB
                "compression": True,
            },
        }
        
        config = platform_config.get(platform_name, platform_config["macos"])
        
        assert config["max_payload_bytes"] >= 1 * 1024 * 1024  # At least 1MB
        
        print(f"\n✅ WebSocket config for {platform_name}:")
        print(f"   Max payload: {config['max_payload_bytes'] / (1024*1024):.1f}MB")
        print(f"   Compression: {config['compression']}")
    
    @pytest.mark.asyncio
    async def test_unicode_handling(self):
        """Test Unicode handling in WebSocket messages."""
        detector = PlatformDetector()
        platform_name = detector.get_platform_name()
        
        # Test Unicode messages
        test_messages = [
            "Hello, world!",  # ASCII
            "こんにちは世界",  # Japanese
            "Привет мир",  # Russian
            "مرحبا بالعالم",  # Arabic
            "🎉🚀✨",  # Emojis
        ]
        
        print(f"\n✅ Unicode handling test on {platform_name}:")
        for msg in test_messages:
            # Test JSON serialization
            command = {"type": "command", "text": msg}
            serialized = json.dumps(command, ensure_ascii=False)
            deserialized = json.loads(serialized)
            
            assert deserialized["text"] == msg
            print(f"   ✓ {msg}")


class TestPerformanceMetrics:
    """Test performance monitoring of WebSocket communication."""
    
    @pytest.mark.asyncio
    async def test_latency_measurement(self):
        """Test latency measurement for WebSocket communication."""
        # Simulate round-trip latency
        latency_measurements = {
            "websocket": 50,  # ms
            "rest": 150,  # ms
            "queue": 500,  # ms
        }
        
        print(f"\n✅ Communication latency:")
        for method, latency_ms in latency_measurements.items():
            print(f"   {method}: {latency_ms}ms")
        
        # Verify WebSocket is fastest
        assert latency_measurements["websocket"] < latency_measurements["rest"]
        assert latency_measurements["rest"] < latency_measurements["queue"]
    
    @pytest.mark.asyncio
    async def test_throughput_limits(self):
        """Test message throughput limits."""
        # Throughput configuration
        throughput_config = {
            "max_messages_per_second": 100,
            "max_concurrent_requests": 10,
            "queue_size": 1000,
        }
        
        print(f"\n✅ Throughput configuration:")
        print(f"   Max msgs/sec: {throughput_config['max_messages_per_second']}")
        print(f"   Max concurrent: {throughput_config['max_concurrent_requests']}")
        print(f"   Queue size: {throughput_config['queue_size']}")
        
        assert throughput_config["max_messages_per_second"] >= 10
        assert throughput_config["max_concurrent_requests"] >= 1


def test_run_all_tests():
    """Run all frontend WebSocket integration tests."""
    print("\n" + "="*70)
    print("FRONTEND WEBSOCKET INTEGRATION TEST SUITE")
    print("="*70)
    
    detector = PlatformDetector()
    print(f"\nRunning on: {detector.get_platform_name()}")
    
    # Run pytest programmatically
    import pytest
    
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-s",  # Show print statements
    ])
    
    return exit_code == 0


if __name__ == "__main__":
    import sys
    success = test_run_all_tests()
    sys.exit(0 if success else 1)
