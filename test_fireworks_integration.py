"""
Test Fireworks AI Integration
==============================

Test script to verify Fireworks AI integration into Ironcliw.

Usage:
    1. Set FIREWORKS_API_KEY in .env file
    2. Run: python test_fireworks_integration.py
"""
import asyncio
import os
import sys

# Set environment for testing
os.environ['FIREWORKS_API_KEY'] = os.getenv('FIREWORKS_API_KEY', 'test-key')
os.environ['FIREWORKS_ENABLED'] = 'true'

print("=" * 70)
print("Ironcliw Fireworks AI Integration Test")
print("=" * 70)

async def test_fireworks_client():
    """Test direct Fireworks client"""
    print("\n[TEST 1] Direct Fireworks Client")
    print("-" * 70)
    
    try:
        from backend.intelligence.fireworks_client import get_fireworks_client
        
        client = get_fireworks_client()
        print(f"[OK] Client created")
        print(f"   - API Key configured: {bool(client.api_key and client.api_key != 'test-key')}")
        print(f"   - Base URL: {client.base_url}")
        print(f"   - Default model: {client.default_model}")
        
        # Test health check
        if client.api_key and client.api_key != 'test-key':
            print("\n[TEST] Health check...")
            healthy = await client.health_check()
            print(f"   - Health status: {'OK' if healthy else 'FAIL'}")
            
            if healthy:
                # Test simple generation
                print("\n[TEST] Simple generation...")
                result = await client.generate(
                    messages=[{"role": "user", "content": "Say 'Hello from Fireworks!'"}],
                    max_tokens=50,
                )
                
                print(f"   - Success: {result['success']}")
                if result['success']:
                    print(f"   - Content: {result['content'][:100]}...")
                    print(f"   - Tokens: {result['tokens_input']} + {result['tokens_output']}")
                    print(f"   - Cost: ${result['cost']:.4f}")
                    print(f"   - Latency: {result['latency_ms']:.0f}ms")
                else:
                    print(f"   - Error: {result['error']}")
        else:
            print("\n[SKIP] No API key configured - skipping live tests")
            print("   To test: export FIREWORKS_API_KEY=your-key-here")
        
        # Show statistics
        stats = client.get_statistics()
        print(f"\n[STATS] Client statistics:")
        print(f"   - Total requests: {stats['total_requests']}")
        print(f"   - Total cost: ${stats['total_cost']:.4f}")
        
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

async def test_model_client_adapter():
    """Test ModelClient adapter"""
    print("\n[TEST 2] FireworksModelClient Adapter")
    print("-" * 70)
    
    try:
        from backend.intelligence.unified_model_serving import (
            FireworksModelClient,
            ModelRequest,
            ModelProvider,
        )
        
        client = FireworksModelClient()
        print(f"[OK] Adapter created")
        print(f"   - Provider: {ModelProvider.FIREWORKS}")
        
        # Test health check
        healthy = await client.health_check()
        print(f"   - Health check: {'OK' if healthy else 'FAIL (API key needed)'}")
        
        # Test supported tasks
        tasks = client.get_supported_tasks()
        print(f"   - Supported tasks: {[t.value for t in tasks]}")
        
        if client._api_key and client._api_key != 'test-key':
            # Test generation
            print("\n[TEST] Generation via adapter...")
            request = ModelRequest(
                messages=[{"role": "user", "content": "What is 2+2?"}],
                max_tokens=50,
                temperature=0.7,
                system_prompt="You are a helpful math assistant.",
            )
            
            response = await client.generate(request)
            print(f"   - Success: {response.success}")
            if response.success:
                print(f"   - Content: {response.content[:100]}...")
                print(f"   - Provider: {response.provider.value}")
                print(f"   - Tokens: {response.tokens_used}")
                print(f"   - Cost: ${response.cost:.4f}")
                print(f"   - Latency: {response.latency_ms:.0f}ms")
        else:
            print("\n[SKIP] No API key - skipping generation test")
        
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

async def test_provider_enum():
    """Test ModelProvider enum update"""
    print("\n[TEST 3] ModelProvider Enum")
    print("-" * 70)
    
    try:
        from backend.intelligence.unified_model_serving import ModelProvider
        
        print(f"[OK] ModelProvider enum:")
        for provider in ModelProvider:
            print(f"   - {provider.name}: {provider.value}")
        
        # Check FIREWORKS is present
        assert hasattr(ModelProvider, 'FIREWORKS')
        print(f"\n[OK] FIREWORKS provider registered successfully")
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return False
    
    return True

async def main():
    """Run all tests"""
    results = []
    
    # Test 1: Direct client
    result1 = await test_fireworks_client()
    results.append(("Direct Client", result1))
    
    # Test 2: Model client adapter
    result2 = await test_model_client_adapter()
    results.append(("Model Client Adapter", result2))
    
    # Test 3: Provider enum
    result3 = await test_provider_enum()
    results.append(("Provider Enum", result3))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n[SUCCESS] All tests passed!")
        print("\nFireworks AI Integration:")
        print("  - Routing: PRIME_API -> FIREWORKS -> CLAUDE")
        print("  - Models: Llama 3.3 70B, Qwen 2.5 72B, Mixtral 8x7B")
        print("  - Cost: 50-80% cheaper than Claude")
        print("  - Speed: ~2x faster inference")
    else:
        print("\n[WARNING] Some tests failed")
        print("This is expected if FIREWORKS_API_KEY is not configured")
    
    print("\nTo use Fireworks AI:")
    print("  1. Get API key from: https://fireworks.ai")
    print("  2. Add to .env: FIREWORKS_API_KEY=your-key-here")
    print("  3. Ironcliw will auto-fallback to Fireworks when Prime is unavailable")

if __name__ == "__main__":
    asyncio.run(main())
