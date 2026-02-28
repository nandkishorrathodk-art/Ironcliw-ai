#!/usr/bin/env python3
"""
Test Enhanced Vision Capabilities - Specific UI Element Detection
Tests Ironcliw's ability to provide specific answers about battery, time, etc.
"""

import asyncio
import sys
import os
import logging

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from backend.chatbots.claude_vision_chatbot import ClaudeVisionChatbot

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_vision_queries():
    """Test various vision queries for specific UI elements"""
    
    # Initialize chatbot
    chatbot = ClaudeVisionChatbot()
    
    if not chatbot.is_available():
        logger.error("Claude API not available. Please set ANTHROPIC_API_KEY")
        return
    
    # Test queries
    test_queries = [
        "Can you see my battery percentage?",
        "What's my battery level?",
        "What time is it?",
        "Can you see the time on my screen?",
        "What do you see in my status bar?",
        "Describe my menu bar",
        "What's on my screen right now?",
        "Can you see what applications I have open?"
    ]
    
    print("\n🚀 Testing Enhanced Vision Capabilities\n")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📌 Test {i}: {query}")
        print("-" * 60)
        
        try:
            # Check if it's a vision command
            is_vision = chatbot.is_vision_command(query)
            print(f"Vision command detected: {is_vision}")
            
            if is_vision:
                print("\n🔍 Analyzing screen...")
                response = await chatbot.analyze_screen_with_vision(query)
                print(f"\n🤖 Ironcliw: {response}")
            else:
                print("❌ Not detected as vision command")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            logger.error(f"Test failed for query '{query}': {e}", exc_info=True)
        
        print("\n" + "=" * 60)
        
        # Wait a bit between tests
        if i < len(test_queries):
            await asyncio.sleep(2)
    
    print("\n✅ Vision tests completed!")


async def test_monitoring_mode():
    """Test continuous monitoring with specific queries"""
    
    chatbot = ClaudeVisionChatbot()
    
    if not chatbot.is_available():
        logger.error("Claude API not available")
        return
    
    print("\n\n🔄 Testing Monitoring Mode\n")
    print("=" * 60)
    
    # Start monitoring
    print("Starting continuous monitoring...")
    response = await chatbot.generate_response("Start monitoring my screen")
    print(f"Ironcliw: {response}")
    
    await asyncio.sleep(3)
    
    # Test specific queries while monitoring
    monitoring_queries = [
        "Can you see my battery percentage?",
        "What time is showing on my screen?",
        "What applications are open?"
    ]
    
    for query in monitoring_queries:
        print(f"\n📍 Query: {query}")
        response = await chatbot.generate_response(query)
        print(f"🤖 Ironcliw: {response}")
        await asyncio.sleep(2)
    
    # Stop monitoring
    print("\nStopping monitoring...")
    response = await chatbot.generate_response("Stop monitoring")
    print(f"Ironcliw: {response}")


async def main():
    """Run all vision tests"""
    
    print("🎯 Ironcliw Enhanced Vision Test Suite")
    print("Testing specific UI element detection capabilities")
    print("=" * 60)
    
    # Run basic vision tests
    await test_vision_queries()
    
    # Ask if user wants to test monitoring mode
    user_input = input("\n\nDo you want to test monitoring mode? (y/n): ")
    if user_input.lower() == 'y':
        await test_monitoring_mode()
    
    print("\n\n✨ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())