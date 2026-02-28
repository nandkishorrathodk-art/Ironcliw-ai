#!/usr/bin/env python3
"""
Test Ironcliw Vision Response
Shows the expected response when you ask "Hey Ironcliw, can you see my screen?"
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.vision.screen_capture_fallback import capture_with_intelligence


def test_vision_response():
    """Test what Ironcliw should say when asked about screen visibility"""
    
    print("🤖 Testing Ironcliw Vision Response")
    print("=" * 60)
    print()
    
    # Test 1: Basic permission check
    print("Test 1: Checking if Ironcliw can see your screen...")
    result = capture_with_intelligence(use_claude=False)
    
    if result["success"]:
        print("✅ Screen recording permission: GRANTED")
        print(f"✅ Screen resolution: {result['image'].shape[1]}x{result['image'].shape[0]}")
        print()
        
        # Expected Ironcliw response
        print("When you say: 'Hey Ironcliw, can you see my screen?'")
        print("Ironcliw responds:")
        print()
        print("🎙️ 'Yes sir, I can see your screen perfectly. I'm viewing your " +
              f"{result['image'].shape[1]}x{result['image'].shape[0]} display. ")
        
        # Check if Claude API is available
        if os.getenv("ANTHROPIC_API_KEY"):
            print("      With Claude Vision enabled, I can analyze what you're working on, ")
            print("      detect errors, find UI elements, and provide intelligent assistance.'")
            
            # Test 2: With Claude intelligence
            print()
            print("Test 2: Testing Claude-enhanced vision...")
            try:
                claude_result = capture_with_intelligence(
                    query="What applications are open and what is the user doing?",
                    use_claude=True
                )
                
                if claude_result.get("intelligence_used"):
                    print("✅ Claude Vision: ACTIVE")
                    print()
                    print("Enhanced response example:")
                    print(f"🎙️ 'Yes sir, I can see your screen perfectly. {claude_result.get('analysis', '')}'")
                else:
                    print("❌ Claude Vision not used")
                    if claude_result.get("error"):
                        print(f"   Reason: {claude_result['error']}")
            except Exception as e:
                print(f"❌ Claude Vision analysis failed: {e}")
        else:
            print("      I can capture your screen and perform basic text extraction.")
            print("      To unlock my full visual intelligence capabilities,")
            print("      consider adding an Anthropic API key for Claude Vision.'")
            print()
            print("💡 To enable Claude Vision:")
            print("   echo 'ANTHROPIC_API_KEY=your-key-here' >> backend/.env")
    else:
        print("❌ Screen recording permission: NOT GRANTED")
        print()
        print("When you say: 'Hey Ironcliw, can you see my screen?'")
        print("Ironcliw responds:")
        print()
        print("🎙️ 'I'm unable to see your screen at the moment, sir.")
        print("      Please grant me screen recording permission in")
        print("      System Preferences → Security & Privacy → Privacy → Screen Recording.")
        print("      Once granted, I'll be able to help you with visual tasks.'")
    
    print()
    print("=" * 60)
    print("✨ Summary:")
    print(f"   Permission Status: {'✅ GRANTED' if result['success'] else '❌ NOT GRANTED'}")
    print(f"   Claude Vision: {'✅ AVAILABLE' if os.getenv('ANTHROPIC_API_KEY') else '⚠️  NOT CONFIGURED'}")
    print(f"   Ready for intelligent vision: {'YES' if result['success'] and os.getenv('ANTHROPIC_API_KEY') else 'PARTIAL' if result['success'] else 'NO'}")


if __name__ == "__main__":
    test_vision_response()