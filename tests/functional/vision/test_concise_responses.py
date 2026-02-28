#!/usr/bin/env python3
"""
Test script to verify concise response improvements
"""

import asyncio
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from backend.vision.jarvis_workspace_integration import IroncliwWorkspaceIntelligence


async def test_concise_responses():
    """Test that responses are now concise and focused"""
    print("🧪 Testing Concise Response Improvements")
    print("=" * 60)
    
    workspace_intel = IroncliwWorkspaceIntelligence()
    
    # Test commands that previously generated verbose responses
    test_cases = [
        {
            "query": "What am I working on?",
            "description": "Current work query"
        },
        {
            "query": "Do I have any messages?",
            "description": "Messages check"
        },
        {
            "query": "What windows are open?",
            "description": "Window list"
        },
        {
            "query": "Are there any errors?",
            "description": "Error check"
        },
        {
            "query": "Describe my workspace",
            "description": "General workspace query"
        }
    ]
    
    print("\n📊 Response Length Analysis:")
    print("-" * 60)
    
    for test in test_cases:
        print(f"\n🎯 {test['description']}:")
        print(f"   Query: \"{test['query']}\"")
        
        try:
            response = await workspace_intel.handle_workspace_command(test['query'])
            
            # Analyze response
            word_count = len(response.split())
            char_count = len(response)
            
            print(f"   Response: {response}")
            print(f"   📏 Length: {word_count} words, {char_count} characters")
            
            # Check for verbose patterns
            verbose_patterns = [
                "Based on the information provided",
                "I can see that",
                "Looking at your workspace",
                "would you like",
                "appears to be"
            ]
            
            verbose_found = [p for p in verbose_patterns if p.lower() in response.lower()]
            
            if verbose_found:
                print(f"   ⚠️  Verbose patterns found: {verbose_found}")
            else:
                print(f"   ✅ No verbose patterns detected")
                
            # Check if response is concise (under 25 words ideal)
            if word_count <= 25:
                print(f"   ✅ Concise response ({word_count} words)")
            elif word_count <= 40:
                print(f"   ⚠️  Slightly verbose ({word_count} words)")
            else:
                print(f"   ❌ Too verbose ({word_count} words)")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Concise response test complete!")
    print("\n💡 Improvements implemented:")
    print("   - Removed verbose preambles in response parsing")
    print("   - Simplified response formatting methods")
    print("   - Reduced Claude API max_tokens to 150")
    print("   - Added concise prompt instructions")
    print("   - Focused on essential information only")


if __name__ == "__main__":
    asyncio.run(test_concise_responses())