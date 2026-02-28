#!/usr/bin/env python3
"""Test Ironcliw real-time responses"""

import asyncio
import sys
sys.path.append('.')

from voice.jarvis_agent_voice import IroncliwAgentVoice

async def test_jarvis():
    print("🤖 Testing Ironcliw Real-Time Responses\n")
    
    jarvis = IroncliwAgentVoice()
    
    # Test queries - including wake word
    queries = [
        "jarvis what time is it",
        "jarvis what's the date today", 
        "jarvis what's the weather for today",
        "jarvis what is the weather in Toronto"
    ]
    
    for query in queries:
        print(f"\n📢 You: {query}")
        
        # Process the query through Ironcliw
        response = await jarvis.process_voice_input(query)
        
        print(f"🎙️ Ironcliw: {response}")
    
    print("\n✅ Test complete!")

if __name__ == "__main__":
    asyncio.run(test_jarvis())