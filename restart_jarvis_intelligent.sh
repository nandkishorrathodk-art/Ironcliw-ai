#!/bin/bash
# Restart Ironcliw with Full Intelligent Vision
# This ensures Ironcliw can understand "what am I working on?" queries

echo "🧠 Restarting Ironcliw with Full Screen Comprehension..."
echo "============================================"

# Kill existing processes
echo "📍 Stopping existing Ironcliw..."
pkill -f "python.*start_system.py" 2>/dev/null || true
pkill -f "python.*main.py" 2>/dev/null || true
sleep 2

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Load the API key
echo "🔑 Loading API key..."
if [ -f "backend/.env" ]; then
    export $(cat backend/.env | grep ANTHROPIC_API_KEY | xargs)
    echo "✅ API key loaded"
else
    echo "⚠️  No .env file found in backend/"
    exit 1
fi

# Quick test of vision commands
echo ""
echo "🔍 Testing vision command recognition..."
python -c "
import sys
sys.path.append('backend')
from voice.jarvis_agent_voice import IroncliwAgentVoice
jarvis = IroncliwAgentVoice()

# Test command detection
test_cmds = ['what am i working on', 'can you see my screen']
for cmd in test_cmds:
    is_system = jarvis._is_system_command(cmd)
    print(f'  • \"{cmd}\" → Vision: {\"✅\" if is_system else \"❌\"}')"

echo ""
echo "🚀 Starting Ironcliw with intelligent vision..."
echo "============================================"
echo ""
echo "🎯 Try these commands:"
echo "  • 'Hey Ironcliw, what am I working on?'"
echo "  • 'What can you see in Cursor?'"
echo "  • 'Describe what I'm doing'"
echo "  • 'Can you see my screen?'"
echo ""
echo "Ironcliw will now provide intelligent, contextual responses!"
echo ""

# Start Ironcliw
python start_system.py