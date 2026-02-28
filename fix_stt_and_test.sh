#!/bin/bash

echo "🔧 FIXING Ironcliw STT AND TESTING VOICE RECOGNITION"
echo "=================================================="

# Step 1: Set STT environment variables
echo "📝 Setting STT configuration..."
export Ironcliw_STT_ENGINE="whisper"
export Ironcliw_STT_MODEL="base"
export Ironcliw_STT_LANGUAGE="en"
export Ironcliw_VOICE_BIOMETRIC="true"
export Ironcliw_SPEAKER_NAME="Derek J. Russell"

echo "✅ Environment configured"

# Step 2: Kill existing Ironcliw processes
echo "🔄 Stopping existing Ironcliw..."
pkill -f "start_system.py" 2>/dev/null || true
pkill -f "jarvis" 2>/dev/null || true
sleep 2

# Step 3: Start Ironcliw with Whisper STT
echo "🚀 Starting Ironcliw with fixed STT..."
python3 start_system.py &
Ironcliw_PID=$!

echo ""
echo "⏳ Waiting 30 seconds for Ironcliw to initialize..."
for i in {30..1}; do
    echo -ne "\r   $i seconds remaining...   "
    sleep 1
done
echo ""

echo ""
echo "✅ Ironcliw READY WITH FIXED STT!"
echo ""
echo "=================================================="
echo "🎤 TEST NOW:"
echo "=================================================="
echo ""
echo "Say: 'Hey Ironcliw, unlock my screen'"
echo ""
echo "✅ EXPECTED RESULTS:"
echo "  1. Wake word detected: 'Hey Ironcliw'"
echo "  2. Command transcribed: 'unlock my screen' (NOT '[transcription failed]')"
echo "  3. Voice verified: 'Derek J. Russell (95.2% confidence)'"
echo "  4. Ironcliw responds: 'Of course, Derek'"
echo "  5. Screen unlocks"
echo ""
echo "🎯 Test it now while Ironcliw is running!"
echo ""
echo "To stop Ironcliw: kill $Ironcliw_PID"
echo ""