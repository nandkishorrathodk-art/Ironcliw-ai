#!/bin/bash
# Complete Voice Unlock System Test
# =================================

echo "🧪 Ironcliw Voice Unlock - Complete System Test"
echo "============================================="
echo

# Function to check if screen is locked
check_screen_locked() {
    # Check if screensaver is running
    if pgrep -x "ScreenSaver" > /dev/null; then
        echo "✅ Screen is locked (ScreenSaver running)"
        return 0
    else
        echo "❌ Screen is NOT locked"
        return 1
    fi
}

# 1. Check prerequisites
echo "1️⃣ Checking Prerequisites..."
echo "-----------------------------"

# Check if password is stored
echo -n "Checking Keychain for stored password... "
if security find-generic-password -s com.jarvis.voiceunlock -a unlock_token 2>/dev/null; then
    echo "✅ Found"
else
    echo "❌ Not found"
    echo "Please run: ./enable_screen_unlock.sh"
    exit 1
fi

# Check permissions
echo -n "Checking accessibility permissions... "
if osascript -e 'tell application "System Events" to return exists' 2>/dev/null; then
    echo "✅ Granted"
else
    echo "❌ Not granted"
    echo "Grant Terminal accessibility access in System Preferences > Privacy & Security"
fi

# Check microphone permissions  
echo -n "Checking microphone permissions... "
# This is harder to check programmatically
echo "⚠️  Please verify manually"

echo

# 2. Start the system
echo "2️⃣ Starting Voice Unlock System..."
echo "-----------------------------------"

# Kill any existing processes
pkill -f websocket_server.py
pkill -f IroncliwVoiceUnlockDaemon
sleep 2

# Start WebSocket server
echo "Starting WebSocket server..."
/Users/derekjrussell/miniforge3/bin/python3 objc/server/websocket_server.py > /tmp/websocket_test.log 2>&1 &
WS_PID=$!
sleep 3

if ps -p $WS_PID > /dev/null; then
    echo "✅ WebSocket server running (PID: $WS_PID)"
else
    echo "❌ WebSocket server failed to start"
    tail -20 /tmp/websocket_test.log
    exit 1
fi

# Start daemon
echo "Starting Voice Unlock daemon..."
./objc/bin/IroncliwVoiceUnlockDaemon > /tmp/daemon_test.log 2>&1 &
DAEMON_PID=$!
sleep 3

if ps -p $DAEMON_PID > /dev/null; then
    echo "✅ Daemon running (PID: $DAEMON_PID)"
else
    echo "❌ Daemon failed to start"
    tail -20 /tmp/daemon_test.log
    exit 1
fi

echo

# 3. Test screen lock detection
echo "3️⃣ Testing Screen Lock Detection..."
echo "------------------------------------"
echo "Current screen status:"
check_screen_locked

echo
echo "📝 Instructions:"
echo "1. Lock your screen now (⌘+Control+Q)"
echo "2. Wait 5 seconds"
echo "3. The system will check if it detects the lock"
echo
echo "Press Enter when you've locked the screen..."
read

# Check again
check_screen_locked
if [ $? -eq 0 ]; then
    echo
    echo "4️⃣ Testing Voice Unlock..."
    echo "-------------------------"
    echo "Say one of these phrases:"
    echo "  • 'Hello Ironcliw, unlock my Mac'"
    echo "  • 'Ironcliw, this is Derek'"  
    echo "  • 'Open sesame, Ironcliw'"
    echo
    echo "The system should unlock your screen!"
    echo
    echo "Monitoring daemon output..."
    tail -f /tmp/daemon_test.log &
    TAIL_PID=$!
else
    echo "Screen not locked - cannot proceed with test"
fi

# Cleanup function
cleanup() {
    echo
    echo "🛑 Cleaning up..."
    kill $TAIL_PID 2>/dev/null
    kill $DAEMON_PID 2>/dev/null
    kill $WS_PID 2>/dev/null
    pkill -f websocket_server.py
    pkill -f IroncliwVoiceUnlockDaemon
    echo "✅ Cleanup complete"
}

trap cleanup INT

echo
echo "Press Ctrl+C to stop the test"
echo

# Keep running
while true; do
    sleep 1
done