#!/bin/bash
# Debug Voice Unlock System

echo "🔍 Ironcliw Voice Unlock Debug Mode"
echo "================================="
echo

# Clean up any existing processes
echo "🧹 Cleaning up existing processes..."
pkill -f websocket_server.py
pkill -f IroncliwVoiceUnlockDaemon
sleep 2

# Start WebSocket server with logging
echo
echo "📡 Starting WebSocket server with debug logging..."
WEBSOCKET_LOG="/tmp/websocket_debug.log"
echo "Log file: $WEBSOCKET_LOG"

/Users/derekjrussell/miniforge3/bin/python3 objc/server/websocket_server.py > "$WEBSOCKET_LOG" 2>&1 &
WEBSOCKET_PID=$!
echo "WebSocket server PID: $WEBSOCKET_PID"

# Wait for server to start
sleep 3

# Check if WebSocket is running
if ps -p $WEBSOCKET_PID > /dev/null; then
    echo "✅ WebSocket server is running"
else
    echo "❌ WebSocket server failed to start"
    echo "Last 20 lines of log:"
    tail -20 "$WEBSOCKET_LOG"
    exit 1
fi

# Test WebSocket connection
echo
echo "🔗 Testing WebSocket connection..."
nc -zv localhost 8765

# Now start the daemon with debug output
echo
echo "🚀 Starting Voice Unlock Daemon with debug output..."
DAEMON_LOG="/tmp/daemon_debug.log"
echo "Log file: $DAEMON_LOG"

# Run daemon directly to see output
./objc/bin/IroncliwVoiceUnlockDaemon > "$DAEMON_LOG" 2>&1 &
DAEMON_PID=$!
echo "Daemon PID: $DAEMON_PID"

# Monitor both processes
echo
echo "📊 Monitoring processes..."
echo "Press Ctrl+C to stop"
echo

# Function to check process status
check_status() {
    if ps -p $WEBSOCKET_PID > /dev/null; then
        echo -n "✅ WebSocket "
    else
        echo -n "❌ WebSocket "
    fi
    
    if ps -p $DAEMON_PID > /dev/null; then
        echo "✅ Daemon"
    else
        echo "❌ Daemon"
    fi
}

# Show logs in real-time
tail -f "$WEBSOCKET_LOG" "$DAEMON_LOG" &
TAIL_PID=$!

# Cleanup function
cleanup() {
    echo
    echo "🛑 Stopping debug session..."
    kill $TAIL_PID 2>/dev/null
    kill $DAEMON_PID 2>/dev/null
    kill $WEBSOCKET_PID 2>/dev/null
    pkill -f websocket_server.py
    pkill -f IroncliwVoiceUnlockDaemon
    echo "✅ Cleanup complete"
    exit 0
}

trap cleanup INT

# Monitor loop
while true; do
    sleep 5
    check_status
done