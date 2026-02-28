#!/bin/bash
# Ironcliw Launcher with Robust Cleanup
# DEPRECATED: This wrapper is no longer needed!
# All cleanup logic has been integrated directly into start_system.py
# You can now run: python start_system.py
#
# This script is kept for backward compatibility only.

echo "⚠️  DEPRECATED: jarvis.sh wrapper is no longer needed"
echo "   All cleanup logic is now integrated in start_system.py"
echo "   Recommendation: Use 'python start_system.py' directly"
echo ""
echo "Continuing with python start_system.py..."
echo ""

set -e

Ironcliw_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$Ironcliw_DIR"

# PID file for tracking
PID_FILE="/tmp/jarvis.pid"
LOG_FILE="/tmp/jarvis_launcher.log"

# Cleanup function
cleanup() {
    echo "🧹 Cleaning up Ironcliw..." | tee -a "$LOG_FILE"

    if [ -f "$PID_FILE" ]; then
        Ironcliw_PID=$(cat "$PID_FILE")
        if ps -p "$Ironcliw_PID" > /dev/null 2>&1; then
            echo "Stopping Ironcliw (PID: $Ironcliw_PID)..." | tee -a "$LOG_FILE"
            kill -TERM "$Ironcliw_PID" 2>/dev/null || true

            # Wait for graceful shutdown
            for i in {1..10}; do
                if ! ps -p "$Ironcliw_PID" > /dev/null 2>&1; then
                    echo "✅ Ironcliw stopped gracefully" | tee -a "$LOG_FILE"
                    break
                fi
                sleep 1
            done

            # Force kill if still running
            if ps -p "$Ironcliw_PID" > /dev/null 2>&1; then
                echo "⚠️  Force killing Ironcliw..." | tee -a "$LOG_FILE"
                kill -9 "$Ironcliw_PID" 2>/dev/null || true
            fi
        fi
        rm -f "$PID_FILE"
    fi

    # Clean up any orphaned processes
    pkill -f "start_system.py" 2>/dev/null || true

    echo "✅ Cleanup complete" | tee -a "$LOG_FILE"
}

# Trap signals
trap cleanup EXIT INT TERM HUP

echo "🚀 Starting Ironcliw..." | tee "$LOG_FILE"
echo "📝 Logs: $LOG_FILE"
echo ""

# Start Ironcliw
python start_system.py &
Ironcliw_PID=$!
echo $Ironcliw_PID > "$PID_FILE"

echo "Ironcliw PID: $Ironcliw_PID" | tee -a "$LOG_FILE"
echo ""
echo "Press Ctrl+C to stop, or close this terminal"
echo ""

# Wait for Ironcliw
wait $Ironcliw_PID 2>/dev/null || true

# If we reach here, Ironcliw exited naturally
rm -f "$PID_FILE"
echo "Ironcliw exited" | tee -a "$LOG_FILE"
