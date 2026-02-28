#!/bin/bash
# Start Ironcliw in the background with proper logging

cd /Users/derekjrussell/Documents/repos/Ironcliw-AI-Agent/backend

# Create logs directory if it doesn't exist
mkdir -p logs

# Start Ironcliw in the background
echo "🚀 Starting Ironcliw on port 8000 in background..."
nohup python main.py > logs/jarvis_$(date +%Y%m%d_%H%M%S).log 2>&1 &
PID=$!

echo "✅ Ironcliw started with PID: $PID"
echo "📝 Logs: logs/jarvis_$(date +%Y%m%d_%H%M%S).log"
echo ""
echo "Waiting for Ironcliw to be ready..."

# Wait for Ironcliw to be ready
for i in {1..30}; do
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        echo "✅ Ironcliw is ready!"
        echo ""
        echo "🎤 Test audio: http://localhost:8000/audio/speak/Hello%20from%20Ironcliw"
        echo "🖥️  Frontend: http://localhost:3000"
        exit 0
    fi
    echo -n "."
    sleep 1
done

echo ""
echo "⚠️  Ironcliw didn't start within 30 seconds. Check the logs."
tail -n 20 logs/jarvis_*.log
exit 1