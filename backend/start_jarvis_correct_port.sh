#!/bin/bash
# Start Ironcliw on the correct default port

cd ~/Documents/repos/Ironcliw-AI-Agent/backend

echo "🚀 Starting Ironcliw on port 8010..."
python main.py --port 8010 &

echo "✅ Ironcliw started!"
echo ""
echo "Now go to: http://localhost:8010"
echo "WebSocket available at: ws://localhost:8010/ws"
echo "And try: 'lock my screen'"