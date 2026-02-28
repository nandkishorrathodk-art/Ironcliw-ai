#!/bin/bash
# Start Ironcliw on port 8000 with all the latest audio fixes

echo "🚀 Starting Ironcliw on port 8000..."
echo "This instance will have all the audio fixes including Daniel's voice"
echo ""

cd /Users/derekjrussell/Documents/repos/Ironcliw-AI-Agent/backend

# Export environment variables
export BACKEND_PORT=8000
export PYTHONUNBUFFERED=1

# Start Ironcliw
echo "Starting Ironcliw with updated audio system..."
python main.py