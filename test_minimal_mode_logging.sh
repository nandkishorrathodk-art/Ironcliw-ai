#!/bin/bash
#
# Test script to verify minimal mode logging enhancements
#

echo "🔍 Testing Ironcliw Minimal Mode Logging..."
echo ""

# Check if backend is running
echo "1️⃣ Checking backend status..."
curl -s http://localhost:8010/health | jq '.' || echo "Backend not running on 8010"
curl -s http://localhost:8001/health | jq '.' || echo "Backend not running on 8001"

echo ""
echo "2️⃣ Checking voice status endpoint..."
curl -s http://localhost:8010/voice/jarvis/status | jq '.' || curl -s http://localhost:8001/voice/jarvis/status | jq '.'

echo ""
echo "3️⃣ Testing Ironcliw activation..."
curl -X POST http://localhost:8010/voice/jarvis/activate -H "Content-Type: application/json" || \
curl -X POST http://localhost:8001/voice/jarvis/activate -H "Content-Type: application/json"

echo ""
echo ""
echo "✅ Test complete!"
echo ""
echo "Check the following for enhanced logging:"
echo "  • Browser Console: Look for detailed minimal mode status logs"
echo "  • Terminal: Look for startup messages about minimal mode"
echo "  • UI: Look for orange 'MINIMAL MODE' badge and banner"