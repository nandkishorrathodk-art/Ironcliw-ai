#!/bin/bash

echo "🛠️ Building Ironcliw Weather Tools"

# Use Python fallback as main tool for now
echo "Setting up weather tools..."

# Make Python fallback the primary tool
cp jarvis-weather-fallback.py jarvis-weather
chmod +x jarvis-weather

echo "✅ Weather tools ready!"
echo ""
echo "The weather system now uses:"
echo "  • IP-based geolocation for general location"
echo "  • wttr.in API for accurate weather data"
echo ""
echo "To test: ./jarvis-weather current"