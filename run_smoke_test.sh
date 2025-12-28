#!/bin/bash
#
# Watch & Act Smoke Test Runner
# Uses virtual environment Python with all dependencies
#

echo "🚀 Running Watch & Act Smoke Test with venv python..."
echo

# Run with venv python
./venv/bin/python3 test_watch_and_act.py

# Capture exit code
EXIT_CODE=$?

echo
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Test completed successfully!"
elif [ $EXIT_CODE -eq 1 ]; then
    echo "❌ Test failed - see errors above"
elif [ $EXIT_CODE -eq 2 ]; then
    echo "⏳ Test pending - monitoring in progress"
else
    echo "⚠️  Test exited with code $EXIT_CODE"
fi

exit $EXIT_CODE
