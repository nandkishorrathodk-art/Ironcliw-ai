#!/bin/bash
# Activate JARVIS virtual environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"
export PYTHONPATH="$SCRIPT_DIR/backend:$PYTHONPATH"
echo "🤖 JARVIS environment activated"
echo "   Python: $(which python)"
echo "   To start: python start_system.py"
