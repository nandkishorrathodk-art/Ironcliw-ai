#!/bin/bash
# Activate Ironcliw virtual environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"
export PYTHONPATH="$SCRIPT_DIR/backend:$PYTHONPATH"
echo "🤖 Ironcliw environment activated"
echo "   Python: $(which python)"
echo "   To start: python start_system.py"
