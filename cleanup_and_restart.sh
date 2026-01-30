#!/bin/bash
#
# JARVIS v147.0 Clean Restart Script
# ===================================
# Clears all stale state and restarts with fixes
#

echo "🧹 JARVIS v147.0 Clean Restart"
echo "=============================="

# Step 1: Kill any running supervisor processes
echo ""
echo "Step 1: Killing existing processes..."
pkill -f "run_supervisor.py" 2>/dev/null || true
pkill -f "jarvis.*prime" 2>/dev/null || true
pkill -f "reactor.*core" 2>/dev/null || true
sleep 2

# Step 2: Clear cloud lock (allows fresh GCP provisioning)
echo ""
echo "Step 2: Clearing cloud lock..."
rm -f ~/.jarvis/trinity/cloud_lock.json
echo "   ✅ Cloud lock cleared"

# Step 3: Clear stale service state
echo ""
echo "Step 3: Clearing stale service state..."
rm -f ~/.jarvis/trinity/ports/*.port 2>/dev/null || true
rm -f ~/.jarvis/trinity/process_tree.json 2>/dev/null || true
rm -f ~/.jarvis/trinity/orchestrator_state.json 2>/dev/null || true
echo "   ✅ Service state cleared"

# Step 4: Clear memory pressure signals
echo ""
echo "Step 4: Clearing memory pressure signals..."
rm -f ~/.jarvis/cross_repo/memory_pressure.json 2>/dev/null || true
echo "   ✅ Memory signals cleared"

# Step 5: Force garbage collection on macOS
echo ""
echo "Step 5: Clearing system memory pressure..."
sudo purge 2>/dev/null || echo "   ⚠️ purge requires sudo (optional)"

echo ""
echo "=============================="
echo "✅ Cleanup complete!"
echo ""
echo "Now run: python3 run_supervisor.py"
echo ""
echo "Expected behavior with v147.0 fixes:"
echo "  1. GCP firewall rule will be created (if needed)"
echo "  2. New GCP VM will be created with fast health stub"
echo "  3. Health check should pass in <30 seconds"
echo "  4. jarvis-prime will route heavy inference to GCP"
echo "  5. Local RAM usage should stay low (~300MB)"
