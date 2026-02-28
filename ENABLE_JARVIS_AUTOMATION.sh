#!/bin/bash
# Enable Ironcliw Goal Inference Automation

# Option 1: Add to your shell configuration for permanent settings
echo "
# Ironcliw Goal Inference Settings
export Ironcliw_GOAL_PRESET=balanced      # or 'aggressive' for more proactive behavior
export Ironcliw_GOAL_AUTOMATION=true      # Enable automatic actions
" >> ~/.zshrc

# Reload shell configuration
source ~/.zshrc

echo "✅ Ironcliw automation enabled permanently!"
echo "Settings:"
echo "  - Preset: balanced"
echo "  - Automation: enabled"
echo ""
echo "To use different settings, you can override:"
echo "  export Ironcliw_GOAL_PRESET=aggressive  # More proactive"
echo "  export Ironcliw_GOAL_PRESET=conservative  # Less proactive"