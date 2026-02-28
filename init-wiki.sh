#!/bin/bash

###############################################################################
# Ironcliw GitHub Wiki Initialization Script
#
# This script initializes the GitHub Wiki if it doesn't exist yet
###############################################################################

set -e

echo "🚀 Ironcliw Wiki Initialization"
echo ""
echo "To initialize your GitHub Wiki, you need to:"
echo ""
echo "1. Go to: https://github.com/drussell23/Ironcliw-AI/wiki"
echo "2. Click the green 'Create the first page' button"
echo "3. Add a title: Home"
echo "4. Add content: # Ironcliw AI Agent Wiki"
echo "5. Click 'Save Page'"
echo ""
echo "After that, run:"
echo "  ./deploy-wiki.sh"
echo ""
echo "This will deploy all 11 Wiki pages automatically!"
echo ""
echo "Opening GitHub Wiki in browser..."
sleep 2
open "https://github.com/drussell23/Ironcliw-AI/wiki"
