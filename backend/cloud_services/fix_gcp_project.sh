#!/bin/bash
# =============================================================================
# Fix GCP Project Configuration
# =============================================================================
# This script ensures the correct GCP project is set for all operations

set -e

CORRECT_PROJECT="jarvis-473803"

echo "🔧 Fixing GCP Project Configuration..."
echo ""

# 1. Set gcloud project
echo "1. Setting gcloud project to: $CORRECT_PROJECT"
gcloud config set project "$CORRECT_PROJECT" || {
    echo "❌ Failed to set gcloud project"
    exit 1
}

# 2. Verify project access
echo "2. Verifying project access..."
if gcloud projects describe "$CORRECT_PROJECT" &>/dev/null; then
    echo "✅ Project $CORRECT_PROJECT is accessible"
else
    echo "❌ Cannot access project $CORRECT_PROJECT"
    echo "   Please check your authentication: gcloud auth login"
    exit 1
fi

# 3. Set environment variable
echo "3. Setting GCP_PROJECT_ID environment variable..."
export GCP_PROJECT_ID="$CORRECT_PROJECT"
echo "   GCP_PROJECT_ID=$GCP_PROJECT_ID"

# 4. Show current configuration
echo ""
echo "📋 Current GCP Configuration:"
echo "   Project: $(gcloud config get-value project)"
echo "   Account: $(gcloud config get-value account)"
echo ""
echo "✅ GCP project configuration fixed!"
echo ""
echo "💡 To make this permanent, add to your shell profile:"
echo "   export GCP_PROJECT_ID=$CORRECT_PROJECT"

