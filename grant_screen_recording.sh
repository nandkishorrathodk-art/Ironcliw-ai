#!/bin/bash
# Automated Screen Recording Permission Helper
# This script automates as much as possible, but you'll still need to click the checkbox

echo "======================================================================"
echo "🔓 JARVIS Screen Recording Permission Helper"
echo "======================================================================"
echo ""
echo "This script will:"
echo "  1. Open System Preferences to the Screen Recording section"
echo "  2. Guide you through granting permission"
echo "  3. Verify permission was granted"
echo ""
echo "======================================================================"
echo ""

# Step 1: Open System Preferences directly to Screen Recording
echo "📂 Step 1: Opening System Preferences → Privacy → Screen Recording..."
echo ""

# Try different methods based on macOS version
if [[ $(sw_vers -productVersion | cut -d. -f1) -ge 13 ]]; then
    # macOS Ventura (13) and later
    echo "   Detected macOS 13+ (Ventura or later)"
    open "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"
else
    # macOS Monterey (12) and earlier
    echo "   Detected macOS 12 or earlier"
    open "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"
fi

sleep 2

echo ""
echo "======================================================================"
echo "👆 ACTION REQUIRED: Please do the following in System Preferences"
echo "======================================================================"
echo ""
echo "  1. ✅ Check the box next to 'Terminal' (or 'Python' if you see it)"
echo "  2. 🔒 If needed: Click the lock icon at bottom-left and enter password"
echo "  3. ✅ Click 'OK' on any popup that appears"
echo "  4. ❌ Close System Preferences window"
echo ""
echo "======================================================================"
echo ""

# Wait for user to grant permission
read -p "Press ENTER after you've checked the Terminal box and closed System Preferences..."

echo ""
echo "======================================================================"
echo "🔄 Step 2: Restarting Terminal for changes to take effect..."
echo "======================================================================"
echo ""
echo "⚠️  IMPORTANT: You need to completely quit and reopen Terminal!"
echo ""
echo "Option 1 (Recommended):"
echo "  1. Press Cmd+Q to quit Terminal"
echo "  2. Wait 2 seconds"
echo "  3. Open Terminal again from Applications"
echo "  4. cd back to: $PWD"
echo "  5. Run: python3 check_screen_recording_permission.py"
echo ""
echo "Option 2 (I'll try to restart it for you - may not work):"
read -p "Try automatic restart? (y/n): " AUTO_RESTART

if [[ "$AUTO_RESTART" == "y" ]]; then
    echo ""
    echo "Attempting automatic Terminal restart..."
    echo "This will:"
    echo "  1. Save this directory path"
    echo "  2. Quit Terminal"
    echo "  3. Reopen Terminal"
    echo "  4. Return to this directory"
    echo "  5. Run the permission check"
    echo ""

    # Save current directory
    CURRENT_DIR="$PWD"

    # Create a temporary script that will run after Terminal restarts
    TEMP_SCRIPT="/tmp/jarvis_restart_check.sh"
    cat > "$TEMP_SCRIPT" << 'EOF'
#!/bin/bash
cd "REPLACE_DIR"
echo ""
echo "======================================================================"
echo "🔍 Verifying Screen Recording Permission..."
echo "======================================================================"
echo ""
python3 check_screen_recording_permission.py
EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "======================================================================"
    echo "🎉 SUCCESS! Screen Recording permission is now granted!"
    echo "======================================================================"
    echo ""
    echo "✅ Ready to run the smoke test!"
    echo ""
    echo "Run: ./run_smoke_test.sh"
    echo ""
else
    echo "======================================================================"
    echo "❌ Permission still not granted"
    echo "======================================================================"
    echo ""
    echo "Please manually:"
    echo "  1. Open System Preferences → Security & Privacy → Screen Recording"
    echo "  2. Check the box next to Terminal"
    echo "  3. Completely quit Terminal (Cmd+Q)"
    echo "  4. Reopen Terminal"
    echo "  5. Run: python3 check_screen_recording_permission.py"
    echo ""
fi
rm -f "$TEMP_SCRIPT"
EOF

    # Replace placeholder with actual directory
    sed -i '' "s|REPLACE_DIR|$CURRENT_DIR|g" "$TEMP_SCRIPT"
    chmod +x "$TEMP_SCRIPT"

    # Schedule the script to run when Terminal reopens
    echo "cd \"$CURRENT_DIR\" && bash \"$TEMP_SCRIPT\"" > ~/.jarvis_startup_check

    # Add to .zshrc or .bash_profile to run on startup
    if [ -f ~/.zshrc ]; then
        echo "" >> ~/.zshrc
        echo "# JARVIS temporary startup check" >> ~/.zshrc
        echo "if [ -f ~/.jarvis_startup_check ]; then" >> ~/.zshrc
        echo "    bash ~/.jarvis_startup_check" >> ~/.zshrc
        echo "    rm -f ~/.jarvis_startup_check" >> ~/.zshrc
        echo "    # Remove these lines from .zshrc" >> ~/.zshrc
        echo "    sed -i '' '/# JARVIS temporary startup check/,+6d' ~/.zshrc" >> ~/.zshrc
        echo "fi" >> ~/.zshrc
    fi

    echo ""
    echo "Setup complete! Now quitting Terminal in 3 seconds..."
    echo "After it reopens, the permission check will run automatically."
    sleep 3

    # Quit Terminal (this will close this script too)
    osascript -e 'tell application "Terminal" to quit'
else
    echo ""
    echo "======================================================================"
    echo "📋 Manual Steps:"
    echo "======================================================================"
    echo ""
    echo "1. Quit Terminal completely (Cmd+Q)"
    echo "2. Wait 2-3 seconds"
    echo "3. Open Terminal again"
    echo "4. cd $PWD"
    echo "5. python3 check_screen_recording_permission.py"
    echo ""
    echo "======================================================================"
fi
