#!/bin/bash
# Force macOS to refresh Screen Recording permissions

echo "======================================================================"
echo "🔄 Force Screen Recording Permission Refresh"
echo "======================================================================"
echo ""

# Check current process info
echo "📊 Current process information:"
echo "   Terminal PID: $(pgrep -x Terminal)"
echo "   Python PIDs: $(pgrep -x python3)"
echo ""

# Check which apps have Screen Recording permission
echo "🔍 Apps with Screen Recording permission:"
echo ""

if command -v sqlite3 >/dev/null 2>&1; then
    # Query TCC database (macOS permission database)
    TCC_DB="$HOME/Library/Application Support/com.apple.TCC/TCC.db"

    if [ -f "$TCC_DB" ]; then
        echo "   Checking TCC database..."
        sqlite3 "$TCC_DB" "SELECT client, auth_value FROM access WHERE service='kTCCServiceScreenCapture';" 2>/dev/null || echo "   (Cannot read TCC database - normal for newer macOS)"
    fi
fi

echo ""
echo "======================================================================"
echo "⚠️  CRITICAL: You must COMPLETELY QUIT Terminal"
echo "======================================================================"
echo ""
echo "The issue: Your current Terminal session still has OLD permissions."
echo "The fix: You MUST quit Terminal app completely (not just close window)"
echo ""
echo "======================================================================"
echo ""
echo "Option 1: Manual Quit (MOST RELIABLE)"
echo "======================================================================"
echo ""
echo "  1. Press: Cmd + Q (or click Terminal menu → Quit Terminal)"
echo "  2. Wait 3 seconds (make sure NO Terminal windows are open)"
echo "  3. Open Terminal again from Applications → Utilities"
echo "  4. cd $PWD"
echo "  5. python3 check_screen_recording_permission.py"
echo ""
echo "======================================================================"
echo ""
echo "Option 2: Kill all Python processes (NUCLEAR)"
echo "======================================================================"
echo ""
echo "This will force-kill all Python processes to refresh permissions:"
echo ""
read -p "Kill all Python processes and quit Terminal? (y/n): " KILL_OPTION

if [[ "$KILL_OPTION" == "y" ]]; then
    echo ""
    echo "Killing all Python processes..."
    killall python3 2>/dev/null
    killall Python 2>/dev/null
    sleep 1

    echo "Creating auto-run script for next Terminal launch..."

    # Create startup script
    cat > ~/.jarvis_check_on_startup << 'EOFCHECK'
#!/bin/bash
cd "CURRENT_DIR_PLACEHOLDER"
echo ""
echo "======================================================================"
echo "🔍 Checking Screen Recording Permission After Restart..."
echo "======================================================================"
echo ""
python3 check_screen_recording_permission.py
EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "🎉 SUCCESS! Permission granted!"
    echo ""
    echo "Ready to run: ./run_smoke_test.sh"
else
    echo "❌ Still not working. Try these:"
    echo ""
    echo "1. System Preferences → Security & Privacy → Screen Recording"
    echo "2. UNCHECK Terminal, then CHECK it again"
    echo "3. Also add 'Python' if you see it in the list"
    echo "4. Click lock, enter password, close prefs"
    echo "5. Quit Terminal COMPLETELY (Cmd+Q)"
    echo "6. Reopen and try again"
fi
rm -f ~/.jarvis_check_on_startup
EOFCHECK

    # Replace placeholder
    sed -i '' "s|CURRENT_DIR_PLACEHOLDER|$PWD|g" ~/.jarvis_check_on_startup
    chmod +x ~/.jarvis_check_on_startup

    # Add to shell RC file
    if [ -f ~/.zshrc ]; then
        echo "" >> ~/.zshrc
        echo "# JARVIS permission check (auto-remove)" >> ~/.zshrc
        echo "if [ -f ~/.jarvis_check_on_startup ]; then" >> ~/.zshrc
        echo "    bash ~/.jarvis_check_on_startup" >> ~/.zshrc
        echo "    sed -i '' '/# JARVIS permission check/,+3d' ~/.zshrc" >> ~/.zshrc
        echo "fi" >> ~/.zshrc
    elif [ -f ~/.bash_profile ]; then
        echo "" >> ~/.bash_profile
        echo "# JARVIS permission check (auto-remove)" >> ~/.bash_profile
        echo "if [ -f ~/.jarvis_check_on_startup ]; then" >> ~/.bash_profile
        echo "    bash ~/.jarvis_check_on_startup" >> ~/.bash_profile
        echo "    sed -i '' '/# JARVIS permission check/,+3d' ~/.bash_profile" >> ~/.bash_profile
        echo "fi" >> ~/.bash_profile
    fi

    echo ""
    echo "✅ Setup complete!"
    echo ""
    echo "Now quitting Terminal in 3 seconds..."
    echo "Permission check will run automatically when you reopen Terminal."
    echo ""
    sleep 3

    # Force quit Terminal
    osascript -e 'tell application "Terminal" to quit' &
    exit 0
fi

echo ""
echo "======================================================================"
echo ""
echo "Option 3: Advanced - Reset TCC Database (LAST RESORT)"
echo "======================================================================"
echo ""
echo "⚠️  WARNING: This will reset ALL privacy permissions!"
echo ""
read -p "Reset TCC database? (y/n): " RESET_TCC

if [[ "$RESET_TCC" == "y" ]]; then
    echo ""
    echo "Resetting TCC database..."

    # This requires System Integrity Protection to be disabled
    # We'll just provide the command
    echo ""
    echo "Run this command in Terminal (requires admin password):"
    echo ""
    echo "sudo tccutil reset ScreenCapture"
    echo ""
    echo "Then:"
    echo "1. Quit Terminal completely (Cmd+Q)"
    echo "2. Reopen Terminal"
    echo "3. You'll be prompted for Screen Recording permission again"
    echo "4. Grant it, then quit and reopen Terminal again"
    echo ""
fi

echo ""
echo "======================================================================"
echo "🔧 Additional Debugging"
echo "======================================================================"
echo ""
echo "If nothing works, try adding BOTH of these to Screen Recording:"
echo "  ☑️  Terminal"
echo "  ☑️  Python (or python3)"
echo ""
echo "Location: System Preferences → Security & Privacy → Screen Recording"
echo ""
echo "You may need to manually add Python:"
echo "  1. Click the '+' button"
echo "  2. Navigate to: /usr/bin/python3"
echo "  3. (or find it with: which python3)"
echo ""
echo "======================================================================"
