#!/bin/bash

echo "🎯 Testing Ironcliw Context Awareness"
echo "=================================="
echo ""
echo "This script will demonstrate the context awareness feature."
echo "Make sure your React app is running and connected to Ironcliw."
echo ""
echo "Press Enter to continue..."
read

echo "1. First, let's lock your screen..."
echo "   Say: 'Hey Ironcliw, lock my screen'"
echo ""
echo "Press Enter after your screen is locked..."
read

echo "2. Now, with your screen locked, try a command that needs screen access:"
echo "   Say: 'Hey Ironcliw, open Safari and search for puppies'"
echo ""
echo "Expected behavior:"
echo "   - Ironcliw detects the screen is locked"
echo "   - Ironcliw says: 'Your screen is locked. I'll unlock it now by typing in the password.'"
echo "   - Ironcliw unlocks your screen"
echo "   - Ironcliw opens Safari and searches"
echo "   - Ironcliw confirms what was done"
echo ""
echo "Press Enter after testing..."
read

echo "✅ Test complete!"
echo ""
echo "If the context awareness didn't work, check:"
echo "1. Backend log: tail -f jarvis_backend.log"
echo "2. Make sure the backend restarted properly"
echo "3. Check browser console for errors"