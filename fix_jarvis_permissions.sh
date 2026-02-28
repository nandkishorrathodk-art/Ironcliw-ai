#!/bin/bash

echo "🔧 Ironcliw Screen Recording Permission Fix Script"
echo "================================================"
echo ""

# Step 1: Kill any existing Swift processes
echo "1️⃣  Stopping existing Swift capture processes..."
pkill -f "swift.*capture" 2>/dev/null
pkill -f "swift-frontend" 2>/dev/null
echo "   ✅ Swift processes stopped"
echo ""

# Step 2: Test current permissions
echo "2️⃣  Testing current permissions..."
python3 backend/test_screen_recording.py
echo ""

# Step 3: Create a simple Swift test that will trigger the permission dialog
echo "3️⃣  Creating permission trigger script..."
cat > test_swift_permission.swift << 'EOF'
import AVFoundation
import Foundation

print("🔍 Testing Swift screen recording permission...")

// Try to create screen input - this will trigger permission dialog
if let screenInput = AVCaptureScreenInput(displayID: CGMainDisplayID()) {
    print("✅ Screen recording permission granted for Swift!")
    print("   Swift processes can now capture the screen")
} else {
    print("❌ Screen recording permission NOT granted for Swift")
    print("\n⚠️  macOS should have shown a permission dialog")
    print("   If no dialog appeared, please manually add Swift to Screen Recording:")
    print("   1. Open System Settings > Privacy & Security > Screen Recording")
    print("   2. Click the + button")  
    print("   3. Navigate to /usr/bin/swift or /Applications/Xcode.app")
    print("   4. Add it to the list and ensure it's checked")
}
EOF

echo "4️⃣  Triggering Swift permission check..."
echo "   ⚠️  IMPORTANT: A permission dialog may appear - please click 'Allow'"
echo ""
swift test_swift_permission.swift
echo ""

# Step 4: Test if Ironcliw backend can now use Swift capture
echo "5️⃣  Testing Ironcliw Swift integration..."
python3 -c "
import asyncio
import sys
sys.path.insert(0, 'backend')

async def test_swift():
    try:
        from vision.swift_video_bridge import SwiftVideoBridge
        bridge = SwiftVideoBridge()
        
        # Check permission
        result = await bridge.check_permission()
        if result.get('permissionStatus') == 'authorized':
            print('✅ Ironcliw can now use Swift video capture!')
            return True
        else:
            print('❌ Swift still needs permission')
            print('   Permission status:', result.get('permissionStatus'))
            return False
    except Exception as e:
        print(f'❌ Error testing Swift bridge: {e}')
        return False

success = asyncio.run(test_swift())
sys.exit(0 if success else 1)
" 2>/dev/null

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ SUCCESS! Swift permissions are configured correctly"
    echo ""
    echo "6️⃣  Next steps:"
    echo "   1. Restart Ironcliw: ./restart_jarvis_intelligent.sh"
    echo "   2. Try the command again: 'Hey Ironcliw, start monitoring my screen'"
else
    echo ""
    echo "⚠️  Swift permissions still need to be configured"
    echo ""
    echo "Manual steps required:"
    echo "1. Open System Settings (⌘ + Space, type 'System Settings')"
    echo "2. Go to Privacy & Security > Screen Recording"
    echo "3. Look for these apps and ensure they're checked:"
    echo "   - Terminal (or your terminal app)"
    echo "   - swift or swift-frontend"
    echo "   - Python or Python3"
    echo ""
    echo "4. If Swift is not in the list:"
    echo "   a. Click the + button"
    echo "   b. Press ⌘ + Shift + G and enter: /usr/bin/"
    echo "   c. Look for 'swift' and add it"
    echo ""
    echo "5. After adding/checking Swift, restart Ironcliw"
fi

# Cleanup
rm -f test_swift_permission.swift

echo ""
echo "================================================"
echo "Script complete!"
