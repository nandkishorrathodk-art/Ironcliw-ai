# Ironcliw Voice Unlock System - Complete Guide
============================================

## Overview

The Ironcliw Voice Unlock system allows you to unlock your Mac using voice commands when the screen is locked. It integrates with Ironcliw AI to provide natural language voice unlock capabilities.

## Current Status

### ✅ Completed Components

1. **Objective-C Daemon** - Background service that monitors for voice commands
2. **Voice Authentication Engine** - Handles voice recognition and anti-spoofing
3. **Screen Unlock Integration** - Interfaces with macOS security framework
4. **Voice Monitor** - Continuous audio monitoring service
5. **WebSocket Bridge** - Communication between components
6. **Permission Manager** - Handles macOS permissions
7. **Python Bridge** - Voice processing and ML capabilities
8. **WebSocket Server** - API for Ironcliw integration
9. **Ironcliw Integration** - Voice commands through Ironcliw

### 🔧 Setup Instructions

1. **Build the daemon:**
   ```bash
   cd backend/voice_unlock/objc
   make clean && make daemon
   ```

2. **Start the Voice Unlock system:**
   ```bash
   cd backend/voice_unlock
   ./start_voice_unlock_system.sh
   ```

3. **Test with Ironcliw:**
   - Say: "Hey Ironcliw, enable voice unlock"
   - Say: "Hey Ironcliw, voice unlock status"

## Voice Commands

When talking to Ironcliw, you can use these commands:

- **"enable voice unlock"** - Start voice monitoring
- **"disable voice unlock"** - Stop voice monitoring  
- **"voice unlock status"** - Check system status
- **"test voice unlock"** - Test the system

When your screen is locked and voice monitoring is enabled:

- **"Hello Ironcliw, unlock my Mac"**
- **"Ironcliw, this is Derek"**
- **"Open sesame, Ironcliw"**

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Ironcliw API    │────▶│ WebSocket Server │────▶│  Objective-C    │
│ (Python/FastAPI)│     │   (Port 8765)    │     │    Daemon       │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                           │
                                                           ▼
                              ┌────────────────────────────┴────────────┐
                              │                                         │
                         ┌────▼──────┐                        ┌────────▼────────┐
                         │   Voice    │                        │     Screen      │
                         │  Monitor   │                        │    Unlock       │
                         └────────────┘                        └─────────────────┘
```

## Technical Details

### WebSocket API

The Voice Unlock daemon exposes a WebSocket API on port 8765:

```javascript
// Connect
ws://localhost:8765/voice-unlock

// Handshake
{
  "type": "command",
  "command": "handshake",
  "parameters": {
    "client": "jarvis-api",
    "version": "1.0"
  }
}

// Get Status
{
  "type": "command", 
  "command": "get_status"
}

// Start Monitoring
{
  "type": "command",
  "command": "start_monitoring"  
}
```

### Permissions Required

The daemon requires these macOS permissions:
- ✅ Microphone Access
- ✅ Accessibility Access  
- ✅ Keychain Access
- ⚠️ System Events (optional)

## Current Limitations

1. **Voice Enrollment**: Currently using a test user profile. Real voice enrollment needs to be implemented.
2. **Screen Unlock Token**: The secure token for actual screen unlock needs to be stored in Keychain.
3. **WebSocket Server**: Using Python bridge instead of native Objective-C WebSocket server.

## Next Steps

To complete the system:

1. **Enroll Your Voice**
   - Record voice samples
   - Create voiceprint
   - Store securely

2. **Store Unlock Token**
   - Get user password securely
   - Store in Keychain
   - Use for screen unlock

3. **Test Full Flow**
   - Lock screen
   - Say unlock phrase
   - Verify unlock works

## Troubleshooting

### WebSocket won't start
```bash
# Check if port 8765 is in use
lsof -i :8765

# Kill any process using it
kill -9 <PID>
```

### Daemon won't build
```bash
# Clean and rebuild
cd backend/voice_unlock/objc
make clean
mkdir -p build bin
make daemon
```

### Voice commands not recognized
1. Check WebSocket server is running
2. Verify Ironcliw is connected
3. Check daemon logs: `/tmp/voice_unlock_ws.log`

## Security Notes

- Voice authentication alone is not sufficient for high-security scenarios
- Always use in combination with other authentication methods
- Voice samples are stored encrypted in Keychain
- Anti-spoofing measures are implemented but not foolproof

## Development

To modify the system:

1. **Objective-C code**: Edit files in `backend/voice_unlock/objc/`
2. **Python bridge**: Edit `backend/voice_unlock/objc/server/websocket_server.py`
3. **Ironcliw integration**: Edit `backend/api/voice_unlock_handler.py`

After changes, rebuild with:
```bash
cd backend/voice_unlock/objc
make clean && make daemon
```

## Contact

For issues or questions, check the logs:
- Daemon log: `/tmp/daemon.log`
- WebSocket log: `/tmp/voice_unlock_ws.log`
- Ironcliw log: Check Ironcliw console output