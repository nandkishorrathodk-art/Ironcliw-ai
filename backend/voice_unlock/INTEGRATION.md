# Voice Unlock - Ironcliw Integration

## Overview

The Voice Unlock system is now fully integrated into Ironcliw's main startup process. When you start Ironcliw, the Voice Unlock WebSocket server automatically starts if you have configured your password.

## What Happens During Ironcliw Startup

1. **Automatic Check**: Ironcliw checks if you've stored your password (via `enable_screen_unlock.sh`)
2. **WebSocket Server**: If password is configured, the Voice Unlock WebSocket server starts automatically on port 8765
3. **Ready to Use**: You can immediately use voice commands like "unlock my mac"
4. **No Manual Steps**: No need to run `start_voice_unlock_system.sh` separately anymore

## How It Works Now

### Starting Ironcliw
```bash
# From the root of the project
python start_system.py
```

This automatically:
- ✅ Starts all Ironcliw components
- ✅ Starts Voice Unlock WebSocket server
- ✅ Initializes voice unlock integration
- ✅ Makes "unlock my mac" commands work immediately

### Using Voice Unlock

1. **Direct Commands** (when talking to Ironcliw):
   - "Hey Ironcliw, unlock my mac"
   - "Hey Ironcliw, unlock my screen"
   - "Hey Ironcliw, unlock the mac"

2. **What Happens**:
   - Ironcliw receives your command
   - Sends unlock request to Voice Unlock system
   - Retrieves password from Keychain
   - Types password and unlocks screen

3. **No More Instructions**: 
   - Previously: Ironcliw would tell you to say another phrase
   - Now: Ironcliw directly unlocks your screen

## Configuration Required

### One-Time Setup
```bash
cd backend/voice_unlock
./enable_screen_unlock.sh
```

This stores your password securely in macOS Keychain.

## Technical Details

### Integration Points

1. **main.py**:
   - Imports `voice_unlock_startup_integration`
   - Starts WebSocket server in `lifespan` function
   - Cleanly shuts down on exit

2. **voice_unlock_integration.py**:
   - Handles "unlock my mac" commands
   - Sends `unlock_screen` command to daemon
   - No longer gives instructions

3. **websocket_server.py**:
   - Added `unlock_screen` command handler
   - Retrieves password from Keychain
   - Performs keyboard simulation

### Architecture
```
User Voice → Ironcliw → voice_unlock_integration → WebSocket → Screen Unlock
```

## Troubleshooting

### Voice Unlock Not Working?

1. **Check if password is stored**:
   ```bash
   security find-generic-password -s com.jarvis.voiceunlock -a unlock_token -g
   ```

2. **Check WebSocket server**:
   ```bash
   lsof -i :8765
   ```

3. **Check logs**:
   - Ironcliw startup log shows Voice Unlock status
   - Look for "🔐 Voice Unlock system started"

### Common Issues

- **"No password stored"**: Run `enable_screen_unlock.sh`
- **WebSocket not starting**: Check if port 8765 is already in use
- **Screen not unlocking**: Verify accessibility permissions

## Benefits of Integration

1. **Seamless Experience**: Voice Unlock is part of Ironcliw, not a separate system
2. **Automatic Startup**: No manual steps required
3. **Direct Commands**: Say "unlock my mac" and it happens
4. **Clean Shutdown**: Properly stops when Ironcliw stops
5. **Status in Logs**: See Voice Unlock status in Ironcliw startup logs