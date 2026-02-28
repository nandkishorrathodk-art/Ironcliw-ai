# 🎤 Quick Start: Voice-Activated God Mode

## What You Can Do Now

**Say this:**
```
"Ironcliw, watch all Chrome windows for bouncing ball"
```

**Ironcliw will:**
1. Parse your command (extract: Chrome, "bouncing ball", all_spaces=True)
2. Initialize VisualMonitorAgent (if first time)
3. Scan entire workspace via Yabai
4. Find ALL Chrome windows across ALL spaces
5. Spawn Ferrari Engine for EACH window (10-60 FPS)
6. Run OCR on parallel video streams
7. Detect "bouncing ball" text in real-time
8. Auto-switch to the space where trigger detected
9. Announce: "Success! I detected 'bouncing ball' in Chrome on Space 3"

---

## Try These Commands

### 1. Single Window Mode
```
"Ironcliw, watch Terminal for Build Complete"
"Ironcliw, monitor Chrome for Error"
"Ironcliw, notify me when Terminal says DONE"
```
→ Watches first matching window only

### 2. God Mode (All Spaces)
```
"Ironcliw, watch all Terminal windows for DONE"
"Ironcliw, monitor Chrome across all spaces for Error"
"Ironcliw, watch every Chrome window for SUCCESS"
```
→ Watches EVERY matching window across ALL desktop spaces in parallel

### 3. With Timeout
```
"Ironcliw, watch Terminal for 5 minutes when it says finished"
"Ironcliw, monitor Chrome for 2 hours for ready"
```
→ Auto-stops after specified duration

---

## Quick Test

### Setup (30 seconds):
1. Open these in separate browser windows/tabs:
   - `backend/tests/visual_test/vertical.html`
   - `backend/tests/visual_test/horizontal.html`

2. Move them to different spaces (optional)

3. Make sure both show "BOUNCE COUNT: X" text

### Run Command:
```
"Ironcliw, watch all Chrome windows for BOUNCE COUNT"
```

### Expected Result:
```
🗣️  Ironcliw: "Understood, Derek. Activating God Mode surveillance -
             I'll watch ALL Chrome windows across every desktop space
             for 'BOUNCE COUNT'."

[2-3 seconds later]

🗣️  Ironcliw: "Success! I detected 'BOUNCE COUNT' in Chrome on Space 2.
             Confidence: 94%. I was monitoring 2 Chrome windows in parallel.
             This one triggered first."
```

---

## Voice Command Format

### Pattern:
```
[Action] [all/every?] [App] [windows?] [across all spaces?] [for/when] [Trigger]
```

### Examples:
| Command | App | Trigger | All Spaces? |
|---------|-----|---------|-------------|
| "Watch Terminal for DONE" | Terminal | DONE | No |
| "Monitor all Chrome windows for Error" | Chrome | Error | Yes |
| "Watch Chrome across all spaces for ready" | Chrome | ready | Yes |
| "Notify me when Terminal says SUCCESS" | Terminal | SUCCESS | No |

---

## What Happens Under the Hood

```
You Say: "Watch all Chrome windows for Error"
    ↓
IntelligentCommandHandler.handle_command()
    ↓
_parse_watch_command()
    → Extracts: {app: "Chrome", trigger: "Error", all_spaces: True}
    ↓
_execute_surveillance_command()
    ↓
VisualMonitorAgent.watch(all_spaces=True)
    ↓
watch_app_across_all_spaces()
    ↓
MultiSpaceWindowDetector.get_all_windows_across_spaces()
    → Finds: 5 Chrome windows on Spaces 1, 2, 4, 6, 8
    ↓
For each window: _spawn_ferrari_watcher()
    → Spawns: 5 parallel Ferrari Engines @ 60 FPS
    ↓
_coordinate_watchers() - Race condition
    → Monitors all 5 streams in parallel
    → First trigger detection wins
    → Cancels other 4 watchers
    ↓
SpatialAwarenessAgent.switch_to_space(winner_space)
    → Auto-switches to Space 4 (where trigger found)
    ↓
_execute_trigger_action()
    → Voice announcement via TTS
    ↓
Return results to Voice Handler
    ↓
_format_surveillance_response()
    → Generates human-friendly response
    ↓
Ironcliw speaks result
```

---

## Troubleshooting

### "God Mode surveillance is not available"
→ VisualMonitorAgent failed to initialize
→ Check: `PYTHONPATH` includes backend directory
→ Check: Ferrari Engine dependencies installed

### "I couldn't find any Chrome windows to monitor"
→ No Chrome windows detected by Yabai
→ Make sure: Chrome is actually open
→ Make sure: Yabai is running (`yabai -m query --windows`)

### "Surveillance timeout after 300 seconds"
→ Trigger text was never detected
→ Check: OCR can read the text on screen (try larger font)
→ Check: Trigger text matches exactly (case-sensitive)

### No voice response
→ Check: Daniel voice installed (`say -v Daniel "test"`)
→ Check: System audio not muted
→ Check: Speech state manager not suppressing

---

## File Locations

### Production Code
- `backend/voice/intelligent_command_handler.py` - Main implementation
- `backend/neural_mesh/agents/visual_monitor_agent.py` - God Mode agent
- `backend/vision/multi_space_window_detector.py` - Multi-space discovery

### Tests
- `test_voice_god_mode.py` - Full integration test with TTS
- `test_voice_parsing_only.py` - Quick parsing test (no dependencies)
- `test_stereo_vision_realtime.py` - Visual test with bouncing balls

### Documentation
- `VOICE_GOD_MODE_INTEGRATION.md` - Complete technical documentation
- `QUICK_START_VOICE_GOD_MODE.md` - This file

---

## Next Steps

1. **Test Basic Parsing:**
   ```bash
   python3 test_voice_parsing_only.py
   ```

2. **Test Full Integration:**
   - Open bouncing ball HTML files
   - Say: "Ironcliw, watch all Chrome windows for BOUNCE COUNT"
   - Verify: Ferrari Engines spawn and detect text

3. **Use in Daily Workflow:**
   - "Ironcliw, watch Terminal for Build Complete"
   - "Ironcliw, monitor Chrome for Error"
   - "Ironcliw, notify me when tests say PASSED"

---

## Pro Tips

### 🎯 Best Practices
- Use **large, high-contrast text** for best OCR accuracy
- Be **specific** with trigger text ("Error:" not "error")
- Use **all_spaces** when you don't know which space has the window
- Add **timeouts** for long-running processes

### 🚀 Advanced Usage
- Chain commands: "Watch for 'DONE', then run tests"
- Multi-trigger: "Watch for 'Error' OR 'Warning'"
- Regex patterns: "Watch for any line matching '^ERROR:.*'"
- Custom actions: Configure what happens when trigger detected

---

## What's New vs. Before

### Before "The Final Wire"
❌ Could only manually run `test_god_mode.py`
❌ No voice control for surveillance
❌ Manual window selection required
❌ No natural language interface

### After "The Final Wire"
✅ Voice-activated God Mode: "Watch all Chrome for Error"
✅ Natural language parsing: Multiple phrasings supported
✅ Automatic window discovery: Ironcliw finds windows himself
✅ Parallel monitoring: N windows simultaneously
✅ Auto space switching: Jumps to window with trigger
✅ Voice feedback: Human-like responses throughout

---

🔌 **THE FINAL WIRE IS COMPLETE** 🔌

Now go ahead and say:
```
"Ironcliw, watch all Chrome windows for bouncing ball"
```
