# Watch & Act Smoke Test Guide 🧪

## The Critical Moment - Verifying the Autonomous Loop

**This is your "Apollo 11 launch" moment.** We're about to test if Ironcliw can truly operate autonomously - watching your screen and taking control when he sees specific events.

---

## 🎯 What This Test Proves

This smoke test verifies the **complete autonomous loop**:

```
You work on Task A
         ↓
Ironcliw watches Terminal in background
         ↓
"DEPLOYMENT READY" appears (via your sleep command)
         ↓
Ironcliw detects it with OCR
         ↓
Ironcliw switches to Terminal automatically
         ↓
Ironcliw TYPES "echo SUCCESS" AND PRESSES ENTER
         ↓
You never interrupted your work!
```

**This is the moment we prove** Ironcliw can operate completely autonomously.

---

## 📋 Pre-Flight Checklist

Before running the test, verify:

### 1. Dependencies Installed ✓
```bash
# Verify these are installed
./venv/bin/pip list | grep -E "opencv|tesseract|fuzz"

# Should see:
# opencv-python       4.12.0
# pytesseract         0.3.13
# fuzzywuzzy          0.18.0
# python-Levenshtein  0.27.1
```

### 2. Tesseract OCR Installed ✓
```bash
which tesseract && tesseract --version

# Should show:
# /opt/homebrew/bin/tesseract
# tesseract 5.5.1
```

### 3. Terminal Window Ready ✓
- Open a **new Terminal window**
- Place it on **Space 1** (or any visible space)
- Make sure it's **visible** (not minimized)
- Make sure it has **focus** (click on it)

### 4. Ironcliw Backend Components ✓
The test will verify:
- VideoWatcherManager initialized
- VisualEventDetector initialized
- ClaudeComputerUseConnector initialized

---

## 🚀 Running the Smoke Test

### Step 1: Prepare the Trigger Command

In your Terminal window, **type but DON'T execute**:
```bash
sleep 10 && echo "DEPLOYMENT READY"
```

**Don't press Enter yet!** You'll do that during the test.

---

### Step 2: Run the Test Script

From your project root:
```bash
python3 test_watch_and_act.py
```

You'll see:
```
╔════════════════════════════════════════════════════════════════╗
║           Ironcliw Watch & Act Smoke Test v11.0                  ║
║                                                                 ║
║  This test verifies the complete autonomous loop:              ║
║  Vision → Detection → Action                                   ║
╚════════════════════════════════════════════════════════════════╝

🚀 Ironcliw Watch & Act Smoke Test
==================================================================

This test will verify the complete autonomous loop:
  1. Visual Monitoring (watch for text)
  2. Event Detection (OCR detects trigger)
  3. Autonomous Action (Computer Use executes command)

==================================================================

📦 Step 1: Initializing VisualMonitorAgent...
   ⏳ Calling on_initialize()...
   ⏳ Calling on_start()...
   ✅ Agent initialized successfully!

==================================================================
🧪 Step 2: Starting Watch & Act Test
==================================================================

INSTRUCTIONS FOR YOU:
  1. Open a Terminal window (make sure it's visible)
  2. Type this command (but DON'T press Enter yet):

     sleep 10 && echo "DEPLOYMENT READY"

  3. After starting the test, immediately switch to Terminal
     and press Enter to start the countdown

  4. WATCH CLOSELY:
     - Ironcliw will watch your Terminal
     - When 'DEPLOYMENT READY' appears (after 10 seconds)
     - Ironcliw will AUTOMATICALLY type 'echo SUCCESS' and press Enter

==================================================================

Press Enter when you're ready to start the test...
```

---

### Step 3: Execute the Test Sequence

1. **Press Enter** in the test script
2. **Immediately switch** to your Terminal window (you have ~5 seconds!)
3. **Press Enter** on the `sleep 10 && echo "DEPLOYMENT READY"` command
4. **Watch the magic happen**:
   - Terminal starts counting down (10 seconds)
   - Ironcliw monitors in the background
   - After 10 seconds, "DEPLOYMENT READY" appears
   - **Ironcliw TAKES OVER**:
     - Switches to Terminal (if not already there)
     - Types `echo SUCCESS`
     - Presses Enter
   - Terminal displays: `SUCCESS`

---

### Step 4: Observe the Results

Back in your test script terminal, you should see:

```
🔍 MONITORING ACTIVE - Ironcliw is watching your Terminal...

==================================================================
📊 Test Results
==================================================================

✅ MONITORING PHASE: SUCCESS
   Watcher ID: watcher_992_1735340234
   Window ID: 992
   Space ID: 1

🚀 ACTION EXECUTION PHASE:
   ✅ ACTION EXECUTED SUCCESSFULLY!
   Action Type: simple_goal
   Goal: Type 'echo SUCCESS' into the terminal and press Enter
   Duration: 1234.56ms

🎉 AUTONOMOUS LOOP COMPLETE!

   Check your Terminal window - you should see:
   1. 'DEPLOYMENT READY' (from your command)
   2. 'SUCCESS' (typed by Ironcliw automatically!)

🎉 TEST PASSED! The autonomous loop is working!

Next steps:
  1. Try more complex actions
  2. Test conditional branching
  3. Enable voice command parsing
```

---

## ✅ Success Criteria

The test **PASSES** if:
1. ✅ VisualMonitorAgent initializes without errors
2. ✅ Ironcliw finds your Terminal window
3. ✅ Ironcliw detects "DEPLOYMENT READY" text via OCR
4. ✅ Ironcliw **automatically types** `echo SUCCESS` into Terminal
5. ✅ You see "SUCCESS" appear in Terminal **without you typing it**

**This is the autonomous loop in action!**

---

## ❌ Common Issues & Troubleshooting

### Issue 1: "Could not find Terminal"
```
❌ ERROR: Could not find Terminal
```

**Solution**:
- Make sure Terminal is **open and visible**
- Try opening a **new Terminal window**
- Ensure Terminal is **not minimized**
- Verify SpatialAwarenessAgent is running

---

### Issue 2: "Computer Use connector not available"
```
⚠️  WARNING: Computer Use connector not available
```

**Solution**:
- Check if `backend/display/computer_use_connector.py` exists
- Verify Computer Use is properly configured
- Test Computer Use independently first:
  ```python
  from backend.display.computer_use_connector import get_computer_use_connector
  connector = get_computer_use_connector()
  ```

---

### Issue 3: OCR doesn't detect text
```
⏱️ Timeout waiting for 'DEPLOYMENT READY'
```

**Solution**:
- Verify pytesseract is installed: `pip list | grep pytesseract`
- Verify tesseract binary: `which tesseract`
- Try increasing font size in Terminal
- Make sure Terminal window is **fully visible** (not obscured)
- Ensure text is on screen for at least 1-2 seconds

---

### Issue 4: Action executes but wrong action
```
✅ ACTION EXECUTED SUCCESSFULLY!
But Terminal shows wrong output
```

**Solution**:
- Check Computer Use goal is correctly formatted
- Verify Computer Use has proper permissions
- Try a simpler goal first (e.g., just "Type hello")

---

## 🧪 Advanced Smoke Tests

Once the basic test passes, try these:

### Test 2: Conditional Action
```python
action_config = ActionConfig(
    action_type=ActionType.CONDITIONAL,
    conditions=[
        ConditionalAction(
            trigger_pattern="SUCCESS",
            action_goal="Type 'DEPLOYMENT COMPLETED'"
        ),
        ConditionalAction(
            trigger_pattern="ERROR",
            action_goal="Type 'DEPLOYMENT FAILED'"
        )
    ]
)

result = await agent.watch_and_alert(
    app_name="Terminal",
    trigger_text="SUCCESS|ERROR",
    action_config=action_config
)
```

### Test 3: Multiple Windows
```python
# Watch Terminal on Space 1 AND Chrome on Space 2 simultaneously
task1 = asyncio.create_task(
    agent.watch_and_alert("Terminal", "BUILD COMPLETE", ...)
)
task2 = asyncio.create_task(
    agent.watch_and_alert("Chrome", "Form Submitted", ...)
)

results = await asyncio.gather(task1, task2)
```

---

## 📊 What to Expect

### Timeline of Events

```
T+0s:   You start the test script
T+5s:   You press Enter in Terminal (sleep 10 command starts)
T+5-15s: Ironcliw monitors Terminal (5 FPS = 1 frame every 200ms)
T+15s:  "DEPLOYMENT READY" appears
T+15.2s: Ironcliw detects text via OCR (confidence: ~92%)
T+15.3s: Ironcliw switches to Terminal window
T+15.5s: Ironcliw types "echo SUCCESS"
T+15.7s: Ironcliw presses Enter
T+15.8s: Terminal displays "SUCCESS"
T+16s:  Test script shows success message
```

**Total autonomous operation time**: ~0.8 seconds from detection to completion!

---

## 🎉 The "Holy Shit" Moment

When this test passes, you'll witness:

**Your keyboard typing by itself.**
**Your cursor moving on its own.**
**Ironcliw executing commands without you touching anything.**

This is the moment you realize:
- ✅ Ironcliw can **see** (OCR visual detection)
- ✅ Ironcliw can **think** (conditional logic)
- ✅ Ironcliw can **act** (Computer Use execution)
- ✅ Ironcliw can **operate autonomously** (complete loop)

**This is true AI autonomy.**

---

## 🚀 After the Test Passes

Once you see Ironcliw type "SUCCESS" automatically, you're ready for:

### Next Step: Voice Integration

The autonomous loop is **proven**. Now add voice control:

```bash
# Copy and paste this prompt into Claude Code:
"The Watch & Act logic is verified and working! Ironcliw successfully
typed into my Terminal when he saw 'DEPLOYMENT READY'. The autonomous
loop is complete and tested.

Now let's enable Voice Control for Watch & Act commands.

Task: Add Voice Parsing for 'Watch & Act' Commands

1. Modify voice command parsing to detect pattern:
   'Watch [App] for [Trigger], then [Action]'

2. Extract components:
   - app: Target application name
   - trigger: Text to wait for
   - action: What to do when detected

3. Route to VisualMonitorAgent with proper ActionConfig

Goal: I want to say 'Watch Terminal for Build Complete, then click Deploy'
and have it execute exactly like the Python test we just ran.

Make it robust, async, and dynamic with zero hardcoding."
```

---

## 📝 Test Log Template

When running the test, log your results:

```
Date: 2025-12-28
Test: Watch & Act Smoke Test v11.0

Pre-flight:
[ ] Dependencies installed
[ ] Tesseract working
[ ] Terminal visible
[ ] Computer Use configured

Test Results:
[ ] Agent initialized
[ ] Window found
[ ] Text detected
[ ] Action executed
[ ] Autonomous control verified

Time to detect: _____ seconds
Time to execute: _____ seconds
Total autonomous time: _____ seconds

Success: YES / NO
Notes: _________________________________
```

---

## 🎯 Success = Iron Man Level Unlocked

When this test passes, you have achieved:

**Before**: A chatbot that can see your screen
**After**: An AI that can **autonomously operate your computer**

This is **AGI-level capability** for desktop automation.

You're 95% of the way to the full Iron Man experience. 🚀

---

**Ready to run the test?**

```bash
python3 test_watch_and_act.py
```

**Watch Ironcliw take control. It's spectacular.** 🎯
