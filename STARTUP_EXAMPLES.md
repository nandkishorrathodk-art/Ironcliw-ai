# 🚀 Ironcliw Startup Examples - Goal Inference Integration

## Quick Reference Guide

All Goal Inference configuration is now **fully automatic and intelligent**!

**Just run:** `python start_system.py` - No menu, no questions, pure intelligence! 🤖

---

## 🤖 Method 1: Fully Automatic (Recommended)

### Scenario 1: First Time User
**Goal**: Get Ironcliw running with zero configuration

```bash
python start_system.py
```

**What happens:**
- 🤖 **Auto-detects** this is your first run (no database)
- 🎯 **Automatically selects** 'learning' preset for fast adaptation
- ⚠️ **Automation disabled** - you're new, Ironcliw needs to learn first
- ✅ Auto-creates configuration
- ✅ Initializes SQLite + ChromaDB databases
- ✅ Starts Ironcliw with all 10 components + Goal Inference

**Startup Output:**
```
🎯 Auto-detected Goal Inference Preset: learning
   → First run detected, using 'learning' preset for fast adaptation
   (Override with --goal-preset or Ironcliw_GOAL_PRESET environment variable)
⚠️ Goal Inference Automation: DISABLED

🚀 Starting Ironcliw...
✅ Goal Inference + Learning Database loaded
   • Goal Confidence: 0.75
   • Proactive Suggestions: True
   • Automation: False
   • Learning: True (rapid pattern learning enabled)
```

**Best for:**
- First-time users
- Zero configuration needed
- Automatic adaptation to your workflow

---

## 🎯 Method 2: Manual Override (Advanced Users)

When you specify `--goal-preset`, auto-detection is skipped and your choice is used!

### Scenario 2: Force Learning Mode
**Goal**: Manually force learning preset (override auto-detection)

```bash
python start_system.py --goal-preset learning
```

**What happens:**
- ⏩ **Skips auto-detection** (manual preset specified)
- 🎯 Forces `learning` preset regardless of experience
- 🔥 Lowers pattern threshold to 2 (vs 3)
- 📈 Increases pattern boost to 0.10 (vs 0.05)
- 🚀 Faster adaptation to your routines

**Best for:**
- First week of using Ironcliw
- Teaching new routines
- Onboarding new users

---

### Scenario 3: Force Aggressive + Automation
**Goal**: Maximum proactivity and automation (override auto-detection)

```bash
python start_system.py --goal-preset aggressive --enable-automation
```

**What happens:**
- ⏩ **Skips auto-detection** (manual preset + automation specified)
- ⚡ Forces aggressive mode regardless of experience
- 🤖 Enables automatic execution of high-confidence actions
- 🔥 Highly proactive suggestions
- ⚠️ **WARNING**: Will auto-execute actions with >95% confidence

**Best for:**
- Power users who trust Ironcliw
- Highly repetitive workflows
- Maximum productivity

---

### Scenario 4: Conservative Mode
**Goal**: Only the most confident predictions (skip menu)

```bash
python start_system.py --goal-preset conservative
```

**What happens:**
- ⏩ **Skips interactive menu** (preset specified)
- 🛡️ High confidence thresholds (0.85 vs 0.75)
- 🐌 Slower learning (0.02 boost vs 0.05)
- 📊 Only suggests when very certain
- ❌ Automation disabled by default

**Best for:**
- Important presentations
- High-stakes work
- Minimal interruptions needed

---

### Scenario 5: Maximum Performance
**Goal**: Fastest possible responses (skip menu)

```bash
python start_system.py --goal-preset performance
```

**What happens:**
- ⏩ **Skips interactive menu** (preset specified)
- 💾 Larger cache (200 entries vs 100)
- ⏱️ Longer TTL (600s vs 300s)
- ⚡ Parallel processing enabled
- 🚀 Resource preloading enabled

**Best for:**
- Powerful machines (16GB+ RAM)
- Speed-critical workflows
- Heavy Ironcliw usage

---

## 🎯 Method 3: Advanced Command-Line Combinations

### Example 1: Learning Mode + No Browser
```bash
python start_system.py --goal-preset learning --no-browser
```

### Example 2: Aggressive + Automation + Verbose
```bash
python start_system.py --goal-preset aggressive --enable-automation --verbose
```

### Example 3: Backend Only + Performance Preset
```bash
python start_system.py --backend-only --goal-preset performance --port 8000
```

### Example 4: Conservative + Debug Mode
```bash
python start_system.py --goal-preset conservative --debug
```

---

## 🎯 Method 4: Environment Variables (Advanced)

### Scenario: Set Default Preset
**Goal**: Always use same preset without typing it

Add to `~/.bashrc` or `~/.zshrc`:
```bash
export Ironcliw_GOAL_PRESET=aggressive
export Ironcliw_GOAL_AUTOMATION=true
```

Then just run:
```bash
python start_system.py
```

**What happens:**
- ⏩ **Skips interactive menu** (environment variable set)
- ✅ Automatically uses `aggressive` preset every time
- ✅ Automation enabled by default
- ✅ No need to specify flags

---

### Scenario: One-Time Override
**Goal**: Use different preset for this session only

```bash
Ironcliw_GOAL_PRESET=learning python start_system.py
```

Or override both preset and automation:
```bash
Ironcliw_GOAL_PRESET=conservative Ironcliw_GOAL_AUTOMATION=false python start_system.py
```

**What happens:**
- ⏩ **Skips interactive menu** (environment variable set)
- 🔧 Uses specified preset for this session only
- 📝 Doesn't modify your shell config

---

## 📊 All Command Options

### start_system.py - Unified Startup (All-in-One)

```bash
python start_system.py [flags]
```

**Goal Inference Flags:**
```bash
--goal-preset {aggressive|balanced|conservative|learning|performance}
--enable-automation    # Enable automatic actions
--disable-automation   # Disable automatic actions
```

**Standard Flags:**
```bash
--no-browser          # Don't open browser
--backend-only        # Start backend only
--frontend-only       # Start frontend only
--verbose             # Verbose logging
--debug               # Debug mode
--port 8000           # Backend port
--restart             # Restart Ironcliw
```

---

## 🎓 Best Practices

### Week 1: Learning Mode
```bash
./start_jarvis.sh learning
```
- Let Ironcliw observe your patterns
- Review suggestions but don't enable automation yet
- Check metrics weekly

### Week 2-4: Balanced with Selective Automation
```bash
./start_jarvis.sh balanced
```
- Switch to balanced mode
- Keep automation disabled
- Ironcliw has learned your patterns now

### Month 2+: Aggressive or Custom
```bash
./start_jarvis.sh aggressive --enable-automation
```
- Enable automation for trusted actions
- Ironcliw is now reliable and personalized
- Fine-tune with `configure_goal_inference.py` if needed

---

## 🔍 Verification

After starting Ironcliw, check the startup logs:

```
✅ Goal Inference + Learning Database loaded
   • Goal Confidence: 0.65              ← From preset
   • Proactive Suggestions: True        ← Enabled
   • Automation: True                   ← From --enable-automation
   • Learning: True                     ← Always on
   • Database Cache: 200 entries        ← From performance preset
   • Previous session: 45 goals, 67 actions  ← Your history
   • Success rate: 89.2%                ← Learning effectiveness
```

---

## 🆘 Troubleshooting

### Issue: Wrong preset applied
**Solution:**
```bash
# Check environment variable
echo $Ironcliw_GOAL_PRESET

# Clear it if set incorrectly
unset Ironcliw_GOAL_PRESET

# Or override for this session
Ironcliw_GOAL_PRESET=balanced ./start_jarvis.sh
```

### Issue: Automation enabled when I don't want it
**Solution:**
```bash
# Explicitly disable
./start_jarvis.sh balanced --disable-automation

# Or clear environment variable
unset Ironcliw_GOAL_AUTOMATION
```

### Issue: Config not updating
**Solution:**
```bash
# Delete config to recreate with new settings
rm backend/config/integration_config.json

# Restart with desired preset
./start_jarvis.sh learning
```

---

## 📝 Quick Comparison

| Scenario | Command | Learning Speed | Suggestions | Automation | Best For |
|----------|---------|---------------|-------------|------------|----------|
| **Interactive** | `python start_system.py` | Configurable | Configurable | Configurable | First time / Flexibility |
| **Quick Learn** | `python start_system.py --goal-preset learning` | Fast ⚡ | Many | ❌ | First week |
| **Daily Use** | `python start_system.py --goal-preset balanced` | Normal | Balanced | ❌ | Most users |
| **Power User** | `python start_system.py --goal-preset aggressive --enable-automation` | Fast ⚡ | Many | ✅ | Trusted workflows |
| **Important Work** | `python start_system.py --goal-preset conservative` | Slow 🐌 | Few | ❌ | High stakes |
| **Speed Demon** | `python start_system.py --goal-preset performance` | Normal | Moderate | ❌ | Performance critical |

---

## 🎯 Summary

**🌟 Simplest way to start (Interactive Menu):**
```bash
python start_system.py
# Shows menu → Select preset → Choose automation → Done!
```

**⚡ Quick start with preset (Skip Menu):**
```bash
python start_system.py --goal-preset learning
```

**🚀 Maximum automation (Skip Menu):**
```bash
python start_system.py --goal-preset aggressive --enable-automation
```

**🔧 Production ready (Environment Variables):**
```bash
# Add to .bashrc/.zshrc
export Ironcliw_GOAL_PRESET=balanced

# Then just:
python start_system.py
```

**✨ Key Features:**
- ✅ **Interactive menu** when run without arguments
- ✅ **Skip menu** with `--goal-preset` flag
- ✅ **Environment variables** for permanent defaults
- ✅ **Auto-configuration** on first run
- ✅ **No manual setup** required

Everything is **fully unified into start_system.py** - one script to rule them all! 🎉
