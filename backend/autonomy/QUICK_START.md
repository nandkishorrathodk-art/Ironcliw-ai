# Ironcliw Autonomous System - Quick Start Guide

## ✅ System Status: WORKING

The Ironcliw Autonomous System has been successfully integrated and tested. Here's how to use it:

## 🚀 Quick Test

```bash
# Run the simple test to verify everything works
cd backend
python3 test_autonomous_simple.py
```

## 🎤 Voice Commands

Enable autonomous mode through Ironcliw:
```
"Hey Ironcliw, enable autonomous mode"
"Hey Ironcliw, activate automatic mode"
```

Check status:
```
"Hey Ironcliw, what's your autonomous status?"
"Hey Ironcliw, show permission statistics"
```

Manage actions:
```
"Hey Ironcliw, rollback last action"
"Hey Ironcliw, disable autonomous mode"
```

## 🤖 What It Does

When enabled, Ironcliw will autonomously:

1. **Monitor Your Workspace** - Continuously analyze open windows
2. **Detect Actionable Situations**:
   - Discord (5 new messages) → Handle notifications
   - Meeting in 3 minutes → Prepare workspace
   - 30+ windows open → Suggest organization
   - Sensitive content visible → Security protection

3. **Check Context** - Won't interrupt during meetings or focus time
4. **Request Permission** - For new actions until it learns your preferences
5. **Execute Actions** - Handle notifications, organize windows, prepare for meetings
6. **Learn From Feedback** - Improve decisions based on your approvals/denials

## 📊 Key Components

- **Decision Engine** - Makes intelligent decisions (no hardcoding!)
- **Permission Manager** - Learns what you approve/deny
- **Context Engine** - Understands when to act
- **Action Executor** - Safely executes actions with rollback

## 🔧 Configuration

The system uses intelligent defaults but you can adjust:
- Notification threshold: 3+ notifications trigger action
- Meeting prep time: 5 minutes before meeting
- Quiet hours: 10 PM - 8 AM (reduced actions)
- Confidence thresholds: 85% for auto-approval

## 🎯 Example Scenarios

### Scenario 1: Message Management
```
Ironcliw detects: "Discord (12 new messages)"
Context: You're not in a meeting, interruption score is good
Permission: First time asks, learns your preference
Action: Focuses Discord or marks as read based on past behavior
```

### Scenario 2: Meeting Preparation
```
Ironcliw detects: "Team Standup starts in 3 minutes"
Context: High priority, time-sensitive
Action: Hides 1Password, banking apps, opens Zoom
Rollback: Can restore previous state after meeting
```

### Scenario 3: Workspace Organization
```
Ironcliw detects: 35 windows open across 15 apps
Context: You're available, not focused
Action: Groups windows by project, minimizes distractions
Learning: Remembers your preferred layouts
```

## 🛡️ Safety Features

- Always asks permission for security-related actions
- Respects quiet hours and focus time
- Limits on bulk actions (max 5 windows closed at once)
- Full rollback capability for recent actions
- Learns from your decisions to improve over time

## 🐛 Troubleshooting

If Ironcliw isn't taking autonomous actions:
1. Ensure autonomous mode is enabled
2. Check if monitoring is active (workspace intelligence must be on)
3. Review context - might be waiting for better timing
4. Check permission stats - might need more learning

## 📈 Next Steps

1. Enable autonomous mode and use Ironcliw normally
2. When prompted for permissions, provide clear yes/no responses
3. After ~5 similar decisions, Ironcliw will start auto-approving
4. Monitor execution stats to see what's being automated
5. Use rollback if any action is unwanted

The more you use it, the smarter it gets!