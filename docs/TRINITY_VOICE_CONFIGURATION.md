# Trinity Voice Coordinator - Environment Configuration Guide

**Version:** 1.0
**Author:** Ironcliw Trinity System
**Last Updated:** 2025-01-10
**Ironcliw Voice:** ⭐ **UK Daniel (Signature)** ⭐

## Overview

The Trinity Voice Coordinator is a **zero-hardcoding, ultra-robust, cross-repo voice announcement system** that provides intelligent voice feedback across Ironcliw (Body), Ironcliw-Prime (Brain), and Reactor-Core (Self-Improvement).

**All configuration is driven by environment variables, allowing complete customization without code changes.**

---

## ⭐ Ironcliw's Signature Voice: UK Daniel ⭐

**The Trinity Voice System uses UK Daniel as Ironcliw's canonical voice.**

- **Name:** Daniel (United Kingdom)
- **Gender:** Male
- **Accent:** British English (UK)
- **Tone:** Professional, deep, authoritative
- **Why:** This is Ironcliw's signature voice across all contexts

**Auto-Detection:**
The system automatically detects and uses UK Daniel if installed. If not found, you'll see:
```
⚠️ UK Daniel voice not found!
Install it via: System Settings → Accessibility → Spoken Content →
System Voice → Download 'Daniel (United Kingdom)'
```

**Verify UK Daniel is installed:**
```bash
say -v "?" | grep -i daniel
```

Expected output:
```
Daniel              en_GB    # Daniel from United Kingdom
```

**Install UK Daniel (if missing):**
1. Open **System Settings** (macOS)
2. Go to **Accessibility** → **Spoken Content**
3. Click **System Voice** → **Manage Voices...**
4. Find **Daniel (United Kingdom)**
5. Click **Download**
6. Wait for completion
7. Restart Ironcliw

---

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                  Trinity Voice Coordinator                         │
│                  (Ironcliw Body - Central Hub)                       │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Multi-Engine TTS Fallback Chain:                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │ MacOS Say   │ ─▶ │  pyttsx3    │ ─▶ │  Edge TTS   │          │
│  │ (Primary)   │    │ (Fallback)  │    │ (Cloud)     │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│                                                                    │
│  Voice Personalities (Context-Aware):                             │
│  • STARTUP  - Formal, professional (system initialization)        │
│  • NARRATOR - Clear, informative (progress updates)               │
│  • RUNTIME  - Friendly, conversational (normal operations)        │
│  • ALERT    - Urgent, attention-grabbing (errors, warnings)       │
│  • SUCCESS  - Celebratory, upbeat (completions)                   │
│  • TRINITY  - Cross-repo coordination (J-Prime, Reactor)          │
│                                                                    │
│  Intelligent Queue:                                                │
│  • Priority-based (CRITICAL → HIGH → NORMAL → LOW → BACKGROUND)  │
│  • Deduplication (hash-based, 30s window)                         │
│  • Rate limiting (5 announcements per 10s)                        │
│  • Coalescing (batch similar messages)                            │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Environment Variables Reference

### **Ironcliw Body (Trinity Voice Coordinator)**

#### General Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `Ironcliw_VOICE_ENABLED` | boolean | `true` | Master enable/disable for all voice announcements |
| `Ironcliw_VOICE_MAX_QUEUE_SIZE` | integer | `100` | Maximum announcements queued before dropping |
| `Ironcliw_VOICE_RATE_LIMIT_WINDOW` | integer | `10` | Rate limiting window in seconds |
| `Ironcliw_VOICE_RATE_LIMIT_MAX` | integer | `5` | Max announcements within rate limit window |
| `Ironcliw_VOICE_DEDUP_WINDOW` | integer | `30` | Deduplication time window in seconds |

#### Voice Personality Configuration (by Context)

**Startup Voice:**
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `Ironcliw_STARTUP_VOICE_NAME` | string | Auto-detected | Voice name for startup context |
| `Ironcliw_STARTUP_VOICE_RATE` | integer | `175` | Speech rate (words per minute) |
| `Ironcliw_STARTUP_VOICE_PITCH` | integer | `50` | Voice pitch (0-100) |
| `Ironcliw_STARTUP_VOICE_VOLUME` | float | `0.9` | Volume (0.0-1.0) |

**Narrator Voice:**
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `Ironcliw_NARRATOR_VOICE_NAME` | string | Auto-detected | Voice for progress narration |
| `Ironcliw_NARRATOR_VOICE_RATE` | integer | `180` | Speech rate |
| `Ironcliw_NARRATOR_VOICE_PITCH` | integer | `50` | Voice pitch |
| `Ironcliw_NARRATOR_VOICE_VOLUME` | float | `0.85` | Volume |

**Runtime Voice:**
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `Ironcliw_RUNTIME_VOICE_NAME` | string | Auto-detected | Voice for normal operations |
| `Ironcliw_RUNTIME_VOICE_RATE` | integer | `170` | Speech rate |
| `Ironcliw_RUNTIME_VOICE_PITCH` | integer | `55` | Voice pitch |
| `Ironcliw_RUNTIME_VOICE_VOLUME` | float | `0.8` | Volume |

**Alert Voice:**
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `Ironcliw_ALERT_VOICE_NAME` | string | Auto-detected | Voice for errors/warnings |
| `Ironcliw_ALERT_VOICE_RATE` | integer | `190` | Speech rate (faster for urgency) |
| `Ironcliw_ALERT_VOICE_PITCH` | integer | `60` | Voice pitch |
| `Ironcliw_ALERT_VOICE_VOLUME` | float | `1.0` | Volume (full for alerts) |

**Success Voice:**
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `Ironcliw_SUCCESS_VOICE_NAME` | string | Auto-detected | Voice for success messages |
| `Ironcliw_SUCCESS_VOICE_RATE` | integer | `165` | Speech rate |
| `Ironcliw_SUCCESS_VOICE_PITCH` | integer | `55` | Voice pitch |
| `Ironcliw_SUCCESS_VOICE_VOLUME` | float | `0.9` | Volume |

**Trinity Voice (Cross-Repo):**
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `Ironcliw_TRINITY_VOICE_NAME` | string | Auto-detected | Voice for cross-repo events |
| `Ironcliw_TRINITY_VOICE_RATE` | integer | `175` | Speech rate |
| `Ironcliw_TRINITY_VOICE_PITCH` | integer | `50` | Voice pitch |
| `Ironcliw_TRINITY_VOICE_VOLUME` | float | `0.9` | Volume |

---

### **Ironcliw-Prime (Brain) Voice Configuration**

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `JPRIME_VOICE_ENABLED` | boolean | `true` | Enable J-Prime voice announcements |
| `JPRIME_VOICE_MODEL_LOAD` | boolean | `true` | Announce model load success/failure |
| `JPRIME_VOICE_TIER_ROUTING` | boolean | `false` | Announce tier routing (local vs cloud) |
| `JPRIME_VOICE_CLOUD_FALLBACK` | boolean | `true` | Announce fallback to cloud |
| `JPRIME_VOICE_HEALTH` | boolean | `true` | Announce health status changes |
| `JPRIME_VOICE_SOURCE` | string | `jarvis_prime` | Source identifier for tracking |
| `Ironcliw_BODY_PATH` | path | Auto-detected | Path to Ironcliw body repo (for Trinity import) |

---

### **Reactor-Core (Self-Improvement) Voice Configuration**

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `REACTOR_VOICE_ENABLED` | boolean | `true` | Enable Reactor Core voice announcements |
| `REACTOR_VOICE_TRAINING_START` | boolean | `true` | Announce training start |
| `REACTOR_VOICE_TRAINING_COMPLETE` | boolean | `true` | Announce training completion |
| `REACTOR_VOICE_EXPORT` | boolean | `true` | Announce model export (GGUF) |
| `REACTOR_VOICE_DEPLOYMENT` | boolean | `true` | Announce model deployment to J-Prime |
| `REACTOR_VOICE_FAILURES` | boolean | `true` | Announce training failures |
| `REACTOR_VOICE_SOURCE` | string | `reactor_core` | Source identifier for tracking |
| `Ironcliw_BODY_PATH` | path | Auto-detected | Path to Ironcliw body repo (for Trinity import) |

---

## Configuration Examples

### Example 1: Production Configuration (Minimal Announcements)

**`.env` file:**
```bash
# Ironcliw Body - Trinity Voice Coordinator
Ironcliw_VOICE_ENABLED=true
Ironcliw_STARTUP_VOICE_RATE=180
Ironcliw_ALERT_VOICE_VOLUME=1.0

# Ironcliw-Prime
JPRIME_VOICE_ENABLED=true
JPRIME_VOICE_MODEL_LOAD=true
JPRIME_VOICE_TIER_ROUTING=false  # Don't announce every routing decision
JPRIME_VOICE_CLOUD_FALLBACK=true
JPRIME_VOICE_HEALTH=true

# Reactor-Core
REACTOR_VOICE_ENABLED=true
REACTOR_VOICE_TRAINING_START=true
REACTOR_VOICE_TRAINING_COMPLETE=true
REACTOR_VOICE_EXPORT=false  # Skip export announcements
REACTOR_VOICE_DEPLOYMENT=true
```

### Example 2: Development Configuration (Verbose)

```bash
# Ironcliw Body
Ironcliw_VOICE_ENABLED=true
Ironcliw_VOICE_RATE_LIMIT_MAX=10  # Allow more frequent announcements
Ironcliw_VOICE_DEDUP_WINDOW=10    # Shorter dedup window for testing

# Ironcliw-Prime
JPRIME_VOICE_ENABLED=true
JPRIME_VOICE_MODEL_LOAD=true
JPRIME_VOICE_TIER_ROUTING=true  # Announce all routing decisions
JPRIME_VOICE_CLOUD_FALLBACK=true
JPRIME_VOICE_HEALTH=true

# Reactor-Core
REACTOR_VOICE_ENABLED=true
REACTOR_VOICE_TRAINING_START=true
REACTOR_VOICE_TRAINING_COMPLETE=true
REACTOR_VOICE_EXPORT=true       # Announce everything
REACTOR_VOICE_DEPLOYMENT=true
REACTOR_VOICE_FAILURES=true
```

### Example 3: Silent Mode (All Disabled)

```bash
# Completely disable all voice announcements
Ironcliw_VOICE_ENABLED=false
JPRIME_VOICE_ENABLED=false
REACTOR_VOICE_ENABLED=false
```

### Example 4: Custom Voice Selection (macOS)

```bash
# Use specific macOS voices for different contexts
Ironcliw_STARTUP_VOICE_NAME="Daniel"    # UK male (professional)
Ironcliw_NARRATOR_VOICE_NAME="Samantha" # US female (clear)
Ironcliw_RUNTIME_VOICE_NAME="Alex"      # US male (friendly)
Ironcliw_ALERT_VOICE_NAME="Victoria"    # UK female (urgent)
Ironcliw_SUCCESS_VOICE_NAME="Tom"       # US male (upbeat)
Ironcliw_TRINITY_VOICE_NAME="Daniel"    # UK male (formal)
```

**Available macOS Voices:**
Run `say -v "?"` to list all available voices. Popular choices:
- **Daniel** (UK Male) - Professional, deep
- **Samantha** (US Female) - Clear, friendly
- **Alex** (US Male) - Natural, conversational
- **Victoria** (UK Female) - Urgent, attention-grabbing
- **Tom** (US Male) - Upbeat, enthusiastic
- **Fiona** (UK Female) - Formal, clear

---

## API Endpoints

### Get Trinity Voice Status

**Endpoint:** `GET /api/trinity-voice/status`

**Description:** Get comprehensive status and metrics from Trinity Voice Coordinator.

**Response:**
```json
{
  "status": "ok",
  "voice_coordinator": {
    "running": true,
    "queue_size": 3,
    "active_engines": ["MacOSSayEngine", "Pyttsx3Engine"],
    "metrics": {
      "total_announcements": 127,
      "successful_announcements": 125,
      "failed_announcements": 2,
      "deduplicated_announcements": 8,
      "dropped_announcements": 1
    },
    "engines": [
      {
        "name": "MacOSSayEngine",
        "available": true,
        "health_score": 0.99,
        "success_count": 120,
        "failure_count": 1
      },
      {
        "name": "Pyttsx3Engine",
        "available": true,
        "health_score": 1.0,
        "success_count": 5,
        "failure_count": 0
      }
    ]
  },
  "timestamp": "2025-01-10T14:32:11.234Z"
}
```

### Test Trinity Voice

**Endpoint:** `POST /api/trinity-voice/test`

**Description:** Send a test announcement to verify voice system is working.

**Response:**
```json
{
  "status": "ok",
  "test_result": "success",
  "message": "Test announcement queued",
  "timestamp": "2025-01-10T14:33:45.678Z"
}
```

---

## Voice Auto-Detection

If voice names are not specified, Trinity Voice Coordinator auto-detects the best available voice:

### macOS
1. Checks for "Daniel" (UK Male, professional)
2. Falls back to "Alex" (US Male, default)
3. Falls back to "Samantha" (US Female)
4. Falls back to first available voice

### Linux/Windows (pyttsx3)
1. Uses system default voice
2. Falls back to first available voice

### Cloud (Edge TTS)
1. Uses "en-US-GuyNeural" (US Male, natural)
2. Falls back to "en-US-JennyNeural" (US Female)

---

## Monitoring and Metrics

### Real-Time Monitoring

Access the Trinity Voice status API to monitor:
- Queue depth
- Success/failure rates
- Engine health scores
- Recent announcements

```bash
curl http://localhost:8010/api/trinity-voice/status | jq
```

### Example Healthy Output

```json
{
  "voice_coordinator": {
    "running": true,
    "queue_size": 0,
    "active_engines": ["MacOSSayEngine"],
    "metrics": {
      "total_announcements": 1247,
      "successful_announcements": 1245,
      "failed_announcements": 2,
      "success_rate": 0.9984
    }
  }
}
```

### Example Degraded Output

```json
{
  "voice_coordinator": {
    "running": true,
    "queue_size": 12,
    "active_engines": ["Pyttsx3Engine"],  // Fell back from MacOS Say
    "metrics": {
      "total_announcements": 532,
      "successful_announcements": 498,
      "failed_announcements": 34,
      "success_rate": 0.9361,
      "engines": [
        {
          "name": "MacOSSayEngine",
          "available": false,  // Primary engine unavailable
          "health_score": 0.0
        },
        {
          "name": "Pyttsx3Engine",
          "available": true,   // Using fallback
          "health_score": 0.94
        }
      ]
    }
  }
}
```

---

## Troubleshooting

### Voice Not Working

**1. Check Trinity Coordinator Status:**
```bash
curl http://localhost:8010/api/trinity-voice/status
```

**2. Test Voice Manually:**
```bash
curl -X POST http://localhost:8010/api/trinity-voice/test
```

**3. Check Environment Variables:**
```bash
echo $Ironcliw_VOICE_ENABLED
echo $JPRIME_VOICE_ENABLED
echo $REACTOR_VOICE_ENABLED
```

**4. Check Logs:**
```bash
tail -f ~/.jarvis/logs/voice_coordinator.log
```

### Common Issues

#### Issue: "Trinity Voice Coordinator not available"
- **Cause:** Trinity Voice Coordinator not imported correctly
- **Fix:** Ensure `Ironcliw_BODY_PATH` environment variable points to Ironcliw body repo
- **Command:**
  ```bash
  export Ironcliw_BODY_PATH=/path/to/Ironcliw-AI-Agent
  ```

#### Issue: "All TTS engines failed"
- **Cause:** No TTS engines available
- **Fix:** Install at least one TTS engine:
  ```bash
  # macOS (built-in `say` command)
  which say  # Should return /usr/bin/say

  # Cross-platform (pyttsx3)
  pip install pyttsx3

  # Cloud (Edge TTS)
  pip install edge-tts
  ```

#### Issue: Voice announcements too frequent
- **Cause:** Rate limiting disabled or set too high
- **Fix:** Adjust rate limiting:
  ```bash
  export Ironcliw_VOICE_RATE_LIMIT_MAX=3
  export Ironcliw_VOICE_RATE_LIMIT_WINDOW=15
  ```

#### Issue: Duplicate announcements
- **Cause:** Deduplication window too short
- **Fix:** Increase deduplication window:
  ```bash
  export Ironcliw_VOICE_DEDUP_WINDOW=60  # 60 seconds
  ```

---

## Advanced Features

### Priority Announcement

Critical announcements (errors, alerts) will interrupt lower-priority announcements:

```python
from backend.core.trinity_voice_coordinator import announce, VoiceContext, VoicePriority

# This will interrupt any NORMAL priority announcements
await announce(
    message="Critical error detected!",
    context=VoiceContext.ALERT,
    priority=VoicePriority.CRITICAL,
    source="error_handler"
)
```

### Cross-Repo Event Tracking

Use the `metadata` field to track events across repos:

```python
await announce(
    message="Model training complete",
    context=VoiceContext.SUCCESS,
    priority=VoicePriority.HIGH,
    source="reactor_core",
    metadata={
        "repo": "reactor-core",
        "event": "training_complete",
        "model": "TinyLlama-1.1B",
        "steps": 1000,
        "loss": 0.245
    }
)
```

### Engine Health Monitoring

Engines are automatically ranked by health score (success/failure ratio). The healthiest engine is used first:

```json
{
  "engines": [
    {
      "name": "MacOSSayEngine",
      "health_score": 0.99,  // 99% success rate
      "priority": 1           // Will be used first
    },
    {
      "name": "Pyttsx3Engine",
      "health_score": 0.95,  // 95% success rate
      "priority": 2           // Fallback
    }
  ]
}
```

---

## Performance Tuning

### Reduce Latency

```bash
# Use faster speech rate
export Ironcliw_STARTUP_VOICE_RATE=200
export Ironcliw_NARRATOR_VOICE_RATE=210

# Reduce queue processing delay
export Ironcliw_VOICE_QUEUE_POLL_INTERVAL=0.05
```

### Reduce Verbosity

```bash
# Disable low-priority announcements
export JPRIME_VOICE_TIER_ROUTING=false
export REACTOR_VOICE_EXPORT=false

# Increase rate limiting
export Ironcliw_VOICE_RATE_LIMIT_MAX=2
export Ironcliw_VOICE_RATE_LIMIT_WINDOW=20
```

### Increase Reliability

```bash
# Allow more retries
export Ironcliw_VOICE_MAX_RETRIES=5

# Increase queue size
export Ironcliw_VOICE_MAX_QUEUE_SIZE=200

# Longer timeout for slow TTS engines
export Ironcliw_VOICE_TTS_TIMEOUT=60
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01-10 | Initial Trinity Voice Coordinator release |

---

## Support

For issues or questions:
1. Check the API status: `GET /api/trinity-voice/status`
2. Review logs: `~/.jarvis/logs/voice_coordinator.log`
3. Test manually: `POST /api/trinity-voice/test`
4. Open an issue on GitHub

---

**End of Trinity Voice Coordinator Configuration Guide**
