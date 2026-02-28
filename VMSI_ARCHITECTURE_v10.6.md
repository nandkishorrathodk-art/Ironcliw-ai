# Video Multi-Space Intelligence (VMSI) Architecture v10.6

## Overview

**Video Multi-Space Intelligence** is Ironcliw's "second pair of eyes" - a background visual monitoring system that watches specific windows across macOS Spaces while you work on other tasks.

**Relationship to Multi-Space Visual Intelligence (MSVI):**
- **MSVI (The Map)** = SpatialAwarenessAgent - Knows *where* things are
- **VMSI (The Watcher)** = VisualMonitorAgent - Watches *what happens* there

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                 Video Multi-Space Intelligence (VMSI)                │
│                         "The Watcher"                                │
└─────────────────────────────────────────────────────────────────────┘

User Voice Command:
"Watch the Terminal on Space 4 for 'Build Successful'"
                    ↓
        ┌───────────────────────┐
        │  VisualMonitorAgent   │
        │  (Neural Mesh Agent)  │
        └───────────────────────┘
                    ↓
        ┌───────────────────────┐
        │ SpatialAwarenessAgent │  ← "Where is Terminal?"
        │    (The Map)          │  → "Space 4, Window ID 992"
        └───────────────────────┘
                    ↓
        ┌───────────────────────┐
        │   VideoWatcherManager │
        │ (Background Streams)  │
        └───────────────────────┘
                    ↓
        spawn_watcher(window_id=992, fps=5)
                    ↓
        ┌───────────────────────┐
        │  Low-FPS Video Stream │  ← 5 FPS (saves GPU/CPU)
        │   (Background Only)   │  ← Targets Window 992 only
        │   (Low Priority Thread)│ ← Doesn't block main work
        └───────────────────────┘
                    ↓
        ┌───────────────────────┐
        │ VisualEventDetector   │
        │ (OCR + Element Match) │
        └───────────────────────┘
                    ↓
        Analyzes frames for "Build Successful"
                    ↓
        ┌───────────────────────┐
        │  Text Found! Alert!   │
        └───────────────────────┘
                    ↓
    Voice: "Build successful on Space 4"
    Notification: "Terminal build complete"
```

---

## Components

### 1. VideoWatcherManager (macos_video_capture_advanced.py)

**Purpose:** Spawn and manage low-FPS background video streams for specific windows.

**Key Methods:**

```python
async def spawn_watcher(
    window_id: int,
    fps: int = 5,
    priority: str = "low"
) -> VideoWatcher:
    """
    Create background video stream for a specific window.

    Args:
        window_id: macOS window ID to watch
        fps: Frame rate (1-10, default 5 for efficiency)
        priority: Thread priority ("low", "normal", "high")

    Returns:
        VideoWatcher instance streaming frames
    """

async def wait_for_visual_event(
    watcher: VideoWatcher,
    trigger: Union[str, Dict],
    timeout: float = 300.0
) -> VisualEventResult:
    """
    Wait for visual event to occur in watcher stream.

    Args:
        watcher: Active VideoWatcher instance
        trigger: Text to find (str) or element spec (dict)
        timeout: Max wait time in seconds

    Returns:
        VisualEventResult with detection details
    """
```

**Features:**
- ✅ Window-specific capture (not full display)
- ✅ Low-FPS (1-10 FPS) to save resources
- ✅ Low-priority thread execution
- ✅ Parallel watchers (monitor multiple windows)
- ✅ Automatic cleanup when done
- ✅ Resource monitoring and throttling

### 2. VisualMonitorAgent (neural_mesh/agents/visual_monitor_agent.py)

**Purpose:** Neural Mesh agent that manages background visual surveillance.

**Key Methods:**

```python
async def watch_and_alert(
    app_name: str,
    trigger_text: str,
    space_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Watch an app for specific text/event and alert when found.

    Args:
        app_name: App to monitor (e.g., "Terminal", "Chrome")
        trigger_text: Text to wait for (e.g., "Build Successful")
        space_id: Optional specific space (auto-detect if None)

    Returns:
        Result with detection timestamp and alert status
    """
```

**Capabilities:**
- `watch_and_alert`: Main monitoring capability
- `watch_multiple`: Monitor multiple windows in parallel
- `stop_watching`: Cancel active watchers
- `list_watchers`: Get status of all active watchers

**Integration:**
- Uses SpatialAwarenessAgent to locate windows
- Uses VideoWatcherManager for background streaming
- Uses VisualEventDetector for OCR/element detection
- Sends voice alerts via Ironcliw Voice API

### 3. VisualEventDetector (vision/visual_event_detector.py)

**Purpose:** Analyze video frames for text and UI elements.

**Detection Methods:**

```python
class VisualEventDetector:
    async def detect_text(
        self,
        frame: np.ndarray,
        target_text: str
    ) -> TextDetectionResult:
        """OCR-based text detection using pytesseract."""

    async def detect_element(
        self,
        frame: np.ndarray,
        element_spec: Dict
    ) -> ElementDetectionResult:
        """Computer vision element detection (buttons, icons)."""

    async def detect_color_pattern(
        self,
        frame: np.ndarray,
        color_range: Tuple
    ) -> ColorDetectionResult:
        """Detect color changes (e.g., progress bars)."""
```

**Features:**
- ✅ OCR text detection (pytesseract)
- ✅ Element detection (OpenCV template matching)
- ✅ Color pattern detection (progress indicators)
- ✅ Fuzzy matching (typo tolerance)
- ✅ Multi-region analysis (focus areas)
- ✅ Confidence scoring

---

## Use Cases

### Use Case 1: Build Monitor
```
User: "Watch the Terminal for 'Build Successful'"

Flow:
1. VisualMonitorAgent locates Terminal (Space 4, Window 992)
2. Spawns low-FPS watcher (5 FPS) on Window 992
3. Runs OCR on each frame looking for "Build Successful"
4. When found: Voice alert "Build successful on Space 4"
5. Auto-stops watcher (cleanup)

User stays on Space 1 working on code the whole time.
```

### Use Case 2: Multi-Site Parallel Monitoring
```
User: "Watch Chrome for 'Application Submitted' and Terminal for 'Error'"

Flow:
1. VisualMonitorAgent.watch_multiple([
     {"app": "Chrome", "trigger": "Application Submitted"},
     {"app": "Terminal", "trigger": "Error"}
   ])
2. Spawns 2 parallel watchers (5 FPS each)
3. Monitors both simultaneously
4. First to trigger: Alert and report which one
5. Continue monitoring the other

Proactive parallelism - multiple eyes at once.
```

### Use Case 3: Progress Monitoring
```
User: "Watch Safari until the page loads 100%"

Flow:
1. VisualMonitorAgent locates Safari
2. Spawns watcher looking for "100%" or full progress bar
3. Detects progress via OCR or color pattern
4. When complete: Alert "Safari page loaded"

You can browse other tabs while it waits.
```

---

## Data Flow

### Frame Processing Pipeline

```
Window 992 (Background) → AVFoundation Capture → Frame Buffer (5 FPS)
                                                        ↓
                                            ┌───────────────────┐
                                            │ Low Priority Queue│
                                            └───────────────────┘
                                                        ↓
                                            ┌───────────────────┐
                                            │ Visual Detector   │
                                            │   - OCR (CPU)     │
                                            │   - CV (GPU)      │
                                            └───────────────────┘
                                                        ↓
                                            Match "Build Successful"?
                                                        ↓
                                            YES → Alert + Cleanup
                                            NO  → Continue monitoring
```

### Resource Usage

**Low-FPS Design (5 FPS):**
- CPU: ~5-10% (vs 30 FPS = 40-60%)
- GPU: ~50 MB VRAM (vs 30 FPS = 200-400 MB)
- Network: 0 (local only)

**Parallel Watchers:**
- 1 watcher: ~10% CPU
- 3 watchers: ~25% CPU (still reasonable)
- 5 watchers: ~40% CPU (near limit)

**Recommendation:** Max 3 parallel watchers for smooth operation.

---

## Cross-Repo Integration

### Shared State Structure

**Location:** `~/.jarvis/cross_repo/vmsi_state.json`

```json
{
  "active_watchers": [
    {
      "watcher_id": "watcher_992_1234567890",
      "window_id": 992,
      "app_name": "Terminal",
      "space_id": 4,
      "trigger_text": "Build Successful",
      "started_at": "2025-12-27T23:00:00Z",
      "status": "watching",
      "frames_processed": 1523,
      "repo": "Ironcliw-AI-Agent"
    }
  ],
  "recent_detections": [
    {
      "detection_id": "det_123456",
      "window_id": 992,
      "trigger_text": "Build Successful",
      "detected_at": "2025-12-27T23:15:32Z",
      "confidence": 0.94,
      "frame_number": 2301,
      "repo": "Ironcliw-AI-Agent"
    }
  ],
  "stats": {
    "total_watchers_spawned": 47,
    "total_detections": 23,
    "average_detection_time_seconds": 127.4,
    "total_frames_analyzed": 45892
  }
}
```

### Ironcliw ↔ Ironcliw Prime Integration

**Ironcliw (Main System):**
- Runs VisualMonitorAgent
- Spawns watchers
- Writes state to `~/.jarvis/cross_repo/vmsi_state.json`

**Ironcliw Prime (Reasoning System):**
- Reads watcher state
- Can query: "What is Ironcliw currently watching?"
- Can reason: "Build monitor has been running 15 min, check progress"
- Can suggest: "Consider adding error watcher too"

**Reactor Core (Event System):**
- Subscribes to detection events
- Can trigger workflows when events occur
- Can aggregate stats across all repos

### Event Broadcasting

**Event Format:**
```json
{
  "event_type": "visual_event_detected",
  "timestamp": "2025-12-27T23:15:32Z",
  "source_repo": "Ironcliw-AI-Agent",
  "window_id": 992,
  "app_name": "Terminal",
  "trigger_text": "Build Successful",
  "confidence": 0.94,
  "action_taken": "voice_alert",
  "narration": "Build successful on Space 4"
}
```

---

## Configuration

**Environment Variables:**

```bash
# Video Watcher Configuration
export Ironcliw_WATCHER_DEFAULT_FPS=5         # Default FPS for watchers
export Ironcliw_WATCHER_MIN_FPS=1             # Minimum allowed FPS
export Ironcliw_WATCHER_MAX_FPS=10            # Maximum allowed FPS
export Ironcliw_WATCHER_MAX_PARALLEL=3        # Max parallel watchers
export Ironcliw_WATCHER_PRIORITY="low"        # Thread priority
export Ironcliw_WATCHER_TIMEOUT=300           # Default timeout (5 min)

# Visual Detection Configuration
export Ironcliw_OCR_ENGINE="pytesseract"      # OCR engine
export Ironcliw_OCR_LANG="eng"                # OCR language
export Ironcliw_DETECTION_CONFIDENCE=0.75     # Min confidence threshold
export Ironcliw_FUZZY_MATCH_RATIO=0.85        # Fuzzy string match threshold

# Cross-Repo Integration
export Ironcliw_CROSS_REPO_DIR="~/.jarvis/cross_repo"
export Ironcliw_VMSI_STATE_FILE="vmsi_state.json"
export Ironcliw_VMSI_SYNC_INTERVAL=5          # Sync interval (seconds)
```

---

## Implementation Phases

### Phase 1: Core Video Watcher ✅ (In Progress)
- Upgrade `macos_video_capture_advanced.py`
- Add `spawn_watcher(window_id)`
- Add window-specific capture
- Add low-FPS streaming
- Add priority queue system

### Phase 2: Visual Event Detection ✅ (Next)
- Create `vision/visual_event_detector.py`
- Implement OCR text detection
- Implement element detection
- Add confidence scoring

### Phase 3: Visual Monitor Agent ✅ (Next)
- Create `neural_mesh/agents/visual_monitor_agent.py`
- Implement `watch_and_alert()`
- Integrate with SpatialAwarenessAgent
- Add parallel watcher support

### Phase 4: Cross-Repo Integration ✅ (Next)
- Create shared state structure
- Add event broadcasting
- Integrate with Ironcliw Prime
- Integrate with Reactor Core

### Phase 5: Voice Integration ✅ (Next)
- Add voice alerts
- Add notification system
- Test complete workflow

---

## Testing Plan

### Test 1: Single Window Monitor
```bash
# Start Ironcliw
python3 backend/main.py

# In Ironcliw voice interface:
"Watch the Terminal for 'Build Successful'"

# In Terminal:
echo "Build Successful"

# Expected: Voice alert within 1-2 seconds
```

### Test 2: Parallel Monitoring
```bash
# Voice command:
"Watch Chrome for 'Submitted' and Terminal for 'Error'"

# Trigger one:
echo "Error occurred" (in Terminal)

# Expected: Immediate alert for Terminal, Chrome watcher continues
```

### Test 3: Cross-Repo State
```bash
# Ironcliw watching Terminal
# Check state from Ironcliw Prime:
cat ~/.jarvis/cross_repo/vmsi_state.json

# Expected: See active watcher details
```

---

## Status

**Version:** v10.6
**Status:** In Development
**Target:** Production-grade universal monitoring system

**Features Implemented:**
- ⏳ Window-specific video capture
- ⏳ Low-FPS background watchers
- ⏳ Visual event detection (OCR)
- ⏳ VisualMonitorAgent
- ⏳ SpatialAwareness integration
- ⏳ Cross-repo state sharing
- ⏳ Voice alerts

**Next Steps:**
1. Implement window-specific capture in VideoWatcher
2. Build VisualEventDetector with OCR
3. Create VisualMonitorAgent
4. Add cross-repo integration
5. Test complete workflow

---

## Notes

**Why Low-FPS (5 FPS)?**
- Text/UI doesn't change every frame
- 5 FPS = 1 frame every 200ms (plenty fast for text detection)
- Saves 80% GPU/CPU vs 30 FPS
- Enables parallel monitoring without lag

**Why Window-Specific Capture?**
- Only capture the window we care about
- Ignore irrelevant visual noise
- Faster OCR (smaller region)
- Better privacy (don't capture everything)

**Why Low Priority Threads?**
- Watchers run in background
- Don't interrupt main work
- OS scheduler deprioritizes automatically
- Smooth user experience

This is "God Mode" - Ironcliw watches multiple things simultaneously while you focus on what matters. 🚀
