# Phase 1.1 Multi-Monitor Support - Verification & Completion Plan

## 📊 Current Implementation Status

### ✅ IMPLEMENTED (Partial)
- Core `MultiMonitorDetector` class exists
- Async/await architecture
- Display detection structure
- Screenshot capture structure
- Performance tracking

### ❌ CRITICAL ISSUES FOUND

#### 1. Core Graphics API Bug 🔴
**Location:** `backend/vision/multi_monitor_detector.py:135-144`

**Current Code (BROKEN):**
```python
display_count = Quartz.CGGetActiveDisplayList(0, None, None)
if display_count == 0:
    logger.warning("No displays detected")
    return []

display_ids = (Quartz.CGDirectDisplayID * display_count)()
actual_count = Quartz.CGGetActiveDisplayList(
    display_count, display_ids, None
)
```

**Problem:** `CGGetActiveDisplayList` returns tuple `(error_code, display_ids)`, not int

**Fix Required:**
```python
# Get display count first
error_code, display_ids = Quartz.CGGetActiveDisplayList(32, None, None)
if error_code != 0:
    logger.error(f"CGGetActiveDisplayList failed with error: {error_code}")
    return []

display_count = len(display_ids) if display_ids else 0
if display_count == 0:
    logger.warning("No displays detected")
    return []

logger.info(f"Detected {display_count} displays")

for display_id in display_ids:
    # Get display bounds
    bounds = Quartz.CGDisplayBounds(display_id)
    # ... rest of implementation
```

#### 2. Yabai JSON Parsing Incomplete 🔴
**Location:** `backend/vision/multi_monitor_detector.py:233-270`

**Current:** Simplified stub, doesn't actually parse Yabai JSON

**Fix Required:**
```python
async def _get_yabai_space_mappings(self) -> Dict[int, SpaceDisplayMapping]:
    """Get space mappings from Yabai CLI with proper JSON parsing"""
    try:
        # Query Yabai for spaces
        result = await asyncio.create_subprocess_exec(
            self.yabai_path, "-m", "query", "--spaces",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await result.communicate()
        
        if result.returncode != 0:
            logger.error(f"Yabai spaces query failed: {stderr.decode()}")
            return {}
        
        # Parse JSON output
        import json
        spaces_data = json.loads(stdout.decode())
        
        # Build mappings
        mappings = {}
        for space in spaces_data:
            space_id = space.get("index", 0)
            display_id = space.get("display", 1)
            is_active = space.get("has-focus", False)
            space_name = space.get("label", f"Space {space_id}")
            
            mapping = SpaceDisplayMapping(
                space_id=space_id,
                display_id=display_id,
                space_name=space_name,
                is_active=is_active
            )
            mappings[space_id] = mapping
        
        logger.info(f"Parsed {len(mappings)} space mappings from Yabai")
        return mappings
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Yabai JSON output: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error querying Yabai for space mappings: {e}")
        return {}
```

#### 3. NOT Integrated with Intelligent Orchestrator 🔴
**Location:** `backend/vision/intelligent_orchestrator.py`

**Current:** No reference to MultiMonitorDetector

**Fix Required:** Add display-aware workspace scouting

#### 4. NO Query Routing for Multi-Monitor Questions 🔴
**Location:** `backend/api/vision_command_handler.py`

**Current:** No handling for "What's on my second monitor?"

**Fix Required:** Add query detection and routing

#### 5. NO API Endpoint 🔴
**Location:** `backend/api/vision_api.py` or routes

**Current:** No `/vision/displays` endpoint

**Fix Required:** Create REST endpoint for display data

---

## 🎯 PRD Requirements Verification

| Requirement | Status | Notes |
|-------------|--------|-------|
| **G1: Detect all monitors** | ⚠️ Partial | Core Graphics bug prevents detection |
| **G2: Map spaces to displays** | ⚠️ Partial | Yabai parsing incomplete |
| **G3: Capture per-monitor** | ✅ Yes | Structure exists, needs testing |
| **G4: Display-aware summaries** | ❌ No | Not integrated with orchestrator |
| **G5: User queries "What's on monitor 2?"** | ❌ No | No query routing |
| **Async/await** | ✅ Yes | All methods are async |
| **No hardcoding** | ✅ Yes | Dynamic display detection |
| **Robust error handling** | ⚠️ Partial | Basic try/catch, needs enhancement |
| **Ambiguity handling** | ❌ No | No handling for "second monitor" vs "monitor 2" |

**Overall Status: 40% Complete** ❌

---

## 🔧 Complete Implementation Plan

### Task 1: Fix Core Graphics API (30 min)
**Priority:** 🔴 CRITICAL

**File:** `backend/vision/multi_monitor_detector.py`

**Changes:**
1. Fix `detect_displays()` method (lines 108-187)
2. Correct API call to `CGGetActiveDisplayList`
3. Add proper error handling
4. Test with `python3 backend/vision/multi_monitor_detector.py`

### Task 2: Fix Yabai JSON Parsing (20 min)
**Priority:** 🔴 CRITICAL

**File:** `backend/vision/multi_monitor_detector.py`

**Changes:**
1. Fix `_get_yabai_space_mappings()` method (lines 233-270)
2. Add proper JSON parsing
3. Map Yabai `display` field to display_id
4. Handle missing fields gracefully

### Task 3: Integrate with Yabai Space Detector (30 min)
**Priority:** 🔴 CRITICAL

**File:** `backend/vision/yabai_space_detector.py`

**Changes:**
1. Add `get_display_for_space(space_id)` method
2. Add `enumerate_spaces_by_display()` method
3. Cache display mappings
4. Update `enumerate_all_spaces()` to include display info

**New Methods:**
```python
def get_display_for_space(self, space_id: int) -> Optional[int]:
    """Get display ID for a given space"""
    pass

def enumerate_spaces_by_display(self) -> Dict[int, List[Dict[str, Any]]]:
    """Group spaces by display ID"""
    pass
```

### Task 4: Integrate with Intelligent Orchestrator (45 min)
**Priority:** 🟡 HIGH

**File:** `backend/vision/intelligent_orchestrator.py`

**Changes:**
1. Import `MultiMonitorDetector`
2. Add `_detect_monitors()` method
3. Update `_scout_workspace()` to include display info
4. Add display awareness to `WorkspaceSnapshot`

**New Code:**
```python
from .multi_monitor_detector import MultiMonitorDetector

class IntelligentOrchestrator:
    def __init__(self):
        self.monitor_detector = MultiMonitorDetector()
        # ... existing init
    
    async def _scout_workspace(self) -> WorkspaceSnapshot:
        """Scout workspace with display awareness"""
        # ... existing code
        
        # Add display detection
        displays = await self.monitor_detector.detect_displays()
        space_display_mapping = await self.monitor_detector.get_space_display_mapping()
        
        # Enhance workspace snapshot with display info
        snapshot.displays = displays
        snapshot.space_display_mapping = space_display_mapping
        
        return snapshot
```

### Task 5: Add Query Routing for Multi-Monitor (60 min)
**Priority:** 🟡 HIGH

**File:** `backend/api/vision_command_handler.py`

**Changes:**
1. Add `_is_multi_monitor_query()` method
2. Add `_handle_multi_monitor_query()` method
3. Update `analyze_screen()` to route multi-monitor queries
4. Handle ambiguous queries ("second monitor", "primary monitor")

**New Methods:**
```python
def _is_multi_monitor_query(self, command_text: str) -> bool:
    """Check if query is about multiple monitors"""
    query_lower = command_text.lower()
    monitor_keywords = [
        "monitor", "display", "screen",
        "second monitor", "primary monitor", "main monitor",
        "monitor 1", "monitor 2", "monitor 3",
        "all monitors", "all displays"
    ]
    return any(keyword in query_lower for keyword in monitor_keywords)

async def _handle_multi_monitor_query(self, command_text: str) -> Dict[str, Any]:
    """Handle multi-monitor specific queries"""
    # Detect monitors
    # Parse which monitor user is asking about
    # Capture specific monitor
    # Analyze with Claude
    pass
```

### Task 6: Add Ambiguity Handling (30 min)
**Priority:** 🟡 HIGH

**File:** `backend/vision/query_disambiguation.py` (NEW)

**Create:**
```python
class QueryDisambiguator:
    """Handle ambiguous multi-monitor queries"""
    
    async def resolve_monitor_reference(
        self, 
        query: str, 
        available_displays: List[DisplayInfo]
    ) -> Optional[int]:
        """
        Resolve ambiguous monitor references:
        - "second monitor" → display_id of second display
        - "primary monitor" → display_id of primary display
        - "main screen" → display_id of primary display
        - "monitor 2" → second display in list
        """
        query_lower = query.lower()
        
        # Primary/main display
        if "primary" in query_lower or "main" in query_lower:
            for display in available_displays:
                if display.is_primary:
                    return display.display_id
        
        # Ordinal references (second, third, etc.)
        ordinals = {
            "second": 1, "2nd": 1, "monitor 2": 1,
            "third": 2, "3rd": 2, "monitor 3": 2,
            "fourth": 3, "4th": 3, "monitor 4": 3
        }
        
        for ordinal, index in ordinals.items():
            if ordinal in query_lower:
                if index < len(available_displays):
                    return available_displays[index].display_id
        
        return None
    
    async def ask_clarification(
        self, 
        query: str, 
        available_displays: List[DisplayInfo]
    ) -> str:
        """Generate clarification question"""
        return f"Sir, I see {len(available_displays)} displays. Which one? " + \
               ", ".join([f"Monitor {i+1} ({d.resolution[0]}x{d.resolution[1]})" 
                         for i, d in enumerate(available_displays)])
```

### Task 7: Add API Endpoint (30 min)
**Priority:** 🟢 MEDIUM

**File:** `backend/api/routes.py` or create `backend/api/display_routes.py`

**Add:**
```python
from fastapi import APIRouter
from vision.multi_monitor_detector import MultiMonitorDetector

router = APIRouter()

@router.get("/vision/displays")
async def get_displays():
    """Get all connected displays"""
    detector = MultiMonitorDetector()
    summary = await detector.get_display_summary()
    return summary

@router.get("/vision/displays/{display_id}")
async def get_display(display_id: int):
    """Get specific display info"""
    detector = MultiMonitorDetector()
    displays = await detector.detect_displays()
    
    for display in displays:
        if display.display_id == display_id:
            return {
                "display_id": display.display_id,
                "resolution": display.resolution,
                "position": display.position,
                "is_primary": display.is_primary,
                "name": display.name
            }
    
    return {"error": "Display not found"}

@router.post("/vision/displays/{display_id}/capture")
async def capture_display(display_id: int):
    """Capture screenshot of specific display"""
    detector = MultiMonitorDetector()
    result = await detector.capture_all_displays()
    
    if display_id in result.displays_captured:
        return {
            "success": True,
            "display_id": display_id,
            "captured": True
        }
    else:
        return {
            "success": False,
            "error": "Failed to capture display"
        }
```

### Task 8: Add Comprehensive Tests (60 min)
**Priority:** 🟢 MEDIUM

**File:** `backend/tests/test_multi_monitor_detector.py` (NEW)

**Create:**
```python
import pytest
from vision.multi_monitor_detector import MultiMonitorDetector, DisplayInfo

@pytest.mark.asyncio
async def test_detect_displays():
    """Test display detection"""
    detector = MultiMonitorDetector()
    displays = await detector.detect_displays()
    
    assert isinstance(displays, list)
    assert len(displays) > 0
    assert all(isinstance(d, DisplayInfo) for d in displays)

@pytest.mark.asyncio
async def test_space_display_mapping():
    """Test space-to-display mapping"""
    detector = MultiMonitorDetector()
    mappings = await detector.get_space_display_mapping()
    
    assert isinstance(mappings, dict)
    # Should have at least one space mapped
    assert len(mappings) > 0

@pytest.mark.asyncio
async def test_capture_all_displays():
    """Test screenshot capture"""
    detector = MultiMonitorDetector()
    result = await detector.capture_all_displays()
    
    assert result.success or len(result.failed_displays) > 0
    assert result.total_displays > 0

@pytest.mark.asyncio
async def test_query_second_monitor():
    """Test 'second monitor' query handling"""
    # Mock test for query routing
    pass
```

---

## 🧪 Testing Plan

### Unit Tests
- ✅ Display detection with mocked CG APIs
- ✅ Yabai JSON parsing with sample data
- ✅ Space-to-display mapping
- ✅ Screenshot capture logic

### Integration Tests
- ✅ End-to-end display detection on dual-monitor system
- ✅ Query "What's on my second monitor?" 
- ✅ Query "Show me all my displays"
- ✅ Query with ambiguity: "What's on the monitor?"

### Edge Cases
- ✅ Single monitor system (ensure no regression)
- ✅ Monitors added/removed during runtime
- ✅ Yabai not running (graceful fallback)
- ✅ Permission denied for screen recording

---

## 📈 Success Metrics

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Display detection accuracy | 100% | N/A | ❌ Not working |
| Space-to-display mapping | 95% | N/A | ❌ Not working |
| Screenshot latency | <300ms | N/A | ⏸️ Untested |
| Query response accuracy | 95% | N/A | ❌ Not implemented |
| Zero impact on single-monitor | ✅ | N/A | ⏸️ Needs testing |

---

## ⏱️ Estimated Timeline

| Task | Time | Priority |
|------|------|----------|
| Fix Core Graphics API | 30 min | 🔴 |
| Fix Yabai JSON parsing | 20 min | 🔴 |
| Integrate with yabai_space_detector | 30 min | 🔴 |
| Integrate with orchestrator | 45 min | 🟡 |
| Add query routing | 60 min | 🟡 |
| Add ambiguity handling | 30 min | 🟡 |
| Add API endpoint | 30 min | 🟢 |
| Add tests | 60 min | 🟢 |
| **TOTAL** | **~5 hours** | - |

---

## 🚨 Critical Path

**Must Complete (for PRD compliance):**
1. Fix Core Graphics bug (blocks everything)
2. Fix Yabai JSON parsing (blocks space mapping)
3. Integrate with orchestrator (blocks query handling)
4. Add query routing (blocks user stories)

**Nice to Have:**
5. API endpoint (for frontend integration)
6. Comprehensive tests (for robustness)

---

## 🎯 Next Steps

### Immediate (Today)
1. ✅ Fix Core Graphics API bug
2. ✅ Fix Yabai JSON parsing
3. ✅ Test basic detection works

### Short-term (This Week)
4. ✅ Integrate with intelligent_orchestrator
5. ✅ Add query routing
6. ✅ Add ambiguity handling

### Medium-term (Next Week)
7. ✅ Add API endpoint
8. ✅ Add comprehensive tests
9. ✅ Performance optimization

---

## 📝 Verification Checklist

After implementation, verify:

- [ ] `python3 backend/vision/multi_monitor_detector.py` runs without errors
- [ ] Detects all connected monitors correctly
- [ ] Maps spaces to displays via Yabai
- [ ] Can capture screenshots per monitor
- [ ] Query "What's on my second monitor?" works
- [ ] Query "Show me all displays" works
- [ ] Ambiguous queries are clarified
- [ ] API endpoint `/vision/displays` returns data
- [ ] Tests pass
- [ ] Single-monitor systems still work

---

**Status:** 🔴 NOT READY FOR PRODUCTION

**Recommendation:** Complete Tasks 1-4 (critical path) before deploying. Tasks 5-8 can be done iteratively.

---

*Document Version: 1.0*
*Date: 2025-10-14*
*Author: Ironcliw Analysis System*
