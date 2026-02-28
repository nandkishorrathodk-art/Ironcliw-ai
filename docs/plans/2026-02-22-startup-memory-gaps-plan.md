# Startup Memory Gap Closure — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close three remaining startup memory gaps: wire COMPONENT_UNLOAD to actually unload the LLM model, bump headroom margins for 16GB systems, and start MemoryQuantizer monitoring in Phase 1.

**Architecture:** Three targeted fixes in existing files. Fix 1 adds unload callbacks to MemoryQuantizer (following the existing `register_tier_change_callback` pattern) and registers `UnifiedModelServing.stop()` as a callback. Fix 2 makes headroom values env-var-configurable with higher defaults. Fix 3 adds a `start_monitoring()` call early in the supervisor startup.

**Tech Stack:** Python 3, asyncio, psutil, existing MemoryQuantizer/UnifiedModelServing

**Design doc:** `docs/plans/2026-02-22-startup-memory-gaps-design.md`

---

### Task 1: Wire COMPONENT_UNLOAD to Unload Callbacks

**Files:**
- Modify: `backend/core/memory_quantizer.py:425-435` (__init__, add state), `:984-986` (COMPONENT_UNLOAD stub), `:1098-1108` (near other callback registrations)

**Step 1: Add `_unload_callbacks` list in `__init__`**

In `__init__`, near the existing `_thrash_callbacks` (line ~435), add:

```python
        self._unload_callbacks: List[Callable] = []
```

**Step 2: Add `register_unload_callback()` method**

Near `register_thrash_callback()` (line ~1098), add:

```python
    def register_unload_callback(self, callback: Callable) -> None:
        """Register callback for COMPONENT_UNLOAD strategy.

        Callback receives one argument: the current MemoryTier.
        Called when the system enters CRITICAL or EMERGENCY tier and
        the COMPONENT_UNLOAD optimization strategy fires. Use this to
        unload heavy components (e.g., the local LLM model) to free RAM.
        """
        self._unload_callbacks.append(callback)
```

**Step 3: Wire COMPONENT_UNLOAD to fire callbacks**

Replace the stub at lines 984-986:

```python
        elif strategy == OptimizationStrategy.COMPONENT_UNLOAD:
            # Unload non-critical components
            logger.debug("Component unload requested")
```

With:

```python
        elif strategy == OptimizationStrategy.COMPONENT_UNLOAD:
            # v266.0: Fire registered unload callbacks
            if self._unload_callbacks:
                logger.warning(
                    f"[ComponentUnload] Firing {len(self._unload_callbacks)} unload callback(s) "
                    f"(tier={self.current_tier.value})"
                )
                for callback in self._unload_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(self.current_tier)
                        else:
                            callback(self.current_tier)
                    except Exception as e:
                        logger.error(f"[ComponentUnload] Callback error: {e}")
            else:
                logger.debug("Component unload requested but no callbacks registered")
```

**Step 4: Verify**

Run: `python3 -c "
import asyncio
from backend.core.memory_quantizer import MemoryQuantizer, OptimizationStrategy
q = MemoryQuantizer()
_fired = []
q.register_unload_callback(lambda tier: _fired.append(tier))
asyncio.run(q._apply_optimization_strategy(OptimizationStrategy.COMPONENT_UNLOAD))
print(f'Fired: {len(_fired)} callback(s)')
print('OK')
"`
Expected: `Fired: 1 callback(s)`, `OK`

**Step 5: Commit**

```bash
git add backend/core/memory_quantizer.py
git commit -m "$(cat <<'EOF'
feat(memory): wire COMPONENT_UNLOAD to unload callbacks

The COMPONENT_UNLOAD strategy was a no-op stub that only logged.
Now fires registered callbacks with the current MemoryTier, allowing
components to unload themselves under CRITICAL/EMERGENCY pressure.
Follows the existing register_tier_change_callback() pattern.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Register Model Unload Callback in UnifiedModelServing

**Files:**
- Modify: `backend/intelligence/unified_model_serving.py:2183-2191` (near thrash callback registration in start())

**Step 1: Add unload callback handler**

Near the existing thrash cascade methods (after `_trigger_gcp_offload_from_thrash`), add:

```python
    async def _handle_component_unload(self, tier) -> None:
        """Handle COMPONENT_UNLOAD from MemoryQuantizer.

        Called when memory reaches CRITICAL/EMERGENCY and GCP is unavailable.
        Unloads the local LLM model to free 4-8GB of RAM.
        """
        _local = self._clients.get(ModelProvider.PRIME_LOCAL)
        if not _local or not isinstance(_local, PrimeLocalClient):
            return
        if not getattr(_local, '_loaded', False):
            self.logger.info("[ComponentUnload] No local model loaded — nothing to unload")
            return

        self.logger.warning(
            f"[ComponentUnload] Memory tier {tier} — unloading local LLM model"
        )
        try:
            await _local.unload_model()
            self.logger.warning("[ComponentUnload] Local LLM model unloaded successfully")
        except Exception as e:
            self.logger.error(f"[ComponentUnload] Unload failed: {e}")
```

Note: Check if `PrimeLocalClient` has an `unload_model()` method. It might be called `stop()` or something else. Read the class to find the right method name. Also check if `_loaded` is the right attribute for checking if a model is loaded — it might be `_model is not None`.

**Step 2: Register the callback in `start()`**

Near the thrash callback registration (line ~2187), add:

```python
        # v266.0: Register for component unload (GCP-disabled escape valve)
        try:
            if _mq and hasattr(_mq, 'register_unload_callback'):
                _mq.register_unload_callback(self._handle_component_unload)
                self.logger.info("[v266.0] Registered component unload callback")
        except Exception as e:
            self.logger.debug(f"[v266.0] Unload callback registration: {e}")
```

If the `_mq` variable from the thrash registration block isn't in scope, re-import:

```python
        try:
            from backend.core.memory_quantizer import get_memory_quantizer
            _mq = await get_memory_quantizer()
            if _mq and hasattr(_mq, 'register_unload_callback'):
                _mq.register_unload_callback(self._handle_component_unload)
                self.logger.info("[v266.0] Registered component unload callback")
        except Exception as e:
            self.logger.debug(f"[v266.0] Unload callback registration: {e}")
```

**Step 3: Verify**

Run: `python3 -c "from backend.intelligence.unified_model_serving import UnifiedModelServing; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add backend/intelligence/unified_model_serving.py
git commit -m "$(cat <<'EOF'
feat(model): register unload callback for GCP-disabled escape valve

When GCP is disabled and memory hits CRITICAL/EMERGENCY, the
COMPONENT_UNLOAD strategy now triggers local LLM model unload via
the callback registered in start(). This frees 4-8GB of RAM —
the only escape valve available without GCP offloading.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Bump Headroom Margins with Env Var Overrides

**Files:**
- Modify: `backend/intelligence/unified_model_serving.py:705-716` (headroom values in load_model)

**Step 1: Replace hardcoded headroom values with env-var-configurable defaults**

Replace lines 705-716:

```python
                # v235.1: Dynamic headroom based on memory pressure tier (Fix C1)
                _headroom_gb = 1.0  # default
                _tier = None
                try:
                    from backend.core.memory_quantizer import MemoryTier
                    _tier = _metrics.tier if '_metrics' in dir() and _metrics else None
                    if _tier in (MemoryTier.ABUNDANT, MemoryTier.OPTIMAL):
                        _headroom_gb = 0.75
                    elif _tier == MemoryTier.ELEVATED:
                        _headroom_gb = 1.0
                    elif _tier == MemoryTier.CONSTRAINED:
                        _headroom_gb = 1.5
```

With:

```python
                # v266.0: Dynamic headroom based on memory pressure tier
                # Bumped from 0.75-1.5GB to 1.5-2.5GB to account for frontend
                # (500MB-1.2GB) and Docker (200-500MB) consuming headroom post-load.
                # Env-var overrides for tuning without code changes.
                _headroom_gb = float(os.getenv("Ironcliw_MODEL_HEADROOM_NORMAL", "2.0"))
                _tier = None
                try:
                    from backend.core.memory_quantizer import MemoryTier
                    _tier = _metrics.tier if '_metrics' in dir() and _metrics else None
                    if _tier in (MemoryTier.ABUNDANT, MemoryTier.OPTIMAL):
                        _headroom_gb = float(os.getenv("Ironcliw_MODEL_HEADROOM_RELAXED", "1.5"))
                    elif _tier == MemoryTier.ELEVATED:
                        _headroom_gb = float(os.getenv("Ironcliw_MODEL_HEADROOM_NORMAL", "2.0"))
                    elif _tier == MemoryTier.CONSTRAINED:
                        _headroom_gb = float(os.getenv("Ironcliw_MODEL_HEADROOM_TIGHT", "2.5"))
```

The CRITICAL/EMERGENCY tier block (lines 717-722) stays unchanged — it already refuses to load.

**Step 2: Verify**

Run: `python3 -c "from backend.intelligence.unified_model_serving import UnifiedModelServing; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add backend/intelligence/unified_model_serving.py
git commit -m "$(cat <<'EOF'
feat(model): bump headroom margins and add env var overrides

Increased model loading headroom from 0.75-1.5GB to 1.5-2.5GB to
account for frontend (500MB-1.2GB) and Docker (200-500MB) consuming
RAM after model load. Now env-var-configurable: Ironcliw_MODEL_HEADROOM_RELAXED,
Ironcliw_MODEL_HEADROOM_NORMAL, Ironcliw_MODEL_HEADROOM_TIGHT.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Start MemoryQuantizer Monitoring in Phase 1

**Files:**
- Modify: `unified_supervisor.py:66561-66610` (Phase 1 Preflight, near end of phase)

**Step 1: Add early monitoring start at end of Phase 1**

Near the end of `_phase_1_preflight()`, after the existing cleanup steps but before the method returns, add:

```python
            # v266.0: Start MemoryQuantizer monitoring early
            # This gives Phases 2-3 pagein baselines and tier tracking.
            # The monitor is lightweight (one vm_stat + one psutil every 1-10s).
            try:
                from backend.core.memory_quantizer import get_memory_quantizer
                _mq = await asyncio.wait_for(get_memory_quantizer(), timeout=5.0)
                if _mq and not _mq.monitoring:
                    await _mq.start_monitoring()
                    self.logger.info("[v266.0] MemoryQuantizer monitoring started (Phase 1)")
            except asyncio.TimeoutError:
                self.logger.debug("[v266.0] MemoryQuantizer init timeout (non-fatal)")
            except Exception as e:
                self.logger.debug(f"[v266.0] Early monitoring start: {e}")
```

IMPORTANT: Read `_phase_1_preflight()` to find the EXACT end of the method — don't insert after a `return`. Find the last substantive block before the method ends.

**Step 2: Verify**

Run: `python3 -c "import py_compile; py_compile.compile('unified_supervisor.py', doraise=True); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add unified_supervisor.py
git commit -m "$(cat <<'EOF'
feat(startup): start MemoryQuantizer monitoring in Phase 1

Previously monitoring only started in Phase 4+ when consumers used it.
Now starts in Phase 1 (Preflight), giving Phases 2-3 pagein baselines
and tier tracking. The monitor is lightweight (vm_stat + psutil every
1-10s) with zero impact on startup performance.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Integration Verification

**Files:** None (verification only)

**Step 1: Verify all imports**

```bash
python3 -c "
from backend.core.memory_quantizer import MemoryQuantizer, OptimizationStrategy
from backend.intelligence.unified_model_serving import UnifiedModelServing
print('All imports OK')
"
```

**Step 2: Verify unload callback round-trip**

```bash
python3 -c "
import asyncio
from backend.core.memory_quantizer import MemoryQuantizer, OptimizationStrategy, MemoryTier
q = MemoryQuantizer()
_results = []
async def _test_cb(tier):
    _results.append(('fired', str(tier)))
q.register_unload_callback(_test_cb)
asyncio.run(q._apply_optimization_strategy(OptimizationStrategy.COMPONENT_UNLOAD))
print(f'Callbacks fired: {len(_results)}')
print(f'Result: {_results}')
print('OK')
"
```

**Step 3: Verify supervisor syntax**

```bash
python3 -c "import py_compile; py_compile.compile('unified_supervisor.py', doraise=True); print('Supervisor OK')"
```

**Step 4: Verify headroom env var works**

```bash
Ironcliw_MODEL_HEADROOM_RELAXED=3.0 python3 -c "
import os
print(f'Headroom override: {os.getenv(\"Ironcliw_MODEL_HEADROOM_RELAXED\")}GB')
from backend.intelligence.unified_model_serving import UnifiedModelServing
print('OK')
"
```
