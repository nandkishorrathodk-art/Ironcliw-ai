# Voice Biometric Authentication Flow Documentation

## Complete Flow Diagram: "Unlock My Screen"

This document provides a comprehensive trace of what happens when you say "unlock my screen" to Ironcliw, with special focus on diagnosing the **"Voice verification failed (confidence: 0.0%)"** error.

---

## Visual Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           USER SAYS: "unlock my screen"                         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: AUDIO CAPTURE                                                           │
│ ─────────────────────                                                           │
│ Component: Microphone → VAD (Voice Activity Detection)                          │
│ File: backend/voice/unified_vad_api.py                                          │
│                                                                                 │
│ Actions:                                                                        │
│   1. Captures audio via pyaudio/sounddevice                                     │
│   2. Detects speech boundaries (silero-vad)                                     │
│   3. Truncates to 2-second window for unlock commands                           │
│   4. Converts to 16kHz, 16-bit PCM format                                       │
│                                                                                 │
│ Output: audio_data (bytes) - typically 32,000-64,000 bytes                      │
│                                                                                 │
│ FAILURE POINT 1: No audio or silent audio                                       │
│   → Energy < 0.0001 indicates silence                                           │
│   → Check: Is microphone working? Permission granted?                           │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: INTENT DETECTION                                                        │
│ ─────────────────────────                                                       │
│ Component: AsyncPipeline                                                        │
│ File: backend/core/async_pipeline.py:1091-1094                                  │
│                                                                                 │
│ Actions:                                                                        │
│   1. Transcribes audio via Whisper/Vosk/Wav2Vec                                 │
│   2. Matches against unlock patterns:                                           │
│      - "unlock my screen"                                                       │
│      - "unlock screen"                                                          │
│      - "unlock the screen"                                                      │
│   3. Detects "unlock" intent                                                    │
│                                                                                 │
│ Output: action_type = "unlock", audio_data passed through                       │
│                                                                                 │
│ FAILURE POINT 2: Transcription fails                                            │
│   → Returns "Unknown command" instead of unlock                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: FAST LOCK/UNLOCK HANDLER                                                │
│ ────────────────────────────────                                                │
│ Component: _fast_lock_unlock()                                                  │
│ File: backend/core/async_pipeline.py:1236-1506                                  │
│                                                                                 │
│ Actions:                                                                        │
│   1. Detects screen is locked                                                   │
│   2. Gets SpeakerVerificationService                                            │
│   3. Calls verify_speaker_enhanced() with audio_data                            │
│                                                                                 │
│ Key Code (line 1394):                                                           │
│   verification_result = await speaker_service.verify_speaker_enhanced(          │
│       audio_data,                                                               │
│       speaker_name or user_name,                                                │
│       context={"environment": "default", "source": "unlock_fallback"}           │
│   )                                                                             │
│                                                                                 │
│ Output: verification_result dict with "verified" and "confidence" keys          │
│                                                                                 │
│ FAILURE POINT 3: SpeakerVerificationService not initialized                     │
│   → Returns {"verified": False, "confidence": 0.0}                              │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: ENHANCED SPEAKER VERIFICATION                                           │
│ ─────────────────────────────────────                                           │
│ Component: SpeakerVerificationService.verify_speaker_enhanced()                 │
│ File: backend/voice/speaker_verification_service.py:2525-2750                   │
│                                                                                 │
│ Actions:                                                                        │
│   Phase 1: Check cache for recent verification                                  │
│   Phase 2: Audio quality analysis (SNR check)                                   │
│   Phase 3: Anti-spoofing checks (replay detection)                              │
│   Phase 4: Core speaker verification ← WHERE 0.0 COMES FROM                     │
│   Phase 5: Multi-factor fusion (voice + behavioral + context)                   │
│   Phase 6: Final decision                                                       │
│                                                                                 │
│ Key Code (line 2626):                                                           │
│   base_result = await self.verify_speaker(audio_data, speaker_name)             │
│   voice_confidence = base_result.get("confidence", 0.0)                         │
│                                                                                 │
│ FAILURE POINT 4: Phase 4 returns 0.0                                            │
│   → Traced to verify_speaker() method                                           │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: CORE SPEAKER VERIFICATION (CRITICAL PATH)                               │
│ ─────────────────────────────────────────────────                               │
│ Component: SpeakerVerificationService.verify_speaker()                          │
│ File: backend/voice/speaker_verification_service.py:4440-4730                   │
│                                                                                 │
│ FAST PATH - Unified Cache (lines 4496-4576):                                    │
│   1. Check if unified_cache is ready                                            │
│   2. Call unified_cache.verify_voice_from_audio()                               │
│   3. If similarity >= 0.85: INSTANT MATCH ✓                                     │
│   4. If similarity >= 0.40: UNLOCK MATCH ✓                                      │
│                                                                                 │
│ FAILURE POINT 5A: Unified cache not ready                                       │
│   → unified_cache.is_ready = False                                              │
│   → Falls through to SpeechBrain path                                           │
│                                                                                 │
│ FAILURE POINT 5B: Cache extraction fails                                        │
│   → ECAPA encoder not loaded → embedding = None                                 │
│   → Returns {"similarity": 0.0, "matched": False}                               │
│   → Falls through to SpeechBrain path                                           │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 5.1: UNIFIED VOICE CACHE VERIFICATION                                      │
│ ──────────────────────────────────────────                                      │
│ Component: UnifiedVoiceCacheManager.verify_voice_from_audio()                   │
│ File: backend/voice_unlock/unified_voice_cache_manager.py                       │
│                                                                                 │
│ Actions:                                                                        │
│   1. Extract embedding from audio via ECAPA-TDNN                                │
│   2. Compare against stored voiceprints                                         │
│   3. Calculate cosine similarity                                                │
│                                                                                 │
│ Key Code:                                                                       │
│   embedding = await self.extract_embedding(audio_data, sample_rate)             │
│   if embedding is None:                                                         │
│       return MatchResult(matched=False, similarity=0.0, ...)                    │
│                                                                                 │
│ FAILURE POINT 6: extract_embedding() returns None                               │
│   → ECAPA encoder not loaded                                                    │
│   → This is THE ROOT CAUSE of 0.0% confidence!                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 5.2: EMBEDDING EXTRACTION (THE CRITICAL COMPONENT)                         │
│ ───────────────────────────────────────────────────────                         │
│ Component: UnifiedVoiceCacheManager.extract_embedding()                         │
│ File: backend/voice_unlock/unified_voice_cache_manager.py:1785-1900             │
│                                                                                 │
│ Actions:                                                                        │
│   Step 0: ensure_encoder_available() ← CRITICAL FIX                             │
│   Step 1: Try hot cache (direct ECAPA encoder)                                  │
│   Step 2: Try model loader                                                      │
│   Step 3: Try ML Registry                                                       │
│   Step 4: Process audio through ECAPA-TDNN                                      │
│   Step 5: L2-normalize embedding (192 dimensions)                               │
│                                                                                 │
│ FAILURE POINT 7: No encoder available in any path                               │
│   → All encoder attempts fail                                                   │
│   → Returns None                                                                │
│   → Verification returns 0.0% confidence                                        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 5.3: ECAPA ENCODER ACQUISITION                                             │
│ ────────────────────────────────────                                            │
│ Component: ensure_encoder_available() / ensure_ecapa_available()                │
│ Files: backend/voice_unlock/unified_voice_cache_manager.py                      │
│        backend/voice_unlock/ml_engine_registry.py                               │
│                                                                                 │
│ Strategy Chain:                                                                 │
│   1. Check _direct_ecapa_encoder (cached locally)                               │
│   2. Check _model_loader.get_model("ecapa_encoder")                             │
│   3. Call ensure_ecapa_available() from ml_engine_registry                      │
│      a. Get ML Registry singleton                                               │
│      b. Try registry.get_ecapa_encoder()                                        │
│      c. Try registry.load_ecapa() on-demand                                     │
│      d. Try cloud fallback if local fails                                       │
│                                                                                 │
│ FAILURE POINT 8: ML Registry never created                                      │
│   → get_ml_registry_sync() returns None                                         │
│   → No registry → No ECAPA → No embedding → 0.0%                                │
│                                                                                 │
│ FAILURE POINT 9: ECAPA load fails                                               │
│   → Model files missing or corrupted                                            │
│   → Not enough RAM (ECAPA needs ~500MB)                                         │
│   → PyTorch/SpeechBrain import error                                            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: ML ENGINE REGISTRY (STARTUP DEPENDENCY)                                 │
│ ───────────────────────────────────────────────                                 │
│ Component: MLEngineRegistry                                                     │
│ File: backend/voice_unlock/ml_engine_registry.py                                │
│       backend/main.py:1279-1436 (startup initialization)                        │
│                                                                                 │
│ Startup Flow in main.py:                                                        │
│   1. _create_registry_robust() - 3 fallback strategies                          │
│   2. Strategy 1: get_ml_registry() async                                        │
│   3. Strategy 2: get_ml_registry_sync(auto_create=True)                         │
│   4. Strategy 3: Direct MLEngineRegistry()                                      │
│   5. Emergency creation in exception handlers                                   │
│   6. Store in app.state.ml_registry                                             │
│                                                                                 │
│ FAILURE POINT 10: All registry creation strategies fail                         │
│   → Import errors in ml_engine_registry.py                                      │
│   → Registry is None for entire server lifetime                                 │
│   → Every unlock attempt returns 0.0%                                           │
│                                                                                 │
│ FAILURE POINT 11: Registry created but ECAPA not prewarmed                      │
│   → First unlock takes 3-5 seconds to load ECAPA                                │
│   → May timeout if timeout < load time                                          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 7: SPEECHBRAIN FALLBACK PATH                                               │
│ ────────────────────────────────                                                │
│ Component: SpeechBrainEngine.verify_speaker()                                   │
│ File: backend/voice/speechbrain_engine.py                                       │
│                                                                                 │
│ This path is used when unified cache fails:                                     │
│   1. Check if speaker_encoder is loaded                                         │
│   2. Extract embedding via speaker_encoder.encode_batch()                       │
│   3. Compare against stored profile embedding                                   │
│   4. Return cosine similarity as confidence                                     │
│                                                                                 │
│ FAILURE POINT 12: speaker_encoder is None                                       │
│   → SpeechBrain not initialized                                                 │
│   → Returns confidence = 0.0                                                    │
│                                                                                 │
│ FAILURE POINT 13: Audio conversion fails                                        │
│   → _audio_bytes_to_tensor() fails                                              │
│   → Returns confidence = 0.0                                                    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 8: PROFILE LOOKUP                                                          │
│ ──────────────────────                                                          │
│ Component: SpeakerVerificationService.speaker_profiles                          │
│ File: backend/voice/speaker_verification_service.py:4577-4616                   │
│                                                                                 │
│ Profile Checks:                                                                 │
│   1. Is speaker_name in speaker_profiles? (line 4579)                           │
│   2. Does profile have embedding? (line 4581)                                   │
│   3. Does profile require enrollment? (line 4584)                               │
│   4. Is embedding norm valid? (line 4607)                                       │
│                                                                                 │
│ FAILURE POINT 14: Profile not found                                             │
│   → speaker_name not in speaker_profiles                                        │
│   → Falls through to unknown speaker path                                       │
│                                                                                 │
│ FAILURE POINT 15: Profile requires enrollment                                   │
│   → Line 4584: requires_enrollment=True or embedding=None                       │
│   → Returns: {"verified": False, "confidence": 0.0,                             │
│              "error": "enrollment_required"}                                    │
│                                                                                 │
│ FAILURE POINT 16: Corrupted profile                                             │
│   → Line 4607: Embedding norm is 0 or near-zero                                 │
│   → Returns: {"verified": False, "confidence": 0.0,                             │
│              "error": "corrupted_profile"}                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 9: FINAL DECISION                                                          │
│ ──────────────────────                                                          │
│ Component: async_pipeline.py:1421-1432                                          │
│                                                                                 │
│ If verification_result.verified == False:                                       │
│   confidence = verification_result.get("confidence", 0.0)                       │
│   message = f"Voice verification failed (confidence: {confidence:.1%})"         │
│                                                                                 │
│ This is the error message you see!                                              │
│                                                                                 │
│ The 0.0% confidence means:                                                      │
│   - ECAPA encoder was not available, OR                                         │
│   - Embedding extraction returned None, OR                                      │
│   - Profile was missing/corrupted, OR                                           │
│   - Audio was silent/corrupted                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Root Cause Analysis Tree

```
                     ┌─────────────────────────────────────┐
                     │   "confidence: 0.0%" ERROR          │
                     └─────────────────────────────────────┘
                                        │
                ┌───────────────────────┼───────────────────────┐
                │                       │                       │
                ▼                       ▼                       ▼
    ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
    │ AUDIO PROBLEM     │   │ ENCODER PROBLEM   │   │ PROFILE PROBLEM   │
    │                   │   │ (MOST LIKELY)     │   │                   │
    └───────────────────┘   └───────────────────┘   └───────────────────┘
            │                       │                       │
    ┌───────┴───────┐       ┌───────┴───────┐       ┌───────┴───────┐
    │               │       │               │       │               │
    ▼               ▼       ▼               ▼       ▼               ▼
┌───────┐     ┌───────┐ ┌───────┐     ┌───────┐ ┌───────┐     ┌───────┐
│Silent │     │Corrupt│ │Registry│    │ECAPA  │ │No     │     │Zero   │
│Audio  │     │Audio  │ │=None  │    │Not    │ │Enroll-│     │Norm   │
│       │     │       │ │       │    │Loaded │ │ment   │     │Embed- │
│       │     │       │ │       │    │       │ │       │     │ding   │
└───────┘     └───────┘ └───────┘    └───────┘ └───────┘     └───────┘
    │               │       │            │         │              │
    │               │       │            │         │              │
    ▼               ▼       ▼            ▼         ▼              ▼
"Energy      "Frame    "main.py    "Model     "Say        "Profile
<0.0001"     decode    startup     loading   'learn my    needs
             error"    failed"    timed      voice'"     re-enroll"
                                  out"
```

---

## Diagnostic Checklist

### Quick Diagnosis Commands

```bash
# 1. Test if ECAPA encoder loads
PYTHONPATH="$PWD:$PWD/backend" ./backend/venv/bin/python3 << 'EOF'
from voice_unlock.ml_engine_registry import ensure_ecapa_available
import asyncio
result = asyncio.run(ensure_ecapa_available(timeout=45))
print(f"ECAPA Available: {result[0]}, Message: {result[1]}")
EOF

# 2. Test embedding extraction
PYTHONPATH="$PWD:$PWD/backend" ./backend/venv/bin/python3 << 'EOF'
import asyncio
import numpy as np
from voice_unlock.unified_voice_cache_manager import get_unified_cache_manager

async def test():
    cache = get_unified_cache_manager()
    test_audio = np.random.randn(24000).astype(np.float32)
    embedding = await cache.extract_embedding(test_audio, sample_rate=16000)
    print(f"Embedding: {embedding.shape if embedding is not None else 'None'}")

asyncio.run(test())
EOF

# 3. Check ML Registry status
PYTHONPATH="$PWD:$PWD/backend" ./backend/venv/bin/python3 << 'EOF'
from voice_unlock.ml_engine_registry import get_ml_registry_sync
registry = get_ml_registry_sync(auto_create=True)
print(f"Registry: {'Exists' if registry else 'None'}")
if registry:
    print(f"  is_ready: {registry.is_ready}")
    print(f"  engines: {list(registry._engines.keys())}")
EOF

# 4. Check voiceprint profiles
PYTHONPATH="$PWD:$PWD/backend" ./backend/venv/bin/python3 << 'EOF'
from voice.speaker_verification_service import get_speaker_verification_service
import asyncio

async def check():
    svc = await get_speaker_verification_service()
    print(f"Profiles loaded: {len(svc.speaker_profiles)}")
    for name, profile in svc.speaker_profiles.items():
        has_emb = profile.get('embedding') is not None
        print(f"  - {name}: embedding={'Yes' if has_emb else 'NO!'}")

asyncio.run(check())
EOF
```

---

## Common Failure Scenarios and Fixes

### Scenario 1: Registry Never Created

**Symptom:** 0.0% confidence on every unlock attempt
**Diagnosis:** Check Ironcliw startup logs for "ML Engine Registry was NOT created"
**Fix:** The fix in main.py:1279-1436 adds 3 fallback strategies
**Verify:** Restart Ironcliw and check for "Registry created via..."

### Scenario 2: ECAPA Never Loads

**Symptom:** First unlock takes forever, then fails
**Diagnosis:** Check for "ECAPA not available" in logs
**Fix:** ensure_encoder_available() added to extract_embedding()
**Verify:** Run diagnostic command #1 above

### Scenario 3: No Voiceprint Enrolled

**Symptom:** 0.0% with "enrollment_required" in logs
**Diagnosis:** Check diagnostic command #4 above
**Fix:** Say "Ironcliw, learn my voice" to enroll voiceprint

### Scenario 4: Corrupted Profile

**Symptom:** 0.0% with "corrupted_profile" in logs
**Diagnosis:** Embedding norm is 0 or near-zero
**Fix:** Delete profile and re-enroll

---

## File Reference Cross-Index

| Component | File | Key Lines | Responsibility |
|-----------|------|-----------|----------------|
| Audio Capture | `voice/unified_vad_api.py` | - | Microphone input, VAD |
| Intent Detection | `core/async_pipeline.py` | 1091-1094 | Unlock pattern matching |
| Fast Unlock Handler | `core/async_pipeline.py` | 1236-1506 | Orchestrates verification |
| Enhanced Verification | `voice/speaker_verification_service.py` | 2525-2750 | Multi-factor fusion |
| Core Verification | `voice/speaker_verification_service.py` | 4440-4730 | Profile matching |
| Unified Cache | `voice_unlock/unified_voice_cache_manager.py` | - | Fast path, embedding extraction |
| Embedding Extraction | `voice_unlock/unified_voice_cache_manager.py` | 1785-1900 | ECAPA inference |
| ML Registry | `voice_unlock/ml_engine_registry.py` | - | ECAPA model management |
| Startup Init | `main.py` | 1279-1436 | Registry creation |
| SpeechBrain Fallback | `voice/speechbrain_engine.py` | - | Legacy verification path |

---

## Logging Locations to Check

When debugging 0.0% confidence, check these log entries:

1. **Audio Debug:**
   - Look for: `🎤 AUDIO DEBUG: Energy level = X.XXXXXX`
   - If < 0.0001, audio is silent

2. **Unified Cache:**
   - Look for: `⚡ UNIFIED CACHE INSTANT MATCH` or `🔐 UNIFIED CACHE UNLOCK MATCH`
   - If missing, cache path failed

3. **ECAPA Encoder:**
   - Look for: `✅ ECAPA available` or `❌ ECAPA not available`
   - This tells you if embeddings can be extracted

4. **Profile Check:**
   - Look for: `🔍 DEBUG: Verifying [name]`
   - Look for: `🔍 DEBUG: Stored embedding shape:`

5. **Final Result:**
   - Look for: `✅ Verification complete: X.X% (PASS/FAIL)`
   - If you don't see this, verification didn't complete

---

*Document created: 2024-12-05*
*Last updated: 2024-12-05*
*Related files: See cross-index above*
