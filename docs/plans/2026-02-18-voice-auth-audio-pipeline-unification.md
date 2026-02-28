# Voice Auth ↔ Audio Pipeline Unification

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate 7 architectural gaps between the voice unlock/VBIA stack and the real-time voice conversation pipeline, creating a unified audio routing layer where biometric authentication flows through AudioBus (AEC-cleaned audio), ConversationPipeline pauses during auth, and continuous speaker verification feeds the VBIA adapter transparently.

**Architecture:** All mic audio flows through AudioBus (AEC-cleaned, 16kHz float32). ModeDispatcher becomes the integration hub: entering BIOMETRIC mode pauses ConversationPipeline, captures audio via AudioBus consumer, runs authentication through IntelligentVoiceUnlockService, speaks challenges/results via TTS→AudioBus (preserving AEC reference), and resumes the previous mode on completion. During CONVERSATION mode, a background speaker verification monitor feeds cached results to TieredVBIAAdapter via `set_verification_result()`, enabling transparent Tier 2 escalation without interrupting conversation flow.

**Tech Stack:** asyncio, AudioBus mic consumer API, ConversationPipeline, ModeDispatcher, TieredVBIAAdapter, IntelligentVoiceUnlockService, existing TTS engine

---

## Gap-to-Task Mapping

| Gap | Description | Task |
|-----|-------------|------|
| 3 | ConversationPipeline has no pause/resume | Task 1 |
| 1 | ModeDispatcher BIOMETRIC is a `pass` stub | Task 2 |
| 2 | VBIA and AudioBus have competing mic capture | Task 2 |
| 5 | Voice unlock TTS bypasses AudioBus (no AEC ref) | Task 2 |
| 4 | No continuous speaker verification in conversation | Task 3 |
| 7 | No conversation-aware transparent auth escalation | Task 3 |
| 6 | Supervisor doesn't pass refs after two-tier init | Task 4, 5 |

## Scope & Non-Goals

**In scope:**
- `backend/audio/conversation_pipeline.py` — pause/resume mechanism
- `backend/audio/mode_dispatcher.py` — BIOMETRIC wiring, continuous speaker verification, AudioBus consumer
- `backend/audio/audio_pipeline_bootstrap.py` — PipelineHandle fields, wiring updates, shutdown cleanup
- `unified_supervisor.py` — pass VBIA adapter + voice unlock service refs to ModeDispatcher after two-tier init

**Out of scope:**
- New files (all changes go into existing files)
- Voice unlock module internals (we use its public API)
- Cross-repo changes (Ironcliw Prime, Reactor Core)
- Audio capture refactoring inside voice_unlock/audio_capture.py

---

### Task 1: ConversationPipeline Pause/Resume (Gap 3)

**Files:**
- Modify: `backend/audio/conversation_pipeline.py`

**Why:** When biometric authentication starts, the conversation loop must pause (stop listening/responding) but NOT end the session. After auth completes, the conversation resumes from where it left off — preserving the full session transcript and sliding window context.

**Step 1: Add pause/resume state and methods**

In `ConversationPipeline.__init__()` (after line 208 `self._running = False`), add:

```python
        self._resume_event = asyncio.Event()
        self._resume_event.set()  # Start unpaused
```

Add two new methods after `end_session()` (after line 244):

```python
    async def pause(self) -> None:
        """
        Pause the conversation loop without ending the session.

        The loop stops listening/responding but the session transcript
        is preserved. Call resume() to continue.
        """
        if not self._running:
            return
        self._resume_event.clear()
        logger.info("[ConvPipeline] Paused")

    async def resume(self) -> None:
        """
        Resume a paused conversation loop.

        The loop continues from where it left off with the full
        session transcript intact.
        """
        self._resume_event.set()
        logger.info("[ConvPipeline] Resumed")

    @property
    def is_paused(self) -> bool:
        """Whether the conversation loop is currently paused."""
        return self._running and not self._resume_event.is_set()
```

**Step 2: Add pause check to conversation loop**

In `_conversation_loop()` (line 287), insert a pause check at the top of the while loop body, immediately after the `while self._running and self._session is not None:` line and before the `if self._session.is_expired:` check:

```python
            # Wait if paused (biometric auth in progress)
            if not self._resume_event.is_set():
                logger.debug("[ConvPipeline] Waiting for resume...")
                await self._resume_event.wait()
                if not self._running:
                    break
```

**Step 3: Update get_status() to include paused state**

In `get_status()` (line 632), add `"paused"` to the returned dict:

```python
            "paused": self.is_paused,
```

**Step 4: Commit**

```bash
git add backend/audio/conversation_pipeline.py
git commit -m "feat: add pause/resume to ConversationPipeline for biometric auth"
```

---

### Task 2: Wire ModeDispatcher BIOMETRIC Mode (Gaps 1, 2, 5)

**Files:**
- Modify: `backend/audio/mode_dispatcher.py`

**Why:** The BIOMETRIC `_enter_mode` is a `pass` stub. We need to:
1. Pause ConversationPipeline when entering biometric mode
2. Capture audio via AudioBus mic consumer (not competing mic path)
3. Run authentication through IntelligentVoiceUnlockService
4. Speak challenges/results via TTS→AudioBus for AEC reference
5. Return to previous mode when done

**Step 1: Add imports and biometric state fields**

At the top of `mode_dispatcher.py`, after the existing imports (line 25), add:

```python
from typing import Any, Callable, Dict, List, Optional
```

Replace the existing `from typing import Optional` (line 24).

In `ModeDispatcher.__init__()`, after `self._mode_history = []` (line 82), add:

```python
        # Biometric authentication state
        self._biometric_task: Optional[asyncio.Task] = None
        self._biometric_audio_buffer: List[Any] = []
        self._biometric_audio_consumer: Optional[Callable] = None

        # Lazy-loaded service references (set by supervisor after two-tier init)
        self._audio_bus = None
        self._tts_engine = None
        self._voice_unlock_service = None
        self._vbia_adapter = None
```

Add setters after `set_speech_state()` (after line 94):

```python
    def set_audio_bus(self, audio_bus) -> None:
        """Set the AudioBus reference for mic consumer registration."""
        self._audio_bus = audio_bus

    def set_tts_engine(self, tts_engine) -> None:
        """Set the TTS engine reference for biometric challenges."""
        self._tts_engine = tts_engine

    def set_voice_unlock_service(self, service) -> None:
        """Set the IntelligentVoiceUnlockService reference."""
        self._voice_unlock_service = service

    def set_vbia_adapter(self, adapter) -> None:
        """Set the TieredVBIAAdapter reference."""
        self._vbia_adapter = adapter
```

**Step 2: Wire `_enter_mode(BIOMETRIC)`**

Replace the BIOMETRIC stub in `_enter_mode()` (lines 179-180):

```python
        elif mode == VoiceMode.BIOMETRIC:
            # Pause conversation if it was running
            if self._conversation_pipeline is not None and hasattr(
                self._conversation_pipeline, 'pause'
            ):
                await self._conversation_pipeline.pause()

            # Launch biometric authentication as background task
            self._biometric_task = asyncio.ensure_future(
                self._run_biometric_authentication()
            )
            self._biometric_task.add_done_callback(
                self._on_biometric_done
            )
```

**Step 3: Wire `_leave_mode(BIOMETRIC)`**

In `_leave_mode()`, after the CONVERSATION block (after line 159), add the BIOMETRIC block:

```python
        elif mode == VoiceMode.BIOMETRIC:
            # Cancel biometric task if still running
            if self._biometric_task is not None:
                self._biometric_task.cancel()
                try:
                    await self._biometric_task
                except asyncio.CancelledError:
                    pass
                self._biometric_task = None

            # Unregister AudioBus mic consumer
            self._unregister_biometric_consumer()

            # Clear audio buffer
            self._biometric_audio_buffer.clear()

            # Resume conversation if it was paused
            if self._conversation_pipeline is not None and hasattr(
                self._conversation_pipeline, 'resume'
            ):
                await self._conversation_pipeline.resume()
```

**Step 4: Add biometric authentication runner**

Add the following methods after `_on_conversation_done()` (after line 268):

```python
    async def _run_biometric_authentication(self) -> None:
        """
        Execute voice biometric authentication flow.

        1. Register as AudioBus mic consumer to capture AEC-cleaned audio
        2. Speak challenge prompt via TTS→AudioBus (AEC reference)
        3. Capture user's voice response from AudioBus frames
        4. Run verification through IntelligentVoiceUnlockService
        5. Speak result via TTS→AudioBus
        6. Return to previous mode
        """
        import numpy as np

        try:
            # 1. Register mic consumer on AudioBus for AEC-cleaned capture
            self._biometric_audio_buffer.clear()

            def _on_biometric_frame(frame: np.ndarray) -> None:
                """Accumulate AEC-cleaned audio frames for biometric verification."""
                if frame.size > 0:
                    self._biometric_audio_buffer.append(frame.copy())

            self._biometric_audio_consumer = _on_biometric_frame
            if self._audio_bus is not None:
                self._audio_bus.register_mic_consumer(_on_biometric_frame)

            # 2. Speak challenge via TTS→AudioBus for AEC reference
            await self._speak_biometric("Verifying your voice now.")

            # 3. Capture audio for verification (2.5 seconds of AEC-cleaned audio)
            capture_duration = float(
                os.getenv("Ironcliw_BIOMETRIC_CAPTURE_DURATION", "2.5")
            )
            await asyncio.sleep(capture_duration)

            # 4. Concatenate captured frames
            if not self._biometric_audio_buffer:
                await self._speak_biometric(
                    "I couldn't capture your voice. Please try again."
                )
                return

            audio_data = np.concatenate(self._biometric_audio_buffer)

            # Unregister consumer before processing (stop capturing)
            self._unregister_biometric_consumer()

            # 5. Run authentication
            auth_result = await self._authenticate_voice(audio_data)

            # 6. Speak result via TTS→AudioBus
            if auth_result.get("success"):
                speaker = auth_result.get("speaker", "")
                await self._speak_biometric(
                    f"Voice verified. Welcome back, {speaker}." if speaker
                    else "Voice verified. Unlocking now."
                )
            else:
                reason = auth_result.get("reason", "verification failed")
                await self._speak_biometric(
                    f"Voice verification unsuccessful. {reason}"
                )

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"[ModeDispatcher] Biometric auth error: {e}")
            await self._speak_biometric(
                "Voice authentication encountered an error. "
                "Please try again or use password unlock."
            )
        finally:
            self._unregister_biometric_consumer()
            self._biometric_audio_buffer.clear()

    async def _authenticate_voice(self, audio_data) -> Dict[str, Any]:
        """
        Run voice authentication through IntelligentVoiceUnlockService.

        Falls back to VBIA adapter if the unlock service is unavailable.
        """
        # Primary: IntelligentVoiceUnlockService
        if self._voice_unlock_service is not None:
            try:
                result = await asyncio.wait_for(
                    self._voice_unlock_service.process_voice_unlock_command(
                        audio_data=audio_data,
                        context={
                            "source": "mode_dispatcher",
                            "previous_mode": (
                                self._previous_mode.value
                                if self._previous_mode else "unknown"
                            ),
                            "aec_cleaned": True,
                        },
                    ),
                    timeout=float(os.getenv("Ironcliw_BIOMETRIC_AUTH_TIMEOUT", "25")),
                )
                return result
            except asyncio.TimeoutError:
                logger.warning("[ModeDispatcher] Voice unlock timed out")
                return {"success": False, "reason": "Authentication timed out"}
            except Exception as e:
                logger.error(f"[ModeDispatcher] Voice unlock error: {e}")

        # Fallback: TieredVBIAAdapter
        if self._vbia_adapter is not None:
            try:
                threshold = float(os.getenv("Ironcliw_BIOMETRIC_THRESHOLD", "0.85"))
                passed, confidence = await self._vbia_adapter.verify_speaker(threshold)
                return {
                    "success": passed,
                    "confidence": confidence,
                    "reason": "verified" if passed else "below threshold",
                }
            except Exception as e:
                logger.error(f"[ModeDispatcher] VBIA fallback error: {e}")

        return {"success": False, "reason": "No authentication service available"}

    async def _speak_biometric(self, text: str) -> None:
        """
        Speak biometric feedback through TTS→AudioBus for AEC reference.

        Falls back to direct TTS if AudioBus is unavailable.
        """
        if self._tts_engine is None:
            logger.info(f"[ModeDispatcher] Biometric (no TTS): {text}")
            return

        try:
            import io
            import numpy as np

            if (
                hasattr(self._tts_engine, 'synthesize')
                and self._audio_bus is not None
            ):
                tts_result = await self._tts_engine.synthesize(text)
                if tts_result is not None:
                    audio_bytes = getattr(tts_result, 'audio_data', None)
                    sample_rate = getattr(tts_result, 'sample_rate', 22050)

                    if audio_bytes is not None:
                        try:
                            import soundfile as sf
                            audio_np, file_sr = sf.read(
                                io.BytesIO(audio_bytes), dtype='float32',
                            )
                            sample_rate = file_sr
                        except Exception:
                            audio_np = np.frombuffer(
                                audio_bytes, dtype=np.int16,
                            ).astype(np.float32) / 32767.0

                        _chunk_size = int(sample_rate * 0.1)

                        async def _chunks():
                            for i in range(0, len(audio_np), _chunk_size):
                                yield audio_np[i:i + _chunk_size]

                        await self._audio_bus.play_stream(
                            _chunks(), sample_rate,
                        )
                        return

            # Fallback: direct TTS (no AEC reference)
            if hasattr(self._tts_engine, 'speak'):
                await self._tts_engine.speak(text, play_audio=True, source="biometric")
        except Exception as e:
            logger.debug(f"[ModeDispatcher] Biometric TTS error: {e}")

    def _unregister_biometric_consumer(self) -> None:
        """Unregister biometric mic consumer from AudioBus."""
        if self._biometric_audio_consumer is not None and self._audio_bus is not None:
            try:
                self._audio_bus.unregister_mic_consumer(
                    self._biometric_audio_consumer
                )
            except Exception:
                pass
            self._biometric_audio_consumer = None

    def _on_biometric_done(self, task: asyncio.Task) -> None:
        """Handle biometric task completion — return to previous mode."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(f"[ModeDispatcher] Biometric task failed: {exc}")
        # Return to previous mode
        if self._current_mode == VoiceMode.BIOMETRIC:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.return_from_biometric())
            except RuntimeError:
                self._current_mode = self._previous_mode or VoiceMode.COMMAND
```

**Step 5: Update get_status() to include biometric state**

In `get_status()` (line 241), add:

```python
            "biometric_active": self._current_mode == VoiceMode.BIOMETRIC,
            "biometric_task_running": (
                self._biometric_task is not None
                and not self._biometric_task.done()
            ),
```

**Step 6: Commit**

```bash
git add backend/audio/mode_dispatcher.py
git commit -m "feat: wire ModeDispatcher BIOMETRIC mode with AudioBus capture and TTS"
```

---

### Task 3: Continuous Speaker Verification Monitor (Gaps 4, 7)

**Files:**
- Modify: `backend/audio/mode_dispatcher.py`

**Why:** During CONVERSATION mode, we should continuously verify the speaker's identity in the background using AudioBus mic frames. Results are cached in TieredVBIAAdapter via `set_verification_result()`, enabling transparent Tier 2 escalation (e.g., "delete all files" requires confidence >0.90) without interrupting conversation flow.

If confidence drops below threshold during conversation, ModeDispatcher can trigger automatic biometric re-verification.

**Step 1: Add speaker verification state**

In `ModeDispatcher.__init__()`, after the biometric state fields added in Task 2, add:

```python
        # Continuous speaker verification state
        self._speaker_verification_task: Optional[asyncio.Task] = None
        self._speaker_audio_buffer: List[Any] = []
        self._speaker_audio_consumer: Optional[Callable] = None
        self._last_speaker_confidence: float = 0.0
        self._speaker_verification_interval = float(
            os.getenv("Ironcliw_SPEAKER_VERIFY_INTERVAL", "15")
        )
        self._speaker_confidence_threshold = float(
            os.getenv("Ironcliw_SPEAKER_CONFIDENCE_THRESHOLD", "0.70")
        )
```

**Step 2: Start verification in CONVERSATION enter, stop in CONVERSATION leave**

In `_enter_mode()`, inside the CONVERSATION block (after line 177, after the conversation task launch), add:

```python
                # Start continuous speaker verification
                self._start_speaker_verification()

```

In `_leave_mode()`, inside the CONVERSATION block (before `if self._speech_state is not None:` at line 154), add:

```python
            # Stop continuous speaker verification
            self._stop_speaker_verification()

```

**Step 3: Add speaker verification methods**

Add after `_on_biometric_done()`:

```python
    def _start_speaker_verification(self) -> None:
        """Start continuous speaker verification during conversation mode."""
        if self._audio_bus is None:
            return

        import numpy as np

        self._speaker_audio_buffer.clear()

        def _on_speaker_frame(frame: np.ndarray) -> None:
            """Accumulate frames for periodic speaker verification."""
            if frame.size > 0:
                self._speaker_audio_buffer.append(frame.copy())
                # Cap buffer at ~5 seconds of audio (16kHz * 5s = 80000 samples)
                total = sum(f.size for f in self._speaker_audio_buffer)
                while total > 80000 and len(self._speaker_audio_buffer) > 1:
                    total -= self._speaker_audio_buffer[0].size
                    self._speaker_audio_buffer.pop(0)

        self._speaker_audio_consumer = _on_speaker_frame
        self._audio_bus.register_mic_consumer(_on_speaker_frame)

        self._speaker_verification_task = asyncio.ensure_future(
            self._speaker_verification_loop()
        )
        logger.info("[ModeDispatcher] Speaker verification monitor started")

    def _stop_speaker_verification(self) -> None:
        """Stop continuous speaker verification."""
        if self._speaker_verification_task is not None:
            self._speaker_verification_task.cancel()
            self._speaker_verification_task = None

        if self._speaker_audio_consumer is not None and self._audio_bus is not None:
            try:
                self._audio_bus.unregister_mic_consumer(
                    self._speaker_audio_consumer
                )
            except Exception:
                pass
            self._speaker_audio_consumer = None

        self._speaker_audio_buffer.clear()
        logger.info("[ModeDispatcher] Speaker verification monitor stopped")

    async def _speaker_verification_loop(self) -> None:
        """
        Periodically verify speaker identity during conversation mode.

        Results feed into TieredVBIAAdapter via set_verification_result()
        for transparent Tier 2 escalation.
        """
        import numpy as np

        try:
            while self._current_mode == VoiceMode.CONVERSATION:
                await asyncio.sleep(self._speaker_verification_interval)

                if not self._speaker_audio_buffer:
                    continue

                # Snapshot and clear buffer
                audio_data = np.concatenate(self._speaker_audio_buffer)
                self._speaker_audio_buffer.clear()

                # Skip if too little audio (less than 0.5s)
                if audio_data.size < 8000:
                    continue

                # Run verification
                try:
                    confidence = await self._verify_speaker_identity(audio_data)
                    self._last_speaker_confidence = confidence

                    # Feed result to VBIA adapter cache
                    if self._vbia_adapter is not None:
                        self._vbia_adapter.set_verification_result(
                            confidence=confidence,
                            speaker_id="owner",
                            is_owner=confidence >= self._speaker_confidence_threshold,
                            verified=confidence >= self._speaker_confidence_threshold,
                            metadata={
                                "source": "continuous_verification",
                                "mode": "conversation",
                                "aec_cleaned": True,
                            },
                        )

                    # Auto-escalate if confidence drops
                    if confidence < self._speaker_confidence_threshold:
                        logger.warning(
                            f"[ModeDispatcher] Speaker confidence dropped: "
                            f"{confidence:.2f} < {self._speaker_confidence_threshold}"
                        )
                        # Switch to biometric for re-verification
                        await self.switch_mode(VoiceMode.BIOMETRIC)
                        return

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.debug(f"[ModeDispatcher] Speaker verify error: {e}")

        except asyncio.CancelledError:
            pass

    async def _verify_speaker_identity(self, audio_data) -> float:
        """
        Verify speaker identity from audio data.

        Returns confidence score (0.0 to 1.0).
        Uses voice unlock service or VBIA adapter.
        """
        # Try VBIA adapter first (lighter weight, designed for continuous use)
        if self._vbia_adapter is not None:
            try:
                threshold = self._speaker_confidence_threshold
                passed, confidence = await asyncio.wait_for(
                    self._vbia_adapter.verify_speaker(threshold),
                    timeout=5.0,
                )
                return confidence
            except Exception:
                pass

        # Fallback: voice unlock service
        if self._voice_unlock_service is not None:
            try:
                result = await asyncio.wait_for(
                    self._voice_unlock_service.process_voice_unlock_command(
                        audio_data=audio_data,
                        context={
                            "source": "continuous_verification",
                            "mode": "conversation",
                        },
                    ),
                    timeout=10.0,
                )
                return result.get("confidence", 0.0)
            except Exception:
                pass

        return 0.0
```

**Step 4: Update get_status()**

Add to `get_status()`:

```python
            "speaker_confidence": self._last_speaker_confidence,
            "speaker_verification_active": (
                self._speaker_verification_task is not None
                and not self._speaker_verification_task.done()
            ),
```

**Step 5: Commit**

```bash
git add backend/audio/mode_dispatcher.py
git commit -m "feat: add continuous speaker verification during conversation mode"
```

---

### Task 4: Bootstrap Phase 2 Wiring Updates (Gap 6 — partial)

**Files:**
- Modify: `backend/audio/audio_pipeline_bootstrap.py`

**Why:** The `PipelineHandle` needs to carry AudioBus and TTS engine references so the supervisor can pass them to ModeDispatcher after two-tier security init. Also, the ModeDispatcher created in bootstrap should receive the AudioBus and TTS engine references immediately.

**Step 1: Pass audio_bus and tts_engine to ModeDispatcher in bootstrap**

In `wire_conversation_pipeline()`, step 5 (ModeDispatcher creation, lines 174-184), update to also pass audio_bus and tts_engine:

```python
    # 5. ModeDispatcher
    try:
        from backend.audio.mode_dispatcher import ModeDispatcher

        handle.mode_dispatcher = ModeDispatcher(
            conversation_pipeline=handle.conversation_pipeline,
            speech_state=speech_state,
        )
        # Wire AudioBus and TTS for biometric mode
        if audio_bus is not None:
            handle.mode_dispatcher.set_audio_bus(audio_bus)
        if handle.tts_engine is not None:
            handle.mode_dispatcher.set_tts_engine(handle.tts_engine)
        await handle.mode_dispatcher.start()
        logger.info("[Bootstrap] ModeDispatcher started")
    except Exception as e:
        logger.warning(f"[Bootstrap] ModeDispatcher init skipped: {e}")
```

**Step 2: Add biometric consumer cleanup to shutdown**

In `shutdown()`, after step 1 (ModeDispatcher stop, line 237), add cleanup for speaker verification consumer:

```python
    # 1b. Stop speaker verification (unregisters its AudioBus consumer)
    if handle.mode_dispatcher is not None:
        try:
            if hasattr(handle.mode_dispatcher, '_stop_speaker_verification'):
                handle.mode_dispatcher._stop_speaker_verification()
        except Exception:
            pass
```

**Step 3: Commit**

```bash
git add backend/audio/audio_pipeline_bootstrap.py
git commit -m "feat: wire AudioBus and TTS into ModeDispatcher via bootstrap"
```

---

### Task 5: Supervisor Pass-Through After Two-Tier Init (Gap 6 — complete)

**Files:**
- Modify: `unified_supervisor.py`

**Why:** After `_initialize_two_tier_security()` completes (line 63768-63770), the VBIA adapter (`self._vbia_adapter`) is available. We need to pass it + the voice unlock service to ModeDispatcher so biometric auth and continuous speaker verification can use them.

**Step 1: Find insertion point**

After the two-tier security init block completes (after `_two_tier_ok = await asyncio.wait_for(self._initialize_two_tier_security(), timeout=...)` and its exception handlers), insert:

```python
            # v242.0: Wire VBIA adapter + voice unlock service into ModeDispatcher
            # ModeDispatcher needs these for biometric auth (Gap 1) and continuous
            # speaker verification (Gap 4). These are available after two-tier init.
            if self._mode_dispatcher is not None:
                if self._vbia_adapter is not None:
                    self._mode_dispatcher.set_vbia_adapter(self._vbia_adapter)
                    self.logger.debug("[Phase4] VBIA adapter → ModeDispatcher")
                if hasattr(self, '_voice_unlock_service') and self._voice_unlock_service:
                    self._mode_dispatcher.set_voice_unlock_service(
                        self._voice_unlock_service
                    )
                    self.logger.debug("[Phase4] Voice unlock service → ModeDispatcher")
```

Note: `self._voice_unlock_service` may not be set yet at Phase 4 (it's set in Phase 6). That's OK — the mode dispatcher handles `None` gracefully and falls back to VBIA adapter. When Phase 6 runs, we also wire it:

In the Phase 6 voice unlock block (after `self._voice_unlock_service = _voice_svc` around line 71773), insert:

```python
                        # Wire into ModeDispatcher for biometric mode
                        if self._mode_dispatcher is not None:
                            self._mode_dispatcher.set_voice_unlock_service(
                                self._voice_unlock_service
                            )
                            self.logger.debug(
                                "[Zone6/VoiceUnlock] Wired into ModeDispatcher"
                            )
```

**Step 2: Commit**

```bash
git add unified_supervisor.py
git commit -m "feat: wire VBIA adapter + voice unlock into ModeDispatcher lifecycle"
```

---

## Risk Assessment

| Change | Risk | Mitigation |
|--------|------|------------|
| ConversationPipeline pause/resume | LOW | asyncio.Event is standard pattern, no external side effects |
| ModeDispatcher biometric wiring | MEDIUM | Lazy-loaded refs (all `None` guards), AudioBus consumer reg/unreg is idempotent |
| Continuous speaker verification | MEDIUM | Background task with cancellation, capped buffer, graceful timeout |
| Bootstrap TTS/AudioBus wiring | LOW | Additive — passes existing refs through setters |
| Supervisor pass-through | LOW | `None` checks on both ends, no behavior change if refs unavailable |

## Success Criteria

1. `ModeDispatcher._enter_mode(BIOMETRIC)` pauses conversation, captures via AudioBus, authenticates, speaks via TTS→AudioBus, returns to previous mode
2. `ConversationPipeline.pause()` / `resume()` preserves session transcript
3. Continuous speaker verification feeds `TieredVBIAAdapter.set_verification_result()` every 15s during conversation
4. Low speaker confidence auto-escalates to biometric mode
5. All TTS during biometric flows through AudioBus for AEC reference
6. No competing mic capture paths — everything through AudioBus
7. Supervisor passes VBIA adapter + voice unlock service refs after two-tier init
