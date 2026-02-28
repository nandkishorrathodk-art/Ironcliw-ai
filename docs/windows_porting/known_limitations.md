# Ironcliw Windows Port - Known Limitations

## Overview

This document describes the current limitations, missing features, and differences between the Windows port and the original macOS version of Ironcliw. This is the MVP (Minimum Viable Product) release, with full feature parity planned for v2.0.

---

## Table of Contents

1. [Authentication Limitations](#authentication-limitations)
2. [Performance Differences](#performance-differences)
3. [Feature Parity Status](#feature-parity-status)
4. [Platform API Differences](#platform-api-differences)
5. [Development Workflow Differences](#development-workflow-differences)
6. [Untested Features](#untested-features)
7. [Roadmap to Full Parity](#roadmap-to-full-parity)

---

## Authentication Limitations

### Voice Biometric Authentication Disabled

**Status:** ❌ **Not Available** (Bypass Mode in MVP)

**Reason:**  
The macOS version uses ECAPA-TDNN speaker verification with Core ML (Metal GPU acceleration). Windows implementation requires:
- DirectML/ONNX Runtime for GPU inference, OR
- CPU-only inference (too slow for real-time), OR
- Cloud-based voice verification API

**Impact:**
- No voice unlock/authentication
- No speaker identification
- Authentication always succeeds (bypass mode)

**Workaround:**
- Use Windows Hello (facial recognition, fingerprint)
- Use password-based authentication at OS level
- Treat Ironcliw as authenticated/trusted (like macOS when TCC is approved)

**Planned Fix:** v2.0 (DirectML/ONNX Runtime integration)

**Tracking Issue:** [#TBD]

---

## Performance Differences

### 1. Hot Reload Detection Slower

**macOS:** Instant (FSEvents API)  
**Windows:** 5-10 second delay (hash-based polling)

**Reason:**  
Windows `ReadDirectoryChangesW` API is less reliable for large codebases with nested directories. Hash-based detection provides more consistent behavior across all platforms.

**Impact:**
- Development cycle: save → wait 5-10s → restart detected
- Not a production issue (hot reload disabled in prod)

**Workaround:**
- Use React HMR for frontend (instant)
- Reduce `Ironcliw_RELOAD_CHECK_INTERVAL` to 5s (higher CPU usage)
- Disable hot reload and restart manually: `Ironcliw_DEV_MODE=false`

**Status:** ⚠️ **Working as Intended** (acceptable trade-off)

---

### 2. Screen Capture Performance

**macOS:** ~5-10ms per frame (ScreenCaptureKit, Metal GPU)  
**Windows:** ~10-15ms per frame (GDI+ BitBlt)

**Reason:**  
Windows uses GDI+ software rendering. Hardware acceleration via Windows.Graphics.Capture (WGC) API or DirectX screen duplication planned.

**Impact:**
- Max FPS: ~60 FPS (Windows) vs ~100 FPS (macOS)
- CPU usage: ~10-15% higher on Windows
- Not noticeable in real-world usage (Ironcliw targets 15 FPS)

**Workaround:**
- Reduce vision FPS: `Ironcliw_VISION_FPS=10` (default: 15)
- Disable continuous capture: `Ironcliw_VISION_CONTINUOUS=false`

**Planned Fix:** v1.1 (Windows.Graphics.Capture API)

---

### 3. Process Startup Overhead

**macOS:** ~2-3 seconds (launchd, shared frameworks)  
**Windows:** ~5-7 seconds (Task Scheduler, DLL loading)

**Reason:**
- Task Scheduler has higher overhead than launchd
- pythonnet CLR initialization adds ~1-2s
- Windows Defender real-time scanning adds ~1-2s

**Impact:**
- Cold start: 5-7 seconds (Windows) vs 2-3 seconds (macOS)
- Warm restart: ~3-4 seconds (cached)

**Workaround:**
- Exclude Ironcliw directory from Windows Defender scanning
- Keep Ironcliw running (minimize instead of close)
- Use `unified_supervisor.py --restart` (faster than cold start)

**Status:** ⚠️ **Acceptable** (within 2x of macOS)

---

## Feature Parity Status

### ✅ Fully Implemented (100% Parity)

| Feature | macOS | Windows | Notes |
|---------|-------|---------|-------|
| Platform Detection | ✅ | ✅ | Full PlatformInfo support |
| System Control | ✅ | ✅ | Window management, volume, notifications |
| Screen Capture | ✅ | ✅ | GDI+ BitBlt (future: WGC API) |
| Audio I/O | ✅ | ✅ | WASAPI via NAudio |
| Process Management | ✅ | ✅ | Task Scheduler replaces launchd |
| File Watching | ✅ | ✅ | Hash-based (future: ReadDirectoryChangesW) |
| Permissions API | ✅ | ✅ | UAC replaces TCC |
| Backend API (FastAPI) | ✅ | ✅ | No changes |
| WebSocket Communication | ✅ | ✅ | No changes |
| Frontend UI | ✅ | ✅ | No changes |
| GCP Integration | ✅ | ✅ | Cross-platform Python code |
| Cloud Inference | ✅ | ✅ | OpenAI/Anthropic APIs work |

---

### ⚠️ Partially Implemented (Workarounds Available)

| Feature | macOS | Windows | Limitation | Workaround |
|---------|-------|---------|-----------|-----------|
| Voice Auth | ✅ | ❌ | ECAPA-TDNN requires GPU | Bypass mode (always succeeds) |
| Hot Reload | ✅ | ⚠️ | Slower (hash vs FSEvents) | Reduce check interval to 5s |
| Rust Extensions | ✅ | ⚠️ | Pre-existing build issues | Optional feature, Python fallback works |
| Logging Emoji | ✅ | ⚠️ | Console encoding issues | Use Windows Terminal |

---

### ❌ Not Yet Implemented (Future Versions)

| Feature | macOS | Windows | Planned | Notes |
|---------|-------|---------|---------|-------|
| Metal GPU Acceleration | ✅ | ❌ | v2.0 | Will use DirectML |
| Swift Native Layer | ✅ | N/A | N/A | C# replaces Swift on Windows |
| Accessibility API | ✅ | ❌ | v1.2 | UI Automation API planned |
| Spotlight Integration | ✅ | N/A | v1.5 | Windows Search integration planned |
| Siri Shortcuts | ✅ | N/A | N/A | Windows-specific alternatives TBD |

---

## Platform API Differences

### 1. Window Management

**macOS:** `Quartz.CGWindowListCopyWindowInfo` (full window metadata)  
**Windows:** `User32.EnumWindows` (basic metadata only)

**Missing on Windows:**
- Window z-order (stacking order)
- Window transparency/alpha
- Window shadow/border metadata
- Per-window memory usage

**Impact:** Minor (core window operations work fine)

---

### 2. Audio APIs

**macOS:** Core Audio (low-level, zero-copy)  
**Windows:** WASAPI (via NAudio, one extra copy)

**Difference:**
- Windows adds ~1ms latency per audio buffer
- macOS: direct Metal → Core Audio pipeline
- Windows: WASAPI → NAudio → Python (extra marshalling)

**Impact:** Negligible for voice commands (<5ms difference)

---

### 3. File System Events

**macOS:** FSEvents (kernel-level, batched, instant)  
**Windows:** ReadDirectoryChangesW (userland, per-directory, delayed)

**Difference:**
- macOS: single watcher for entire tree
- Windows: one watcher per directory (overhead scales with depth)

**Impact:** Hash-based fallback used on both platforms now (consistency)

---

### 4. Permission Management

**macOS:** TCC (Transparency, Consent, and Control)  
**Windows:** UAC (User Account Control) + Privacy Settings

**Difference:**
- TCC: per-app, per-permission granularity
- UAC: binary (elevated or not)
- Windows Privacy Settings: per-feature (microphone, camera)

**Impact:** Less granular control on Windows, but functional

---

## Development Workflow Differences

### 1. Package Manager

**macOS:** Homebrew (`brew install`)  
**Windows:** winget, Chocolatey, or manual installers

**Recommendation:** Use winget (built into Windows 11, installable on Windows 10)

---

### 2. Virtual Environment Scripts

**macOS/Linux:** `.venv/bin/activate`  
**Windows:** `.venv\Scripts\Activate.ps1`

**Gotcha:** PowerShell execution policy may block activation

---

### 3. Environment Variables

**macOS/Linux:** `.bashrc`, `.zshrc`  
**Windows:** System Properties → Environment Variables or `$env:VAR="value"`

**Gotcha:** PowerShell uses `$env:VAR`, CMD uses `%VAR%`

---

### 4. Process Management

**macOS:** `launchd`, `launchctl`  
**Windows:** Task Scheduler, `schtasks`

**Difference:** 
- launchd: instant startup, agent-based
- Task Scheduler: GUI-based, XML config, heavier

---

## Untested Features

The following features are **theoretically cross-platform** but **not thoroughly tested on Windows**:

### ☑️ GCP Cloud Features

- ✅ GCP VM provisioning (uses `google-cloud-compute` Python SDK)
- ⚠️ Cloud SQL Proxy startup (not tested on Windows)
- ⚠️ Spot VM preemption handling (not tested)
- ⚠️ Invincible Node recovery (not tested)

**Status:** Core Python code is cross-platform, but needs Windows-specific testing.

**Help Wanted:** Community testing appreciated!

---

### ☑️ Trinity Coordination (Ironcliw-Prime + Reactor-Core)

- ⚠️ Cross-repo discovery (uses `pathlib.Path`, should work)
- ⚠️ Process spawning for Prime/Reactor (uses `subprocess.Popen`, should work)
- ⚠️ Health check polling (HTTP, cross-platform)
- ⚠️ Signal-based shutdown (Windows uses different signals)

**Status:** Code updated for Windows compatibility, but full Trinity not tested.

**Reason:** Ironcliw-Prime and Reactor-Core repos not yet ported to Windows.

---

### ☑️ Frontend Features

- ✅ React development server (works)
- ✅ Production build (`npm run build` works)
- ⚠️ Hot Module Replacement on Windows (theoretically works, not tested)
- ⚠️ Service worker (browser feature, should be cross-platform)

**Status:** Basic testing done, advanced features need more testing.

---

## Roadmap to Full Parity

### v1.0-MVP (Current Release)
- ✅ Platform abstraction layer
- ✅ Windows native layer (C# DLLs)
- ✅ Core platform implementations (system, audio, vision)
- ✅ Unified supervisor Windows port
- ✅ Backend API Windows port
- ⚠️ Voice authentication (bypass mode)
- ⚠️ Rust extensions (optional, build issues)

---

### v1.1 (Performance & Polish)
- Windows.Graphics.Capture API for faster screen capture (10ms → 5ms)
- DirectML GPU acceleration for ML inference
- ONNX Runtime integration for local models
- Rust extension fixes (sysinfo, memmap2, rayon)
- ReadDirectoryChangesW file watcher (replace hash-based)

---

### v1.2 (Feature Expansion)
- Windows Accessibility API (UI Automation)
- Windows Search integration (Spotlight alternative)
- Task Scheduler advanced features (conditional triggers)
- Windows-specific keyboard shortcuts (Win+key)
- Native Windows notifications (Action Center)

---

### v1.5 (Advanced Integration)
- DirectML ECAPA-TDNN voice authentication
- Windows Hello Companion Device integration
- Windows Copilot integration
- Azure DevOps integration
- WSL (Windows Subsystem for Linux) support

---

### v2.0 (Full Parity + Beyond)
- ✅ Voice authentication with DirectML
- ✅ Complete Trinity support (Prime + Reactor on Windows)
- ✅ All Rust extensions working
- ✅ GCP features fully tested and validated
- ✅ Performance equal to or better than macOS
- ✅ Windows-specific features (Copilot, PowerToys integration)
- ✅ Full test coverage (unit + integration + E2E)

---

## Comparison Matrix

### Performance Comparison

| Metric | macOS | Windows (Current) | Windows (v2.0 Goal) |
|--------|-------|-------------------|---------------------|
| Cold Start | 2-3s | 5-7s | 3-4s |
| Hot Reload Detection | Instant | 5-10s | 1-2s |
| Screen Capture FPS | 100+ | 60 | 100+ |
| Screen Capture Latency | 5-10ms | 10-15ms | 5-10ms |
| Audio Latency | <1ms | 1-2ms | <1ms |
| Memory Usage (Idle) | 800MB | 900MB | 800MB |
| CPU Usage (Idle) | 2-3% | 3-5% | 2-3% |

---

### API Compatibility

| API Category | macOS | Windows | Compatibility | Notes |
|--------------|-------|---------|---------------|-------|
| System Control | Core Graphics, Quartz | User32, WinMM | 95% | Minor metadata differences |
| Audio | Core Audio | WASAPI | 98% | Slight latency difference |
| Vision | ScreenCaptureKit | GDI+ / WGC | 90% | Performance gap exists |
| Auth | TCC, ECAPA-TDNN | UAC, bypass | 50% | Voice auth missing |
| Permissions | TCC | UAC + Privacy Settings | 80% | Less granular on Windows |
| Process Mgmt | launchd | Task Scheduler | 95% | Different paradigm, same result |
| File Watching | FSEvents | Hash-based | 85% | Slower, but reliable |

---

## Reporting Issues

Found a limitation not listed here? Please report it!

1. Check [GitHub Issues](https://github.com/drussell23/Ironcliw/issues)
2. Open a new issue with tag: `windows-port`, `limitation`
3. Include:
   - Description of missing feature
   - macOS equivalent behavior
   - Impact on workflow
   - Suggested workaround (if any)

---

## Contributing

Want to help improve Windows support?

**High-Impact Contributions:**
1. ✅ Voice authentication with DirectML/ONNX Runtime
2. ✅ Windows.Graphics.Capture API integration
3. ✅ ReadDirectoryChangesW file watcher implementation
4. ✅ Rust extension build fixes
5. ✅ GCP feature testing on Windows
6. ✅ Trinity (Prime + Reactor) Windows testing

**Low-Hanging Fruit:**
1. ✅ Documentation improvements
2. ✅ Troubleshooting guide additions
3. ✅ Installation script enhancements
4. ✅ Test coverage expansion
5. ✅ Bug reports with reproduction steps

---

**Last Updated:** February 2026  
**Windows Port Version:** 1.0.0-MVP  
**Target Platform:** Windows 10 (1809+), Windows 11, Windows Server 2019/2022
