# Security Policy — Ironcliw-AI / Ironcliw

## 🔒 Security Overview

Ironcliw is an AI assistant with **system-level access** to your computer (screen, keyboard, microphone, file system). Security is critical. This document outlines security policies, known risks, and how to report vulnerabilities.

---

## 📋 Supported Versions

| Version | Port Status | Security Updates |
|---------|-------------|------------------|
| **Phase 11 (Windows Port)** | ✅ Active Development | Yes — this repo |
| Phase 1–10 (Windows Port) | ⚠️ Superseded | Upgrade to Phase 11 |
| Original macOS | ⚠️ Upstream Only | See [drussell23/Ironcliw](https://github.com/drussell23/Ironcliw) |

---

## 🚨 Reporting a Vulnerability

**DO NOT** open public GitHub Issues for security vulnerabilities.

### How to Report

1. **GitHub Security Advisory**: Go to [Security tab](https://github.com/nandkishorrathodk-art/Ironcliw-ai/security) → "Report a vulnerability" (private)
2. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact / severity
   - Suggested fix (optional)

### Response Timeline

| Severity | Initial Response | Fix Target |
|----------|-----------------|------------|
| **Critical** (RCE, auth bypass) | < 24 hours | 24–72 hours |
| **High** (data leak, privilege escalation) | < 48 hours | 1–2 weeks |
| **Medium** | < 72 hours | 2–4 weeks |
| **Low** | < 1 week | Next release |

---

## ⚠️ Current Security Status (Phase 11 — Windows MVP)

### Authentication Mode

```env
Ironcliw_AUTO_BYPASS_WINDOWS=true    # Auth bypassed on Windows
Ironcliw_VOICE_BIOMETRIC_ENABLED=false
```

**What this means:**
- ❌ No voice biometric verification (speechbrain/GPU required)
- ❌ No Windows Hello integration (Phase 12+)
- ✅ Suitable for **single-user development machines only**
- ⚠️ **NOT production-ready** for multi-user or shared systems

**Mitigation**: Run Ironcliw only on your personal, trusted machine. Do not expose any ports to external networks.

---

## 🔐 Security Features

### ✅ Implemented (Phase 11)

| Feature | Status | Implementation |
|---------|--------|---------------|
| API Key Encryption | ✅ | Environment variables only — never in code |
| Secure Logging | ✅ | CWE-117/532 log injection prevention (`secure_logging.py`) |
| Atomic Write Permissions | ✅ | `0o600` (owner read/write only) |
| Input Validation | ✅ | FastAPI Pydantic validators on all endpoints |
| HTTPS for External APIs | ✅ | Required for Claude, Fireworks, GCP |
| Temp File Cleanup | ✅ | `tempfile` + `finally: os.unlink()` pattern |
| Windows Credential Guard | ✅ | Keychain replaced with Windows Credential Manager |
| UNIQUE Constraint Prevention | ✅ | Pre-check hash before DB insert (no log spam) |
| Secure TTS temp files | ✅ | edge-tts MP3 deleted immediately after playback |
| Process isolation | ✅ | Backend/frontend run as separate processes |

### 🔧 Fixed in Phase 11

| Vulnerability | Fix Applied | Commit |
|---------------|------------|--------|
| API keys in `.env.windows` committed to git | Redacted + gitignored | `a77933aa` |
| Log injection (CWE-117) via user input in logs | `secure_logging.py` sanitizer | Phase 11 Session 8 |
| UNIQUE constraint spam (DB timing race) | Hash pre-check in `learning_database.py` | `2c22880f` |
| `WinError 2` on Keychain access | Windows platform guard | `2c22880f` |

### ⏳ Planned (Phase 12+)

| Feature | Priority | Target Phase |
|---------|----------|--------------|
| Windows Hello biometric auth | High | Phase 12 |
| Voice biometric (ECAPA-TDNN) | High | Phase 13 (GPU machine) |
| TPM key storage | Medium | Phase 14 |
| End-to-end encryption (local data) | Medium | Phase 14 |
| Rate limiting (API abuse prevention) | Medium | Phase 12 |
| Audit logging (full trail) | High | Phase 12 |
| Dependency vulnerability scanning (CI) | High | Active (CodeQL) |

---

## 🛡️ Security Best Practices

### 1. API Key Management

**DO:**
```powershell
# Store in .env (already in .gitignore)
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx
FIREWORKS_API_KEY=fw-xxxxxxxxxxxxx
```

**DO NOT:**
```python
# ❌ NEVER hardcode in source files
api_key = "sk-ant-xxxxxxxxxxxxx"  # This will be scanned and flagged!
```

**Verify your .env is gitignored:**
```powershell
git check-ignore -v .env     # Should output: .gitignore:X:.env
```

### 2. File Permissions (Windows)

```powershell
# Restrict .env to your user only
icacls .env /inheritance:r /grant:r "%USERNAME%:R"
icacls .env   # Verify
```

### 3. Network Security

Ironcliw defaults to **localhost only**. Keep it that way:

```env
BACKEND_HOST=127.0.0.1    # Never 0.0.0.0
BACKEND_PORT=8010
CORS_ORIGINS=http://localhost:3000
```

**Windows Firewall — block external access:**
```powershell
# Block inbound on port 8010 from any external IP
New-NetFirewallRule -DisplayName "Ironcliw Block External" `
  -Direction Inbound -LocalPort 8010 -Protocol TCP `
  -Action Block -RemoteAddress Any

# Allow only localhost
New-NetFirewallRule -DisplayName "Ironcliw Allow Localhost" `
  -Direction Inbound -LocalPort 8010 -Protocol TCP `
  -Action Allow -RemoteAddress 127.0.0.1
```

### 4. GCP / Cloud Security

If using GCP Spot VM for inference:
- ✅ Use service accounts with **minimal IAM permissions**
- ✅ Enable VPC firewall: allow only your IP
- ✅ Never expose VM to `0.0.0.0/0`
- ✅ Use Workload Identity — no service account JSON files
- ✅ Enable Cloud Armor (DDoS protection)
- ❌ Do NOT use root/admin service accounts

### 5. Dependency Updates

```powershell
# Check for known vulnerabilities
pip install pip-audit
pip-audit

# Update all packages
pip install --upgrade -r requirements.txt

# JavaScript (frontend)
cd frontend
npm audit
npm audit fix
```

---

## 🔍 Known Security Considerations

### 1. System-Level Access

Ironcliw has broad system access by design:

| Permission | Purpose | Risk | Mitigation |
|------------|---------|------|-----------|
| Screen capture (mss) | Vision / context awareness | Medium | Only captures when Ironcliw is active |
| Keyboard/mouse (pyautogui) | Ghost Hands automation | High | User must explicitly trigger |
| File system (pathlib) | Project management | Medium | Scoped to working directories |
| Process management (psutil) | Orchestration | High | Only Ironcliw-owned processes |
| Microphone | Voice commands | Medium | No recording stored by default |
| Network | API calls, GCP | Medium | HTTPS only, keys in env |

**Recommendation**: Disable features you don't use via `.env`:
```env
Ironcliw_DISABLE_GHOST_HANDS=true      # Disable keyboard/mouse control
Ironcliw_DISABLE_SCREEN_CAPTURE=true   # Disable vision
Ironcliw_SKIP_GCP=true                 # Disable cloud routing
```

### 2. Third-Party APIs — Data Sent

| Service | What Is Sent | Privacy Policy |
|---------|-------------|----------------|
| Claude API (Anthropic) | User prompts, screen context | [anthropic.com/privacy](https://www.anthropic.com/privacy) |
| Fireworks AI | User prompts | [fireworks.ai/privacy](https://fireworks.ai/privacy) |
| edge-tts (Microsoft) | Text to speak | [microsoft.com/privacy](https://privacy.microsoft.com) |
| GCP (optional) | Inference requests | [cloud.google.com/privacy](https://cloud.google.com/privacy) |

**Recommendation**: Avoid sending PII (names, passwords, financial data) in voice commands or prompts.

### 3. Log Files

Logs may contain sensitive context. They are gitignored but check locally:

```powershell
# Verify logs are not tracked
git check-ignore -v backend\logs\
git check-ignore -v .jarvis\

# Clear old logs
Remove-Item -Recurse -Force backend\logs\
```

### 4. Windows Native Layer (C# DLLs)

`backend/windows_native/` uses P/Invoke for Win32 APIs:

- **Risk**: Buffer overflow via .NET interop, DLL injection
- **Mitigation**: Source is auditable in this repo; compiled from source
- **Mitigation**: .NET runtime provides bounds checking
- **Note**: DLLs require UAC elevation for privileged operations

---

## 🔧 Recommended `.env` Security Settings

```env
# ─── Authentication ─────────────────────────────────────────
Ironcliw_AUTO_BYPASS_WINDOWS=true        # ⚠️ MVP only — disable in production
Ironcliw_VOICE_BIOMETRIC_ENABLED=false   # ⚠️ Enable when GPU + speechbrain available

# ─── Network (NEVER change these unless you know what you're doing) ───
BACKEND_HOST=127.0.0.1                 # ✅ Localhost ONLY
BACKEND_PORT=8010
FRONTEND_PORT=3000
CORS_ORIGINS=http://localhost:3000     # ✅ Restrict CORS

# ─── API Keys (NEVER commit — use .env only) ─────────────────
ANTHROPIC_API_KEY=sk-ant-xxxxx
FIREWORKS_API_KEY=fw-xxxxx

# ─── Logging ─────────────────────────────────────────────────
LOG_LEVEL=INFO                         # ⚠️ Never DEBUG in production
Ironcliw_ENABLE_AUDIT_LOG=true           # ✅ Keep audit trail

# ─── GCP (optional) ──────────────────────────────────────────
GCP_PROJECT_ID=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=path\to\service-account.json
```

---

## 🧪 Security Testing

### Pre-Deployment Checklist

- [ ] All API keys in `.env` — not in any source file
- [ ] `.env` has restricted file permissions (`icacls`)
- [ ] `.env` is gitignored and not in git history
- [ ] Backend bound to `127.0.0.1` only
- [ ] Windows Firewall rules configured
- [ ] `pip-audit` returns no critical vulnerabilities
- [ ] `npm audit` returns no critical vulnerabilities
- [ ] Logs cleared of sensitive data
- [ ] GCP VM firewall rules configured (if used)

### Security Scan Commands

```powershell
# 1. Scan Python for vulnerabilities
pip-audit

# 2. Scan Python code for security issues (bandit)
pip install bandit
bandit -r backend\ -ll

# 3. Scan for accidentally committed secrets
pip install detect-secrets
detect-secrets scan --baseline .secrets.baseline

# 4. Check JS vulnerabilities
cd frontend && npm audit

# 5. Verify no secrets in git history
git log --all --full-history -- "*.env" "*.key" "*.pem"

# 6. Check open ports
netstat -an | findstr ":8010 "
netstat -an | findstr ":3000 "
```

---

## 🔄 Security Update Policy

| Type | Frequency |
|------|-----------|
| Critical vulnerabilities | Immediate (24–72h) |
| High severity | Within 1–2 weeks |
| Medium/Low severity | Monthly releases |
| Dependency updates | Bi-weekly |

Updates announced via:
- GitHub Releases
- GitHub Security Advisories
- CHANGELOG.md

---

## 📜 Compliance Notes

| Regulation | Status |
|------------|--------|
| **GDPR** | User data stays local unless cloud APIs are used. No data sold. |
| **CCPA** | No personal data sold to third parties. |
| **SOC 2** | Not applicable (personal/development project). |

---

## ✅ Security Acknowledgments

We thank the following for responsible disclosure:

*No vulnerabilities reported yet. Be the first — and get credited here!*

---

## 📞 Contact

**Maintainer**: Nandkishor Rathod
**Repository**: [nandkishorrathodk-art/Ironcliw-ai](https://github.com/nandkishorrathodk-art/Ironcliw-ai)
**Security Reports**: [GitHub Security Advisory](https://github.com/nandkishorrathodk-art/Ironcliw-ai/security/advisories/new)

---

<div align="center">

**🔒 Security is everyone's responsibility. Report issues responsibly.**

Last Updated: February 2026 | Phase 11 (Windows Port Active)

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

</div>
