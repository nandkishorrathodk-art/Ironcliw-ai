# Security Policy

## 🔒 Security Overview

JARVIS is an AI assistant with system-level access to your computer. Security is critical. This document outlines our security policies, practices, and how to report vulnerabilities.

---

## 📋 Supported Versions

| Version | Status | Security Updates |
|---------|--------|------------------|
| Phase 1-5 (Windows Port) | ✅ Active Development | Yes |
| Original macOS | ⚠️ Upstream Only | See [original repo](https://github.com/drussell23/JARVIS-AI-Agent) |

---

## 🚨 Reporting a Vulnerability

**DO NOT** open public issues for security vulnerabilities.

### How to Report

1. **Email**: Send details to your repository maintainer (nandkishorrathodk-art)
2. **GitHub Security Advisory**: Use "Security" tab → "Report a vulnerability"
3. **Include**:
   - Description of vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if available)

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Depends on severity (Critical: 24-72h, High: 1-2 weeks, Medium: 2-4 weeks)

---

## ⚠️ Current Security Status (MVP Phase)

### ⚠️ BYPASS MODE ACTIVE

**Authentication is currently BYPASSED for Windows MVP**:
```env
JARVIS_AUTH_MODE=BYPASS
JARVIS_SKIP_VOICE_AUTH=true
```

**What This Means**:
- ❌ No voice biometric verification
- ❌ No password authentication
- ❌ Anyone with local access can use JARVIS
- ✅ Suitable for single-user development environments
- ⚠️ **NOT production-ready for multi-user systems**

**Mitigation**: Only run JARVIS on trusted, single-user machines.

---

## 🔐 Security Features

### ✅ Implemented

| Feature | Status | Platform |
|---------|--------|----------|
| API Key Encryption | ✅ Environment Variables | All |
| Credential Manager | ✅ Windows Credential API | Windows |
| UAC Elevation Detection | ✅ Platform Layer | Windows |
| Secure Temp File Handling | ✅ `tempfile` Module | All |
| Input Validation | ✅ FastAPI Validators | All |
| HTTPS for External APIs | ✅ Required | All |

### ⏳ Planned (Phase 6+)

| Feature | Priority | Target Phase |
|---------|----------|--------------|
| Voice Biometric Auth | High | Phase 6 |
| Windows Hello Integration | High | Phase 7 |
| TPM Key Storage | Medium | Phase 8 |
| End-to-End Encryption | Medium | Phase 9 |
| Audit Logging | High | Phase 6 |
| Rate Limiting | Medium | Phase 7 |

---

## 🛡️ Security Best Practices

### For Users

#### 1. **API Key Management**

**DO**:
```bash
# Use .env file (gitignored)
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx
FIREWORKS_API_KEY=fw-xxxxxxxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxxxxxxx
```

**DON'T**:
```python
# ❌ NEVER hardcode keys in source code
api_key = "sk-ant-xxxxxxxxxxxxx"  # BAD!
```

#### 2. **File Permissions**

```powershell
# Windows: Restrict .env file access
icacls .env /inheritance:r /grant:r "%USERNAME%:R"

# Verify
icacls .env
```

#### 3. **Network Security**

- Run JARVIS on localhost only (default: `127.0.0.1:8010`)
- Do NOT expose ports to public internet
- Use firewall to block unauthorized access:
  ```powershell
  # Windows Firewall: Block inbound on port 8010 from external networks
  New-NetFirewallRule -DisplayName "JARVIS Local Only" -Direction Inbound -LocalPort 8010 -Protocol TCP -Action Block -RemoteAddress Any
  New-NetFirewallRule -DisplayName "JARVIS Localhost" -Direction Inbound -LocalPort 8010 -Protocol TCP -Action Allow -RemoteAddress 127.0.0.1
  ```

#### 4. **GCP VM Security**

If using GCP golden image for inference:
- ✅ Use service accounts with minimal permissions
- ✅ Enable VPC firewall rules (allow only your IP)
- ✅ Enable Cloud Armor for DDoS protection
- ✅ Use Workload Identity for authentication
- ❌ Do NOT expose VM to `0.0.0.0/0`

#### 5. **Update Dependencies**

```bash
# Check for security vulnerabilities
pip-audit

# Update packages
pip install --upgrade -r requirements.txt
```

---

## 🔍 Known Security Considerations

### 1. **System-Level Access**

JARVIS requires extensive system permissions:

| Permission | Purpose | Risk Level |
|------------|---------|------------|
| Screen Capture | Vision/automation | Medium |
| Keyboard/Mouse Control | Ghost Hands automation | High |
| File System Access | Project management | High |
| Process Management | Task orchestration | High |
| Microphone Access | Voice commands | Medium |

**Mitigation**: Only grant permissions you need. Disable unused features in `windows_config.yaml`.

### 2. **Third-Party APIs**

JARVIS sends data to external services:

| Service | Data Sent | Privacy Policy |
|---------|-----------|----------------|
| Claude API | User prompts, context | [Anthropic](https://www.anthropic.com/privacy) |
| Fireworks AI | User prompts | [Fireworks](https://fireworks.ai/privacy) |
| OpenAI API | User prompts (optional) | [OpenAI](https://openai.com/privacy) |
| GCP VMs | Inference requests | [Google Cloud](https://cloud.google.com/privacy) |

**Mitigation**: 
- Review each provider's privacy policy
- Use local inference when possible (PRIME_LOCAL)
- Avoid sending sensitive data in prompts

### 3. **Log Files**

Logs may contain sensitive information:

```yaml
# .gitignore includes:
*.log
logs/
.jarvis/
```

**Action**: Regularly clean logs and NEVER commit them to version control.

### 4. **C# Native DLLs**

Windows native layer uses P/Invoke for system APIs:

**Risks**:
- Buffer overflows (mitigated by .NET runtime)
- Privilege escalation (requires UAC)
- DLL injection attacks

**Mitigation**:
- Source code auditable (`backend/windows_native/`)
- Compiled DLLs should be reproducible
- Verify DLL signatures before running

---

## 🔧 Security Configuration

### Recommended `.env` Settings

```env
# Authentication (CHANGE THIS FOR PRODUCTION!)
JARVIS_AUTH_MODE=BYPASS                    # ⚠️ Change to VOICE or PASSWORD
JARVIS_SKIP_VOICE_AUTH=true                # ⚠️ Set to false for production

# Network
BACKEND_HOST=127.0.0.1                     # ✅ Localhost only
BACKEND_PORT=8010                          # ✅ Non-standard port
CORS_ORIGINS=http://localhost:3000         # ✅ Restrict CORS

# API Keys (NEVER commit these!)
ANTHROPIC_API_KEY=sk-ant-xxxxx             # ✅ Use .env, not code
FIREWORKS_API_KEY=fw-xxxxx                 # ✅ Rotate regularly

# Logging
LOG_LEVEL=INFO                             # ⚠️ Don't use DEBUG in production
JARVIS_ENABLE_AUDIT_LOG=true               # ✅ Enable audit trail

# GCP (if used)
GCP_PROJECT_ID=your-project                # ✅ Use dedicated project
GOOGLE_APPLICATION_CREDENTIALS=path.json   # ✅ Use service account

# Rate Limiting (Phase 7+)
JARVIS_RATE_LIMIT_ENABLED=true             # ✅ Prevent abuse
JARVIS_MAX_REQUESTS_PER_MINUTE=60          # ✅ Adjust as needed
```

---

## 🧪 Security Testing

### Checklist Before Deployment

- [ ] All API keys in `.env` (not in code)
- [ ] `.env` file has restricted permissions
- [ ] Authentication enabled (not BYPASS)
- [ ] Running on localhost only
- [ ] Firewall rules configured
- [ ] Dependencies updated (`pip-audit` clean)
- [ ] Logs cleared of sensitive data
- [ ] GCP VM has firewall rules (if used)
- [ ] CORS origins restricted
- [ ] Audit logging enabled

### Testing Commands

```powershell
# 1. Check for hardcoded secrets
python -m detect-secrets scan --baseline .secrets.baseline

# 2. Check for vulnerable dependencies
pip-audit

# 3. Verify .env permissions
icacls .env

# 4. Test authentication bypass (should fail if disabled)
curl http://localhost:8010/api/command -H "Content-Type: application/json" -d '{"text":"test"}'

# 5. Check exposed ports
netstat -an | findstr :8010
```

---

## 📚 Resources

### Security Tools

- [**pip-audit**](https://pypi.org/project/pip-audit/) - Find vulnerable dependencies
- [**detect-secrets**](https://github.com/Yelp/detect-secrets) - Prevent secret commits
- [**bandit**](https://bandit.readthedocs.io/) - Python security linter
- [**Safety**](https://pyup.io/safety/) - Check known vulnerabilities

### Installation

```bash
pip install pip-audit detect-secrets bandit safety
```

### Usage

```bash
# Scan for secrets
detect-secrets scan

# Find vulnerabilities
pip-audit
bandit -r backend/

# Check dependency safety
safety check
```

---

## 🔄 Security Update Policy

### Update Frequency

- **Critical vulnerabilities**: Immediate (within 24-72 hours)
- **High severity**: Weekly
- **Medium/Low severity**: Monthly
- **Dependency updates**: Bi-weekly

### Notification Channels

- GitHub Security Advisories
- Release notes (CHANGELOG.md)
- Repository README.md

---

## 📜 Compliance

### Data Privacy

- **GDPR**: User data stays local (unless using cloud APIs)
- **CCPA**: No personal data sold to third parties
- **SOC 2**: Not applicable (development project)

### Licenses

- **Windows Port**: MIT License (See [LICENSE](LICENSE))
- **Original JARVIS**: All Rights Reserved (upstream)
- **Dependencies**: Various (see `requirements.txt`)

---

## ✅ Security Acknowledgments

We thank the following for responsible disclosure:

- *No vulnerabilities reported yet*

---

## 📞 Contact

**Maintainer**: Nandkishor Rathod  
**Repository**: [nandkishorrathodk-art/Ironcliw-ai](https://github.com/nandkishorrathodk-art/Ironcliw-ai)  
**Security Email**: [Use GitHub Security Advisory]

---

<div align="center">

**🔒 Security is a shared responsibility. Report issues responsibly.**

Last Updated: February 2026  
Version: Phase 1-5 (Windows Port)

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

</div>
