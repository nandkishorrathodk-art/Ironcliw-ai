# JARVIS Postman Flows

## Voice Unlock Authentication Flow

The primary authentication pipeline for JARVIS voice biometric unlock system.

> **Note:** Postman Flows cannot be exported/imported as JSON files. Flows are created directly in the Postman visual editor. This document provides:
> 1. A **Collection-based workflow** that can be imported and run with Collection Runner
> 2. **Step-by-step instructions** for building the Flow manually in Postman

---

## Option 1: Collection-Based Workflow (Recommended)

Import the collection that executes the same logic as a Flow using Collection Runner.

### Import Instructions

1. Open Postman
2. Click **Import** in the sidebar
3. Select file: `postman/collections/JARVIS_Voice_Unlock_Flow_Collection.postman_collection.json`
4. The collection will appear in your workspace

### Run with Collection Runner

1. Click on the collection **"JARVIS Voice Unlock Flow (Sequential)"**
2. Click **Run** button (or right-click → "Run collection")
3. Configure:
   - **Iterations:** 1
   - **Delay:** 0 ms
4. Click **Run JARVIS Voice Unlock Flow**

### What Happens

The Collection Runner executes requests sequentially with test scripts that:
- Check conditions and route to different requests
- Store variables between requests
- Skip to error handlers when needed
- Log results to the console

```
1. Health Check → [Pass] → 2. Start Audit
                  [Fail] → Error: Unavailable

2. Start Audit → 3. Anti-Spoofing Check

3. Anti-Spoofing → [No Attack] → 4. Voice Auth
                   [Detected] → Error: Security Alert

4. Voice Auth → [≥90%] → 6. Unlock (High Confidence)
                [85-90%] → 6. Unlock (Passed)
                [75-85%] → 5. Multi-Factor Fusion
                [<75%] → Error: Auth Failed

5. Multi-Factor → [≥80%] → 6. Unlock
                  [<80%] → Error: Challenge Required

6. Unlock → 7. JARVIS Feedback → 8. End Audit
```

---

## Option 2: Build Flow Manually in Postman

### Step-by-Step Flow Creation

#### 1. Create New Flow

1. Open Postman
2. Go to **Flows** tab in sidebar
3. Click **+ New Flow**
4. Name it: "JARVIS Voice Unlock Authentication"

#### 2. Add Start Block

- Drag **Start** block onto canvas
- This is your entry point

#### 3. Add Health Check

1. Drag **Send Request** block
2. Connect from Start
3. Configure:
   - **Request:** GET `{{base_url}}/api/voice-auth-intelligence/health`
   - **Name:** "Check System Health"

#### 4. Add Condition Block

1. Drag **If** block
2. Connect from Health Check
3. Configure condition: `/status = "healthy"`
4. Create two branches: **Then** (healthy) and **Else** (unhealthy)

#### 5. Add Anti-Spoofing Check (Then branch)

1. Drag **Send Request** block
2. Connect to Then branch
3. Configure:
   - **Request:** POST `{{base_url}}/api/voice-auth-intelligence/patterns/detect-replay`
   - **Body:** `{"check_exact_match": true, "check_spectral_fingerprint": true}`

#### 6. Add Replay Check Condition

1. Drag **If** block
2. Connect from Anti-Spoofing
3. Configure: `/is_replay = true`
4. **Then:** Security Alert (Output block with error)
5. **Else:** Continue to Voice Auth

#### 7. Add Voice Authentication

1. Drag **Send Request** block
2. Connect from Else branch
3. Configure:
   - **Request:** POST `{{base_url}}/api/voice-auth-intelligence/authenticate/enhanced`
   - **Body:**
   ```json
   {
     "speaker_name": "{{speaker_name}}",
     "use_adaptive": true,
     "max_attempts": 1
   }
   ```

#### 8. Add Confidence Router (Select/Switch)

1. Drag **Select** block (or multiple **If** blocks)
2. Connect from Voice Auth
3. Create conditions:
   - `/confidence >= 0.90` → High Confidence Path
   - `/confidence >= 0.85` → Pass Path
   - `/confidence >= 0.75` → Borderline Path
   - `else` → Failed Path

#### 9. Add Multi-Factor Fusion (Borderline Path)

1. Drag **Send Request** block
2. Connect from Borderline condition
3. Configure:
   - **Request:** POST `{{base_url}}/api/voice-auth-intelligence/fusion/calculate`
   - **Body:**
   ```json
   {
     "voice_confidence": "{{/voice-auth/response/body/confidence}}",
     "weights": {"voice": 0.50, "behavioral": 0.35, "context": 0.15}
   }
   ```

#### 10. Add Fusion Check

1. Drag **If** block
2. Connect from Fusion
3. Condition: `/fused_confidence >= 0.80`
4. **Then:** Proceed to Unlock
5. **Else:** Challenge Required

#### 11. Add Screen Unlock

1. Drag **Send Request** block
2. Connect from all success paths
3. Configure:
   - **Request:** POST `{{base_url}}/api/screen/unlock`
   - **Body:** `{"method": "keychain", "reason": "voice_authenticated"}`

#### 12. Add JARVIS Feedback

1. Drag **Send Request** block
2. Connect from Unlock
3. Configure:
   - **Request:** POST `{{base_url}}/voice/jarvis/speak`
   - **Body:** `{"text": "Unlocking for you, Derek."}`

#### 13. Add Output Blocks

Create output blocks for:
- **Success:** Authentication passed, screen unlocked
- **Security Alert:** Replay attack detected
- **Auth Failed:** Voice didn't match
- **Challenge Required:** Need secondary verification
- **System Unavailable:** Health check failed

#### 14. Set Variables

In Flow settings, add:
- `base_url`: `http://localhost:8010`
- `speaker_name`: `Derek`

---

## Flow Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         JARVIS VOICE UNLOCK AUTHENTICATION FLOW                      │
└─────────────────────────────────────────────────────────────────────────────────────┘

                                    ┌──────────────────┐
                                    │   Voice Unlock   │
                                    │     Request      │
                                    │   (Trigger)      │
                                    └────────┬─────────┘
                                             │
                                             ▼
                                    ┌──────────────────┐
                                    │  Check System    │
                                    │     Health       │
                                    └────────┬─────────┘
                                             │
                              ┌──────────────┴──────────────┐
                              │                             │
                         [Healthy]                    [Unhealthy]
                              │                             │
                              ▼                             ▼
                    ┌──────────────────┐          ┌──────────────────┐
                    │   Anti-Spoofing  │          │  ❌ Unavailable  │
                    │    Detection     │          │     Output       │
                    └────────┬─────────┘          └──────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
         [No Attack]                 [Replay Detected]
              │                             │
              ▼                             ▼
    ┌──────────────────┐          ┌──────────────────┐
    │  Voice Biometric │          │  🚨 Security     │
    │  Authentication  │          │  Alert Output    │
    └────────┬─────────┘          └──────────────────┘
             │
    ┌────────┴────────────────────────────────────┐
    │              │                │              │
 [≥90%]        [85-90%]         [75-85%]        [<75%]
    │              │                │              │
    │              │                ▼              ▼
    │              │      ┌──────────────┐  ┌──────────────┐
    │              │      │Multi-Factor  │  │ ❌ Auth      │
    │              │      │   Fusion     │  │ Failed Output│
    │              │      └──────┬───────┘  └──────────────┘
    │              │             │
    │              │      ┌──────┴──────┐
    │              │   [≥80%]        [<80%]
    │              │      │              │
    │              │      │              ▼
    │              │      │      ┌──────────────┐
    │              │      │      │ ❓ Challenge │
    │              │      │      │    Output    │
    │              │      │      └──────────────┘
    │              │      │
    └──────────────┴──────┴──────────┐
                                     │
                                     ▼
                           ┌──────────────────┐
                           │  🔓 Unlock       │
                           │     Screen       │
                           └────────┬─────────┘
                                    │
                                    ▼
                           ┌──────────────────┐
                           │  🔊 JARVIS       │
                           │    Speaks        │
                           └────────┬─────────┘
                                    │
                                    ▼
                           ┌──────────────────┐
                           │  ✅ Success      │
                           │     Output       │
                           └──────────────────┘
```

---

## API Endpoints Reference

| Step | Endpoint | Method | Purpose |
|------|----------|--------|---------|
| Health | `/api/voice-auth-intelligence/health` | GET | System status |
| Audit Start | `/api/voice-auth-intelligence/audit/session/start` | POST | Begin logging |
| Anti-Spoof | `/api/voice-auth-intelligence/patterns/detect-replay` | POST | Replay detection |
| Auth | `/api/voice-auth-intelligence/authenticate/enhanced` | POST | Voice biometric |
| Fusion | `/api/voice-auth-intelligence/fusion/calculate` | POST | Multi-factor |
| Unlock | `/api/screen/unlock` | POST | Screen unlock |
| Speak | `/voice/jarvis/speak` | POST | TTS feedback |
| Audit End | `/api/voice-auth-intelligence/audit/session/end` | POST | Close logging |

---

## Testing with Simulation Endpoints

Test different scenarios without real audio:

```bash
# Success scenario
curl -X POST http://localhost:8010/api/voice-auth-intelligence/authenticate/simulate \
  -H "Content-Type: application/json" \
  -d '{"scenario": "success", "speaker_name": "Derek"}'

# Borderline scenario (triggers multi-factor)
curl -X POST http://localhost:8010/api/voice-auth-intelligence/authenticate/simulate \
  -H "Content-Type: application/json" \
  -d '{"scenario": "borderline", "speaker_name": "Derek"}'

# Replay attack scenario
curl -X POST http://localhost:8010/api/voice-auth-intelligence/authenticate/simulate \
  -H "Content-Type: application/json" \
  -d '{"scenario": "replay_attack", "speaker_name": "Derek"}'

# Unknown speaker scenario
curl -X POST http://localhost:8010/api/voice-auth-intelligence/authenticate/simulate \
  -H "Content-Type: application/json" \
  -d '{"scenario": "unknown_speaker", "speaker_name": "Intruder"}'
```

---

## Sources

- [Postman Flows Overview](https://learning.postman.com/docs/postman-flows/overview/)
- [Postman Collection Runner](https://learning.postman.com/docs/collections/running-collections/intro-to-collection-runs/)
- [Postman Schemas Repository](https://github.com/postmanlabs/schemas)
