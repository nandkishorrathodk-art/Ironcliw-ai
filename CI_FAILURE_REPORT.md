# CI/CD Failure Analysis Report

## Executive Summary

- **Workflow**: WebSocket Self-Healing Validation
- **Run Number**: #8
- **Branch**: `main`
- **Commit**: `ce513d74a600e9e7f9b9299af7577d74509e998f`
- **Status**: ❌ FAILED
- **Timestamp**: 2026-02-28T03:47:48Z
- **Triggered By**: @nandkishorrathodk-art
- **Workflow URL**: [View Run](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22512695812)

## Failure Overview

Total Failed Jobs: **6**

| # | Job Name | Category | Severity | Duration |
|---|----------|----------|----------|----------|
| 1 | WebSocket Health Tests (latency-performance, Latency & Performance, ⚡) | test_failure | high | 56s |
| 2 | WebSocket Health Tests (self-healing, Self-Healing & Recovery, 🔄) | test_failure | high | 61s |
| 3 | WebSocket Health Tests (connection-lifecycle, Connection Lifecycle, 🔌) | test_failure | high | 53s |
| 4 | WebSocket Health Tests (message-delivery, Message Delivery & Reliability, 📨) | test_failure | high | 57s |
| 5 | WebSocket Health Tests (heartbeat-monitoring, Heartbeat & Health Monitoring, 💓) | test_failure | high | 44s |
| 6 | WebSocket Health Tests (concurrent-connections, Concurrent Connections, 🔗) | test_failure | high | 57s |

## Detailed Analysis

### 1. WebSocket Health Tests (latency-performance, Latency & Performance, ⚡)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-28T03:48:03Z
**Completed**: 2026-02-28T03:48:59Z
**Duration**: 56 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22512695812/job/65224884616)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 32: `2026-02-28T03:48:56.8103059Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 33: `2026-02-28T03:48:56.8150466Z   error: subprocess-exited-with-error`
    - Line 54: `2026-02-28T03:48:56.8173631Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 58: `2026-02-28T03:48:56.8176434Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 67: `2026-02-28T03:48:56.9279966Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`
    - Line 97: `2026-02-28T03:48:57.0887170Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T03:48:57.0887170Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-28T03:48:53.1082293Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-28T03:48:53.1082293Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 2. WebSocket Health Tests (self-healing, Self-Healing & Recovery, 🔄)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-28T03:48:04Z
**Completed**: 2026-02-28T03:49:05Z
**Duration**: 61 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22512695812/job/65224884619)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 32: `2026-02-28T03:49:01.8234967Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 33: `2026-02-28T03:49:01.8284234Z   error: subprocess-exited-with-error`
    - Line 54: `2026-02-28T03:49:01.8299935Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 58: `2026-02-28T03:49:01.8301492Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 67: `2026-02-28T03:49:02.0125494Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`
    - Line 97: `2026-02-28T03:49:02.1855645Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T03:49:02.1855645Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-28T03:48:58.4969567Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-28T03:48:58.4969567Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 3. WebSocket Health Tests (connection-lifecycle, Connection Lifecycle, 🔌)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-28T03:48:03Z
**Completed**: 2026-02-28T03:48:56Z
**Duration**: 53 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22512695812/job/65224884621)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 32: `2026-02-28T03:48:54.0423499Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 33: `2026-02-28T03:48:54.0476005Z   error: subprocess-exited-with-error`
    - Line 54: `2026-02-28T03:48:54.0496069Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 58: `2026-02-28T03:48:54.0497578Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 67: `2026-02-28T03:48:54.1849464Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`
    - Line 97: `2026-02-28T03:48:54.3514038Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T03:48:54.3514038Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-28T03:48:50.4298361Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-28T03:48:50.4298361Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 4. WebSocket Health Tests (message-delivery, Message Delivery & Reliability, 📨)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-28T03:48:04Z
**Completed**: 2026-02-28T03:49:01Z
**Duration**: 57 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22512695812/job/65224884623)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 32: `2026-02-28T03:48:58.3883029Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 33: `2026-02-28T03:48:58.3932453Z   error: subprocess-exited-with-error`
    - Line 54: `2026-02-28T03:48:58.3948346Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 58: `2026-02-28T03:48:58.3950434Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 67: `2026-02-28T03:48:58.5101191Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`
    - Line 97: `2026-02-28T03:48:58.6723183Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T03:48:58.6723183Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-28T03:48:55.3027140Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-28T03:48:55.3027140Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 5. WebSocket Health Tests (heartbeat-monitoring, Heartbeat & Health Monitoring, 💓)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-28T03:48:03Z
**Completed**: 2026-02-28T03:48:47Z
**Duration**: 44 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22512695812/job/65224884625)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 32: `2026-02-28T03:48:45.3748822Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 33: `2026-02-28T03:48:45.3798928Z   error: subprocess-exited-with-error`
    - Line 54: `2026-02-28T03:48:45.3820888Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 58: `2026-02-28T03:48:45.3823463Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 67: `2026-02-28T03:48:45.5285360Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`
    - Line 97: `2026-02-28T03:48:45.6978749Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T03:48:45.6978749Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-28T03:48:42.1618955Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-28T03:48:42.1618955Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 6. WebSocket Health Tests (concurrent-connections, Concurrent Connections, 🔗)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-28T03:48:03Z
**Completed**: 2026-02-28T03:49:00Z
**Duration**: 57 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22512695812/job/65224884637)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 32: `2026-02-28T03:48:57.7204654Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 33: `2026-02-28T03:48:57.7252627Z   error: subprocess-exited-with-error`
    - Line 54: `2026-02-28T03:48:57.7268304Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 58: `2026-02-28T03:48:57.7269821Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 67: `2026-02-28T03:48:57.8510757Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`
    - Line 97: `2026-02-28T03:48:58.0127794Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T03:48:58.0127794Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-28T03:48:54.1463786Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-28T03:48:54.1463786Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

## Action Items

- [ ] Review detailed logs for each failed job
- [ ] Implement suggested fixes
- [ ] Add or update tests to prevent regression
- [ ] Verify fixes locally before pushing
- [ ] Update CI/CD configuration if needed

## Additional Resources

- [Workflow File](.github/workflows/)
- [CI/CD Documentation](../../docs/ci-cd/)
- [Troubleshooting Guide](../../docs/troubleshooting/)

---

📊 *Report generated on 2026-02-28T03:50:28.419504*
🤖 *JARVIS CI/CD Auto-PR Manager*
