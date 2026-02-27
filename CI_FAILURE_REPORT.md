# CI/CD Failure Analysis Report

## Executive Summary

- **Workflow**: WebSocket Self-Healing Validation
- **Run Number**: #7
- **Branch**: `main`
- **Commit**: `ce513d74a600e9e7f9b9299af7577d74509e998f`
- **Status**: ❌ FAILED
- **Timestamp**: 2026-02-27T04:05:48Z
- **Triggered By**: @nandkishorrathodk-art
- **Workflow URL**: [View Run](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22472291464)

## Failure Overview

Total Failed Jobs: **6**

| # | Job Name | Category | Severity | Duration |
|---|----------|----------|----------|----------|
| 1 | WebSocket Health Tests (heartbeat-monitoring, Heartbeat & Health Monitoring, 💓) | test_failure | high | 50s |
| 2 | WebSocket Health Tests (latency-performance, Latency & Performance, ⚡) | test_failure | high | 49s |
| 3 | WebSocket Health Tests (self-healing, Self-Healing & Recovery, 🔄) | test_failure | high | 63s |
| 4 | WebSocket Health Tests (concurrent-connections, Concurrent Connections, 🔗) | test_failure | high | 47s |
| 5 | WebSocket Health Tests (connection-lifecycle, Connection Lifecycle, 🔌) | test_failure | high | 57s |
| 6 | WebSocket Health Tests (message-delivery, Message Delivery & Reliability, 📨) | test_failure | high | 62s |

## Detailed Analysis

### 1. WebSocket Health Tests (heartbeat-monitoring, Heartbeat & Health Monitoring, 💓)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-27T04:06:03Z
**Completed**: 2026-02-27T04:06:53Z
**Duration**: 50 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22472291464/job/65091672797)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 32: `2026-02-27T04:06:51.4080550Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 33: `2026-02-27T04:06:51.4129470Z   error: subprocess-exited-with-error`
    - Line 54: `2026-02-27T04:06:51.4144878Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 58: `2026-02-27T04:06:51.4146461Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 67: `2026-02-27T04:06:51.5389180Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`
    - Line 97: `2026-02-27T04:06:51.7022617Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-27T04:06:51.7022617Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-27T04:06:47.8897721Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-27T04:06:47.8897721Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 2. WebSocket Health Tests (latency-performance, Latency & Performance, ⚡)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-27T04:06:03Z
**Completed**: 2026-02-27T04:06:52Z
**Duration**: 49 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22472291464/job/65091672798)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 32: `2026-02-27T04:06:50.7940139Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 33: `2026-02-27T04:06:50.7990427Z   error: subprocess-exited-with-error`
    - Line 54: `2026-02-27T04:06:50.8009836Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 58: `2026-02-27T04:06:50.8011667Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 67: `2026-02-27T04:06:50.9154353Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`
    - Line 97: `2026-02-27T04:06:51.0781553Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-27T04:06:51.0781553Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-27T04:06:46.7879405Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-27T04:06:46.7879405Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 3. WebSocket Health Tests (self-healing, Self-Healing & Recovery, 🔄)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-27T04:06:03Z
**Completed**: 2026-02-27T04:07:06Z
**Duration**: 63 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22472291464/job/65091672799)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 32: `2026-02-27T04:07:02.9467016Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 33: `2026-02-27T04:07:02.9515309Z   error: subprocess-exited-with-error`
    - Line 54: `2026-02-27T04:07:02.9539424Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 58: `2026-02-27T04:07:02.9542534Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 67: `2026-02-27T04:07:03.0805923Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`
    - Line 97: `2026-02-27T04:07:03.2429813Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-27T04:07:03.2429813Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-27T04:06:58.4434735Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-27T04:06:58.4434735Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 4. WebSocket Health Tests (concurrent-connections, Concurrent Connections, 🔗)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-27T04:06:03Z
**Completed**: 2026-02-27T04:06:50Z
**Duration**: 47 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22472291464/job/65091672800)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 32: `2026-02-27T04:06:48.4540094Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 33: `2026-02-27T04:06:48.4586261Z   error: subprocess-exited-with-error`
    - Line 54: `2026-02-27T04:06:48.4606062Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 58: `2026-02-27T04:06:48.4608198Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 67: `2026-02-27T04:06:48.5948024Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`
    - Line 97: `2026-02-27T04:06:48.7614032Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-27T04:06:48.7614032Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-27T04:06:45.5031670Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-27T04:06:45.5031670Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 5. WebSocket Health Tests (connection-lifecycle, Connection Lifecycle, 🔌)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-27T04:06:03Z
**Completed**: 2026-02-27T04:07:00Z
**Duration**: 57 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22472291464/job/65091672801)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 32: `2026-02-27T04:06:58.1011795Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 33: `2026-02-27T04:06:58.1062766Z   error: subprocess-exited-with-error`
    - Line 54: `2026-02-27T04:06:58.1079183Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 58: `2026-02-27T04:06:58.1080697Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 67: `2026-02-27T04:06:58.2899663Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`
    - Line 97: `2026-02-27T04:06:58.4612605Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-27T04:06:58.4612605Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-27T04:06:54.3112865Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-27T04:06:54.3112865Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 6. WebSocket Health Tests (message-delivery, Message Delivery & Reliability, 📨)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-27T04:06:03Z
**Completed**: 2026-02-27T04:07:05Z
**Duration**: 62 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22472291464/job/65091672818)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 32: `2026-02-27T04:07:01.5957039Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 33: `2026-02-27T04:07:01.6004542Z   error: subprocess-exited-with-error`
    - Line 54: `2026-02-27T04:07:01.6024493Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 58: `2026-02-27T04:07:01.6026003Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 67: `2026-02-27T04:07:01.7224330Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`
    - Line 97: `2026-02-27T04:07:01.8867234Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-27T04:07:01.8867234Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-27T04:06:58.3685067Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-27T04:06:58.3685067Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

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

📊 *Report generated on 2026-02-27T04:08:25.116004*
🤖 *JARVIS CI/CD Auto-PR Manager*
