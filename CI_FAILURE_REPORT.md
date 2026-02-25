# CI/CD Failure Analysis Report

## Executive Summary

- **Workflow**: WebSocket Self-Healing Validation
- **Run Number**: #5
- **Branch**: `main`
- **Commit**: `e272639237ac2968a887a0c1df7659932a78fffe`
- **Status**: ❌ FAILED
- **Timestamp**: 2026-02-25T04:12:41Z
- **Triggered By**: @nandkishorrathodk-art
- **Workflow URL**: [View Run](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22381847573)

## Failure Overview

Total Failed Jobs: **6**

| # | Job Name | Category | Severity | Duration |
|---|----------|----------|----------|----------|
| 1 | WebSocket Health Tests (heartbeat-monitoring, Heartbeat & Health Monitoring, 💓) | test_failure | high | 47s |
| 2 | WebSocket Health Tests (message-delivery, Message Delivery & Reliability, 📨) | test_failure | high | 55s |
| 3 | WebSocket Health Tests (concurrent-connections, Concurrent Connections, 🔗) | test_failure | high | 57s |
| 4 | WebSocket Health Tests (connection-lifecycle, Connection Lifecycle, 🔌) | test_failure | high | 60s |
| 5 | WebSocket Health Tests (self-healing, Self-Healing & Recovery, 🔄) | test_failure | high | 44s |
| 6 | WebSocket Health Tests (latency-performance, Latency & Performance, ⚡) | test_failure | high | 68s |

## Detailed Analysis

### 1. WebSocket Health Tests (heartbeat-monitoring, Heartbeat & Health Monitoring, 💓)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-25T04:12:59Z
**Completed**: 2026-02-25T04:13:46Z
**Duration**: 47 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22381847573/job/64784085990)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 32: `2026-02-25T04:13:43.9592941Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 33: `2026-02-25T04:13:43.9642116Z   error: subprocess-exited-with-error`
    - Line 54: `2026-02-25T04:13:43.9658466Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 58: `2026-02-25T04:13:43.9660254Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 67: `2026-02-25T04:13:44.0946027Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`
    - Line 97: `2026-02-25T04:13:44.2502820Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-25T04:13:44.2502820Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-25T04:13:41.0442091Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-25T04:13:41.0442091Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 2. WebSocket Health Tests (message-delivery, Message Delivery & Reliability, 📨)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-25T04:13:00Z
**Completed**: 2026-02-25T04:13:55Z
**Duration**: 55 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22381847573/job/64784085996)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 32: `2026-02-25T04:13:52.9571391Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 33: `2026-02-25T04:13:52.9620210Z   error: subprocess-exited-with-error`
    - Line 54: `2026-02-25T04:13:52.9639888Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 58: `2026-02-25T04:13:52.9641361Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 67: `2026-02-25T04:13:53.0786312Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`
    - Line 97: `2026-02-25T04:13:53.2314398Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-25T04:13:53.2314398Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-25T04:13:49.8755071Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-25T04:13:49.8755071Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 3. WebSocket Health Tests (concurrent-connections, Concurrent Connections, 🔗)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-25T04:13:00Z
**Completed**: 2026-02-25T04:13:57Z
**Duration**: 57 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22381847573/job/64784085999)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 32: `2026-02-25T04:13:54.0153362Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 33: `2026-02-25T04:13:54.0203755Z   error: subprocess-exited-with-error`
    - Line 54: `2026-02-25T04:13:54.0227098Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 58: `2026-02-25T04:13:54.0229991Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 67: `2026-02-25T04:13:54.1373756Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`
    - Line 97: `2026-02-25T04:13:54.2917552Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-25T04:13:54.2917552Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-25T04:13:50.8500341Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-25T04:13:50.8500341Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 4. WebSocket Health Tests (connection-lifecycle, Connection Lifecycle, 🔌)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-25T04:13:00Z
**Completed**: 2026-02-25T04:14:00Z
**Duration**: 60 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22381847573/job/64784086001)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 32: `2026-02-25T04:13:56.5807462Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 33: `2026-02-25T04:13:56.5856195Z   error: subprocess-exited-with-error`
    - Line 54: `2026-02-25T04:13:56.5872992Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 58: `2026-02-25T04:13:56.5874720Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 67: `2026-02-25T04:13:56.7898198Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`
    - Line 97: `2026-02-25T04:13:56.9517570Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-25T04:13:56.9517570Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-25T04:13:52.4298322Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-25T04:13:52.4298322Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 5. WebSocket Health Tests (self-healing, Self-Healing & Recovery, 🔄)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-25T04:12:59Z
**Completed**: 2026-02-25T04:13:43Z
**Duration**: 44 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22381847573/job/64784086013)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 32: `2026-02-25T04:13:41.3960192Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 33: `2026-02-25T04:13:41.4005554Z   error: subprocess-exited-with-error`
    - Line 54: `2026-02-25T04:13:41.4025212Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 58: `2026-02-25T04:13:41.4027477Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 67: `2026-02-25T04:13:41.5192187Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`
    - Line 97: `2026-02-25T04:13:41.6643006Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-25T04:13:41.6643006Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-25T04:13:38.6574847Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-25T04:13:38.6574847Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 6. WebSocket Health Tests (latency-performance, Latency & Performance, ⚡)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-25T04:13:00Z
**Completed**: 2026-02-25T04:14:08Z
**Duration**: 68 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22381847573/job/64784086027)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 32: `2026-02-25T04:14:05.1456625Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 33: `2026-02-25T04:14:05.1505041Z   error: subprocess-exited-with-error`
    - Line 54: `2026-02-25T04:14:05.1526680Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 58: `2026-02-25T04:14:05.1529250Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 67: `2026-02-25T04:14:05.2723171Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`
    - Line 97: `2026-02-25T04:14:05.4265663Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-25T04:14:05.4265663Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-25T04:14:01.8556055Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-25T04:14:01.8556055Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

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

📊 *Report generated on 2026-02-25T04:15:24.572278*
🤖 *JARVIS CI/CD Auto-PR Manager*
