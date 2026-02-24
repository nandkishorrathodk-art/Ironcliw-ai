# CI/CD Failure Analysis Report

## Executive Summary

- **Workflow**: WebSocket Self-Healing Validation
- **Run Number**: #4
- **Branch**: `main`
- **Commit**: `3ce7237a675833e142cfadbb33c39828ea904d68`
- **Status**: ❌ FAILED
- **Timestamp**: 2026-02-24T15:51:07Z
- **Triggered By**: @nandkishorrathodk-art
- **Workflow URL**: [View Run](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22358540291)

## Failure Overview

Total Failed Jobs: **6**

| # | Job Name | Category | Severity | Duration |
|---|----------|----------|----------|----------|
| 1 | WebSocket Health Tests (heartbeat-monitoring, Heartbeat & Health Monitoring, 💓) | test_failure | high | 50s |
| 2 | WebSocket Health Tests (latency-performance, Latency & Performance, ⚡) | test_failure | high | 40s |
| 3 | WebSocket Health Tests (concurrent-connections, Concurrent Connections, 🔗) | test_failure | high | 52s |
| 4 | WebSocket Health Tests (connection-lifecycle, Connection Lifecycle, 🔌) | test_failure | high | 55s |
| 5 | WebSocket Health Tests (self-healing, Self-Healing & Recovery, 🔄) | test_failure | high | 58s |
| 6 | WebSocket Health Tests (message-delivery, Message Delivery & Reliability, 📨) | test_failure | high | 47s |

## Detailed Analysis

### 1. WebSocket Health Tests (heartbeat-monitoring, Heartbeat & Health Monitoring, 💓)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-24T15:51:36Z
**Completed**: 2026-02-24T15:52:26Z
**Duration**: 50 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22358540291/job/64705059188)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 32: `2026-02-24T15:52:24.2101374Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 33: `2026-02-24T15:52:24.2149798Z   error: subprocess-exited-with-error`
    - Line 54: `2026-02-24T15:52:24.2172506Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 58: `2026-02-24T15:52:24.2175193Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 67: `2026-02-24T15:52:24.3492088Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`
    - Line 97: `2026-02-24T15:52:24.5046342Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T15:52:24.5046342Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-24T15:52:20.2755376Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-24T15:52:20.2755376Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 2. WebSocket Health Tests (latency-performance, Latency & Performance, ⚡)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-24T15:52:27Z
**Completed**: 2026-02-24T15:53:07Z
**Duration**: 40 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22358540291/job/64705059219)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 32: `2026-02-24T15:53:05.7207635Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 33: `2026-02-24T15:53:05.7255455Z   error: subprocess-exited-with-error`
    - Line 54: `2026-02-24T15:53:05.7270858Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 58: `2026-02-24T15:53:05.7272409Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 67: `2026-02-24T15:53:05.8433324Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`
    - Line 97: `2026-02-24T15:53:05.9949591Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T15:53:05.9949591Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-24T15:53:02.6492062Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-24T15:53:02.6492062Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 3. WebSocket Health Tests (concurrent-connections, Concurrent Connections, 🔗)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-24T15:52:22Z
**Completed**: 2026-02-24T15:53:14Z
**Duration**: 52 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22358540291/job/64705059229)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 32: `2026-02-24T15:53:11.7777529Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 33: `2026-02-24T15:53:11.7825512Z   error: subprocess-exited-with-error`
    - Line 54: `2026-02-24T15:53:11.7841051Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 58: `2026-02-24T15:53:11.7842693Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 67: `2026-02-24T15:53:11.9145235Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`
    - Line 97: `2026-02-24T15:53:12.0666112Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T15:53:12.0666112Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-24T15:53:07.7193660Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-24T15:53:07.7193660Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 4. WebSocket Health Tests (connection-lifecycle, Connection Lifecycle, 🔌)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-24T15:51:40Z
**Completed**: 2026-02-24T15:52:35Z
**Duration**: 55 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22358540291/job/64705059243)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 32: `2026-02-24T15:52:32.8210956Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 33: `2026-02-24T15:52:32.8266873Z   error: subprocess-exited-with-error`
    - Line 54: `2026-02-24T15:52:32.8289299Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 58: `2026-02-24T15:52:32.8291958Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 67: `2026-02-24T15:52:32.9542352Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`
    - Line 97: `2026-02-24T15:52:33.1077024Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T15:52:33.1077024Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-24T15:52:28.4708919Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-24T15:52:28.4708919Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 5. WebSocket Health Tests (self-healing, Self-Healing & Recovery, 🔄)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-24T15:51:30Z
**Completed**: 2026-02-24T15:52:28Z
**Duration**: 58 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22358540291/job/64705059274)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 32: `2026-02-24T15:52:25.9468260Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 33: `2026-02-24T15:52:25.9516884Z   error: subprocess-exited-with-error`
    - Line 54: `2026-02-24T15:52:25.9533456Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 58: `2026-02-24T15:52:25.9535077Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 67: `2026-02-24T15:52:26.0875716Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`
    - Line 97: `2026-02-24T15:52:26.2433610Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T15:52:26.2433610Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-24T15:52:22.8561779Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-24T15:52:22.8561779Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 6. WebSocket Health Tests (message-delivery, Message Delivery & Reliability, 📨)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-24T15:51:52Z
**Completed**: 2026-02-24T15:52:39Z
**Duration**: 47 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22358540291/job/64705059288)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 32: `2026-02-24T15:52:37.1549398Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 33: `2026-02-24T15:52:37.1597997Z   error: subprocess-exited-with-error`
    - Line 54: `2026-02-24T15:52:37.1620140Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 58: `2026-02-24T15:52:37.1622927Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 67: `2026-02-24T15:52:37.3116042Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`
    - Line 97: `2026-02-24T15:52:37.4652312Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T15:52:37.4652312Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-24T15:52:33.4097238Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-24T15:52:33.4097238Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

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

📊 *Report generated on 2026-02-24T15:56:08.976192*
🤖 *JARVIS CI/CD Auto-PR Manager*
