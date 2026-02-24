# CI/CD Failure Analysis Report

## Executive Summary

- **Workflow**: WebSocket Self-Healing Validation
- **Run Number**: #3
- **Branch**: `main`
- **Commit**: `cfa75296be1c6903f03a08e13e37dde76b831efe`
- **Status**: ❌ FAILED
- **Timestamp**: 2026-02-24T04:11:10Z
- **Triggered By**: @nandkishorrathodk-art
- **Workflow URL**: [View Run](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22336264778)

## Failure Overview

Total Failed Jobs: **6**

| # | Job Name | Category | Severity | Duration |
|---|----------|----------|----------|----------|
| 1 | WebSocket Health Tests (concurrent-connections, Concurrent Connections, 🔗) | test_failure | high | 39s |
| 2 | WebSocket Health Tests (self-healing, Self-Healing & Recovery, 🔄) | test_failure | high | 52s |
| 3 | WebSocket Health Tests (connection-lifecycle, Connection Lifecycle, 🔌) | test_failure | high | 41s |
| 4 | WebSocket Health Tests (heartbeat-monitoring, Heartbeat & Health Monitoring, 💓) | test_failure | high | 57s |
| 5 | WebSocket Health Tests (message-delivery, Message Delivery & Reliability, 📨) | test_failure | high | 64s |
| 6 | WebSocket Health Tests (latency-performance, Latency & Performance, ⚡) | test_failure | high | 48s |

## Detailed Analysis

### 1. WebSocket Health Tests (concurrent-connections, Concurrent Connections, 🔗)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-24T04:11:27Z
**Completed**: 2026-02-24T04:12:06Z
**Duration**: 39 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22336264778/job/64629373303)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 32: `2026-02-24T04:12:04.8418021Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 33: `2026-02-24T04:12:04.8465908Z   error: subprocess-exited-with-error`
    - Line 54: `2026-02-24T04:12:04.8479455Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 58: `2026-02-24T04:12:04.8480989Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 67: `2026-02-24T04:12:04.9648540Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`
    - Line 97: `2026-02-24T04:12:05.1256997Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T04:12:05.1256997Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-24T04:12:01.6811961Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-24T04:12:01.6811961Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 2. WebSocket Health Tests (self-healing, Self-Healing & Recovery, 🔄)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-24T04:11:27Z
**Completed**: 2026-02-24T04:12:19Z
**Duration**: 52 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22336264778/job/64629373305)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 32: `2026-02-24T04:12:16.7471077Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 33: `2026-02-24T04:12:16.7517566Z   error: subprocess-exited-with-error`
    - Line 54: `2026-02-24T04:12:16.7532800Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 58: `2026-02-24T04:12:16.7534242Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 67: `2026-02-24T04:12:16.8789474Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`
    - Line 97: `2026-02-24T04:12:17.0212519Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T04:12:17.0212519Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-24T04:12:13.4931047Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-24T04:12:13.4931047Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 3. WebSocket Health Tests (connection-lifecycle, Connection Lifecycle, 🔌)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-24T04:11:27Z
**Completed**: 2026-02-24T04:12:08Z
**Duration**: 41 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22336264778/job/64629373310)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 32: `2026-02-24T04:12:06.2216158Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 33: `2026-02-24T04:12:06.2264656Z   error: subprocess-exited-with-error`
    - Line 54: `2026-02-24T04:12:06.2278720Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 58: `2026-02-24T04:12:06.2280401Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 67: `2026-02-24T04:12:06.3522977Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`
    - Line 97: `2026-02-24T04:12:06.5069652Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T04:12:06.5069652Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-24T04:12:02.9949074Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-24T04:12:02.9949074Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 4. WebSocket Health Tests (heartbeat-monitoring, Heartbeat & Health Monitoring, 💓)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-24T04:11:27Z
**Completed**: 2026-02-24T04:12:24Z
**Duration**: 57 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22336264778/job/64629373314)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 32: `2026-02-24T04:12:22.4306608Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 33: `2026-02-24T04:12:22.4356303Z   error: subprocess-exited-with-error`
    - Line 54: `2026-02-24T04:12:22.4378340Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 58: `2026-02-24T04:12:22.4380922Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 67: `2026-02-24T04:12:22.5979267Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`
    - Line 97: `2026-02-24T04:12:22.7569196Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T04:12:22.7569196Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-24T04:12:18.7153370Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-24T04:12:18.7153370Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 5. WebSocket Health Tests (message-delivery, Message Delivery & Reliability, 📨)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-24T04:11:27Z
**Completed**: 2026-02-24T04:12:31Z
**Duration**: 64 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22336264778/job/64629373328)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 32: `2026-02-24T04:12:28.0177213Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 33: `2026-02-24T04:12:28.0250435Z   error: subprocess-exited-with-error`
    - Line 54: `2026-02-24T04:12:28.0266172Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 58: `2026-02-24T04:12:28.0267817Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 67: `2026-02-24T04:12:28.2091903Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`
    - Line 97: `2026-02-24T04:12:28.3718665Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T04:12:28.3718665Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-24T04:12:24.1271381Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-24T04:12:24.1271381Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 6. WebSocket Health Tests (latency-performance, Latency & Performance, ⚡)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-24T04:11:27Z
**Completed**: 2026-02-24T04:12:15Z
**Duration**: 48 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22336264778/job/64629373334)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 32: `2026-02-24T04:12:13.1717742Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 33: `2026-02-24T04:12:13.1771771Z   error: subprocess-exited-with-error`
    - Line 54: `2026-02-24T04:12:13.1793917Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 58: `2026-02-24T04:12:13.1796653Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 67: `2026-02-24T04:12:13.3470973Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`
    - Line 97: `2026-02-24T04:12:13.5091077Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T04:12:13.5091077Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-24T04:12:09.5544535Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 1
  - Sample matches:
    - Line 1: `2026-02-24T04:12:09.5544535Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

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

📊 *Report generated on 2026-02-24T04:13:50.759103*
🤖 *JARVIS CI/CD Auto-PR Manager*
