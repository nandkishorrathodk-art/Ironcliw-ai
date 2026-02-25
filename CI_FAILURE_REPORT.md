# CI/CD Failure Analysis Report

## Executive Summary

- **Workflow**: Complete Unlock Test Suite (Master)
- **Run Number**: #8
- **Branch**: `main`
- **Commit**: `e272639237ac2968a887a0c1df7659932a78fffe`
- **Status**: ❌ FAILED
- **Timestamp**: 2026-02-25T05:22:28Z
- **Triggered By**: @nandkishorrathodk-art
- **Workflow URL**: [View Run](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22383470345)

## Failure Overview

Total Failed Jobs: **4**

| # | Job Name | Category | Severity | Duration |
|---|----------|----------|----------|----------|
| 1 | Run Unlock Integration E2E / Integration Tests - macOS | dependency_error | high | 49s |
| 2 | Run Biometric Voice E2E / Integration Biometric Tests - macOS | dependency_error | high | 40s |
| 3 | Generate Combined Test Summary | test_failure | high | 3s |
| 4 | Notify Test Status | test_failure | high | 3s |

## Detailed Analysis

### 1. Run Unlock Integration E2E / Integration Tests - macOS

**Status**: ❌ failure
**Category**: Dependency Error
**Severity**: HIGH
**Started**: 2026-02-25T05:22:47Z
**Completed**: 2026-02-25T05:23:36Z
**Duration**: 49 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22383470345/job/64789136880)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 39: `2026-02-25T05:23:31.6954700Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 40: `2026-02-25T05:23:31.6989450Z   error: subprocess-exited-with-error`
    - Line 61: `2026-02-25T05:23:31.7204730Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 2
  - Sample matches:
    - Line 65: `2026-02-25T05:23:31.7209910Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 97: `2026-02-25T05:23:32.4750670Z ##[warning]The process '/opt/homebrew/bin/git' failed with exit code 12`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-25T05:23:31.9708790Z   if-no-files-found: warn`
    - Line 86: `2026-02-25T05:23:32.1613120Z ##[warning]No files were found with the provided path: test-results/unl`
    - Line 97: `2026-02-25T05:23:32.4750670Z ##[warning]The process '/opt/homebrew/bin/git' failed with exit code 12`

- Pattern: `AssertionError|Exception`
  - Occurrences: 2
  - Sample matches:
    - Line 5: `2026-02-25T05:23:25.9919390Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 7: `2026-02-25T05:23:26.9824910Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 2
  - Sample matches:
    - Line 5: `2026-02-25T05:23:25.9919390Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 7: `2026-02-25T05:23:26.9824910Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 2. Run Biometric Voice E2E / Integration Biometric Tests - macOS

**Status**: ❌ failure
**Category**: Dependency Error
**Severity**: HIGH
**Started**: 2026-02-25T05:22:48Z
**Completed**: 2026-02-25T05:23:28Z
**Duration**: 40 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22383470345/job/64789138057)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 39: `2026-02-25T05:23:24.5767770Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 40: `2026-02-25T05:23:24.5797260Z   error: subprocess-exited-with-error`
    - Line 61: `2026-02-25T05:23:24.5865150Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 2
  - Sample matches:
    - Line 65: `2026-02-25T05:23:24.5866820Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 97: `2026-02-25T05:23:25.1033650Z ##[warning]The process '/opt/homebrew/bin/git' failed with exit code 12`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-25T05:23:24.7430690Z   if-no-files-found: warn`
    - Line 86: `2026-02-25T05:23:24.8977620Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-25T05:23:25.1033650Z ##[warning]The process '/opt/homebrew/bin/git' failed with exit code 12`

- Pattern: `AssertionError|Exception`
  - Occurrences: 2
  - Sample matches:
    - Line 5: `2026-02-25T05:23:16.9286900Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 7: `2026-02-25T05:23:21.7264380Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 2
  - Sample matches:
    - Line 5: `2026-02-25T05:23:16.9286900Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 7: `2026-02-25T05:23:21.7264380Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 3. Generate Combined Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-25T05:23:38Z
**Completed**: 2026-02-25T05:23:41Z
**Duration**: 3 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22383470345/job/64789196389)

#### Failed Steps

- **Step 2**: Generate Combined Summary
- **Step 3**: Create Issue on Failure

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 20: `2026-02-25T05:23:39.9125448Z ##[error]Process completed with exit code 1.`
    - Line 40: `2026-02-25T05:23:40.2260101Z RequestError [HttpError]: Resource not accessible by integration`
    - Line 55: `2026-02-25T05:23:40.2305976Z ##[error]Unhandled error: HttpError: Resource not accessible by integra`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 8
  - Sample matches:
    - Line 0: `2026-02-25T05:23:39.8935839Z [36;1mif [ "failure" = "success" ] && [ "failure" = "success" ]; then`
    - Line 4: `2026-02-25T05:23:39.8939271Z [36;1m  FAIL=0[0m`
    - Line 6: `2026-02-25T05:23:39.8940199Z [36;1m  echo "## ❌ Overall Status: TESTS FAILED" >> $GITHUB_STEP_SUMMA`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 4. Notify Test Status

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-25T05:23:44Z
**Completed**: 2026-02-25T05:23:47Z
**Duration**: 3 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22383470345/job/64789202417)

#### Failed Steps

- **Step 3**: Failure Notification

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line -23: `2026-02-25T05:23:45.2676906Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line -31: `2026-02-25T05:23:45.1864547Z ##[group]Run echo "❌ Unlock tests failed - 'unlock my screen' may be br`
    - Line -30: `2026-02-25T05:23:45.1865651Z [36;1mecho "❌ Unlock tests failed - 'unlock my screen' may be broken!"`
    - Line -25: `2026-02-25T05:23:45.2648878Z ❌ Unlock tests failed - 'unlock my screen' may be broken!`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

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

📊 *Report generated on 2026-02-25T05:24:53.431432*
🤖 *JARVIS CI/CD Auto-PR Manager*
