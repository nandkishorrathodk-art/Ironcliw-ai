# CI/CD Failure Analysis Report

## Executive Summary

- **Workflow**: Complete Unlock Test Suite (Master)
- **Run Number**: #17
- **Branch**: `main`
- **Commit**: `ce513d74a600e9e7f9b9299af7577d74509e998f`
- **Status**: ❌ FAILED
- **Timestamp**: 2026-02-28T04:54:44Z
- **Triggered By**: @nandkishorrathodk-art
- **Workflow URL**: [View Run](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22513764024)

## Failure Overview

Total Failed Jobs: **4**

| # | Job Name | Category | Severity | Duration |
|---|----------|----------|----------|----------|
| 1 | Run Unlock Integration E2E / Integration Tests - macOS | dependency_error | high | 33s |
| 2 | Run Biometric Voice E2E / Integration Biometric Tests - macOS | dependency_error | high | 52s |
| 3 | Generate Combined Test Summary | test_failure | high | 4s |
| 4 | Notify Test Status | test_failure | high | 3s |

## Detailed Analysis

### 1. Run Unlock Integration E2E / Integration Tests - macOS

**Status**: ❌ failure
**Category**: Dependency Error
**Severity**: HIGH
**Started**: 2026-02-28T04:55:02Z
**Completed**: 2026-02-28T04:55:35Z
**Duration**: 33 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22513764024/job/65227958123)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 39: `2026-02-28T04:55:32.4130350Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 40: `2026-02-28T04:55:32.4156870Z   error: subprocess-exited-with-error`
    - Line 61: `2026-02-28T04:55:32.4216530Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 2
  - Sample matches:
    - Line 65: `2026-02-28T04:55:32.4217640Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 97: `2026-02-28T04:55:32.9412250Z ##[warning]The process '/opt/homebrew/bin/git' failed with exit code 12`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T04:55:32.5275750Z   if-no-files-found: warn`
    - Line 86: `2026-02-28T04:55:32.7360630Z ##[warning]No files were found with the provided path: test-results/unl`
    - Line 97: `2026-02-28T04:55:32.9412250Z ##[warning]The process '/opt/homebrew/bin/git' failed with exit code 12`

- Pattern: `AssertionError|Exception`
  - Occurrences: 2
  - Sample matches:
    - Line 5: `2026-02-28T04:55:29.1320680Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 7: `2026-02-28T04:55:29.8647610Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 2
  - Sample matches:
    - Line 5: `2026-02-28T04:55:29.1320680Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 7: `2026-02-28T04:55:29.8647610Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 2. Run Biometric Voice E2E / Integration Biometric Tests - macOS

**Status**: ❌ failure
**Category**: Dependency Error
**Severity**: HIGH
**Started**: 2026-02-28T04:55:03Z
**Completed**: 2026-02-28T04:55:55Z
**Duration**: 52 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22513764024/job/65227958137)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 39: `2026-02-28T04:55:51.3883450Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 40: `2026-02-28T04:55:51.4070660Z   error: subprocess-exited-with-error`
    - Line 61: `2026-02-28T04:55:51.4278390Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 2
  - Sample matches:
    - Line 65: `2026-02-28T04:55:51.4282540Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 97: `2026-02-28T04:55:52.2597530Z ##[warning]The process '/opt/homebrew/bin/git' failed with exit code 12`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T04:55:51.6327500Z   if-no-files-found: warn`
    - Line 86: `2026-02-28T04:55:51.8948200Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-28T04:55:52.2597530Z ##[warning]The process '/opt/homebrew/bin/git' failed with exit code 12`

- Pattern: `AssertionError|Exception`
  - Occurrences: 2
  - Sample matches:
    - Line 5: `2026-02-28T04:55:42.1416530Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 7: `2026-02-28T04:55:47.8287090Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 2
  - Sample matches:
    - Line 5: `2026-02-28T04:55:42.1416530Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 7: `2026-02-28T04:55:47.8287090Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 3. Generate Combined Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-28T04:55:57Z
**Completed**: 2026-02-28T04:56:01Z
**Duration**: 4 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22513764024/job/65227992905)

#### Failed Steps

- **Step 2**: Generate Combined Summary
- **Step 3**: Create Issue on Failure

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 20: `2026-02-28T04:55:59.8863005Z ##[error]Process completed with exit code 1.`
    - Line 40: `2026-02-28T04:56:00.2193549Z RequestError [HttpError]: Resource not accessible by integration`
    - Line 97: `2026-02-28T04:56:00.2311114Z ##[error]Unhandled error: HttpError: Resource not accessible by integra`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 8
  - Sample matches:
    - Line 0: `2026-02-28T04:55:59.7999899Z [36;1mif [ "failure" = "success" ] && [ "failure" = "success" ]; then`
    - Line 4: `2026-02-28T04:55:59.8003481Z [36;1m  FAIL=0[0m`
    - Line 6: `2026-02-28T04:55:59.8004998Z [36;1m  echo "## ❌ Overall Status: TESTS FAILED" >> $GITHUB_STEP_SUMMA`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 4. Notify Test Status

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-28T04:56:03Z
**Completed**: 2026-02-28T04:56:06Z
**Duration**: 3 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22513764024/job/65227997028)

#### Failed Steps

- **Step 3**: Failure Notification

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line -23: `2026-02-28T04:56:04.7352950Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line -31: `2026-02-28T04:56:04.6274563Z ##[group]Run echo "❌ Unlock tests failed - 'unlock my screen' may be br`
    - Line -30: `2026-02-28T04:56:04.6275706Z [36;1mecho "❌ Unlock tests failed - 'unlock my screen' may be broken!"`
    - Line -25: `2026-02-28T04:56:04.7334256Z ❌ Unlock tests failed - 'unlock my screen' may be broken!`

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

📊 *Report generated on 2026-02-28T04:57:20.192608*
🤖 *JARVIS CI/CD Auto-PR Manager*
