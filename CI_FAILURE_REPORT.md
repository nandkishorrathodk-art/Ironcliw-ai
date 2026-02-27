# CI/CD Failure Analysis Report

## Executive Summary

- **Workflow**: Complete Unlock Test Suite (Master)
- **Run Number**: #16
- **Branch**: `main`
- **Commit**: `ce513d74a600e9e7f9b9299af7577d74509e998f`
- **Status**: ❌ FAILED
- **Timestamp**: 2026-02-27T05:14:12Z
- **Triggered By**: @nandkishorrathodk-art
- **Workflow URL**: [View Run](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22473786460)

## Failure Overview

Total Failed Jobs: **4**

| # | Job Name | Category | Severity | Duration |
|---|----------|----------|----------|----------|
| 1 | Run Biometric Voice E2E / Integration Biometric Tests - macOS | dependency_error | high | 53s |
| 2 | Run Unlock Integration E2E / Integration Tests - macOS | dependency_error | high | 33s |
| 3 | Generate Combined Test Summary | test_failure | high | 5s |
| 4 | Notify Test Status | test_failure | high | 2s |

## Detailed Analysis

### 1. Run Biometric Voice E2E / Integration Biometric Tests - macOS

**Status**: ❌ failure
**Category**: Dependency Error
**Severity**: HIGH
**Started**: 2026-02-27T05:14:28Z
**Completed**: 2026-02-27T05:15:21Z
**Duration**: 53 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22473786460/job/65096377739)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 39: `2026-02-27T05:15:16.9798170Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 40: `2026-02-27T05:15:16.9875770Z   error: subprocess-exited-with-error`
    - Line 61: `2026-02-27T05:15:16.9905440Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 2
  - Sample matches:
    - Line 65: `2026-02-27T05:15:16.9909030Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 97: `2026-02-27T05:15:17.8766370Z ##[warning]The process '/opt/homebrew/bin/git' failed with exit code 12`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-27T05:15:17.2716520Z   if-no-files-found: warn`
    - Line 86: `2026-02-27T05:15:17.5830560Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-27T05:15:17.8766370Z ##[warning]The process '/opt/homebrew/bin/git' failed with exit code 12`

- Pattern: `AssertionError|Exception`
  - Occurrences: 2
  - Sample matches:
    - Line 5: `2026-02-27T05:15:05.2562530Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 7: `2026-02-27T05:15:13.0492250Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 2
  - Sample matches:
    - Line 5: `2026-02-27T05:15:05.2562530Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 7: `2026-02-27T05:15:13.0492250Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 2. Run Unlock Integration E2E / Integration Tests - macOS

**Status**: ❌ failure
**Category**: Dependency Error
**Severity**: HIGH
**Started**: 2026-02-27T05:14:29Z
**Completed**: 2026-02-27T05:15:02Z
**Duration**: 33 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22473786460/job/65096379245)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 39: `2026-02-27T05:14:59.1029280Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 40: `2026-02-27T05:14:59.1102850Z   error: subprocess-exited-with-error`
    - Line 61: `2026-02-27T05:14:59.1150730Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 2
  - Sample matches:
    - Line 65: `2026-02-27T05:14:59.1153940Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 97: `2026-02-27T05:14:59.9020170Z ##[warning]The process '/opt/homebrew/bin/git' failed with exit code 12`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-27T05:14:59.3618650Z   if-no-files-found: warn`
    - Line 86: `2026-02-27T05:14:59.5899020Z ##[warning]No files were found with the provided path: test-results/unl`
    - Line 97: `2026-02-27T05:14:59.9020170Z ##[warning]The process '/opt/homebrew/bin/git' failed with exit code 12`

- Pattern: `AssertionError|Exception`
  - Occurrences: 2
  - Sample matches:
    - Line 5: `2026-02-27T05:14:55.7367870Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 7: `2026-02-27T05:14:56.7132850Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 2
  - Sample matches:
    - Line 5: `2026-02-27T05:14:55.7367870Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 7: `2026-02-27T05:14:56.7132850Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 3. Generate Combined Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-27T05:15:24Z
**Completed**: 2026-02-27T05:15:29Z
**Duration**: 5 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22473786460/job/65096435974)

#### Failed Steps

- **Step 2**: Generate Combined Summary
- **Step 3**: Create Issue on Failure

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 20: `2026-02-27T05:15:26.3704188Z ##[error]Process completed with exit code 1.`
    - Line 40: `2026-02-27T05:15:26.7587006Z RequestError [HttpError]: Resource not accessible by integration`
    - Line 41: `2026-02-27T05:15:26.7611160Z ##[error]Unhandled error: HttpError: Resource not accessible by integra`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 8
  - Sample matches:
    - Line 0: `2026-02-27T05:15:26.3533494Z [36;1mif [ "failure" = "success" ] && [ "failure" = "success" ]; then`
    - Line 4: `2026-02-27T05:15:26.3537019Z [36;1m  FAIL=0[0m`
    - Line 6: `2026-02-27T05:15:26.3537989Z [36;1m  echo "## ❌ Overall Status: TESTS FAILED" >> $GITHUB_STEP_SUMMA`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 4. Notify Test Status

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-27T05:15:31Z
**Completed**: 2026-02-27T05:15:33Z
**Duration**: 2 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22473786460/job/65096445472)

#### Failed Steps

- **Step 3**: Failure Notification

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line -23: `2026-02-27T05:15:32.1069486Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line -31: `2026-02-27T05:15:32.0068784Z ##[group]Run echo "❌ Unlock tests failed - 'unlock my screen' may be br`
    - Line -30: `2026-02-27T05:15:32.0069865Z [36;1mecho "❌ Unlock tests failed - 'unlock my screen' may be broken!"`
    - Line -25: `2026-02-27T05:15:32.1051092Z ❌ Unlock tests failed - 'unlock my screen' may be broken!`

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

📊 *Report generated on 2026-02-27T05:16:39.890951*
🤖 *JARVIS CI/CD Auto-PR Manager*
