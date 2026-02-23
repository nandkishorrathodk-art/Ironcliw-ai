# CI/CD Failure Analysis Report

## Executive Summary

- **Workflow**: Complete Unlock Test Suite (Master)
- **Run Number**: #3
- **Branch**: `main`
- **Commit**: `9c3daeb317327f11cc2b19653e461bacdcf4cf03`
- **Status**: ❌ FAILED
- **Timestamp**: 2026-02-23T05:25:05Z
- **Triggered By**: @nandkishorrathodk-art
- **Workflow URL**: [View Run](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22294016090)

## Failure Overview

Total Failed Jobs: **4**

| # | Job Name | Category | Severity | Duration |
|---|----------|----------|----------|----------|
| 1 | Run Biometric Voice E2E / Integration Biometric Tests - macOS | dependency_error | high | 56s |
| 2 | Run Unlock Integration E2E / Integration Tests - macOS | dependency_error | high | 43s |
| 3 | Generate Combined Test Summary | test_failure | high | 5s |
| 4 | Notify Test Status | test_failure | high | 2s |

## Detailed Analysis

### 1. Run Biometric Voice E2E / Integration Biometric Tests - macOS

**Status**: ❌ failure
**Category**: Dependency Error
**Severity**: HIGH
**Started**: 2026-02-23T05:25:20Z
**Completed**: 2026-02-23T05:26:16Z
**Duration**: 56 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22294016090/job/64486921653)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 39: `2026-02-23T05:26:12.6804230Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 40: `2026-02-23T05:26:12.6859940Z   error: subprocess-exited-with-error`
    - Line 61: `2026-02-23T05:26:12.6963890Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 2
  - Sample matches:
    - Line 65: `2026-02-23T05:26:12.6965720Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 97: `2026-02-23T05:26:13.9153960Z ##[warning]The process '/opt/homebrew/bin/git' failed with exit code 12`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-23T05:26:13.0633680Z   if-no-files-found: warn`
    - Line 86: `2026-02-23T05:26:13.4420130Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-23T05:26:13.9153960Z ##[warning]The process '/opt/homebrew/bin/git' failed with exit code 12`

- Pattern: `AssertionError|Exception`
  - Occurrences: 2
  - Sample matches:
    - Line 5: `2026-02-23T05:26:01.8205010Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 7: `2026-02-23T05:26:08.8640530Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 2
  - Sample matches:
    - Line 5: `2026-02-23T05:26:01.8205010Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 7: `2026-02-23T05:26:08.8640530Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 2. Run Unlock Integration E2E / Integration Tests - macOS

**Status**: ❌ failure
**Category**: Dependency Error
**Severity**: HIGH
**Started**: 2026-02-23T05:25:23Z
**Completed**: 2026-02-23T05:26:06Z
**Duration**: 43 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22294016090/job/64486924886)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 39: `2026-02-23T05:26:01.8667590Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 40: `2026-02-23T05:26:01.8890970Z   error: subprocess-exited-with-error`
    - Line 61: `2026-02-23T05:26:02.0275570Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 2
  - Sample matches:
    - Line 65: `2026-02-23T05:26:02.0277630Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 97: `2026-02-23T05:26:02.7864030Z ##[warning]The process '/opt/homebrew/bin/git' failed with exit code 12`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-23T05:26:02.1928940Z   if-no-files-found: warn`
    - Line 86: `2026-02-23T05:26:02.4253400Z ##[warning]No files were found with the provided path: test-results/unl`
    - Line 97: `2026-02-23T05:26:02.7864030Z ##[warning]The process '/opt/homebrew/bin/git' failed with exit code 12`

- Pattern: `AssertionError|Exception`
  - Occurrences: 2
  - Sample matches:
    - Line 5: `2026-02-23T05:25:57.9316720Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 7: `2026-02-23T05:25:58.9615410Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 2
  - Sample matches:
    - Line 5: `2026-02-23T05:25:57.9316720Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 7: `2026-02-23T05:25:58.9615410Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 3. Generate Combined Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-23T05:26:19Z
**Completed**: 2026-02-23T05:26:24Z
**Duration**: 5 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22294016090/job/64486972313)

#### Failed Steps

- **Step 2**: Generate Combined Summary
- **Step 3**: Create Issue on Failure

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 20: `2026-02-23T05:26:21.7578856Z ##[error]Process completed with exit code 1.`
    - Line 40: `2026-02-23T05:26:22.0516785Z RequestError [HttpError]: Resource not accessible by integration`
    - Line 97: `2026-02-23T05:26:22.0648229Z ##[error]Unhandled error: HttpError: Resource not accessible by integra`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 8
  - Sample matches:
    - Line 0: `2026-02-23T05:26:21.7391236Z [36;1mif [ "failure" = "success" ] && [ "failure" = "success" ]; then`
    - Line 4: `2026-02-23T05:26:21.7394956Z [36;1m  FAIL=0[0m`
    - Line 6: `2026-02-23T05:26:21.7395973Z [36;1m  echo "## ❌ Overall Status: TESTS FAILED" >> $GITHUB_STEP_SUMMA`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 4. Notify Test Status

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-23T05:26:26Z
**Completed**: 2026-02-23T05:26:28Z
**Duration**: 2 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22294016090/job/64486979652)

#### Failed Steps

- **Step 3**: Failure Notification

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line -23: `2026-02-23T05:26:27.0793916Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line -31: `2026-02-23T05:26:27.0121932Z ##[group]Run echo "❌ Unlock tests failed - 'unlock my screen' may be br`
    - Line -30: `2026-02-23T05:26:27.0123142Z [36;1mecho "❌ Unlock tests failed - 'unlock my screen' may be broken!"`
    - Line -25: `2026-02-23T05:26:27.0775405Z ❌ Unlock tests failed - 'unlock my screen' may be broken!`

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

📊 *Report generated on 2026-02-23T05:27:14.678197*
🤖 *JARVIS CI/CD Auto-PR Manager*
