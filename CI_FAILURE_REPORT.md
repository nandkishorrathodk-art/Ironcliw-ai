# CI/CD Failure Analysis Report

## Executive Summary

- **Workflow**: Complete Unlock Test Suite (Master)
- **Run Number**: #4
- **Branch**: `main`
- **Commit**: `fee806bcecbc169d123ebae35fef4a70c6575f4a`
- **Status**: ❌ FAILED
- **Timestamp**: 2026-02-24T05:20:23Z
- **Triggered By**: @nandkishorrathodk-art
- **Workflow URL**: [View Run](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22337785404)

## Failure Overview

Total Failed Jobs: **4**

| # | Job Name | Category | Severity | Duration |
|---|----------|----------|----------|----------|
| 1 | Run Unlock Integration E2E / Integration Tests - macOS | dependency_error | high | 46s |
| 2 | Run Biometric Voice E2E / Integration Biometric Tests - macOS | dependency_error | high | 36s |
| 3 | Generate Combined Test Summary | test_failure | high | 3s |
| 4 | Notify Test Status | test_failure | high | 3s |

## Detailed Analysis

### 1. Run Unlock Integration E2E / Integration Tests - macOS

**Status**: ❌ failure
**Category**: Dependency Error
**Severity**: HIGH
**Started**: 2026-02-24T05:20:38Z
**Completed**: 2026-02-24T05:21:24Z
**Duration**: 46 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22337785404/job/64634241173)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 39: `2026-02-24T05:21:20.9484300Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 40: `2026-02-24T05:21:20.9524280Z   error: subprocess-exited-with-error`
    - Line 61: `2026-02-24T05:21:20.9647430Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 2
  - Sample matches:
    - Line 65: `2026-02-24T05:21:20.9649910Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 97: `2026-02-24T05:21:21.8642140Z ##[warning]The process '/opt/homebrew/bin/git' failed with exit code 12`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T05:21:21.2027460Z   if-no-files-found: warn`
    - Line 86: `2026-02-24T05:21:21.4691270Z ##[warning]No files were found with the provided path: test-results/unl`
    - Line 97: `2026-02-24T05:21:21.8642140Z ##[warning]The process '/opt/homebrew/bin/git' failed with exit code 12`

- Pattern: `AssertionError|Exception`
  - Occurrences: 2
  - Sample matches:
    - Line 5: `2026-02-24T05:21:15.6116930Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 7: `2026-02-24T05:21:17.1701090Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 2
  - Sample matches:
    - Line 5: `2026-02-24T05:21:15.6116930Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 7: `2026-02-24T05:21:17.1701090Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 2. Run Biometric Voice E2E / Integration Biometric Tests - macOS

**Status**: ❌ failure
**Category**: Dependency Error
**Severity**: HIGH
**Started**: 2026-02-24T05:20:39Z
**Completed**: 2026-02-24T05:21:15Z
**Duration**: 36 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22337785404/job/64634242722)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 39: `2026-02-24T05:21:12.5414300Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 40: `2026-02-24T05:21:12.5457420Z   error: subprocess-exited-with-error`
    - Line 61: `2026-02-24T05:21:12.5878030Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 2
  - Sample matches:
    - Line 65: `2026-02-24T05:21:12.5879290Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 97: `2026-02-24T05:21:13.1714000Z ##[warning]The process '/opt/homebrew/bin/git' failed with exit code 12`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T05:21:12.7247410Z   if-no-files-found: warn`
    - Line 86: `2026-02-24T05:21:12.9381350Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T05:21:13.1714000Z ##[warning]The process '/opt/homebrew/bin/git' failed with exit code 12`

- Pattern: `AssertionError|Exception`
  - Occurrences: 2
  - Sample matches:
    - Line 5: `2026-02-24T05:21:05.6099000Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 7: `2026-02-24T05:21:10.1997740Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 2
  - Sample matches:
    - Line 5: `2026-02-24T05:21:05.6099000Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 7: `2026-02-24T05:21:10.1997740Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 3. Generate Combined Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-24T05:21:27Z
**Completed**: 2026-02-24T05:21:30Z
**Duration**: 3 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22337785404/job/64634296123)

#### Failed Steps

- **Step 2**: Generate Combined Summary
- **Step 3**: Create Issue on Failure

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 20: `2026-02-24T05:21:28.6496252Z ##[error]Process completed with exit code 1.`
    - Line 40: `2026-02-24T05:21:28.9764607Z RequestError [HttpError]: Resource not accessible by integration`
    - Line 41: `2026-02-24T05:21:28.9787128Z ##[error]Unhandled error: HttpError: Resource not accessible by integra`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 8
  - Sample matches:
    - Line 0: `2026-02-24T05:21:28.6312316Z [36;1mif [ "failure" = "success" ] && [ "failure" = "success" ]; then`
    - Line 4: `2026-02-24T05:21:28.6315504Z [36;1m  FAIL=0[0m`
    - Line 6: `2026-02-24T05:21:28.6316374Z [36;1m  echo "## ❌ Overall Status: TESTS FAILED" >> $GITHUB_STEP_SUMMA`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 4. Notify Test Status

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-24T05:21:33Z
**Completed**: 2026-02-24T05:21:36Z
**Duration**: 3 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22337785404/job/64634302945)

#### Failed Steps

- **Step 3**: Failure Notification

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line -23: `2026-02-24T05:21:34.2929487Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line -31: `2026-02-24T05:21:34.2062621Z ##[group]Run echo "❌ Unlock tests failed - 'unlock my screen' may be br`
    - Line -30: `2026-02-24T05:21:34.2063679Z [36;1mecho "❌ Unlock tests failed - 'unlock my screen' may be broken!"`
    - Line -25: `2026-02-24T05:21:34.2903013Z ❌ Unlock tests failed - 'unlock my screen' may be broken!`

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

📊 *Report generated on 2026-02-24T05:22:41.356825*
🤖 *JARVIS CI/CD Auto-PR Manager*
