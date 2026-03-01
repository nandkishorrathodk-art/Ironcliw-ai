# CI/CD Failure Analysis Report

## Executive Summary

- **Workflow**: Complete Unlock Test Suite (Master)
- **Run Number**: #19
- **Branch**: `main`
- **Commit**: `41b1fb38c756c5ecab9f500d4f623f76a57269d1`
- **Status**: ❌ FAILED
- **Timestamp**: 2026-03-01T05:16:42Z
- **Triggered By**: @nandkishorrathodk-art
- **Workflow URL**: [View Run](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22536629382)

## Failure Overview

Total Failed Jobs: **4**

| # | Job Name | Category | Severity | Duration |
|---|----------|----------|----------|----------|
| 1 | Run Biometric Voice E2E / Integration Biometric Tests - macOS | dependency_error | high | 49s |
| 2 | Run Unlock Integration E2E / Integration Tests - macOS | dependency_error | high | 41s |
| 3 | Generate Combined Test Summary | test_failure | high | 5s |
| 4 | Notify Test Status | test_failure | high | 4s |

## Detailed Analysis

### 1. Run Biometric Voice E2E / Integration Biometric Tests - macOS

**Status**: ❌ failure
**Category**: Dependency Error
**Severity**: HIGH
**Started**: 2026-03-01T05:17:03Z
**Completed**: 2026-03-01T05:17:52Z
**Duration**: 49 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22536629382/job/65285288892)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 39: `2026-03-01T05:17:47.5744230Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 40: `2026-03-01T05:17:47.5776750Z   error: subprocess-exited-with-error`
    - Line 61: `2026-03-01T05:17:47.5839070Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 2
  - Sample matches:
    - Line 65: `2026-03-01T05:17:47.5841610Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 97: `2026-03-01T05:17:48.4711510Z ##[warning]The process '/opt/homebrew/bin/git' failed with exit code 12`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-03-01T05:17:47.8198810Z   if-no-files-found: warn`
    - Line 86: `2026-03-01T05:17:48.1251550Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-03-01T05:17:48.4711510Z ##[warning]The process '/opt/homebrew/bin/git' failed with exit code 12`

- Pattern: `AssertionError|Exception`
  - Occurrences: 2
  - Sample matches:
    - Line 5: `2026-03-01T05:17:38.2336680Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 7: `2026-03-01T05:17:44.1366270Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 2
  - Sample matches:
    - Line 5: `2026-03-01T05:17:38.2336680Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 7: `2026-03-01T05:17:44.1366270Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 2. Run Unlock Integration E2E / Integration Tests - macOS

**Status**: ❌ failure
**Category**: Dependency Error
**Severity**: HIGH
**Started**: 2026-03-01T05:17:09Z
**Completed**: 2026-03-01T05:17:50Z
**Duration**: 41 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22536629382/job/65285294011)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 39: `2026-03-01T05:17:46.6793420Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 40: `2026-03-01T05:17:46.6899710Z   error: subprocess-exited-with-error`
    - Line 61: `2026-03-01T05:17:46.7557240Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 2
  - Sample matches:
    - Line 65: `2026-03-01T05:17:46.7560820Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 97: `2026-03-01T05:17:47.3592250Z ##[warning]The process '/opt/homebrew/bin/git' failed with exit code 12`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-03-01T05:17:46.8966220Z   if-no-files-found: warn`
    - Line 86: `2026-03-01T05:17:47.1142830Z ##[warning]No files were found with the provided path: test-results/unl`
    - Line 97: `2026-03-01T05:17:47.3592250Z ##[warning]The process '/opt/homebrew/bin/git' failed with exit code 12`

- Pattern: `AssertionError|Exception`
  - Occurrences: 2
  - Sample matches:
    - Line 5: `2026-03-01T05:17:42.1801190Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 7: `2026-03-01T05:17:43.0301590Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 2
  - Sample matches:
    - Line 5: `2026-03-01T05:17:42.1801190Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 7: `2026-03-01T05:17:43.0301590Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 3. Generate Combined Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-03-01T05:17:55Z
**Completed**: 2026-03-01T05:18:00Z
**Duration**: 5 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22536629382/job/65285321448)

#### Failed Steps

- **Step 2**: Generate Combined Summary
- **Step 3**: Create Issue on Failure

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 20: `2026-03-01T05:17:57.4066265Z ##[error]Process completed with exit code 1.`
    - Line 40: `2026-03-01T05:17:57.8684421Z RequestError [HttpError]: Resource not accessible by integration`
    - Line 62: `2026-03-01T05:17:57.8749649Z ##[error]Unhandled error: HttpError: Resource not accessible by integra`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 8
  - Sample matches:
    - Line 0: `2026-03-01T05:17:57.3815909Z [36;1mif [ "failure" = "success" ] && [ "failure" = "success" ]; then`
    - Line 4: `2026-03-01T05:17:57.3823258Z [36;1m  FAIL=0[0m`
    - Line 6: `2026-03-01T05:17:57.3825253Z [36;1m  echo "## ❌ Overall Status: TESTS FAILED" >> $GITHUB_STEP_SUMMA`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 4. Notify Test Status

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-03-01T05:18:02Z
**Completed**: 2026-03-01T05:18:06Z
**Duration**: 4 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22536629382/job/65285325496)

#### Failed Steps

- **Step 3**: Failure Notification

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line -23: `2026-03-01T05:18:03.8880438Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line -31: `2026-03-01T05:18:03.8022399Z ##[group]Run echo "❌ Unlock tests failed - 'unlock my screen' may be br`
    - Line -30: `2026-03-01T05:18:03.8023347Z [36;1mecho "❌ Unlock tests failed - 'unlock my screen' may be broken!"`
    - Line -25: `2026-03-01T05:18:03.8855731Z ❌ Unlock tests failed - 'unlock my screen' may be broken!`

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

📊 *Report generated on 2026-03-01T05:19:12.654489*
🤖 *Ironcliw CI/CD Auto-PR Manager*
