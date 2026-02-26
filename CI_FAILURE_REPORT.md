# CI/CD Failure Analysis Report

## Executive Summary

- **Workflow**: Complete Unlock Test Suite (Master)
- **Run Number**: #15
- **Branch**: `main`
- **Commit**: `ce513d74a600e9e7f9b9299af7577d74509e998f`
- **Status**: ❌ FAILED
- **Timestamp**: 2026-02-26T05:19:16Z
- **Triggered By**: @nandkishorrathodk-art
- **Workflow URL**: [View Run](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428989636)

## Failure Overview

Total Failed Jobs: **4**

| # | Job Name | Category | Severity | Duration |
|---|----------|----------|----------|----------|
| 1 | Run Biometric Voice E2E / Integration Biometric Tests - macOS | dependency_error | high | 53s |
| 2 | Run Unlock Integration E2E / Integration Tests - macOS | dependency_error | high | 45s |
| 3 | Generate Combined Test Summary | test_failure | high | 4s |
| 4 | Notify Test Status | test_failure | high | 3s |

## Detailed Analysis

### 1. Run Biometric Voice E2E / Integration Biometric Tests - macOS

**Status**: ❌ failure
**Category**: Dependency Error
**Severity**: HIGH
**Started**: 2026-02-26T05:19:32Z
**Completed**: 2026-02-26T05:20:25Z
**Duration**: 53 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428989636/job/64943628304)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 39: `2026-02-26T05:20:20.9932760Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 40: `2026-02-26T05:20:20.9970700Z   error: subprocess-exited-with-error`
    - Line 61: `2026-02-26T05:20:21.1576560Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 2
  - Sample matches:
    - Line 65: `2026-02-26T05:20:21.1988610Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 97: `2026-02-26T05:20:21.9207420Z ##[warning]The process '/opt/homebrew/bin/git' failed with exit code 12`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T05:20:21.2195270Z   if-no-files-found: warn`
    - Line 86: `2026-02-26T05:20:21.5069270Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T05:20:21.9207420Z ##[warning]The process '/opt/homebrew/bin/git' failed with exit code 12`

- Pattern: `AssertionError|Exception`
  - Occurrences: 2
  - Sample matches:
    - Line 5: `2026-02-26T05:20:10.0427310Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 7: `2026-02-26T05:20:16.8924290Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 2
  - Sample matches:
    - Line 5: `2026-02-26T05:20:10.0427310Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 7: `2026-02-26T05:20:16.8924290Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 2. Run Unlock Integration E2E / Integration Tests - macOS

**Status**: ❌ failure
**Category**: Dependency Error
**Severity**: HIGH
**Started**: 2026-02-26T05:19:34Z
**Completed**: 2026-02-26T05:20:19Z
**Duration**: 45 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428989636/job/64943629988)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 6
  - Sample matches:
    - Line 39: `2026-02-26T05:20:15.2865950Z   Getting requirements to build wheel: finished with status 'error'`
    - Line 40: `2026-02-26T05:20:15.2894810Z   error: subprocess-exited-with-error`
    - Line 61: `2026-02-26T05:20:15.2999510Z       ModuleNotFoundError: No module named 'pkg_resources'`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 2
  - Sample matches:
    - Line 65: `2026-02-26T05:20:15.3000760Z ERROR: Failed to build 'openai-whisper' when getting requirements to bu`
    - Line 97: `2026-02-26T05:20:16.0975620Z ##[warning]The process '/opt/homebrew/bin/git' failed with exit code 12`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T05:20:15.5218270Z   if-no-files-found: warn`
    - Line 86: `2026-02-26T05:20:15.7563990Z ##[warning]No files were found with the provided path: test-results/unl`
    - Line 97: `2026-02-26T05:20:16.0975620Z ##[warning]The process '/opt/homebrew/bin/git' failed with exit code 12`

- Pattern: `AssertionError|Exception`
  - Occurrences: 2
  - Sample matches:
    - Line 5: `2026-02-26T05:20:10.1235180Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 7: `2026-02-26T05:20:11.7088660Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 2
  - Sample matches:
    - Line 5: `2026-02-26T05:20:10.1235180Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 7: `2026-02-26T05:20:11.7088660Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 3. Generate Combined Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-26T05:20:27Z
**Completed**: 2026-02-26T05:20:31Z
**Duration**: 4 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428989636/job/64943692717)

#### Failed Steps

- **Step 2**: Generate Combined Summary
- **Step 3**: Create Issue on Failure

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 20: `2026-02-26T05:20:29.0536852Z ##[error]Process completed with exit code 1.`
    - Line 40: `2026-02-26T05:20:29.3666859Z RequestError [HttpError]: Resource not accessible by integration`
    - Line 72: `2026-02-26T05:20:29.3714335Z ##[error]Unhandled error: HttpError: Resource not accessible by integra`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 8
  - Sample matches:
    - Line 0: `2026-02-26T05:20:29.0312162Z [36;1mif [ "failure" = "success" ] && [ "failure" = "success" ]; then`
    - Line 4: `2026-02-26T05:20:29.0315473Z [36;1m  FAIL=0[0m`
    - Line 6: `2026-02-26T05:20:29.0316379Z [36;1m  echo "## ❌ Overall Status: TESTS FAILED" >> $GITHUB_STEP_SUMMA`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 4. Notify Test Status

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-26T05:20:33Z
**Completed**: 2026-02-26T05:20:36Z
**Duration**: 3 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428989636/job/64943700142)

#### Failed Steps

- **Step 3**: Failure Notification

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line -23: `2026-02-26T05:20:34.3189489Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line -31: `2026-02-26T05:20:34.2100225Z ##[group]Run echo "❌ Unlock tests failed - 'unlock my screen' may be br`
    - Line -30: `2026-02-26T05:20:34.2101171Z [36;1mecho "❌ Unlock tests failed - 'unlock my screen' may be broken!"`
    - Line -25: `2026-02-26T05:20:34.3172282Z ❌ Unlock tests failed - 'unlock my screen' may be broken!`

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

📊 *Report generated on 2026-02-26T05:21:47.086231*
🤖 *JARVIS CI/CD Auto-PR Manager*
