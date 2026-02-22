# CI/CD Failure Analysis Report

## Executive Summary

- **Workflow**: Unlock Integration E2E Testing
- **Run Number**: #1
- **Branch**: `main`
- **Commit**: `f0315234349df624ef29f63fdcb8bb7da520a6e6`
- **Status**: ❌ FAILED
- **Timestamp**: 2026-02-22T17:46:42Z
- **Triggered By**: @nandkishorrathodk-art
- **Workflow URL**: [View Run](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140212)

## Failure Overview

Total Failed Jobs: **3**

| # | Job Name | Category | Severity | Duration |
|---|----------|----------|----------|----------|
| 1 | Mock Tests - memory-security | test_failure | high | 15s |
| 2 | Mock Tests - security-checks | test_failure | high | 12s |
| 3 | Generate Test Summary | test_failure | high | 6s |

## Detailed Analysis

### 1. Mock Tests - memory-security

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-22T17:51:05Z
**Completed**: 2026-02-22T17:51:20Z
**Duration**: 15 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140212/job/64454360754)

#### Failed Steps

- **Step 6**: Run Mock Tests

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 5
  - Sample matches:
    - Line 36: `2026-02-22T17:51:17.7266119Z 2026-02-22 17:51:17,726 - __main__ - ERROR - ❌ Memory security test fai`
    - Line 37: `2026-02-22T17:51:17.7268568Z 2026-02-22 17:51:17,726 - __main__ - INFO - ❌ memory_security: Error: p`
    - Line 39: `2026-02-22T17:51:17.7288743Z 2026-02-22 17:51:17,728 - __main__ - ERROR - ❌ 1 test(s) failed`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 4
  - Sample matches:
    - Line 36: `2026-02-22T17:51:17.7266119Z 2026-02-22 17:51:17,726 - __main__ - ERROR - ❌ Memory security test fai`
    - Line 39: `2026-02-22T17:51:17.7288743Z 2026-02-22 17:51:17,728 - __main__ - ERROR - ❌ 1 test(s) failed`
    - Line 48: `2026-02-22T17:51:17.7299678Z ❌ Failed: 1`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 62: `2026-02-22T17:51:17.7510325Z   if-no-files-found: warn`
    - Line 97: `2026-02-22T17:51:18.5259924Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 2. Mock Tests - security-checks

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-22T17:48:50Z
**Completed**: 2026-02-22T17:49:02Z
**Duration**: 12 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140212/job/64454360755)

#### Failed Steps

- **Step 6**: Run Mock Tests

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 2
  - Sample matches:
    - Line 39: `2026-02-22T17:49:00.9522811Z 2026-02-22 17:49:00,952 - __main__ - ERROR - ❌ 1 test(s) failed`
    - Line 56: `2026-02-22T17:49:00.9638906Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 39: `2026-02-22T17:49:00.9522811Z 2026-02-22 17:49:00,952 - __main__ - ERROR - ❌ 1 test(s) failed`
    - Line 48: `2026-02-22T17:49:00.9530487Z ❌ Failed: 1`
    - Line 97: `2026-02-22T17:49:01.5981525Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 62: `2026-02-22T17:49:00.9713192Z   if-no-files-found: warn`
    - Line 97: `2026-02-22T17:49:01.5981525Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 3. Generate Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-22T18:41:02Z
**Completed**: 2026-02-22T18:41:08Z
**Duration**: 6 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140212/job/64454579451)

#### Failed Steps

- **Step 4**: Check Test Status

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T18:41:05.7831330Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 13
  - Sample matches:
    - Line 39: `2026-02-22T18:41:05.6064788Z [36;1mTOTAL_FAILED=0[0m`
    - Line 44: `2026-02-22T18:41:05.6068887Z [36;1m    FAILED=$(jq -r '.summary.failed' "$report")[0m`
    - Line 46: `2026-02-22T18:41:05.6070767Z [36;1m    TOTAL_FAILED=$((TOTAL_FAILED + FAILED))[0m`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 4: `2026-02-22T18:41:05.4330054Z (node:2106) [DEP0005] DeprecationWarning: Buffer() is deprecated due to`
    - Line 5: `2026-02-22T18:41:05.4334636Z (Use `node --trace-deprecation ...` to show where the warning was creat`

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

📊 *Report generated on 2026-02-22T19:52:59.412979*
🤖 *JARVIS CI/CD Auto-PR Manager*
