# CI/CD Failure Analysis Report

## Executive Summary

- **Workflow**: Complete Unlock Test Suite (Master)
- **Run Number**: #1
- **Branch**: `main`
- **Commit**: `f0315234349df624ef29f63fdcb8bb7da520a6e6`
- **Status**: ❌ FAILED
- **Timestamp**: 2026-02-22T17:46:42Z
- **Triggered By**: @nandkishorrathodk-art
- **Workflow URL**: [View Run](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140294)

## Failure Overview

Total Failed Jobs: **23**

| # | Job Name | Category | Severity | Duration |
|---|----------|----------|----------|----------|
| 1 | Run Unlock Integration E2E / Mock Tests - memory-security | test_failure | high | 19s |
| 2 | Run Unlock Integration E2E / Mock Tests - security-checks | test_failure | high | 13s |
| 3 | Run Biometric Voice E2E / Mock Biometric Tests - profile-quality-assessment | timeout | high | 24s |
| 4 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-noise | timeout | high | 21s |
| 5 | Run Biometric Voice E2E / Mock Biometric Tests - wake-word-detection | timeout | high | 24s |
| 6 | Run Biometric Voice E2E / Mock Biometric Tests - dimension-adaptation | timeout | high | 23s |
| 7 | Run Biometric Voice E2E / Mock Biometric Tests - replay-attack-detection | timeout | high | 23s |
| 8 | Run Biometric Voice E2E / Mock Biometric Tests - adaptive-thresholds | timeout | high | 25s |
| 9 | Run Biometric Voice E2E / Mock Biometric Tests - stt-transcription | timeout | high | 26s |
| 10 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-voice-drift | timeout | high | 25s |
| 11 | Run Biometric Voice E2E / Mock Biometric Tests - embedding-validation | timeout | high | 23s |
| 12 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-database-failure | test_failure | high | 20s |
| 13 | Run Biometric Voice E2E / Mock Biometric Tests - voice-verification | timeout | high | 23s |
| 14 | Run Biometric Voice E2E / Mock Biometric Tests - anti-spoofing | timeout | high | 25s |
| 15 | Run Biometric Voice E2E / Mock Biometric Tests - security-validation | timeout | high | 23s |
| 16 | Run Biometric Voice E2E / Mock Biometric Tests - voice-synthesis-detection | timeout | high | 20s |
| 17 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-cold-start | timeout | high | 21s |
| 18 | Run Biometric Voice E2E / Mock Biometric Tests - performance-baseline | timeout | high | 26s |
| 19 | Run Biometric Voice E2E / Mock Biometric Tests - end-to-end-flow | timeout | high | 20s |
| 20 | Run Unlock Integration E2E / Generate Test Summary | test_failure | high | 6s |
| 21 | Run Biometric Voice E2E / Generate Biometric Test Summary | test_failure | high | 4s |
| 22 | Generate Combined Test Summary | test_failure | high | 4s |
| 23 | Notify Test Status | test_failure | high | 3s |

## Detailed Analysis

### 1. Run Unlock Integration E2E / Mock Tests - memory-security

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-22T18:43:19Z
**Completed**: 2026-02-22T18:43:38Z
**Duration**: 19 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140294/job/64454696177)

#### Failed Steps

- **Step 6**: Run Mock Tests

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 5
  - Sample matches:
    - Line 36: `2026-02-22T18:43:35.5275641Z 2026-02-22 18:43:35,527 - __main__ - ERROR - ❌ Memory security test fai`
    - Line 37: `2026-02-22T18:43:35.5277085Z 2026-02-22 18:43:35,527 - __main__ - INFO - ❌ memory_security: Error: p`
    - Line 39: `2026-02-22T18:43:35.5292523Z 2026-02-22 18:43:35,529 - __main__ - ERROR - ❌ 1 test(s) failed`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 4
  - Sample matches:
    - Line 36: `2026-02-22T18:43:35.5275641Z 2026-02-22 18:43:35,527 - __main__ - ERROR - ❌ Memory security test fai`
    - Line 39: `2026-02-22T18:43:35.5292523Z 2026-02-22 18:43:35,529 - __main__ - ERROR - ❌ 1 test(s) failed`
    - Line 48: `2026-02-22T18:43:35.5300890Z ❌ Failed: 1`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 62: `2026-02-22T18:43:35.5489994Z   if-no-files-found: warn`
    - Line 97: `2026-02-22T18:43:36.5691050Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 2. Run Unlock Integration E2E / Mock Tests - security-checks

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-22T18:52:20Z
**Completed**: 2026-02-22T18:52:33Z
**Duration**: 13 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140294/job/64454696181)

#### Failed Steps

- **Step 6**: Run Mock Tests

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 2
  - Sample matches:
    - Line 39: `2026-02-22T18:52:31.1416287Z 2026-02-22 18:52:31,141 - __main__ - ERROR - ❌ 1 test(s) failed`
    - Line 56: `2026-02-22T18:52:31.1534559Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 39: `2026-02-22T18:52:31.1416287Z 2026-02-22 18:52:31,141 - __main__ - ERROR - ❌ 1 test(s) failed`
    - Line 48: `2026-02-22T18:52:31.1423975Z ❌ Failed: 1`
    - Line 97: `2026-02-22T18:52:31.9010171Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 62: `2026-02-22T18:52:31.1606240Z   if-no-files-found: warn`
    - Line 97: `2026-02-22T18:52:31.9010171Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 3. Run Biometric Voice E2E / Mock Biometric Tests - profile-quality-assessment

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T18:52:40Z
**Completed**: 2026-02-22T18:53:04Z
**Duration**: 24 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140294/job/64455481432)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T18:53:01.8621851Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T18:53:01.8631354Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T18:53:01.9026086Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T18:53:02.2802243Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T18:53:01.9137598Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T18:53:02.1282528Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T18:53:02.2802243Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T18:52:51.8491731Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T18:52:51.8505223Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T18:52:54.4387366Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T18:52:52.1670303Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T18:52:52.1682925Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T18:52:53.0558219Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 4. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-noise

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T18:52:55Z
**Completed**: 2026-02-22T18:53:16Z
**Duration**: 21 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140294/job/64455481433)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T18:53:14.5003170Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T18:53:14.5012235Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T18:53:14.5384770Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T18:53:14.9163757Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T18:53:14.5491386Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T18:53:14.7661913Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T18:53:14.9163757Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T18:53:05.6628378Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T18:53:05.6641648Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T18:53:06.8874743Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T18:53:05.8181112Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T18:53:05.8194191Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T18:53:06.6170727Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 5. Run Biometric Voice E2E / Mock Biometric Tests - wake-word-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T18:53:22Z
**Completed**: 2026-02-22T18:53:46Z
**Duration**: 24 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140294/job/64455481434)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T18:53:42.8502795Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T18:53:42.8511695Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T18:53:42.8881469Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T18:53:43.2657463Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T18:53:42.8987864Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T18:53:43.1153966Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T18:53:43.2657463Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T18:53:33.8897992Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T18:53:33.8911673Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T18:53:35.1793579Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T18:53:34.0446438Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T18:53:34.0466317Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T18:53:34.8533987Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 6. Run Biometric Voice E2E / Mock Biometric Tests - dimension-adaptation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T18:51:37Z
**Completed**: 2026-02-22T18:52:00Z
**Duration**: 23 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140294/job/64455481435)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T18:51:57.9114736Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T18:51:57.9124952Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T18:51:57.9578552Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T18:51:58.3260096Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T18:51:57.9683172Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T18:51:58.1800258Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T18:51:58.3260096Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T18:51:48.9983041Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T18:51:48.9998825Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T18:51:50.6087985Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T18:51:49.3310094Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T18:51:49.3325731Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T18:51:50.1567637Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 7. Run Biometric Voice E2E / Mock Biometric Tests - replay-attack-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T18:52:50Z
**Completed**: 2026-02-22T18:53:13Z
**Duration**: 23 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140294/job/64455481438)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T18:53:11.8600603Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T18:53:11.8609123Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T18:53:11.8981222Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T18:53:12.2723665Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T18:53:11.9091687Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T18:53:12.1232509Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T18:53:12.2723665Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T18:53:02.0705129Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T18:53:02.0718623Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T18:53:04.1201931Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T18:53:02.3553254Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T18:53:02.3566642Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T18:53:03.6227011Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 8. Run Biometric Voice E2E / Mock Biometric Tests - adaptive-thresholds

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T18:54:16Z
**Completed**: 2026-02-22T18:54:41Z
**Duration**: 25 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140294/job/64455481441)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T18:54:38.9145424Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T18:54:38.9154796Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T18:54:38.9552965Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T18:54:39.3482632Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T18:54:38.9666310Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T18:54:39.1930141Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T18:54:39.3482632Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T18:54:29.5242316Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T18:54:29.5256506Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T18:54:30.8812011Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T18:54:29.6790286Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T18:54:29.6804577Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T18:54:30.4998907Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 9. Run Biometric Voice E2E / Mock Biometric Tests - stt-transcription

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T18:54:45Z
**Completed**: 2026-02-22T18:55:11Z
**Duration**: 26 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140294/job/64455481444)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T18:55:08.9439566Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T18:55:08.9456235Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T18:55:09.0132116Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T18:55:09.4394306Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T18:55:09.0255376Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T18:55:09.2735466Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T18:55:09.4394306Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T18:54:58.3395211Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T18:54:58.3412096Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T18:55:00.1104148Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T18:54:58.6709947Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T18:54:58.6725018Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T18:54:59.6278469Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 10. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-voice-drift

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T18:54:43Z
**Completed**: 2026-02-22T18:55:08Z
**Duration**: 25 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140294/job/64455481446)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T18:55:05.6215265Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T18:55:05.6223892Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T18:55:05.6609456Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T18:55:06.0405936Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T18:55:05.6719793Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T18:55:05.8908535Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T18:55:06.0405936Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T18:54:55.9046531Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T18:54:55.9060305Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T18:54:57.7185093Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T18:54:56.1353730Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T18:54:56.1367323Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T18:54:56.9795629Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 11. Run Biometric Voice E2E / Mock Biometric Tests - embedding-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T18:56:25Z
**Completed**: 2026-02-22T18:56:48Z
**Duration**: 23 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140294/job/64455481448)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T18:56:46.1866801Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T18:56:46.1876437Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T18:56:46.2298489Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T18:56:46.5884786Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T18:56:46.2407224Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T18:56:46.4526714Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T18:56:46.5884786Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T18:56:37.2827869Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T18:56:37.2841815Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T18:56:39.0454668Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T18:56:37.5141925Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T18:56:37.5155466Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T18:56:38.2578003Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 12. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-database-failure

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-22T18:54:23Z
**Completed**: 2026-02-22T18:54:43Z
**Duration**: 20 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140294/job/64455481451)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T18:54:42.3618878Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T18:54:42.3627048Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T18:54:42.3966374Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 2
  - Sample matches:
    - Line 69: `2026-02-22T18:54:42.4075795Z   name: test-results-biometric-mock-edge-case-database-failure`
    - Line 97: `2026-02-22T18:54:42.7683537Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T18:54:42.4076630Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T18:54:42.6185781Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T18:54:42.7683537Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T18:54:33.4367692Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T18:54:33.4381366Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T18:54:35.1556351Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T18:54:33.6331566Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T18:54:33.6344794Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T18:54:34.4540315Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 13. Run Biometric Voice E2E / Mock Biometric Tests - voice-verification

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T18:56:01Z
**Completed**: 2026-02-22T18:56:24Z
**Duration**: 23 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140294/job/64455481453)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T18:56:22.4370167Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T18:56:22.4380181Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T18:56:22.4783412Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T18:56:22.8612196Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T18:56:22.4888822Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T18:56:22.7109515Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T18:56:22.8612196Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T18:56:13.5295132Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T18:56:13.5308008Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T18:56:14.7821335Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T18:56:13.7006895Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T18:56:13.7020757Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T18:56:14.4934064Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 14. Run Biometric Voice E2E / Mock Biometric Tests - anti-spoofing

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T18:57:45Z
**Completed**: 2026-02-22T18:58:10Z
**Duration**: 25 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140294/job/64455481454)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T18:58:08.0496425Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T18:58:08.0505470Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T18:58:08.0904165Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T18:58:08.4730779Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T18:58:08.1012349Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T18:58:08.3214284Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T18:58:08.4730779Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T18:57:58.2184651Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T18:57:58.2198595Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T18:58:00.4204596Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T18:57:58.5156470Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T18:57:58.5169923Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T18:57:59.4239540Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 15. Run Biometric Voice E2E / Mock Biometric Tests - security-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T18:57:16Z
**Completed**: 2026-02-22T18:57:39Z
**Duration**: 23 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140294/job/64455481461)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T18:57:37.3457762Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T18:57:37.3467302Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T18:57:37.3852041Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T18:57:37.7320511Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T18:57:37.3955311Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T18:57:37.5993364Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T18:57:37.7320511Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T18:57:28.9766620Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T18:57:28.9779600Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T18:57:30.4674318Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T18:57:29.2860944Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T18:57:29.2874131Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T18:57:30.0711736Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 16. Run Biometric Voice E2E / Mock Biometric Tests - voice-synthesis-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T18:58:11Z
**Completed**: 2026-02-22T18:58:31Z
**Duration**: 20 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140294/job/64455481462)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T18:58:29.4201157Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T18:58:29.4210463Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T18:58:29.4579998Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T18:58:29.8300058Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T18:58:29.4684886Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T18:58:29.6838604Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T18:58:29.8300058Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T18:58:20.6508577Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T18:58:20.6522864Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T18:58:22.0319369Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T18:58:20.8184683Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T18:58:20.8197775Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T18:58:21.6109430Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 17. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-cold-start

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T18:55:14Z
**Completed**: 2026-02-22T18:55:35Z
**Duration**: 21 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140294/job/64455481463)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T18:55:33.3965476Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T18:55:33.3974412Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T18:55:33.4369341Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T18:55:33.8179093Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T18:55:33.4475097Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T18:55:33.6656158Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T18:55:33.8179093Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T18:55:23.6884853Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T18:55:23.6898675Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T18:55:25.6913587Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T18:55:23.8486296Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T18:55:23.8500530Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T18:55:24.6623994Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 18. Run Biometric Voice E2E / Mock Biometric Tests - performance-baseline

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T18:55:57Z
**Completed**: 2026-02-22T18:56:23Z
**Duration**: 26 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140294/job/64455481464)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T18:56:20.1626558Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T18:56:20.1637815Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T18:56:20.2180137Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T18:56:20.6375484Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T18:56:20.2295093Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T18:56:20.4654616Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T18:56:20.6375484Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T18:56:10.5630604Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T18:56:10.5646089Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T18:56:11.9531210Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T18:56:10.7243988Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T18:56:10.7258792Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T18:56:11.5666982Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 19. Run Biometric Voice E2E / Mock Biometric Tests - end-to-end-flow

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T18:57:06Z
**Completed**: 2026-02-22T18:57:26Z
**Duration**: 20 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140294/job/64455481472)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T18:57:25.7141897Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T18:57:25.7153155Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T18:57:25.7530695Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T18:57:26.1374209Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T18:57:25.7636623Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T18:57:25.9875229Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T18:57:26.1374209Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T18:57:16.6661585Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T18:57:16.6675293Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T18:57:18.4360888Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T18:57:16.8648166Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T18:57:16.8661732Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T18:57:17.6824330Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 20. Run Unlock Integration E2E / Generate Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-22T19:47:14Z
**Completed**: 2026-02-22T19:47:20Z
**Duration**: 6 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140294/job/64457056790)

#### Failed Steps

- **Step 4**: Check Test Status

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T19:47:18.1904496Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 13
  - Sample matches:
    - Line 39: `2026-02-22T19:47:17.9180991Z [36;1mTOTAL_FAILED=0[0m`
    - Line 44: `2026-02-22T19:47:17.9183924Z [36;1m    FAILED=$(jq -r '.summary.failed' "$report")[0m`
    - Line 46: `2026-02-22T19:47:17.9185082Z [36;1m    TOTAL_FAILED=$((TOTAL_FAILED + FAILED))[0m`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 4: `2026-02-22T19:47:17.7228680Z (node:2114) [DEP0005] DeprecationWarning: Buffer() is deprecated due to`
    - Line 5: `2026-02-22T19:47:17.7232167Z (Use `node --trace-deprecation ...` to show where the warning was creat`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 21. Run Biometric Voice E2E / Generate Biometric Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-22T19:48:00Z
**Completed**: 2026-02-22T19:48:04Z
**Duration**: 4 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140294/job/64457266998)

#### Failed Steps

- **Step 4**: Check Test Status

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 2
  - Sample matches:
    - Line 15: `2026-02-22T19:48:02.8118642Z Starting download of artifact to: /home/runner/work/Ironcliw-ai/Ironcli`
    - Line 97: `2026-02-22T19:48:02.9900213Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 11
  - Sample matches:
    - Line 32: `2026-02-22T19:48:02.8410495Z [36;1mTOTAL_FAILED=0[0m`
    - Line 37: `2026-02-22T19:48:02.8413878Z [36;1m    FAILED=$(jq -r '.summary.failed' "$report")[0m`
    - Line 39: `2026-02-22T19:48:02.8415402Z [36;1m    TOTAL_FAILED=$((TOTAL_FAILED + FAILED))[0m`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 22. Generate Combined Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-22T19:50:18Z
**Completed**: 2026-02-22T19:50:22Z
**Duration**: 4 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140294/job/64459260085)

#### Failed Steps

- **Step 2**: Generate Combined Summary

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 61: `2026-02-22T19:50:20.4036979Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 9
  - Sample matches:
    - Line 25: `2026-02-22T19:50:20.3842695Z [36;1mif [ "failure" = "success" ]; then[0m`
    - Line 28: `2026-02-22T19:50:20.3844600Z [36;1m  echo "- ❌ **Unlock Integration E2E:** failure" >> $GITHUB_STEP`
    - Line 32: `2026-02-22T19:50:20.3846569Z [36;1mif [ "failure" = "success" ]; then[0m`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 23. Notify Test Status

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-22T19:51:42Z
**Completed**: 2026-02-22T19:51:45Z
**Duration**: 3 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140294/job/64459349992)

#### Failed Steps

- **Step 3**: Failure Notification

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line -23: `2026-02-22T19:51:43.5656863Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line -31: `2026-02-22T19:51:43.4715340Z ##[group]Run echo "❌ Unlock tests failed - 'unlock my screen' may be br`
    - Line -30: `2026-02-22T19:51:43.4716569Z [36;1mecho "❌ Unlock tests failed - 'unlock my screen' may be broken!"`
    - Line -25: `2026-02-22T19:51:43.5620134Z ❌ Unlock tests failed - 'unlock my screen' may be broken!`

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

📊 *Report generated on 2026-02-22T19:54:36.820534*
🤖 *JARVIS CI/CD Auto-PR Manager*
