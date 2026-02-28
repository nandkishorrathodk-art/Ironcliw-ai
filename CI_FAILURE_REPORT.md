# CI/CD Failure Analysis Report

## Executive Summary

- **Workflow**: Complete Unlock Test Suite (Master)
- **Run Number**: #18
- **Branch**: `main`
- **Commit**: `41b1fb38c756c5ecab9f500d4f623f76a57269d1`
- **Status**: ❌ FAILED
- **Timestamp**: 2026-02-28T17:41:17Z
- **Triggered By**: @nandkishorrathodk-art
- **Workflow URL**: [View Run](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682753)

## Failure Overview

Total Failed Jobs: **22**

| # | Job Name | Category | Severity | Duration |
|---|----------|----------|----------|----------|
| 1 | Run Unlock Integration E2E / Mock Tests - security-checks | test_failure | high | 45s |
| 2 | Run Biometric Voice E2E / Mock Biometric Tests - wake-word-detection | timeout | high | 40s |
| 3 | Run Biometric Voice E2E / Mock Biometric Tests - stt-transcription | timeout | high | 45s |
| 4 | Run Biometric Voice E2E / Mock Biometric Tests - adaptive-thresholds | timeout | high | 46s |
| 5 | Run Biometric Voice E2E / Mock Biometric Tests - end-to-end-flow | timeout | high | 40s |
| 6 | Run Biometric Voice E2E / Mock Biometric Tests - voice-verification | timeout | high | 42s |
| 7 | Run Biometric Voice E2E / Mock Biometric Tests - anti-spoofing | timeout | high | 55s |
| 8 | Run Biometric Voice E2E / Mock Biometric Tests - dimension-adaptation | timeout | high | 47s |
| 9 | Run Biometric Voice E2E / Mock Biometric Tests - performance-baseline | timeout | high | 41s |
| 10 | Run Biometric Voice E2E / Mock Biometric Tests - profile-quality-assessment | timeout | high | 44s |
| 11 | Run Biometric Voice E2E / Mock Biometric Tests - security-validation | timeout | high | 53s |
| 12 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-database-failure | test_failure | high | 50s |
| 13 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-noise | timeout | high | 50s |
| 14 | Run Biometric Voice E2E / Mock Biometric Tests - voice-synthesis-detection | timeout | high | 42s |
| 15 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-cold-start | timeout | high | 45s |
| 16 | Run Biometric Voice E2E / Mock Biometric Tests - embedding-validation | timeout | high | 45s |
| 17 | Run Biometric Voice E2E / Mock Biometric Tests - replay-attack-detection | timeout | high | 45s |
| 18 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-voice-drift | timeout | high | 43s |
| 19 | Run Unlock Integration E2E / Generate Test Summary | test_failure | high | 3s |
| 20 | Run Biometric Voice E2E / Generate Biometric Test Summary | test_failure | high | 3s |
| 21 | Generate Combined Test Summary | test_failure | high | 4s |
| 22 | Notify Test Status | test_failure | high | 3s |

## Detailed Analysis

### 1. Run Unlock Integration E2E / Mock Tests - security-checks

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-28T17:42:18Z
**Completed**: 2026-02-28T17:43:03Z
**Duration**: 45 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682753/job/65257246085)

#### Failed Steps

- **Step 6**: Run Mock Tests

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 2
  - Sample matches:
    - Line 39: `2026-02-28T17:43:00.1946176Z 2026-02-28 17:43:00,194 - __main__ - ERROR - ❌ 1 test(s) failed`
    - Line 56: `2026-02-28T17:43:00.2061953Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 39: `2026-02-28T17:43:00.1946176Z 2026-02-28 17:43:00,194 - __main__ - ERROR - ❌ 1 test(s) failed`
    - Line 48: `2026-02-28T17:43:00.1954496Z ❌ Failed: 1`
    - Line 97: `2026-02-28T17:43:01.2576179Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 62: `2026-02-28T17:43:00.2137322Z   if-no-files-found: warn`
    - Line 97: `2026-02-28T17:43:01.2576179Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 2. Run Biometric Voice E2E / Mock Biometric Tests - wake-word-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-28T17:43:46Z
**Completed**: 2026-02-28T17:44:26Z
**Duration**: 40 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682753/job/65257246211)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-28T17:44:23.8822190Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-28T17:44:23.8831279Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-28T17:44:23.9206958Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T17:44:24.3091210Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T17:44:23.9315811Z   if-no-files-found: warn`
    - Line 87: `2026-02-28T17:44:24.1544131Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-28T17:44:24.3091210Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-28T17:44:14.4752285Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-28T17:44:14.4765605Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-28T17:44:16.2117364Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-28T17:44:14.6712922Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-28T17:44:14.6726382Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-28T17:44:15.5029210Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 3. Run Biometric Voice E2E / Mock Biometric Tests - stt-transcription

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-28T17:43:58Z
**Completed**: 2026-02-28T17:44:43Z
**Duration**: 45 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682753/job/65257246213)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-28T17:44:41.0861685Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-28T17:44:41.0870942Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-28T17:44:41.1246505Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T17:44:41.5067733Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T17:44:41.1354960Z   if-no-files-found: warn`
    - Line 87: `2026-02-28T17:44:41.3515261Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-28T17:44:41.5067733Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-28T17:44:30.4674523Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-28T17:44:30.4688368Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-28T17:44:33.1114766Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-28T17:44:30.7521891Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-28T17:44:30.7536829Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-28T17:44:31.6344241Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 4. Run Biometric Voice E2E / Mock Biometric Tests - adaptive-thresholds

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-28T17:43:46Z
**Completed**: 2026-02-28T17:44:32Z
**Duration**: 46 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682753/job/65257246214)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-28T17:44:30.2459638Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-28T17:44:30.2469955Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-28T17:44:30.2846097Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T17:44:30.6652623Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T17:44:30.2955408Z   if-no-files-found: warn`
    - Line 87: `2026-02-28T17:44:30.5103649Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-28T17:44:30.6652623Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-28T17:44:20.8669368Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-28T17:44:20.8684215Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-28T17:44:22.1013736Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-28T17:44:21.0323870Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-28T17:44:21.0338761Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-28T17:44:21.8707102Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 5. Run Biometric Voice E2E / Mock Biometric Tests - end-to-end-flow

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-28T17:43:55Z
**Completed**: 2026-02-28T17:44:35Z
**Duration**: 40 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682753/job/65257246216)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-28T17:44:33.0778976Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-28T17:44:33.0787549Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-28T17:44:33.1137054Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T17:44:33.5006604Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T17:44:33.1249544Z   if-no-files-found: warn`
    - Line 87: `2026-02-28T17:44:33.3469620Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-28T17:44:33.5006604Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-28T17:44:23.3875575Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-28T17:44:23.3888680Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-28T17:44:24.9739898Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-28T17:44:23.5768660Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-28T17:44:23.5781269Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-28T17:44:24.4273182Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 6. Run Biometric Voice E2E / Mock Biometric Tests - voice-verification

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-28T17:44:14Z
**Completed**: 2026-02-28T17:44:56Z
**Duration**: 42 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682753/job/65257246217)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-28T17:44:54.8260923Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-28T17:44:54.8270554Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-28T17:44:54.8642175Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T17:44:55.2463991Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T17:44:54.8751005Z   if-no-files-found: warn`
    - Line 87: `2026-02-28T17:44:55.0919744Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-28T17:44:55.2463991Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-28T17:44:44.1537103Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-28T17:44:44.1550401Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-28T17:44:46.5527129Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-28T17:44:44.3155604Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-28T17:44:44.3170155Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-28T17:44:45.1307673Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 7. Run Biometric Voice E2E / Mock Biometric Tests - anti-spoofing

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-28T17:43:58Z
**Completed**: 2026-02-28T17:44:53Z
**Duration**: 55 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682753/job/65257246218)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-28T17:44:50.3487841Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-28T17:44:50.3497023Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-28T17:44:50.3853401Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T17:44:50.7647454Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T17:44:50.3960966Z   if-no-files-found: warn`
    - Line 87: `2026-02-28T17:44:50.6111508Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-28T17:44:50.7647454Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-28T17:44:41.6250281Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-28T17:44:41.6264669Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-28T17:44:42.8508949Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-28T17:44:41.7780829Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-28T17:44:41.7796188Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-28T17:44:42.5753449Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 8. Run Biometric Voice E2E / Mock Biometric Tests - dimension-adaptation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-28T17:44:06Z
**Completed**: 2026-02-28T17:44:53Z
**Duration**: 47 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682753/job/65257246219)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-28T17:44:51.9915111Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-28T17:44:51.9924322Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-28T17:44:52.0378372Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T17:44:52.4363250Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T17:44:52.0491719Z   if-no-files-found: warn`
    - Line 87: `2026-02-28T17:44:52.2725161Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-28T17:44:52.4363250Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-28T17:44:41.0950483Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-28T17:44:41.0966424Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-28T17:44:43.4611367Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-28T17:44:41.3853809Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-28T17:44:41.3870137Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-28T17:44:42.3019869Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 9. Run Biometric Voice E2E / Mock Biometric Tests - performance-baseline

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-28T17:43:56Z
**Completed**: 2026-02-28T17:44:37Z
**Duration**: 41 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682753/job/65257246220)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-28T17:44:35.2872515Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-28T17:44:35.2881516Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-28T17:44:35.3267377Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T17:44:35.7109943Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T17:44:35.3377356Z   if-no-files-found: warn`
    - Line 87: `2026-02-28T17:44:35.5558180Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-28T17:44:35.7109943Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-28T17:44:25.8603781Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-28T17:44:25.8617418Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-28T17:44:27.5110990Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-28T17:44:26.0555594Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-28T17:44:26.0569278Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-28T17:44:26.8883068Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 10. Run Biometric Voice E2E / Mock Biometric Tests - profile-quality-assessment

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-28T17:44:02Z
**Completed**: 2026-02-28T17:44:46Z
**Duration**: 44 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682753/job/65257246221)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-28T17:44:44.7680809Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-28T17:44:44.7691155Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-28T17:44:44.8128681Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T17:44:45.2036797Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T17:44:44.8237408Z   if-no-files-found: warn`
    - Line 87: `2026-02-28T17:44:45.0429450Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-28T17:44:45.2036797Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-28T17:44:35.3481581Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-28T17:44:35.3497614Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-28T17:44:36.6036056Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-28T17:44:35.5175678Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-28T17:44:35.5190672Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-28T17:44:36.3643031Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 11. Run Biometric Voice E2E / Mock Biometric Tests - security-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-28T17:44:04Z
**Completed**: 2026-02-28T17:44:57Z
**Duration**: 53 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682753/job/65257246223)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-28T17:44:55.5093664Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-28T17:44:55.5103385Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-28T17:44:55.5467832Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T17:44:55.9455178Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T17:44:55.5574877Z   if-no-files-found: warn`
    - Line 87: `2026-02-28T17:44:55.7893212Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-28T17:44:55.9455178Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-28T17:44:44.7293461Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-28T17:44:44.7307800Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-28T17:44:47.3749535Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-28T17:44:45.0568824Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-28T17:44:45.0583782Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-28T17:44:45.9914621Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 12. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-database-failure

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-28T17:44:13Z
**Completed**: 2026-02-28T17:45:03Z
**Duration**: 50 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682753/job/65257246224)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-28T17:45:00.2627091Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-28T17:45:00.2637197Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-28T17:45:00.3060366Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 2
  - Sample matches:
    - Line 69: `2026-02-28T17:45:00.3161034Z   name: test-results-biometric-mock-edge-case-database-failure`
    - Line 97: `2026-02-28T17:45:00.6654391Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T17:45:00.3161883Z   if-no-files-found: warn`
    - Line 87: `2026-02-28T17:45:00.5251272Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-28T17:45:00.6654391Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-28T17:44:51.2929932Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-28T17:44:51.2944682Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-28T17:44:52.9970443Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-28T17:44:51.5150416Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-28T17:44:51.5166187Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-28T17:44:52.2661724Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 13. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-noise

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-28T17:44:12Z
**Completed**: 2026-02-28T17:45:02Z
**Duration**: 50 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682753/job/65257246225)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-28T17:45:00.1644196Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-28T17:45:00.1652644Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-28T17:45:00.2005339Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T17:45:00.5820486Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T17:45:00.2118767Z   if-no-files-found: warn`
    - Line 87: `2026-02-28T17:45:00.4296664Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-28T17:45:00.5820486Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-28T17:44:51.2615389Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-28T17:44:51.2628373Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-28T17:44:52.4601074Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-28T17:44:51.4023174Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-28T17:44:51.4036624Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-28T17:44:52.1979423Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 14. Run Biometric Voice E2E / Mock Biometric Tests - voice-synthesis-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-28T17:44:22Z
**Completed**: 2026-02-28T17:45:04Z
**Duration**: 42 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682753/job/65257246226)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-28T17:45:02.0182288Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-28T17:45:02.0192761Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-28T17:45:02.0599534Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T17:45:02.4479880Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T17:45:02.0711428Z   if-no-files-found: warn`
    - Line 87: `2026-02-28T17:45:02.2905131Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-28T17:45:02.4479880Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-28T17:44:52.7831489Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-28T17:44:52.7844542Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-28T17:44:54.0218491Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-28T17:44:52.9436199Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-28T17:44:52.9450110Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-28T17:44:53.7719674Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 15. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-cold-start

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-28T17:44:30Z
**Completed**: 2026-02-28T17:45:15Z
**Duration**: 45 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682753/job/65257246227)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-28T17:45:12.9539677Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-28T17:45:12.9549149Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-28T17:45:12.9919194Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T17:45:13.3729942Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T17:45:13.0027937Z   if-no-files-found: warn`
    - Line 87: `2026-02-28T17:45:13.2184929Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-28T17:45:13.3729942Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-28T17:45:02.7143393Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-28T17:45:02.7157823Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-28T17:45:05.1027791Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-28T17:45:03.0071580Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-28T17:45:03.0086590Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-28T17:45:03.9119810Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 16. Run Biometric Voice E2E / Mock Biometric Tests - embedding-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-28T17:44:27Z
**Completed**: 2026-02-28T17:45:12Z
**Duration**: 45 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682753/job/65257246229)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-28T17:45:10.0239449Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-28T17:45:10.0248310Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-28T17:45:10.0620657Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T17:45:10.4456485Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T17:45:10.0730313Z   if-no-files-found: warn`
    - Line 87: `2026-02-28T17:45:10.2904783Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-28T17:45:10.4456485Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-28T17:45:00.8842176Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-28T17:45:00.8862697Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-28T17:45:02.1877712Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-28T17:45:01.0547800Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-28T17:45:01.0562198Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-28T17:45:01.9030622Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 17. Run Biometric Voice E2E / Mock Biometric Tests - replay-attack-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-28T17:44:21Z
**Completed**: 2026-02-28T17:45:06Z
**Duration**: 45 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682753/job/65257246233)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-28T17:45:04.5694735Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-28T17:45:04.5704055Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-28T17:45:04.6078849Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T17:45:04.9941808Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T17:45:04.6184252Z   if-no-files-found: warn`
    - Line 87: `2026-02-28T17:45:04.8366387Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-28T17:45:04.9941808Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-28T17:44:53.8712555Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-28T17:44:53.8726729Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-28T17:44:56.8959507Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-28T17:44:54.1642968Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-28T17:44:54.1656211Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-28T17:44:55.0637434Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 18. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-voice-drift

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-28T17:44:38Z
**Completed**: 2026-02-28T17:45:21Z
**Duration**: 43 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682753/job/65257246237)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-28T17:45:18.7073352Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-28T17:45:18.7082708Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-28T17:45:18.7505097Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T17:45:19.1334112Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T17:45:18.7611380Z   if-no-files-found: warn`
    - Line 87: `2026-02-28T17:45:18.9777432Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-28T17:45:19.1334112Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-28T17:45:09.1392870Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-28T17:45:09.1407462Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-28T17:45:11.1362366Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-28T17:45:09.2999013Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-28T17:45:09.3012950Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-28T17:45:10.1308864Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 19. Run Unlock Integration E2E / Generate Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-28T17:44:09Z
**Completed**: 2026-02-28T17:44:12Z
**Duration**: 3 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682753/job/65257338723)

#### Failed Steps

- **Step 4**: Check Test Status

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T17:44:11.8867269Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 13
  - Sample matches:
    - Line 39: `2026-02-28T17:44:11.7027795Z [36;1mTOTAL_FAILED=0[0m`
    - Line 44: `2026-02-28T17:44:11.7031310Z [36;1m    FAILED=$(jq -r '.summary.failed' "$report")[0m`
    - Line 46: `2026-02-28T17:44:11.7033039Z [36;1m    TOTAL_FAILED=$((TOTAL_FAILED + FAILED))[0m`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 20. Run Biometric Voice E2E / Generate Biometric Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-28T17:45:23Z
**Completed**: 2026-02-28T17:45:26Z
**Duration**: 3 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682753/job/65257390886)

#### Failed Steps

- **Step 4**: Check Test Status

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T17:45:25.2842103Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 11
  - Sample matches:
    - Line 32: `2026-02-28T17:45:25.0920318Z [36;1mTOTAL_FAILED=0[0m`
    - Line 37: `2026-02-28T17:45:25.0923761Z [36;1m    FAILED=$(jq -r '.summary.failed' "$report")[0m`
    - Line 39: `2026-02-28T17:45:25.0925321Z [36;1m    TOTAL_FAILED=$((TOTAL_FAILED + FAILED))[0m`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 21. Generate Combined Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-28T17:45:29Z
**Completed**: 2026-02-28T17:45:33Z
**Duration**: 4 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682753/job/65257394268)

#### Failed Steps

- **Step 2**: Generate Combined Summary

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 61: `2026-02-28T17:45:31.0892239Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 9
  - Sample matches:
    - Line 25: `2026-02-28T17:45:31.0654173Z [36;1mif [ "failure" = "success" ]; then[0m`
    - Line 28: `2026-02-28T17:45:31.0656319Z [36;1m  echo "- ❌ **Unlock Integration E2E:** failure" >> $GITHUB_STEP`
    - Line 32: `2026-02-28T17:45:31.0658250Z [36;1mif [ "failure" = "success" ]; then[0m`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 22. Notify Test Status

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-28T17:45:35Z
**Completed**: 2026-02-28T17:45:38Z
**Duration**: 3 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682753/job/65257397830)

#### Failed Steps

- **Step 3**: Failure Notification

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line -23: `2026-02-28T17:45:36.6409849Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line -31: `2026-02-28T17:45:36.5288236Z ##[group]Run echo "❌ Unlock tests failed - 'unlock my screen' may be br`
    - Line -30: `2026-02-28T17:45:36.5289704Z [36;1mecho "❌ Unlock tests failed - 'unlock my screen' may be broken!"`
    - Line -25: `2026-02-28T17:45:36.6383950Z ❌ Unlock tests failed - 'unlock my screen' may be broken!`

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

📊 *Report generated on 2026-02-28T17:46:39.645859*
🤖 *Ironcliw CI/CD Auto-PR Manager*
