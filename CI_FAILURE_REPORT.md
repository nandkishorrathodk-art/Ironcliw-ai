# CI/CD Failure Analysis Report

## Executive Summary

- **Workflow**: Complete Unlock Test Suite (Master)
- **Run Number**: #7
- **Branch**: `main`
- **Commit**: `3ce7237a675833e142cfadbb33c39828ea904d68`
- **Status**: ❌ FAILED
- **Timestamp**: 2026-02-24T15:51:14Z
- **Triggered By**: @nandkishorrathodk-art
- **Workflow URL**: [View Run](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22358545081)

## Failure Overview

Total Failed Jobs: **22**

| # | Job Name | Category | Severity | Duration |
|---|----------|----------|----------|----------|
| 1 | Run Unlock Integration E2E / Mock Tests - security-checks | test_failure | high | 44s |
| 2 | Run Biometric Voice E2E / Mock Biometric Tests - embedding-validation | timeout | high | 44s |
| 3 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-cold-start | timeout | high | 42s |
| 4 | Run Biometric Voice E2E / Mock Biometric Tests - voice-verification | timeout | high | 52s |
| 5 | Run Biometric Voice E2E / Mock Biometric Tests - end-to-end-flow | timeout | high | 51s |
| 6 | Run Biometric Voice E2E / Mock Biometric Tests - security-validation | timeout | high | 52s |
| 7 | Run Biometric Voice E2E / Mock Biometric Tests - performance-baseline | timeout | high | 49s |
| 8 | Run Biometric Voice E2E / Mock Biometric Tests - adaptive-thresholds | timeout | high | 47s |
| 9 | Run Biometric Voice E2E / Mock Biometric Tests - anti-spoofing | timeout | high | 55s |
| 10 | Run Biometric Voice E2E / Mock Biometric Tests - replay-attack-detection | timeout | high | 65s |
| 11 | Run Biometric Voice E2E / Mock Biometric Tests - stt-transcription | timeout | high | 56s |
| 12 | Run Biometric Voice E2E / Mock Biometric Tests - wake-word-detection | timeout | high | 56s |
| 13 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-database-failure | test_failure | high | 54s |
| 14 | Run Biometric Voice E2E / Mock Biometric Tests - dimension-adaptation | timeout | high | 45s |
| 15 | Run Biometric Voice E2E / Mock Biometric Tests - profile-quality-assessment | timeout | high | 43s |
| 16 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-noise | timeout | high | 63s |
| 17 | Run Biometric Voice E2E / Mock Biometric Tests - voice-synthesis-detection | timeout | high | 47s |
| 18 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-voice-drift | timeout | high | 49s |
| 19 | Run Unlock Integration E2E / Generate Test Summary | test_failure | high | 8s |
| 20 | Run Biometric Voice E2E / Generate Biometric Test Summary | test_failure | high | 8s |
| 21 | Generate Combined Test Summary | test_failure | high | 2s |
| 22 | Notify Test Status | test_failure | high | 2s |

## Detailed Analysis

### 1. Run Unlock Integration E2E / Mock Tests - security-checks

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-24T15:54:38Z
**Completed**: 2026-02-24T15:55:22Z
**Duration**: 44 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22358545081/job/64705071469)

#### Failed Steps

- **Step 6**: Run Mock Tests

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 2
  - Sample matches:
    - Line 39: `2026-02-24T15:55:20.4857231Z 2026-02-24 15:55:20,485 - __main__ - ERROR - ❌ 1 test(s) failed`
    - Line 56: `2026-02-24T15:55:20.4977670Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 39: `2026-02-24T15:55:20.4857231Z 2026-02-24 15:55:20,485 - __main__ - ERROR - ❌ 1 test(s) failed`
    - Line 48: `2026-02-24T15:55:20.4865814Z ❌ Failed: 1`
    - Line 97: `2026-02-24T15:55:21.1684483Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 62: `2026-02-24T15:55:20.5051988Z   if-no-files-found: warn`
    - Line 97: `2026-02-24T15:55:21.1684483Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 2. Run Biometric Voice E2E / Mock Biometric Tests - embedding-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T15:55:24Z
**Completed**: 2026-02-24T15:56:08Z
**Duration**: 44 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22358545081/job/64705416577)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T15:56:07.0652874Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T15:56:07.0662941Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T15:56:07.1060557Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T15:56:07.4888512Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T15:56:07.1168208Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T15:56:07.3357169Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T15:56:07.4888512Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T15:55:57.7323334Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T15:55:57.7336720Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T15:55:59.1614297Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T15:55:57.8860563Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T15:55:57.8873340Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T15:55:58.9322552Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 3. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-cold-start

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T15:54:10Z
**Completed**: 2026-02-24T15:54:52Z
**Duration**: 42 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22358545081/job/64705416579)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T15:54:51.5887261Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T15:54:51.5895416Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T15:54:51.6238206Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T15:54:52.0031722Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T15:54:51.6341620Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T15:54:51.8550695Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T15:54:52.0031722Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T15:54:42.4164152Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T15:54:42.4177537Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T15:54:43.9531310Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T15:54:42.6049621Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T15:54:42.6062233Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T15:54:43.4079284Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 4. Run Biometric Voice E2E / Mock Biometric Tests - voice-verification

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T15:53:51Z
**Completed**: 2026-02-24T15:54:43Z
**Duration**: 52 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22358545081/job/64705416586)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T15:54:41.1702387Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T15:54:41.1711607Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T15:54:41.2098314Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T15:54:41.5950244Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T15:54:41.2204370Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T15:54:41.4396484Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T15:54:41.5950244Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T15:54:30.3090647Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T15:54:30.3104677Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T15:54:33.0216741Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T15:54:30.4720906Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T15:54:30.4734159Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T15:54:31.6518870Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 5. Run Biometric Voice E2E / Mock Biometric Tests - end-to-end-flow

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T15:55:16Z
**Completed**: 2026-02-24T15:56:07Z
**Duration**: 51 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22358545081/job/64705416592)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T15:56:05.1087361Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T15:56:05.1096027Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T15:56:05.1441923Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T15:56:05.5129614Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T15:56:05.1547105Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T15:56:05.3640547Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T15:56:05.5129614Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T15:55:56.4041536Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T15:55:56.4054690Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T15:55:57.8319037Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T15:55:56.6577652Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T15:55:56.6590934Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T15:55:57.5178017Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 6. Run Biometric Voice E2E / Mock Biometric Tests - security-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T15:54:24Z
**Completed**: 2026-02-24T15:55:16Z
**Duration**: 52 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22358545081/job/64705416598)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T15:55:14.4770091Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T15:55:14.4778959Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T15:55:14.5142651Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T15:55:14.8894090Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T15:55:14.5259355Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T15:55:14.7405542Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T15:55:14.8894090Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T15:55:04.6234992Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T15:55:04.6248404Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T15:55:07.0457655Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T15:55:04.9002313Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T15:55:04.9016250Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T15:55:05.7675872Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 7. Run Biometric Voice E2E / Mock Biometric Tests - performance-baseline

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T15:54:46Z
**Completed**: 2026-02-24T15:55:35Z
**Duration**: 49 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22358545081/job/64705416602)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T15:55:32.5140915Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T15:55:32.5150919Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T15:55:32.5696025Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T15:55:32.9756424Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T15:55:32.5812057Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T15:55:32.8183719Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T15:55:32.9756424Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T15:55:22.5831435Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T15:55:22.5845019Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T15:55:24.4175980Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T15:55:22.8663623Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T15:55:22.8679544Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T15:55:23.7813457Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 8. Run Biometric Voice E2E / Mock Biometric Tests - adaptive-thresholds

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T15:54:14Z
**Completed**: 2026-02-24T15:55:01Z
**Duration**: 47 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22358545081/job/64705416605)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T15:54:59.3903756Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T15:54:59.3913654Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T15:54:59.4288290Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T15:54:59.8018124Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T15:54:59.4395264Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T15:54:59.6532018Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T15:54:59.8018124Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T15:54:50.5876907Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T15:54:50.5890590Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T15:54:52.0285291Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T15:54:50.7435575Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T15:54:50.7448400Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T15:54:51.5446016Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 9. Run Biometric Voice E2E / Mock Biometric Tests - anti-spoofing

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T15:54:33Z
**Completed**: 2026-02-24T15:55:28Z
**Duration**: 55 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22358545081/job/64705416612)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T15:55:25.0072286Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T15:55:25.0081460Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T15:55:25.0471040Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T15:55:25.4195863Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T15:55:25.0579680Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T15:55:25.2706719Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T15:55:25.4195863Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T15:55:16.1920016Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T15:55:16.1933852Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T15:55:17.3738535Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T15:55:16.3382476Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T15:55:16.3395228Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T15:55:17.1293234Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 10. Run Biometric Voice E2E / Mock Biometric Tests - replay-attack-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T15:54:01Z
**Completed**: 2026-02-24T15:55:06Z
**Duration**: 65 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22358545081/job/64705416617)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T15:55:03.2540708Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T15:55:03.2549229Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T15:55:03.3010516Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T15:55:03.6968611Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T15:55:03.3116046Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T15:55:03.5413191Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T15:55:03.6968611Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T15:54:53.9792335Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T15:54:53.9807091Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T15:54:55.3001478Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T15:54:54.1246730Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T15:54:54.1260238Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T15:54:54.9238380Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 11. Run Biometric Voice E2E / Mock Biometric Tests - stt-transcription

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T15:55:25Z
**Completed**: 2026-02-24T15:56:21Z
**Duration**: 56 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22358545081/job/64705416618)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T15:56:19.8742666Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T15:56:19.8751379Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T15:56:19.9121942Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T15:56:20.2925892Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T15:56:19.9228486Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T15:56:20.1373487Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T15:56:20.2925892Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T15:56:10.7048043Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T15:56:10.7062204Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T15:56:11.9281682Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T15:56:10.8598495Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T15:56:10.8612727Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T15:56:11.6789432Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 12. Run Biometric Voice E2E / Mock Biometric Tests - wake-word-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T15:55:26Z
**Completed**: 2026-02-24T15:56:22Z
**Duration**: 56 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22358545081/job/64705416634)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T15:56:19.9896407Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T15:56:19.9906584Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T15:56:20.0286540Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T15:56:20.4052029Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T15:56:20.0412225Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T15:56:20.2540554Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T15:56:20.4052029Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T15:56:10.7557635Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T15:56:10.7571463Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T15:56:12.0002127Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T15:56:10.9139534Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T15:56:10.9153588Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T15:56:11.7220253Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 13. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-database-failure

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-24T15:54:46Z
**Completed**: 2026-02-24T15:55:40Z
**Duration**: 54 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22358545081/job/64705416637)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T15:55:38.1689629Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T15:55:38.1699488Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T15:55:38.2065052Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 2
  - Sample matches:
    - Line 69: `2026-02-24T15:55:38.2170390Z   name: test-results-biometric-mock-edge-case-database-failure`
    - Line 97: `2026-02-24T15:55:38.5821402Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T15:55:38.2171536Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T15:55:38.4316482Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T15:55:38.5821402Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T15:55:28.2617290Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T15:55:28.2630346Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T15:55:30.3420225Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T15:55:28.4811398Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T15:55:28.4825130Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T15:55:29.5939118Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 14. Run Biometric Voice E2E / Mock Biometric Tests - dimension-adaptation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T15:55:30Z
**Completed**: 2026-02-24T15:56:15Z
**Duration**: 45 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22358545081/job/64705416643)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T15:56:13.0802355Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T15:56:13.0810774Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T15:56:13.1187537Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T15:56:13.4877589Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T15:56:13.1294524Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T15:56:13.3405921Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T15:56:13.4877589Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T15:56:04.3537763Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T15:56:04.3551492Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T15:56:05.5347679Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T15:56:04.5090794Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T15:56:04.5103951Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T15:56:05.3130113Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 15. Run Biometric Voice E2E / Mock Biometric Tests - profile-quality-assessment

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T15:55:36Z
**Completed**: 2026-02-24T15:56:19Z
**Duration**: 43 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22358545081/job/64705416648)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T15:56:17.6710602Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T15:56:17.6718735Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T15:56:17.7084262Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T15:56:18.0888591Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T15:56:17.7188435Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T15:56:17.9302585Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T15:56:18.0888591Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T15:56:08.1843753Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T15:56:08.1858240Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T15:56:09.7142279Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T15:56:08.3368959Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T15:56:08.3382554Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T15:56:09.4334992Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 16. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-noise

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T15:55:03Z
**Completed**: 2026-02-24T15:56:06Z
**Duration**: 63 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22358545081/job/64705416650)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T15:56:04.3239584Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T15:56:04.3247521Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T15:56:04.3578602Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T15:56:04.7280950Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T15:56:04.3683361Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T15:56:04.5798693Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T15:56:04.7280950Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T15:55:55.4214066Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T15:55:55.4227150Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T15:55:56.6101750Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T15:55:55.5617204Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T15:55:55.5630204Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T15:55:56.3474987Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 17. Run Biometric Voice E2E / Mock Biometric Tests - voice-synthesis-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T15:55:17Z
**Completed**: 2026-02-24T15:56:04Z
**Duration**: 47 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22358545081/job/64705416651)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T15:56:02.2219618Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T15:56:02.2229382Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T15:56:02.2606849Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T15:56:02.6330227Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T15:56:02.2716166Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T15:56:02.4836677Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T15:56:02.6330227Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T15:55:52.7200273Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T15:55:52.7214089Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T15:55:53.9143628Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T15:55:52.8754781Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T15:55:52.8768330Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T15:55:53.6787362Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 18. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-voice-drift

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T15:55:33Z
**Completed**: 2026-02-24T15:56:22Z
**Duration**: 49 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22358545081/job/64705416702)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T15:56:19.0602807Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T15:56:19.0612503Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T15:56:19.0985165Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T15:56:19.4436260Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T15:56:19.1088032Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T15:56:19.3076620Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T15:56:19.4436260Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T15:56:10.7594310Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T15:56:10.7607354Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T15:56:12.6625169Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T15:56:10.9951595Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T15:56:10.9964269Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T15:56:11.7372418Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 19. Run Unlock Integration E2E / Generate Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-24T15:55:47Z
**Completed**: 2026-02-24T15:55:55Z
**Duration**: 8 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22358545081/job/64705716749)

#### Failed Steps

- **Step 4**: Check Test Status

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T15:55:52.2612791Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 13
  - Sample matches:
    - Line 39: `2026-02-24T15:55:52.0814705Z [36;1mTOTAL_FAILED=0[0m`
    - Line 44: `2026-02-24T15:55:52.0817194Z [36;1m    FAILED=$(jq -r '.summary.failed' "$report")[0m`
    - Line 46: `2026-02-24T15:55:52.0818111Z [36;1m    TOTAL_FAILED=$((TOTAL_FAILED + FAILED))[0m`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 4: `2026-02-24T15:55:51.8449532Z (node:2059) [DEP0005] DeprecationWarning: Buffer() is deprecated due to`
    - Line 5: `2026-02-24T15:55:51.8451585Z (Use `node --trace-deprecation ...` to show where the warning was creat`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 20. Run Biometric Voice E2E / Generate Biometric Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-24T15:56:25Z
**Completed**: 2026-02-24T15:56:33Z
**Duration**: 8 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22358545081/job/64705810376)

#### Failed Steps

- **Step 4**: Check Test Status

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T15:56:29.8968477Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 11
  - Sample matches:
    - Line 32: `2026-02-24T15:56:29.7154866Z [36;1mTOTAL_FAILED=0[0m`
    - Line 37: `2026-02-24T15:56:29.7156794Z [36;1m    FAILED=$(jq -r '.summary.failed' "$report")[0m`
    - Line 39: `2026-02-24T15:56:29.7157646Z [36;1m    TOTAL_FAILED=$((TOTAL_FAILED + FAILED))[0m`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 21. Generate Combined Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-24T15:56:36Z
**Completed**: 2026-02-24T15:56:38Z
**Duration**: 2 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22358545081/job/64705834773)

#### Failed Steps

- **Step 2**: Generate Combined Summary

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 61: `2026-02-24T15:56:37.5905172Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 9
  - Sample matches:
    - Line 25: `2026-02-24T15:56:37.5699124Z [36;1mif [ "failure" = "success" ]; then[0m`
    - Line 28: `2026-02-24T15:56:37.5701207Z [36;1m  echo "- ❌ **Unlock Integration E2E:** failure" >> $GITHUB_STEP`
    - Line 32: `2026-02-24T15:56:37.5703430Z [36;1mif [ "failure" = "success" ]; then[0m`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 22. Notify Test Status

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-24T15:56:41Z
**Completed**: 2026-02-24T15:56:43Z
**Duration**: 2 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22358545081/job/64705848039)

#### Failed Steps

- **Step 3**: Failure Notification

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line -23: `2026-02-24T15:56:42.3028050Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line -31: `2026-02-24T15:56:42.1910864Z ##[group]Run echo "❌ Unlock tests failed - 'unlock my screen' may be br`
    - Line -30: `2026-02-24T15:56:42.1912894Z [36;1mecho "❌ Unlock tests failed - 'unlock my screen' may be broken!"`
    - Line -25: `2026-02-24T15:56:42.3008993Z ❌ Unlock tests failed - 'unlock my screen' may be broken!`

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

📊 *Report generated on 2026-02-24T15:58:12.342489*
🤖 *JARVIS CI/CD Auto-PR Manager*
