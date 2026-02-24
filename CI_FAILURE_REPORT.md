# CI/CD Failure Analysis Report

## Executive Summary

- **Workflow**: Complete Unlock Test Suite (Master)
- **Run Number**: #6
- **Branch**: `main`
- **Commit**: `2c22880fc1e06a4b544e0fc3acfc9517b37c77f1`
- **Status**: ❌ FAILED
- **Timestamp**: 2026-02-24T06:26:16Z
- **Triggered By**: @nandkishorrathodk-art
- **Workflow URL**: [View Run](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347373)

## Failure Overview

Total Failed Jobs: **22**

| # | Job Name | Category | Severity | Duration |
|---|----------|----------|----------|----------|
| 1 | Run Unlock Integration E2E / Mock Tests - security-checks | test_failure | high | 37s |
| 2 | Run Biometric Voice E2E / Mock Biometric Tests - profile-quality-assessment | timeout | high | 38s |
| 3 | Run Biometric Voice E2E / Mock Biometric Tests - stt-transcription | timeout | high | 39s |
| 4 | Run Biometric Voice E2E / Mock Biometric Tests - voice-verification | timeout | high | 52s |
| 5 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-database-failure | test_failure | high | 42s |
| 6 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-cold-start | timeout | high | 41s |
| 7 | Run Biometric Voice E2E / Mock Biometric Tests - embedding-validation | timeout | high | 46s |
| 8 | Run Biometric Voice E2E / Mock Biometric Tests - replay-attack-detection | timeout | high | 44s |
| 9 | Run Biometric Voice E2E / Mock Biometric Tests - anti-spoofing | timeout | high | 60s |
| 10 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-noise | timeout | high | 44s |
| 11 | Run Biometric Voice E2E / Mock Biometric Tests - voice-synthesis-detection | timeout | high | 40s |
| 12 | Run Biometric Voice E2E / Mock Biometric Tests - wake-word-detection | timeout | high | 40s |
| 13 | Run Biometric Voice E2E / Mock Biometric Tests - adaptive-thresholds | timeout | high | 46s |
| 14 | Run Biometric Voice E2E / Mock Biometric Tests - performance-baseline | timeout | high | 43s |
| 15 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-voice-drift | timeout | high | 49s |
| 16 | Run Biometric Voice E2E / Mock Biometric Tests - dimension-adaptation | timeout | high | 41s |
| 17 | Run Biometric Voice E2E / Mock Biometric Tests - end-to-end-flow | timeout | high | 46s |
| 18 | Run Biometric Voice E2E / Mock Biometric Tests - security-validation | timeout | high | 43s |
| 19 | Run Unlock Integration E2E / Generate Test Summary | test_failure | high | 5s |
| 20 | Run Biometric Voice E2E / Generate Biometric Test Summary | test_failure | high | 6s |
| 21 | Generate Combined Test Summary | test_failure | high | 2s |
| 22 | Notify Test Status | test_failure | high | 2s |

## Detailed Analysis

### 1. Run Unlock Integration E2E / Mock Tests - security-checks

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-24T06:28:26Z
**Completed**: 2026-02-24T06:29:03Z
**Duration**: 37 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347373/job/64639006028)

#### Failed Steps

- **Step 6**: Run Mock Tests

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 2
  - Sample matches:
    - Line 39: `2026-02-24T06:29:01.0834364Z 2026-02-24 06:29:01,083 - __main__ - ERROR - ❌ 1 test(s) failed`
    - Line 56: `2026-02-24T06:29:01.0972917Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 39: `2026-02-24T06:29:01.0834364Z 2026-02-24 06:29:01,083 - __main__ - ERROR - ❌ 1 test(s) failed`
    - Line 48: `2026-02-24T06:29:01.0841843Z ❌ Failed: 1`
    - Line 97: `2026-02-24T06:29:01.7066036Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 62: `2026-02-24T06:29:01.1045823Z   if-no-files-found: warn`
    - Line 97: `2026-02-24T06:29:01.7066036Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 2. Run Biometric Voice E2E / Mock Biometric Tests - profile-quality-assessment

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:29:09Z
**Completed**: 2026-02-24T06:29:47Z
**Duration**: 38 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347373/job/64639010473)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:29:46.4681285Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:29:46.4690002Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:29:46.5032692Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:29:46.8737486Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:29:46.5138476Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:29:46.7242822Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:29:46.8737486Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:29:37.9696931Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:29:37.9711107Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:29:39.2098169Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:29:38.1278371Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:29:38.1291673Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:29:38.9287723Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 3. Run Biometric Voice E2E / Mock Biometric Tests - stt-transcription

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:28:43Z
**Completed**: 2026-02-24T06:29:22Z
**Duration**: 39 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347373/job/64639010475)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:29:20.3600595Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:29:20.3609757Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:29:20.3996369Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:29:20.7796529Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:29:20.4103577Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:29:20.6265399Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:29:20.7796529Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:29:11.5274559Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:29:11.5288630Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:29:13.1385005Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:29:11.7162085Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:29:11.7176716Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:29:12.5451153Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 4. Run Biometric Voice E2E / Mock Biometric Tests - voice-verification

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:29:27Z
**Completed**: 2026-02-24T06:30:19Z
**Duration**: 52 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347373/job/64639010476)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:30:16.7390612Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:30:16.7399800Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:30:16.7823266Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:30:17.1610758Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:30:16.7933424Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:30:17.0081100Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:30:17.1610758Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:30:07.7813370Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:30:07.7828017Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:30:09.0324867Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:30:07.9455868Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:30:07.9469883Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:30:08.7647547Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 5. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-database-failure

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-24T06:28:34Z
**Completed**: 2026-02-24T06:29:16Z
**Duration**: 42 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347373/job/64639010481)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:29:14.0832710Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:29:14.0841454Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:29:14.1195183Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 2
  - Sample matches:
    - Line 69: `2026-02-24T06:29:14.1297718Z   name: test-results-biometric-mock-edge-case-database-failure`
    - Line 97: `2026-02-24T06:29:14.4967868Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:29:14.1298554Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:29:14.3463817Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:29:14.4967868Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:29:05.1637673Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:29:05.1650870Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:29:06.3309837Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:29:05.3186934Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:29:05.3199915Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:29:06.1107798Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 6. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-cold-start

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:29:07Z
**Completed**: 2026-02-24T06:29:48Z
**Duration**: 41 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347373/job/64639010486)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:29:46.5467831Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:29:46.5476404Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:29:46.5847545Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:29:46.9620748Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:29:46.5958632Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:29:46.8088237Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:29:46.9620748Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:29:37.6553969Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:29:37.6567692Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:29:39.2189881Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:29:37.8471958Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:29:37.8485972Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:29:38.6744431Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 7. Run Biometric Voice E2E / Mock Biometric Tests - embedding-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:29:04Z
**Completed**: 2026-02-24T06:29:50Z
**Duration**: 46 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347373/job/64639010487)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:29:48.7196034Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:29:48.7205568Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:29:48.7565495Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:29:49.1247391Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:29:48.7669384Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:29:48.9773944Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:29:49.1247391Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:29:39.6359740Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:29:39.6373302Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:29:41.2796050Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:29:39.7907316Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:29:39.7920434Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:29:41.0188430Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 8. Run Biometric Voice E2E / Mock Biometric Tests - replay-attack-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:28:44Z
**Completed**: 2026-02-24T06:29:28Z
**Duration**: 44 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347373/job/64639010492)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:29:26.3369438Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:29:26.3377482Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:29:26.3728277Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:29:26.7483726Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:29:26.3838492Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:29:26.6000835Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:29:26.7483726Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:29:17.1088373Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:29:17.1101585Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:29:18.6541497Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:29:17.2614106Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:29:17.2627625Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:29:18.0544719Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 9. Run Biometric Voice E2E / Mock Biometric Tests - anti-spoofing

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:29:16Z
**Completed**: 2026-02-24T06:30:16Z
**Duration**: 60 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347373/job/64639010495)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:30:13.6969257Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:30:13.6978400Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:30:13.7333886Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:30:14.1024581Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:30:13.7439118Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:30:13.9538149Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:30:14.1024581Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:30:04.0617554Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:30:04.0630254Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:30:05.7242365Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:30:04.2064566Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:30:04.2079314Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:30:05.0068413Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 10. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-noise

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:29:15Z
**Completed**: 2026-02-24T06:29:59Z
**Duration**: 44 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347373/job/64639010497)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:29:57.7741128Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:29:57.7750855Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:29:57.8198741Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:29:58.2105945Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:29:57.8306287Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:29:58.0543837Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:29:58.2105945Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:29:48.2957211Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:29:48.2971591Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:29:49.6689984Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:29:48.4455777Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:29:48.4470639Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:29:49.1647534Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 11. Run Biometric Voice E2E / Mock Biometric Tests - voice-synthesis-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:29:16Z
**Completed**: 2026-02-24T06:29:56Z
**Duration**: 40 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347373/job/64639010500)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:29:54.3977206Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:29:54.3986019Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:29:54.4380266Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:29:54.8130131Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:29:54.4487987Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:29:54.6587690Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:29:54.8130131Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:29:45.1557151Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:29:45.1570994Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:29:46.6937744Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:29:45.3422492Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:29:45.3436163Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:29:46.0739437Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 12. Run Biometric Voice E2E / Mock Biometric Tests - wake-word-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:29:19Z
**Completed**: 2026-02-24T06:29:59Z
**Duration**: 40 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347373/job/64639010502)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:29:58.0802921Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:29:58.0813485Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:29:58.1208669Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:29:58.5025955Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:29:58.1316880Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:29:58.3495043Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:29:58.5025955Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:29:48.4486243Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:29:48.4500674Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:29:50.4753779Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:29:48.6447333Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:29:48.6462104Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:29:49.4843016Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 13. Run Biometric Voice E2E / Mock Biometric Tests - adaptive-thresholds

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:28:54Z
**Completed**: 2026-02-24T06:29:40Z
**Duration**: 46 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347373/job/64639010503)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:29:38.3393746Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:29:38.3402826Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:29:38.3824422Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:29:38.7569904Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:29:38.3927496Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:29:38.6054491Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:29:38.7569904Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:29:27.8041260Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:29:27.8055547Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:29:30.2592983Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:29:27.9629985Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:29:27.9644438Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:29:28.7633386Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 14. Run Biometric Voice E2E / Mock Biometric Tests - performance-baseline

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:29:17Z
**Completed**: 2026-02-24T06:30:00Z
**Duration**: 43 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347373/job/64639010505)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:29:58.6892346Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:29:58.6901078Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:29:58.7269610Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:29:59.1036186Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:29:58.7375886Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:29:58.9501909Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:29:59.1036186Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:29:49.4566083Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:29:49.4579837Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:29:51.2979412Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:29:49.6459841Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:29:49.6473425Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:29:50.7263343Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 15. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-voice-drift

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:29:10Z
**Completed**: 2026-02-24T06:29:59Z
**Duration**: 49 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347373/job/64639010509)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:29:57.3640123Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:29:57.3649769Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:29:57.4070566Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:29:57.7773648Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:29:57.4174175Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:29:57.6275678Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:29:57.7773648Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:29:47.7008728Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:29:47.7022296Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:29:50.1055321Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:29:47.9739714Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:29:47.9754619Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:29:48.8449022Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 16. Run Biometric Voice E2E / Mock Biometric Tests - dimension-adaptation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:29:35Z
**Completed**: 2026-02-24T06:30:16Z
**Duration**: 41 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347373/job/64639010522)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:30:14.5271644Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:30:14.5280041Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:30:14.5664936Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:30:14.9437753Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:30:14.5775097Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:30:14.7924612Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:30:14.9437753Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:30:04.9881483Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:30:04.9895708Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:30:07.0457996Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:30:05.1756409Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:30:05.1770284Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:30:06.0203541Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 17. Run Biometric Voice E2E / Mock Biometric Tests - end-to-end-flow

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:29:24Z
**Completed**: 2026-02-24T06:30:10Z
**Duration**: 46 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347373/job/64639010535)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:30:08.2454855Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:30:08.2465462Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:30:08.2897276Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:30:08.6838985Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:30:08.3003595Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:30:08.5276694Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:30:08.6838985Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:29:58.7654003Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:29:58.7668609Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:30:00.0188811Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:29:58.9267034Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:29:58.9282422Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:29:59.7649279Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 18. Run Biometric Voice E2E / Mock Biometric Tests - security-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:29:35Z
**Completed**: 2026-02-24T06:30:18Z
**Duration**: 43 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347373/job/64639010541)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:30:16.3887512Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:30:16.3896826Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:30:16.4339242Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:30:16.8221157Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:30:16.4453043Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:30:16.6660987Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:30:16.8221157Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:30:07.1332173Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:30:07.1346455Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:30:08.3766441Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:30:07.2910085Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:30:07.2924322Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:30:08.1126276Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 19. Run Unlock Integration E2E / Generate Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-24T06:29:28Z
**Completed**: 2026-02-24T06:29:33Z
**Duration**: 5 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347373/job/64639229932)

#### Failed Steps

- **Step 4**: Check Test Status

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:29:31.1034807Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 13
  - Sample matches:
    - Line 39: `2026-02-24T06:29:30.9070184Z [36;1mTOTAL_FAILED=0[0m`
    - Line 44: `2026-02-24T06:29:30.9074204Z [36;1m    FAILED=$(jq -r '.summary.failed' "$report")[0m`
    - Line 46: `2026-02-24T06:29:30.9075865Z [36;1m    TOTAL_FAILED=$((TOTAL_FAILED + FAILED))[0m`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 4: `2026-02-24T06:29:30.7801545Z (node:2119) [DEP0005] DeprecationWarning: Buffer() is deprecated due to`
    - Line 5: `2026-02-24T06:29:30.7805445Z (Use `node --trace-deprecation ...` to show where the warning was creat`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 20. Run Biometric Voice E2E / Generate Biometric Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-24T06:30:22Z
**Completed**: 2026-02-24T06:30:28Z
**Duration**: 6 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347373/job/64639305500)

#### Failed Steps

- **Step 4**: Check Test Status

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:30:26.3457243Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 11
  - Sample matches:
    - Line 32: `2026-02-24T06:30:26.1739601Z [36;1mTOTAL_FAILED=0[0m`
    - Line 37: `2026-02-24T06:30:26.1746630Z [36;1m    FAILED=$(jq -r '.summary.failed' "$report")[0m`
    - Line 39: `2026-02-24T06:30:26.1749830Z [36;1m    TOTAL_FAILED=$((TOTAL_FAILED + FAILED))[0m`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 21. Generate Combined Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-24T06:30:31Z
**Completed**: 2026-02-24T06:30:33Z
**Duration**: 2 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347373/job/64639318402)

#### Failed Steps

- **Step 2**: Generate Combined Summary

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 61: `2026-02-24T06:30:32.2968008Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 9
  - Sample matches:
    - Line 25: `2026-02-24T06:30:32.2774838Z [36;1mif [ "failure" = "success" ]; then[0m`
    - Line 28: `2026-02-24T06:30:32.2776846Z [36;1m  echo "- ❌ **Unlock Integration E2E:** failure" >> $GITHUB_STEP`
    - Line 32: `2026-02-24T06:30:32.2778821Z [36;1mif [ "failure" = "success" ]; then[0m`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 22. Notify Test Status

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-24T06:30:35Z
**Completed**: 2026-02-24T06:30:37Z
**Duration**: 2 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347373/job/64639323903)

#### Failed Steps

- **Step 3**: Failure Notification

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line -23: `2026-02-24T06:30:36.0158948Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line -31: `2026-02-24T06:30:35.9321657Z ##[group]Run echo "❌ Unlock tests failed - 'unlock my screen' may be br`
    - Line -30: `2026-02-24T06:30:35.9322732Z [36;1mecho "❌ Unlock tests failed - 'unlock my screen' may be broken!"`
    - Line -25: `2026-02-24T06:30:36.0139319Z ❌ Unlock tests failed - 'unlock my screen' may be broken!`

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

📊 *Report generated on 2026-02-24T06:31:37.549143*
🤖 *JARVIS CI/CD Auto-PR Manager*
