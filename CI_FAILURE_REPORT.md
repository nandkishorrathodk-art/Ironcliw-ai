# CI/CD Failure Analysis Report

## Executive Summary

- **Workflow**: Complete Unlock Test Suite (Master)
- **Run Number**: #2
- **Branch**: `main`
- **Commit**: `9c3daeb317327f11cc2b19653e461bacdcf4cf03`
- **Status**: ❌ FAILED
- **Timestamp**: 2026-02-22T20:24:03Z
- **Triggered By**: @nandkishorrathodk-art
- **Workflow URL**: [View Run](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22284655623)

## Failure Overview

Total Failed Jobs: **22**

| # | Job Name | Category | Severity | Duration |
|---|----------|----------|----------|----------|
| 1 | Run Unlock Integration E2E / Mock Tests - security-checks | test_failure | high | 16s |
| 2 | Run Biometric Voice E2E / Mock Biometric Tests - stt-transcription | timeout | high | 20s |
| 3 | Run Biometric Voice E2E / Mock Biometric Tests - wake-word-detection | timeout | high | 24s |
| 4 | Run Biometric Voice E2E / Mock Biometric Tests - voice-verification | timeout | high | 22s |
| 5 | Run Biometric Voice E2E / Mock Biometric Tests - embedding-validation | timeout | high | 23s |
| 6 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-cold-start | timeout | high | 24s |
| 7 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-voice-drift | timeout | high | 19s |
| 8 | Run Biometric Voice E2E / Mock Biometric Tests - profile-quality-assessment | timeout | high | 22s |
| 9 | Run Biometric Voice E2E / Mock Biometric Tests - end-to-end-flow | timeout | high | 19s |
| 10 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-database-failure | test_failure | high | 24s |
| 11 | Run Biometric Voice E2E / Mock Biometric Tests - dimension-adaptation | timeout | high | 23s |
| 12 | Run Biometric Voice E2E / Mock Biometric Tests - replay-attack-detection | timeout | high | 19s |
| 13 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-noise | timeout | high | 20s |
| 14 | Run Biometric Voice E2E / Mock Biometric Tests - anti-spoofing | timeout | high | 22s |
| 15 | Run Biometric Voice E2E / Mock Biometric Tests - performance-baseline | timeout | high | 25s |
| 16 | Run Biometric Voice E2E / Mock Biometric Tests - adaptive-thresholds | timeout | high | 23s |
| 17 | Run Biometric Voice E2E / Mock Biometric Tests - security-validation | timeout | high | 23s |
| 18 | Run Biometric Voice E2E / Mock Biometric Tests - voice-synthesis-detection | timeout | high | 21s |
| 19 | Run Unlock Integration E2E / Generate Test Summary | test_failure | high | 5s |
| 20 | Run Biometric Voice E2E / Generate Biometric Test Summary | test_failure | high | 4s |
| 21 | Generate Combined Test Summary | test_failure | high | 4s |
| 22 | Notify Test Status | test_failure | high | 2s |

## Detailed Analysis

### 1. Run Unlock Integration E2E / Mock Tests - security-checks

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-22T20:24:41Z
**Completed**: 2026-02-22T20:24:57Z
**Duration**: 16 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22284655623/job/64460797447)

#### Failed Steps

- **Step 6**: Run Mock Tests

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 2
  - Sample matches:
    - Line 39: `2026-02-22T20:24:54.2665677Z 2026-02-22 20:24:54,266 - __main__ - ERROR - ❌ 1 test(s) failed`
    - Line 56: `2026-02-22T20:24:54.2787725Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 39: `2026-02-22T20:24:54.2665677Z 2026-02-22 20:24:54,266 - __main__ - ERROR - ❌ 1 test(s) failed`
    - Line 48: `2026-02-22T20:24:54.2674269Z ❌ Failed: 1`
    - Line 97: `2026-02-22T20:24:55.0809202Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 62: `2026-02-22T20:24:54.2862485Z   if-no-files-found: warn`
    - Line 97: `2026-02-22T20:24:55.0809202Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 2. Run Biometric Voice E2E / Mock Biometric Tests - stt-transcription

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T20:25:09Z
**Completed**: 2026-02-22T20:25:29Z
**Duration**: 20 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22284655623/job/64460797936)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T20:25:27.8748164Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T20:25:27.8756583Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T20:25:27.9102453Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T20:25:28.2884198Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T20:25:27.9210538Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T20:25:28.1375496Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T20:25:28.2884198Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T20:25:18.8394774Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T20:25:18.8409383Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T20:25:20.3988145Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T20:25:19.0269147Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T20:25:19.0282042Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T20:25:19.8525300Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 3. Run Biometric Voice E2E / Mock Biometric Tests - wake-word-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T20:25:12Z
**Completed**: 2026-02-22T20:25:36Z
**Duration**: 24 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22284655623/job/64460797939)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T20:25:33.6986602Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T20:25:33.6996771Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T20:25:33.7377936Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T20:25:34.1149677Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T20:25:33.7487355Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T20:25:33.9624929Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T20:25:34.1149677Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T20:25:24.1048066Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T20:25:24.1062610Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T20:25:26.1591919Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T20:25:24.3399977Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T20:25:24.3413143Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T20:25:25.1930918Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 4. Run Biometric Voice E2E / Mock Biometric Tests - voice-verification

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T20:25:04Z
**Completed**: 2026-02-22T20:25:26Z
**Duration**: 22 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22284655623/job/64460797942)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T20:25:25.6747356Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T20:25:25.6755775Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T20:25:25.7103601Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T20:25:26.0809101Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T20:25:25.7206455Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T20:25:25.9317511Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T20:25:26.0809101Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T20:25:16.2516904Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T20:25:16.2530096Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T20:25:18.1026328Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T20:25:16.4457408Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T20:25:16.4476680Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T20:25:17.2877474Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 5. Run Biometric Voice E2E / Mock Biometric Tests - embedding-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T20:25:09Z
**Completed**: 2026-02-22T20:25:32Z
**Duration**: 23 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22284655623/job/64460797944)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T20:25:29.2157480Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T20:25:29.2166248Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T20:25:29.2562540Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T20:25:29.6355369Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T20:25:29.2669018Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T20:25:29.4829605Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T20:25:29.6355369Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T20:25:20.6058154Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T20:25:20.6071610Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T20:25:21.8911648Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T20:25:20.7565512Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T20:25:20.7578665Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T20:25:21.5661907Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 6. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-cold-start

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T20:25:28Z
**Completed**: 2026-02-22T20:25:52Z
**Duration**: 24 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22284655623/job/64460797945)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T20:25:50.6585840Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T20:25:50.6595184Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T20:25:50.6945122Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T20:25:51.0690591Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T20:25:50.7049150Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T20:25:50.9166390Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T20:25:51.0690591Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T20:25:41.0538760Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T20:25:41.0552336Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T20:25:42.9893641Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T20:25:41.3140946Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T20:25:41.3153801Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T20:25:42.1833042Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 7. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-voice-drift

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T20:24:53Z
**Completed**: 2026-02-22T20:25:12Z
**Duration**: 19 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22284655623/job/64460797946)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T20:25:10.9631415Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T20:25:10.9640528Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T20:25:11.0016471Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T20:25:11.3791355Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T20:25:11.0127750Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T20:25:11.2258285Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T20:25:11.3791355Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T20:25:01.1266257Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T20:25:01.1279827Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T20:25:02.7072900Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T20:25:01.3046197Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T20:25:01.3065519Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T20:25:02.0110131Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 8. Run Biometric Voice E2E / Mock Biometric Tests - profile-quality-assessment

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T20:25:13Z
**Completed**: 2026-02-22T20:25:35Z
**Duration**: 22 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22284655623/job/64460797949)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T20:25:33.3564239Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T20:25:33.3574328Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T20:25:33.3987249Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T20:25:33.8061124Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T20:25:33.4127341Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T20:25:33.6388709Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T20:25:33.8061124Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T20:25:24.0495696Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T20:25:24.0509170Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T20:25:25.7202707Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T20:25:24.2451751Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T20:25:24.2465816Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T20:25:25.1035838Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 9. Run Biometric Voice E2E / Mock Biometric Tests - end-to-end-flow

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T20:24:28Z
**Completed**: 2026-02-22T20:24:47Z
**Duration**: 19 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22284655623/job/64460797951)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T20:24:45.1985855Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T20:24:45.1994798Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T20:24:45.2361306Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T20:24:45.6147549Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T20:24:45.2473036Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T20:24:45.4623709Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T20:24:45.6147549Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T20:24:36.1985143Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T20:24:36.1998305Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T20:24:37.4623071Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T20:24:36.3741514Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T20:24:36.3763102Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T20:24:37.1907604Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 10. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-database-failure

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-22T20:25:08Z
**Completed**: 2026-02-22T20:25:32Z
**Duration**: 24 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22284655623/job/64460797952)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T20:25:29.2147099Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T20:25:29.2156583Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T20:25:29.2515091Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 2
  - Sample matches:
    - Line 69: `2026-02-22T20:25:29.2621856Z   name: test-results-biometric-mock-edge-case-database-failure`
    - Line 97: `2026-02-22T20:25:29.6204701Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T20:25:29.2622698Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T20:25:29.4713364Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T20:25:29.6204701Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T20:25:20.2700837Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T20:25:20.2720493Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T20:25:21.6059769Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T20:25:20.4235029Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T20:25:20.4247901Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T20:25:21.2270699Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 11. Run Biometric Voice E2E / Mock Biometric Tests - dimension-adaptation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T20:24:44Z
**Completed**: 2026-02-22T20:25:07Z
**Duration**: 23 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22284655623/job/64460797953)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T20:25:06.0869493Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T20:25:06.0878817Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T20:25:06.1241012Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T20:25:06.5026243Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T20:25:06.1345139Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T20:25:06.3517897Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T20:25:06.5026243Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T20:24:52.6349504Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T20:24:52.6363323Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T20:24:58.0952846Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T20:24:52.8009253Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T20:24:52.8022856Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T20:24:53.6098966Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 12. Run Biometric Voice E2E / Mock Biometric Tests - replay-attack-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T20:24:47Z
**Completed**: 2026-02-22T20:25:06Z
**Duration**: 19 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22284655623/job/64460797954)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T20:25:04.9231325Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T20:25:04.9241470Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T20:25:04.9674076Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T20:25:05.3579316Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T20:25:04.9782549Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T20:25:05.1995338Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T20:25:05.3579316Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T20:24:55.5788628Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T20:24:55.5802648Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T20:24:57.0545933Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T20:24:55.7401345Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T20:24:55.7415015Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T20:24:56.5736893Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 13. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-noise

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T20:25:11Z
**Completed**: 2026-02-22T20:25:31Z
**Duration**: 20 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22284655623/job/64460797956)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T20:25:30.1945643Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T20:25:30.1953836Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T20:25:30.2317988Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T20:25:30.6808795Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T20:25:30.2427888Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T20:25:30.4613939Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T20:25:30.6808795Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T20:25:21.2255662Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T20:25:21.2269794Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T20:25:22.6120763Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T20:25:21.3774420Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T20:25:21.3787620Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T20:25:22.1745804Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 14. Run Biometric Voice E2E / Mock Biometric Tests - anti-spoofing

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T20:25:29Z
**Completed**: 2026-02-22T20:25:51Z
**Duration**: 22 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22284655623/job/64460797958)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T20:25:49.2604724Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T20:25:49.2613654Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T20:25:49.3004836Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T20:25:49.6834224Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T20:25:49.3108637Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T20:25:49.5281974Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T20:25:49.6834224Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T20:25:39.8354447Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T20:25:39.8368476Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T20:25:41.1966195Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T20:25:39.9858599Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T20:25:39.9878790Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T20:25:40.8033685Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 15. Run Biometric Voice E2E / Mock Biometric Tests - performance-baseline

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T20:25:23Z
**Completed**: 2026-02-22T20:25:48Z
**Duration**: 25 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22284655623/job/64460797959)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T20:25:45.9481334Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T20:25:45.9490582Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T20:25:45.9870868Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T20:25:46.3648718Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T20:25:45.9980901Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T20:25:46.2119089Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T20:25:46.3648718Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T20:25:36.9718155Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T20:25:36.9732382Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T20:25:38.2641071Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T20:25:37.1151250Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T20:25:37.1164794Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T20:25:37.9479739Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 16. Run Biometric Voice E2E / Mock Biometric Tests - adaptive-thresholds

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T20:25:09Z
**Completed**: 2026-02-22T20:25:32Z
**Duration**: 23 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22284655623/job/64460797963)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T20:25:30.2902904Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T20:25:30.2912028Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T20:25:30.3365759Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T20:25:30.7284405Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T20:25:30.3479780Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T20:25:30.5732066Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T20:25:30.7284405Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T20:25:21.2116940Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T20:25:21.2131141Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T20:25:22.4772218Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T20:25:21.3812655Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T20:25:21.3826179Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T20:25:22.2081424Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 17. Run Biometric Voice E2E / Mock Biometric Tests - security-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T20:25:14Z
**Completed**: 2026-02-22T20:25:37Z
**Duration**: 23 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22284655623/job/64460797968)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T20:25:34.6628351Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T20:25:34.6636930Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T20:25:34.7023380Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T20:25:35.0759305Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T20:25:34.7131905Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T20:25:34.9240644Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T20:25:35.0759305Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T20:25:25.6961293Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T20:25:25.6975145Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T20:25:27.0303443Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T20:25:25.8591557Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T20:25:25.8604348Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T20:25:26.6921269Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 18. Run Biometric Voice E2E / Mock Biometric Tests - voice-synthesis-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T20:25:17Z
**Completed**: 2026-02-22T20:25:38Z
**Duration**: 21 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22284655623/job/64460797972)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T20:25:36.1477240Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T20:25:36.1485994Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T20:25:36.1875669Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T20:25:36.5327369Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T20:25:36.1978823Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T20:25:36.3973957Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T20:25:36.5327369Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T20:25:27.8138061Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T20:25:27.8150654Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T20:25:29.5455378Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T20:25:28.0316591Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T20:25:28.0330078Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T20:25:28.7737268Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 19. Run Unlock Integration E2E / Generate Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-22T20:25:22Z
**Completed**: 2026-02-22T20:25:27Z
**Duration**: 5 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22284655623/job/64460829468)

#### Failed Steps

- **Step 4**: Check Test Status

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T20:25:25.5562345Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 13
  - Sample matches:
    - Line 39: `2026-02-22T20:25:25.3719691Z [36;1mTOTAL_FAILED=0[0m`
    - Line 44: `2026-02-22T20:25:25.3726917Z [36;1m    FAILED=$(jq -r '.summary.failed' "$report")[0m`
    - Line 46: `2026-02-22T20:25:25.3730104Z [36;1m    TOTAL_FAILED=$((TOTAL_FAILED + FAILED))[0m`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 4: `2026-02-22T20:25:25.2032840Z (node:2126) [DEP0005] DeprecationWarning: Buffer() is deprecated due to`
    - Line 5: `2026-02-22T20:25:25.2036938Z (Use `node --trace-deprecation ...` to show where the warning was creat`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 20. Run Biometric Voice E2E / Generate Biometric Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-22T20:25:54Z
**Completed**: 2026-02-22T20:25:58Z
**Duration**: 4 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22284655623/job/64460858016)

#### Failed Steps

- **Step 4**: Check Test Status

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T20:25:57.0928049Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 11
  - Sample matches:
    - Line 32: `2026-02-22T20:25:56.9295290Z [36;1mTOTAL_FAILED=0[0m`
    - Line 37: `2026-02-22T20:25:56.9298797Z [36;1m    FAILED=$(jq -r '.summary.failed' "$report")[0m`
    - Line 39: `2026-02-22T20:25:56.9300391Z [36;1m    TOTAL_FAILED=$((TOTAL_FAILED + FAILED))[0m`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 21. Generate Combined Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-22T20:26:00Z
**Completed**: 2026-02-22T20:26:04Z
**Duration**: 4 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22284655623/job/64460861866)

#### Failed Steps

- **Step 2**: Generate Combined Summary

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 61: `2026-02-22T20:26:02.6562146Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 9
  - Sample matches:
    - Line 25: `2026-02-22T20:26:02.6388496Z [36;1mif [ "failure" = "success" ]; then[0m`
    - Line 28: `2026-02-22T20:26:02.6390924Z [36;1m  echo "- ❌ **Unlock Integration E2E:** failure" >> $GITHUB_STEP`
    - Line 32: `2026-02-22T20:26:02.6393675Z [36;1mif [ "failure" = "success" ]; then[0m`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 22. Notify Test Status

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-22T20:26:06Z
**Completed**: 2026-02-22T20:26:08Z
**Duration**: 2 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22284655623/job/64460866330)

#### Failed Steps

- **Step 3**: Failure Notification

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line -23: `2026-02-22T20:26:07.3467829Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line -31: `2026-02-22T20:26:07.2645563Z ##[group]Run echo "❌ Unlock tests failed - 'unlock my screen' may be br`
    - Line -30: `2026-02-22T20:26:07.2646690Z [36;1mecho "❌ Unlock tests failed - 'unlock my screen' may be broken!"`
    - Line -25: `2026-02-22T20:26:07.3450302Z ❌ Unlock tests failed - 'unlock my screen' may be broken!`

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

📊 *Report generated on 2026-02-22T20:26:47.266974*
🤖 *JARVIS CI/CD Auto-PR Manager*
