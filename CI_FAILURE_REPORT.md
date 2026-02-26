# CI/CD Failure Analysis Report

## Executive Summary

- **Workflow**: Complete Unlock Test Suite (Master)
- **Run Number**: #13
- **Branch**: `main`
- **Commit**: `4a9c1458bd8eb37901d16713479f9eb74798ca5f`
- **Status**: ❌ FAILED
- **Timestamp**: 2026-02-26T04:47:52Z
- **Triggered By**: @nandkishorrathodk-art
- **Workflow URL**: [View Run](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428290180)

## Failure Overview

Total Failed Jobs: **22**

| # | Job Name | Category | Severity | Duration |
|---|----------|----------|----------|----------|
| 1 | Run Biometric Voice E2E / Mock Biometric Tests - embedding-validation | timeout | high | 38s |
| 2 | Run Biometric Voice E2E / Mock Biometric Tests - voice-verification | timeout | high | 56s |
| 3 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-cold-start | timeout | high | 37s |
| 4 | Run Biometric Voice E2E / Mock Biometric Tests - replay-attack-detection | timeout | high | 46s |
| 5 | Run Biometric Voice E2E / Mock Biometric Tests - anti-spoofing | timeout | high | 50s |
| 6 | Run Biometric Voice E2E / Mock Biometric Tests - wake-word-detection | timeout | high | 52s |
| 7 | Run Biometric Voice E2E / Mock Biometric Tests - stt-transcription | timeout | high | 45s |
| 8 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-database-failure | test_failure | high | 36s |
| 9 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-noise | timeout | high | 44s |
| 10 | Run Biometric Voice E2E / Mock Biometric Tests - adaptive-thresholds | timeout | high | 48s |
| 11 | Run Biometric Voice E2E / Mock Biometric Tests - performance-baseline | timeout | high | 43s |
| 12 | Run Biometric Voice E2E / Mock Biometric Tests - profile-quality-assessment | timeout | high | 52s |
| 13 | Run Biometric Voice E2E / Mock Biometric Tests - end-to-end-flow | timeout | high | 45s |
| 14 | Run Biometric Voice E2E / Mock Biometric Tests - voice-synthesis-detection | timeout | high | 52s |
| 15 | Run Biometric Voice E2E / Mock Biometric Tests - security-validation | timeout | high | 53s |
| 16 | Run Biometric Voice E2E / Mock Biometric Tests - dimension-adaptation | timeout | high | 43s |
| 17 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-voice-drift | timeout | high | 43s |
| 18 | Run Unlock Integration E2E / Mock Tests - security-checks | test_failure | high | 32s |
| 19 | Run Unlock Integration E2E / Generate Test Summary | test_failure | high | 4s |
| 20 | Run Biometric Voice E2E / Generate Biometric Test Summary | test_failure | high | 7s |
| 21 | Generate Combined Test Summary | test_failure | high | 4s |
| 22 | Notify Test Status | test_failure | high | 3s |

## Detailed Analysis

### 1. Run Biometric Voice E2E / Mock Biometric Tests - embedding-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T04:48:59Z
**Completed**: 2026-02-26T04:49:37Z
**Duration**: 38 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428290180/job/64941506073)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T04:49:35.8535450Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T04:49:35.8545106Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T04:49:35.8939371Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T04:49:36.2845831Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T04:49:35.9049315Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T04:49:36.1221342Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T04:49:36.2845831Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T04:49:26.3817067Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T04:49:26.3830469Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T04:49:27.9748385Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T04:49:26.5616351Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T04:49:26.5630571Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T04:49:27.2838902Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 2. Run Biometric Voice E2E / Mock Biometric Tests - voice-verification

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T04:49:22Z
**Completed**: 2026-02-26T04:50:18Z
**Duration**: 56 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428290180/job/64941506077)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T04:50:15.8303272Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T04:50:15.8313183Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T04:50:15.8677934Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T04:50:16.2475884Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T04:50:15.8782147Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T04:50:16.0927687Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T04:50:16.2475884Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T04:50:06.2904884Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T04:50:06.2918693Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T04:50:08.1662382Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T04:50:06.5410657Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T04:50:06.5424472Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T04:50:07.4097537Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 3. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-cold-start

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T04:49:02Z
**Completed**: 2026-02-26T04:49:39Z
**Duration**: 37 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428290180/job/64941506083)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T04:49:37.9163014Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T04:49:37.9171985Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T04:49:37.9519777Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T04:49:38.3302645Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T04:49:37.9630513Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T04:49:38.1777629Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T04:49:38.3302645Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T04:49:28.7669214Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T04:49:28.7683131Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T04:49:30.4271226Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T04:49:28.9651322Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T04:49:28.9665571Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T04:49:29.8103275Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 4. Run Biometric Voice E2E / Mock Biometric Tests - replay-attack-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T04:49:25Z
**Completed**: 2026-02-26T04:50:11Z
**Duration**: 46 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428290180/job/64941506085)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T04:50:09.4959300Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T04:50:09.4968135Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T04:50:09.5349024Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T04:50:09.9246450Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T04:50:09.5457087Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T04:50:09.7687033Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T04:50:09.9246450Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T04:50:00.2693867Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T04:50:00.2708534Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T04:50:01.4981549Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T04:50:00.4266702Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T04:50:00.4281000Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T04:50:01.2524426Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 5. Run Biometric Voice E2E / Mock Biometric Tests - anti-spoofing

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T04:49:41Z
**Completed**: 2026-02-26T04:50:31Z
**Duration**: 50 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428290180/job/64941506088)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T04:50:29.5015424Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T04:50:29.5024142Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T04:50:29.5381799Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T04:50:29.9178618Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T04:50:29.5493046Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T04:50:29.7636312Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T04:50:29.9178618Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T04:50:19.6920773Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T04:50:19.6941388Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T04:50:21.5653995Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T04:50:19.9144039Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T04:50:19.9158293Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T04:50:20.7654313Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 6. Run Biometric Voice E2E / Mock Biometric Tests - wake-word-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T04:50:34Z
**Completed**: 2026-02-26T04:51:26Z
**Duration**: 52 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428290180/job/64941506095)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T04:51:23.7203633Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T04:51:23.7214088Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T04:51:23.7617886Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T04:51:24.1403019Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T04:51:23.7723255Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T04:51:23.9871526Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T04:51:24.1403019Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T04:51:12.7491673Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T04:51:12.7513110Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T04:51:15.7875727Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T04:51:13.0611894Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T04:51:13.0627488Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T04:51:13.9833589Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 7. Run Biometric Voice E2E / Mock Biometric Tests - stt-transcription

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T04:50:18Z
**Completed**: 2026-02-26T04:51:03Z
**Duration**: 45 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428290180/job/64941506096)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T04:51:01.7245495Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T04:51:01.7254829Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T04:51:01.7616721Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T04:51:02.1461741Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T04:51:01.7720207Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T04:51:01.9909922Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T04:51:02.1461741Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T04:50:51.8812844Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T04:50:51.8827336Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T04:50:53.1219380Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T04:50:52.0382701Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T04:50:52.0396358Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T04:50:52.8486778Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 8. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-database-failure

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-26T04:50:39Z
**Completed**: 2026-02-26T04:51:15Z
**Duration**: 36 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428290180/job/64941506097)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T04:51:14.2052528Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T04:51:14.2061639Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T04:51:14.2407645Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 2
  - Sample matches:
    - Line 69: `2026-02-26T04:51:14.2516392Z   name: test-results-biometric-mock-edge-case-database-failure`
    - Line 97: `2026-02-26T04:51:14.6207307Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T04:51:14.2517198Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T04:51:14.4682083Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T04:51:14.6207307Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T04:51:04.9902358Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T04:51:04.9916003Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T04:51:06.5794442Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T04:51:05.1777546Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T04:51:05.1791109Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T04:51:06.0085963Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 9. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-noise

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T04:49:43Z
**Completed**: 2026-02-26T04:50:27Z
**Duration**: 44 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428290180/job/64941506098)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T04:50:26.0728632Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T04:50:26.0737955Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T04:50:26.1091530Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T04:50:26.4928328Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T04:50:26.1202656Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T04:50:26.3371813Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T04:50:26.4928328Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T04:50:16.8726497Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T04:50:16.8740597Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T04:50:18.1378487Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T04:50:17.0471691Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T04:50:17.0485774Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T04:50:17.8731626Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 10. Run Biometric Voice E2E / Mock Biometric Tests - adaptive-thresholds

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T04:49:52Z
**Completed**: 2026-02-26T04:50:40Z
**Duration**: 48 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428290180/job/64941506101)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T04:50:37.4824378Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T04:50:37.4834869Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T04:50:37.5244237Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T04:50:37.9128041Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T04:50:37.5353122Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T04:50:37.7531953Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T04:50:37.9128041Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T04:50:27.8930221Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T04:50:27.8945130Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T04:50:29.8100882Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T04:50:28.1450668Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T04:50:28.1465688Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T04:50:29.0210248Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 11. Run Biometric Voice E2E / Mock Biometric Tests - performance-baseline

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T04:50:00Z
**Completed**: 2026-02-26T04:50:43Z
**Duration**: 43 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428290180/job/64941506102)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T04:50:41.7421673Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T04:50:41.7431254Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T04:50:41.7795330Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T04:50:42.1648323Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T04:50:41.7905633Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T04:50:42.0117359Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T04:50:42.1648323Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T04:50:33.1218638Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T04:50:33.1232891Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T04:50:34.3765494Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T04:50:33.2822877Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T04:50:33.2837671Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T04:50:34.1039193Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 12. Run Biometric Voice E2E / Mock Biometric Tests - profile-quality-assessment

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T04:49:54Z
**Completed**: 2026-02-26T04:50:46Z
**Duration**: 52 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428290180/job/64941506105)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T04:50:43.7808197Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T04:50:43.7816977Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T04:50:43.8182096Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T04:50:44.1958941Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T04:50:43.8288675Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T04:50:44.0421615Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T04:50:44.1958941Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T04:50:35.0828786Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T04:50:35.0843228Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T04:50:36.2855137Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T04:50:35.2317800Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T04:50:35.2330866Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T04:50:36.0387855Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 13. Run Biometric Voice E2E / Mock Biometric Tests - end-to-end-flow

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T04:49:47Z
**Completed**: 2026-02-26T04:50:32Z
**Duration**: 45 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428290180/job/64941506106)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T04:50:30.2607529Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T04:50:30.2618235Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T04:50:30.2984991Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T04:50:30.6871934Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T04:50:30.3088958Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T04:50:30.5307386Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T04:50:30.6871934Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T04:50:20.9463475Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T04:50:20.9477851Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T04:50:22.3950032Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T04:50:21.1187805Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T04:50:21.1202483Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T04:50:21.9455252Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 14. Run Biometric Voice E2E / Mock Biometric Tests - voice-synthesis-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T04:50:21Z
**Completed**: 2026-02-26T04:51:13Z
**Duration**: 52 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428290180/job/64941506109)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T04:51:11.0992054Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T04:51:11.1000456Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T04:51:11.1357145Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T04:51:11.5249459Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T04:51:11.1460645Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T04:51:11.3649668Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T04:51:11.5249459Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T04:51:01.8953771Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T04:51:01.8967912Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T04:51:03.6696143Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T04:51:02.2805871Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T04:51:02.2819531Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T04:51:03.2112108Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 15. Run Biometric Voice E2E / Mock Biometric Tests - security-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T04:50:15Z
**Completed**: 2026-02-26T04:51:08Z
**Duration**: 53 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428290180/job/64941506110)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T04:51:05.5952335Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T04:51:05.5961834Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T04:51:05.6317088Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T04:51:06.0170929Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T04:51:05.6422240Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T04:51:05.8631913Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T04:51:06.0170929Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T04:50:55.5120277Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T04:50:55.5133435Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T04:50:57.4457163Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T04:50:55.7454186Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T04:50:55.7467053Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T04:50:56.5932048Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 16. Run Biometric Voice E2E / Mock Biometric Tests - dimension-adaptation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T04:50:37Z
**Completed**: 2026-02-26T04:51:20Z
**Duration**: 43 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428290180/job/64941506115)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T04:51:18.5682208Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T04:51:18.5690868Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T04:51:18.6088194Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T04:51:18.9946497Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T04:51:18.6199724Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T04:51:18.8371486Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T04:51:18.9946497Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T04:51:09.4284685Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T04:51:09.4297792Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T04:51:10.8521815Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T04:51:09.5811524Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T04:51:09.5825975Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T04:51:10.3776009Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 17. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-voice-drift

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T04:50:39Z
**Completed**: 2026-02-26T04:51:22Z
**Duration**: 43 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428290180/job/64941506166)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T04:51:19.9767957Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T04:51:19.9776680Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T04:51:20.0138290Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T04:51:20.3922537Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T04:51:20.0247586Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T04:51:20.2386612Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T04:51:20.3922537Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T04:51:09.8851626Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T04:51:09.8865304Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T04:51:12.2596664Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T04:51:10.0516512Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T04:51:10.0530821Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T04:51:10.8455172Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 18. Run Unlock Integration E2E / Mock Tests - security-checks

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-26T04:49:48Z
**Completed**: 2026-02-26T04:50:20Z
**Duration**: 32 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428290180/job/64941560534)

#### Failed Steps

- **Step 6**: Run Mock Tests

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 2
  - Sample matches:
    - Line 39: `2026-02-26T04:50:18.2474425Z 2026-02-26 04:50:18,247 - __main__ - ERROR - ❌ 1 test(s) failed`
    - Line 56: `2026-02-26T04:50:18.2605795Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 39: `2026-02-26T04:50:18.2474425Z 2026-02-26 04:50:18,247 - __main__ - ERROR - ❌ 1 test(s) failed`
    - Line 48: `2026-02-26T04:50:18.2486920Z ❌ Failed: 1`
    - Line 97: `2026-02-26T04:50:18.8899739Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 62: `2026-02-26T04:50:18.2696988Z   if-no-files-found: warn`
    - Line 97: `2026-02-26T04:50:18.8899739Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 19. Run Unlock Integration E2E / Generate Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-26T04:51:16Z
**Completed**: 2026-02-26T04:51:20Z
**Duration**: 4 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428290180/job/64941712821)

#### Failed Steps

- **Step 4**: Check Test Status

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 2
  - Sample matches:
    - Line 1: `2026-02-26T04:51:18.7267972Z Starting download of artifact to: /home/runner/work/Ironcliw-ai/Ironcli`
    - Line 97: `2026-02-26T04:51:18.9837987Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 13
  - Sample matches:
    - Line 39: `2026-02-26T04:51:18.8123379Z [36;1mTOTAL_FAILED=0[0m`
    - Line 44: `2026-02-26T04:51:18.8126959Z [36;1m    FAILED=$(jq -r '.summary.failed' "$report")[0m`
    - Line 46: `2026-02-26T04:51:18.8128512Z [36;1m    TOTAL_FAILED=$((TOTAL_FAILED + FAILED))[0m`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 20. Run Biometric Voice E2E / Generate Biometric Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-26T04:51:28Z
**Completed**: 2026-02-26T04:51:35Z
**Duration**: 7 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428290180/job/64941725415)

#### Failed Steps

- **Step 4**: Check Test Status

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T04:51:32.4414954Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 11
  - Sample matches:
    - Line 32: `2026-02-26T04:51:32.2380860Z [36;1mTOTAL_FAILED=0[0m`
    - Line 37: `2026-02-26T04:51:32.2388449Z [36;1m    FAILED=$(jq -r '.summary.failed' "$report")[0m`
    - Line 39: `2026-02-26T04:51:32.2391796Z [36;1m    TOTAL_FAILED=$((TOTAL_FAILED + FAILED))[0m`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 21. Generate Combined Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-26T04:51:37Z
**Completed**: 2026-02-26T04:51:41Z
**Duration**: 4 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428290180/job/64941736245)

#### Failed Steps

- **Step 2**: Generate Combined Summary

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 61: `2026-02-26T04:51:39.8788860Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 9
  - Sample matches:
    - Line 25: `2026-02-26T04:51:39.8553234Z [36;1mif [ "failure" = "success" ]; then[0m`
    - Line 28: `2026-02-26T04:51:39.8555190Z [36;1m  echo "- ❌ **Unlock Integration E2E:** failure" >> $GITHUB_STEP`
    - Line 32: `2026-02-26T04:51:39.8557160Z [36;1mif [ "failure" = "success" ]; then[0m`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 22. Notify Test Status

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-26T04:51:44Z
**Completed**: 2026-02-26T04:51:47Z
**Duration**: 3 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428290180/job/64941744055)

#### Failed Steps

- **Step 3**: Failure Notification

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line -23: `2026-02-26T04:51:45.4734766Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line -31: `2026-02-26T04:51:45.3444364Z ##[group]Run echo "❌ Unlock tests failed - 'unlock my screen' may be br`
    - Line -30: `2026-02-26T04:51:45.3446391Z [36;1mecho "❌ Unlock tests failed - 'unlock my screen' may be broken!"`
    - Line -25: `2026-02-26T04:51:45.4714617Z ❌ Unlock tests failed - 'unlock my screen' may be broken!`

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

📊 *Report generated on 2026-02-26T04:53:08.892055*
🤖 *JARVIS CI/CD Auto-PR Manager*
