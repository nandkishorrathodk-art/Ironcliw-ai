# CI/CD Failure Analysis Report

## Executive Summary

- **Workflow**: Complete Unlock Test Suite (Master)
- **Run Number**: #14
- **Branch**: `main`
- **Commit**: `55ab4b398ad5e7f5d900623b5e1fe8367677d099`
- **Status**: ❌ FAILED
- **Timestamp**: 2026-02-26T04:57:29Z
- **Triggered By**: @nandkishorrathodk-art
- **Workflow URL**: [View Run](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428505954)

## Failure Overview

Total Failed Jobs: **21**

| # | Job Name | Category | Severity | Duration |
|---|----------|----------|----------|----------|
| 1 | Run Biometric Voice E2E / Mock Biometric Tests - profile-quality-assessment | timeout | high | 50s |
| 2 | Run Biometric Voice E2E / Mock Biometric Tests - embedding-validation | timeout | high | 53s |
| 3 | Run Biometric Voice E2E / Mock Biometric Tests - voice-verification | timeout | high | 48s |
| 4 | Run Biometric Voice E2E / Mock Biometric Tests - wake-word-detection | timeout | high | 53s |
| 5 | Run Biometric Voice E2E / Mock Biometric Tests - stt-transcription | timeout | high | 50s |
| 6 | Run Biometric Voice E2E / Mock Biometric Tests - dimension-adaptation | timeout | high | 52s |
| 7 | Run Biometric Voice E2E / Mock Biometric Tests - security-validation | timeout | high | 40s |
| 8 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-noise | timeout | high | 50s |
| 9 | Run Biometric Voice E2E / Mock Biometric Tests - voice-synthesis-detection | timeout | high | 64s |
| 10 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-voice-drift | timeout | high | 46s |
| 11 | Run Biometric Voice E2E / Mock Biometric Tests - end-to-end-flow | timeout | high | 43s |
| 12 | Run Biometric Voice E2E / Mock Biometric Tests - anti-spoofing | timeout | high | 44s |
| 13 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-cold-start | timeout | high | 44s |
| 14 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-database-failure | test_failure | high | 37s |
| 15 | Run Biometric Voice E2E / Mock Biometric Tests - performance-baseline | timeout | high | 43s |
| 16 | Run Biometric Voice E2E / Mock Biometric Tests - replay-attack-detection | timeout | high | 43s |
| 17 | Run Biometric Voice E2E / Mock Biometric Tests - adaptive-thresholds | timeout | high | 42s |
| 18 | Run Unlock Integration E2E / Mock Tests - security-checks | test_failure | high | 49s |
| 19 | Run Unlock Integration E2E / Generate Test Summary | test_failure | high | 4s |
| 20 | Generate Combined Test Summary | test_failure | high | 2s |
| 21 | Notify Test Status | test_failure | high | 4s |

## Detailed Analysis

### 1. Run Biometric Voice E2E / Mock Biometric Tests - profile-quality-assessment

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T04:57:46Z
**Completed**: 2026-02-26T04:58:36Z
**Duration**: 50 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428505954/job/64942146880)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T04:58:33.9817042Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T04:58:33.9826458Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T04:58:34.0190456Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T04:58:34.3944221Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T04:58:34.0301193Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T04:58:34.2414585Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T04:58:34.3944221Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T04:58:22.7438261Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T04:58:22.7451888Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T04:58:26.0657152Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T04:58:23.0535702Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T04:58:23.0548810Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T04:58:23.9463369Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 2. Run Biometric Voice E2E / Mock Biometric Tests - embedding-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T04:57:46Z
**Completed**: 2026-02-26T04:58:39Z
**Duration**: 53 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428505954/job/64942146887)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T04:58:37.4633720Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T04:58:37.4643627Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T04:58:37.5004811Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T04:58:37.8945598Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T04:58:37.5113427Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T04:58:37.7383049Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T04:58:37.8945598Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T04:58:27.5892466Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T04:58:27.5906469Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T04:58:29.5833610Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T04:58:27.8324137Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T04:58:27.8338626Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T04:58:28.6800101Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 3. Run Biometric Voice E2E / Mock Biometric Tests - voice-verification

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T04:57:46Z
**Completed**: 2026-02-26T04:58:34Z
**Duration**: 48 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428505954/job/64942146889)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T04:58:32.1477583Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T04:58:32.1488057Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T04:58:32.1898900Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T04:58:32.5762930Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T04:58:32.2008893Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T04:58:32.4191871Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T04:58:32.5762930Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T04:58:22.6605743Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T04:58:22.6620068Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T04:58:24.3188717Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T04:58:22.9802221Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T04:58:22.9817064Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T04:58:23.8949059Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 4. Run Biometric Voice E2E / Mock Biometric Tests - wake-word-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T04:58:09Z
**Completed**: 2026-02-26T04:59:02Z
**Duration**: 53 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428505954/job/64942146891)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T04:58:59.4198621Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T04:58:59.4208337Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T04:58:59.4574946Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T04:58:59.8345241Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T04:58:59.4680810Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T04:58:59.6821088Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T04:58:59.8345241Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T04:58:50.3962552Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T04:58:50.3976173Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T04:58:51.6574665Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T04:58:50.5565872Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T04:58:50.5579456Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T04:58:51.3695688Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 5. Run Biometric Voice E2E / Mock Biometric Tests - stt-transcription

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T04:58:37Z
**Completed**: 2026-02-26T04:59:27Z
**Duration**: 50 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428505954/job/64942146892)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T04:59:25.4345926Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T04:59:25.4355726Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T04:59:25.4879139Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T04:59:25.9015795Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T04:59:25.4990791Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T04:59:25.7277345Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T04:59:25.9015795Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T04:59:09.0202223Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T04:59:09.0219143Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T04:59:16.9612709Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T04:59:09.2360313Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T04:59:09.2377318Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T04:59:10.1550021Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 6. Run Biometric Voice E2E / Mock Biometric Tests - dimension-adaptation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T04:57:48Z
**Completed**: 2026-02-26T04:58:40Z
**Duration**: 52 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428505954/job/64942146899)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T04:58:38.3490195Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T04:58:38.3500228Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T04:58:38.4010021Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T04:58:38.7802913Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T04:58:38.4117207Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T04:58:38.6315098Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T04:58:38.7802913Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T04:58:27.0968458Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T04:58:27.0984933Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T04:58:30.7649203Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T04:58:27.4231241Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T04:58:27.4245822Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T04:58:28.2553808Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 7. Run Biometric Voice E2E / Mock Biometric Tests - security-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T04:57:47Z
**Completed**: 2026-02-26T04:58:27Z
**Duration**: 40 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428505954/job/64942146908)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T04:58:25.3419821Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T04:58:25.3429099Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T04:58:25.3788432Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T04:58:25.7550988Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T04:58:25.3894510Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T04:58:25.6025219Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T04:58:25.7550988Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T04:58:16.5429677Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T04:58:16.5442964Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T04:58:17.8527151Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T04:58:16.6931358Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T04:58:16.6944507Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T04:58:17.5044469Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 8. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-noise

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T04:58:11Z
**Completed**: 2026-02-26T04:59:01Z
**Duration**: 50 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428505954/job/64942146916)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T04:58:59.1067935Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T04:58:59.1077130Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T04:58:59.1429825Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T04:58:59.5229284Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T04:58:59.1537183Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T04:58:59.3694950Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T04:58:59.5229284Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T04:58:49.2916041Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T04:58:49.2929911Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T04:58:51.2513446Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T04:58:49.5413671Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T04:58:49.5427009Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T04:58:50.3892238Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 9. Run Biometric Voice E2E / Mock Biometric Tests - voice-synthesis-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T04:58:56Z
**Completed**: 2026-02-26T05:00:00Z
**Duration**: 64 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428505954/job/64942146917)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T04:59:57.5067406Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T04:59:57.5076186Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T04:59:57.5439152Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T04:59:57.9227513Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T04:59:57.5547200Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T04:59:57.7690992Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T04:59:57.9227513Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T04:59:48.7292428Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T04:59:48.7305562Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T04:59:50.0801249Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T04:59:48.8832044Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T04:59:48.8845686Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T04:59:49.6823187Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 10. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-voice-drift

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T04:58:43Z
**Completed**: 2026-02-26T04:59:29Z
**Duration**: 46 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428505954/job/64942146918)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T04:59:27.3729880Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T04:59:27.3739973Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T04:59:27.4113622Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T04:59:27.7564271Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T04:59:27.4216232Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T04:59:27.6194425Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T04:59:27.7564271Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T04:59:18.4719841Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T04:59:18.4733633Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T04:59:20.5359653Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T04:59:18.7164226Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T04:59:18.7178273Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T04:59:19.4682507Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 11. Run Biometric Voice E2E / Mock Biometric Tests - end-to-end-flow

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T04:59:03Z
**Completed**: 2026-02-26T04:59:46Z
**Duration**: 43 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428505954/job/64942146919)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T04:59:44.5803802Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T04:59:44.5812456Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T04:59:44.6151523Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T04:59:44.9928886Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T04:59:44.6256198Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T04:59:44.8394313Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T04:59:44.9928886Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T04:59:35.7320410Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T04:59:35.7334051Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T04:59:36.9330496Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T04:59:35.8899962Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T04:59:35.8918035Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T04:59:36.6867326Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 12. Run Biometric Voice E2E / Mock Biometric Tests - anti-spoofing

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T04:59:07Z
**Completed**: 2026-02-26T04:59:51Z
**Duration**: 44 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428505954/job/64942146920)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T04:59:49.2601969Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T04:59:49.2612734Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T04:59:49.2992194Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T04:59:49.6833703Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T04:59:49.3097985Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T04:59:49.5308941Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T04:59:49.6833703Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T04:59:38.9214199Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T04:59:38.9227998Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T04:59:40.6695793Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T04:59:39.1572324Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T04:59:39.1586467Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T04:59:40.0161105Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 13. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-cold-start

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T04:59:21Z
**Completed**: 2026-02-26T05:00:05Z
**Duration**: 44 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428505954/job/64942146924)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T05:00:03.1212331Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T05:00:03.1222186Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T05:00:03.1634279Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T05:00:03.5476401Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T05:00:03.1749089Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T05:00:03.3922144Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T05:00:03.5476401Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T04:59:54.0028034Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T04:59:54.0042289Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T04:59:55.2389904Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T04:59:54.1618732Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T04:59:54.1631826Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T04:59:54.9988324Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 14. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-database-failure

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-26T04:58:41Z
**Completed**: 2026-02-26T04:59:18Z
**Duration**: 37 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428505954/job/64942146935)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T04:59:16.7013972Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T04:59:16.7023677Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T04:59:16.7464686Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 2
  - Sample matches:
    - Line 69: `2026-02-26T04:59:16.7577566Z   name: test-results-biometric-mock-edge-case-database-failure`
    - Line 97: `2026-02-26T04:59:17.1403253Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T04:59:16.7578406Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T04:59:16.9830608Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T04:59:17.1403253Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T04:59:07.4101009Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T04:59:07.4114938Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T04:59:08.7265388Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T04:59:07.5720574Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T04:59:07.5734714Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T04:59:08.4110291Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 15. Run Biometric Voice E2E / Mock Biometric Tests - performance-baseline

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T04:58:11Z
**Completed**: 2026-02-26T04:58:54Z
**Duration**: 43 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428505954/job/64942146939)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T04:58:53.2621927Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T04:58:53.2630730Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T04:58:53.3045846Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T04:58:53.6553334Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T04:58:53.3149192Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T04:58:53.5169164Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T04:58:53.6553334Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T04:58:44.9940806Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T04:58:44.9954456Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T04:58:46.0089329Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T04:58:45.1427764Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T04:58:45.1441757Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T04:58:45.8415159Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 16. Run Biometric Voice E2E / Mock Biometric Tests - replay-attack-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T04:59:28Z
**Completed**: 2026-02-26T05:00:11Z
**Duration**: 43 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428505954/job/64942146943)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T05:00:09.1663258Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T05:00:09.1672391Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T05:00:09.2013777Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T05:00:09.5761208Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T05:00:09.2120245Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T05:00:09.4247934Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T05:00:09.5761208Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T05:00:00.4247199Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T05:00:00.4260410Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T05:00:01.6346548Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T05:00:00.5805414Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T05:00:00.5819220Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T05:00:01.3948116Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 17. Run Biometric Voice E2E / Mock Biometric Tests - adaptive-thresholds

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T04:58:28Z
**Completed**: 2026-02-26T04:59:10Z
**Duration**: 42 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428505954/job/64942146945)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T04:59:08.3830949Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T04:59:08.3839649Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T04:59:08.4205989Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T04:59:08.7998650Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T04:59:08.4311184Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T04:59:08.6454144Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T04:59:08.7998650Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T04:58:59.4544586Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T04:58:59.4558572Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T04:59:00.6914310Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T04:58:59.6175076Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T04:58:59.6188354Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T04:59:00.4441195Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 18. Run Unlock Integration E2E / Mock Tests - security-checks

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-26T04:59:34Z
**Completed**: 2026-02-26T05:00:23Z
**Duration**: 49 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428505954/job/64942152124)

#### Failed Steps

- **Step 6**: Run Mock Tests

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 2
  - Sample matches:
    - Line 39: `2026-02-26T05:00:20.4171832Z 2026-02-26 05:00:20,417 - __main__ - ERROR - ❌ 1 test(s) failed`
    - Line 56: `2026-02-26T05:00:20.4299668Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 39: `2026-02-26T05:00:20.4171832Z 2026-02-26 05:00:20,417 - __main__ - ERROR - ❌ 1 test(s) failed`
    - Line 48: `2026-02-26T05:00:20.4182502Z ❌ Failed: 1`
    - Line 97: `2026-02-26T05:00:21.5305820Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 62: `2026-02-26T05:00:20.4370842Z   if-no-files-found: warn`
    - Line 97: `2026-02-26T05:00:21.5305820Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 19. Run Unlock Integration E2E / Generate Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-26T05:00:32Z
**Completed**: 2026-02-26T05:00:36Z
**Duration**: 4 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428505954/job/64942332643)

#### Failed Steps

- **Step 4**: Check Test Status

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T05:00:35.4249554Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 13
  - Sample matches:
    - Line 39: `2026-02-26T05:00:35.2417501Z [36;1mTOTAL_FAILED=0[0m`
    - Line 44: `2026-02-26T05:00:35.2421711Z [36;1m    FAILED=$(jq -r '.summary.failed' "$report")[0m`
    - Line 46: `2026-02-26T05:00:35.2423403Z [36;1m    TOTAL_FAILED=$((TOTAL_FAILED + FAILED))[0m`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 4: `2026-02-26T05:00:35.1331548Z (node:2178) [DEP0005] DeprecationWarning: Buffer() is deprecated due to`
    - Line 5: `2026-02-26T05:00:35.1335083Z (Use `node --trace-deprecation ...` to show where the warning was creat`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 20. Generate Combined Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-26T05:00:39Z
**Completed**: 2026-02-26T05:00:41Z
**Duration**: 2 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428505954/job/64942341871)

#### Failed Steps

- **Step 2**: Generate Combined Summary

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 61: `2026-02-26T05:00:40.7424269Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 9
  - Sample matches:
    - Line 25: `2026-02-26T05:00:40.7181926Z [36;1mif [ "failure" = "success" ]; then[0m`
    - Line 28: `2026-02-26T05:00:40.7184208Z [36;1m  echo "- ❌ **Unlock Integration E2E:** failure" >> $GITHUB_STEP`
    - Line 32: `2026-02-26T05:00:40.7186166Z [36;1mif [ "failure" = "success" ]; then[0m`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 21. Notify Test Status

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-26T05:00:43Z
**Completed**: 2026-02-26T05:00:47Z
**Duration**: 4 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22428505954/job/64942348033)

#### Failed Steps

- **Step 3**: Failure Notification

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line -23: `2026-02-26T05:00:44.9896811Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line -31: `2026-02-26T05:00:44.8797552Z ##[group]Run echo "❌ Unlock tests failed - 'unlock my screen' may be br`
    - Line -30: `2026-02-26T05:00:44.8798822Z [36;1mecho "❌ Unlock tests failed - 'unlock my screen' may be broken!"`
    - Line -25: `2026-02-26T05:00:44.9877309Z ❌ Unlock tests failed - 'unlock my screen' may be broken!`

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

📊 *Report generated on 2026-02-26T05:01:57.702318*
🤖 *JARVIS CI/CD Auto-PR Manager*
