# CI/CD Failure Analysis Report

## Executive Summary

- **Workflow**: Complete Unlock Test Suite (Master)
- **Run Number**: #12
- **Branch**: `main`
- **Commit**: `7941ff9c445f017bf9af6d4f1137f75cc0ccce3c`
- **Status**: ❌ FAILED
- **Timestamp**: 2026-02-26T02:52:59Z
- **Triggered By**: @nandkishorrathodk-art
- **Workflow URL**: [View Run](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425760002)

## Failure Overview

Total Failed Jobs: **22**

| # | Job Name | Category | Severity | Duration |
|---|----------|----------|----------|----------|
| 1 | Run Biometric Voice E2E / Mock Biometric Tests - voice-verification | timeout | high | 37s |
| 2 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-noise | timeout | high | 53s |
| 3 | Run Biometric Voice E2E / Mock Biometric Tests - adaptive-thresholds | timeout | high | 53s |
| 4 | Run Biometric Voice E2E / Mock Biometric Tests - wake-word-detection | timeout | high | 53s |
| 5 | Run Biometric Voice E2E / Mock Biometric Tests - stt-transcription | timeout | high | 46s |
| 6 | Run Biometric Voice E2E / Mock Biometric Tests - anti-spoofing | timeout | high | 54s |
| 7 | Run Biometric Voice E2E / Mock Biometric Tests - profile-quality-assessment | timeout | high | 48s |
| 8 | Run Biometric Voice E2E / Mock Biometric Tests - dimension-adaptation | timeout | high | 51s |
| 9 | Run Biometric Voice E2E / Mock Biometric Tests - voice-synthesis-detection | timeout | high | 41s |
| 10 | Run Biometric Voice E2E / Mock Biometric Tests - embedding-validation | timeout | high | 55s |
| 11 | Run Biometric Voice E2E / Mock Biometric Tests - security-validation | timeout | high | 39s |
| 12 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-voice-drift | timeout | high | 45s |
| 13 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-cold-start | timeout | high | 42s |
| 14 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-database-failure | test_failure | high | 50s |
| 15 | Run Biometric Voice E2E / Mock Biometric Tests - replay-attack-detection | timeout | high | 43s |
| 16 | Run Biometric Voice E2E / Mock Biometric Tests - performance-baseline | timeout | high | 43s |
| 17 | Run Biometric Voice E2E / Mock Biometric Tests - end-to-end-flow | timeout | high | 54s |
| 18 | Run Unlock Integration E2E / Mock Tests - security-checks | test_failure | high | 42s |
| 19 | Run Unlock Integration E2E / Generate Test Summary | test_failure | high | 5s |
| 20 | Run Biometric Voice E2E / Generate Biometric Test Summary | test_failure | high | 4s |
| 21 | Generate Combined Test Summary | test_failure | high | 5s |
| 22 | Notify Test Status | test_failure | high | 3s |

## Detailed Analysis

### 1. Run Biometric Voice E2E / Mock Biometric Tests - voice-verification

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:54:21Z
**Completed**: 2026-02-26T02:54:58Z
**Duration**: 37 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425760002/job/64933419073)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:54:57.0332111Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:54:57.0341933Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:54:57.0720651Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:54:57.4545022Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:54:57.0825435Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:54:57.2971847Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:54:57.4545022Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:54:48.0350753Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:54:48.0364437Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:54:49.7050087Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:54:48.2281038Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:54:48.2294571Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:54:49.0710849Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 2. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-noise

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:53:56Z
**Completed**: 2026-02-26T02:54:49Z
**Duration**: 53 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425760002/job/64933419076)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:54:47.4533135Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:54:47.4543054Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:54:47.4948814Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:54:47.8795547Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:54:47.5058504Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:54:47.7240158Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:54:47.8795547Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:54:37.5967968Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:54:37.5982430Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:54:39.6495456Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:54:37.8515114Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:54:37.8529737Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:54:38.7335701Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 3. Run Biometric Voice E2E / Mock Biometric Tests - adaptive-thresholds

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:54:37Z
**Completed**: 2026-02-26T02:55:30Z
**Duration**: 53 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425760002/job/64933419077)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:55:27.0143802Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:55:27.0152670Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:55:27.0511213Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:55:27.4329513Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:55:27.0620528Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:55:27.2772294Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:55:27.4329513Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:55:17.8903178Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:55:17.8916294Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:55:19.1931822Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:55:18.0504107Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:55:18.0517999Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:55:18.8609174Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 4. Run Biometric Voice E2E / Mock Biometric Tests - wake-word-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:54:00Z
**Completed**: 2026-02-26T02:54:53Z
**Duration**: 53 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425760002/job/64933419079)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:54:51.4827974Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:54:51.4836549Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:54:51.5218043Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:54:51.9020311Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:54:51.5327640Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:54:51.7479312Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:54:51.9020311Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:54:42.8892603Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:54:42.8906695Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:54:44.0825862Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:54:43.0386516Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:54:43.0399949Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:54:43.8439101Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 5. Run Biometric Voice E2E / Mock Biometric Tests - stt-transcription

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:54:31Z
**Completed**: 2026-02-26T02:55:17Z
**Duration**: 46 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425760002/job/64933419080)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:55:15.3407335Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:55:15.3416378Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:55:15.3782331Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:55:15.7568185Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:55:15.3888977Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:55:15.6044909Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:55:15.7568185Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:55:03.8134787Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:55:03.8148590Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:55:08.0941425Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:55:03.9861943Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:55:03.9875701Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:55:04.7985333Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 6. Run Biometric Voice E2E / Mock Biometric Tests - anti-spoofing

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:54:39Z
**Completed**: 2026-02-26T02:55:33Z
**Duration**: 54 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425760002/job/64933419081)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:55:30.5377488Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:55:30.5387693Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:55:30.5780775Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:55:30.9675051Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:55:30.5894594Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:55:30.8103007Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:55:30.9675051Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:55:21.1723061Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:55:21.1738473Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:55:22.5197119Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:55:21.3337550Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:55:21.3352656Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:55:22.1555908Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 7. Run Biometric Voice E2E / Mock Biometric Tests - profile-quality-assessment

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:55:06Z
**Completed**: 2026-02-26T02:55:54Z
**Duration**: 48 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425760002/job/64933419082)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:55:52.9058963Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:55:52.9068961Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:55:52.9479443Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:55:53.3447886Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:55:52.9586454Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:55:53.1844950Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:55:53.3447886Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:55:42.9755385Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:55:42.9772319Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:55:44.8330883Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:55:43.2656921Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:55:43.2670484Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:55:44.1769271Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 8. Run Biometric Voice E2E / Mock Biometric Tests - dimension-adaptation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:54:04Z
**Completed**: 2026-02-26T02:54:55Z
**Duration**: 51 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425760002/job/64933419083)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:54:52.0392804Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:54:52.0401753Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:54:52.0761918Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:54:52.4557831Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:54:52.0866116Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:54:52.2972342Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:54:52.4557831Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:54:42.3797256Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:54:42.3811568Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:54:44.0278137Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:54:42.7057561Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:54:42.7071735Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:54:43.6091393Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 9. Run Biometric Voice E2E / Mock Biometric Tests - voice-synthesis-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:53:39Z
**Completed**: 2026-02-26T02:54:20Z
**Duration**: 41 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425760002/job/64933419086)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:54:18.3761534Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:54:18.3771271Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:54:18.4140886Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:54:18.8042344Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:54:18.4252295Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:54:18.6501702Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:54:18.8042344Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:54:08.2869625Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:54:08.2884011Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:54:09.8671341Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:54:08.4819222Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:54:08.4833626Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:54:09.3149254Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 10. Run Biometric Voice E2E / Mock Biometric Tests - embedding-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:55:39Z
**Completed**: 2026-02-26T02:56:34Z
**Duration**: 55 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425760002/job/64933419088)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:56:31.6474233Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:56:31.6484478Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:56:31.6913507Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:56:32.1198585Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:56:31.7025244Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:56:31.9478353Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:56:32.1198585Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:56:22.3464041Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:56:22.3489698Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:56:24.0225013Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:56:22.5209877Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:56:22.5227903Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:56:23.3655689Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 11. Run Biometric Voice E2E / Mock Biometric Tests - security-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:53:41Z
**Completed**: 2026-02-26T02:54:20Z
**Duration**: 39 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425760002/job/64933419092)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:54:18.3294048Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:54:18.3302274Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:54:18.3662644Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:54:18.7034600Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:54:18.3763338Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:54:18.5725269Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:54:18.7034600Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:54:10.8263110Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:54:10.8275382Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:54:11.9859032Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:54:11.0143246Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:54:11.0155924Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:54:11.7030498Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 12. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-voice-drift

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:55:02Z
**Completed**: 2026-02-26T02:55:47Z
**Duration**: 45 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425760002/job/64933419093)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:55:45.4882049Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:55:45.4893302Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:55:45.5356430Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:55:45.9441776Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:55:45.5471453Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:55:45.7795767Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:55:45.9441776Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:55:35.6842918Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:55:35.6858462Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:55:37.0572253Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:55:35.8724465Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:55:35.8740977Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:55:36.7378003Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 13. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-cold-start

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:54:18Z
**Completed**: 2026-02-26T02:55:00Z
**Duration**: 42 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425760002/job/64933419097)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:54:58.4826229Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:54:58.4835832Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:54:58.5225459Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:54:58.9085717Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:54:58.5331923Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:54:58.7479657Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:54:58.9085717Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:54:49.1318380Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:54:49.1331668Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:54:50.2620487Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:54:49.2833232Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:54:49.2846308Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:54:50.0122959Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 14. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-database-failure

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-26T02:54:53Z
**Completed**: 2026-02-26T02:55:43Z
**Duration**: 50 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425760002/job/64933419100)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:55:41.2652326Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:55:41.2661911Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:55:41.3084609Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 2
  - Sample matches:
    - Line 69: `2026-02-26T02:55:41.3189957Z   name: test-results-biometric-mock-edge-case-database-failure`
    - Line 97: `2026-02-26T02:55:41.6994067Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:55:41.3190834Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:55:41.5383894Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:55:41.6994067Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:55:29.4530092Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:55:29.4545189Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:55:33.0826648Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:55:29.7259512Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:55:29.7275300Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:55:30.6401323Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 15. Run Biometric Voice E2E / Mock Biometric Tests - replay-attack-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:54:28Z
**Completed**: 2026-02-26T02:55:11Z
**Duration**: 43 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425760002/job/64933419105)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:55:08.8119817Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:55:08.8129884Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:55:08.8525823Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:55:09.2449679Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:55:08.8637756Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:55:09.0878814Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:55:09.2449679Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:55:00.0497036Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:55:00.0510203Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:55:01.3132059Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:55:00.2085850Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:55:00.2099278Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:55:01.0112532Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 16. Run Biometric Voice E2E / Mock Biometric Tests - performance-baseline

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:54:19Z
**Completed**: 2026-02-26T02:55:02Z
**Duration**: 43 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425760002/job/64933419110)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:55:00.1505539Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:55:00.1514609Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:55:00.1888639Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:55:00.5691874Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:55:00.1995723Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:55:00.4149006Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:55:00.5691874Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:54:50.7511783Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:54:50.7525298Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:54:52.1233245Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:54:50.9072230Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:54:50.9086227Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:54:51.7450946Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 17. Run Biometric Voice E2E / Mock Biometric Tests - end-to-end-flow

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:55:35Z
**Completed**: 2026-02-26T02:56:29Z
**Duration**: 54 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425760002/job/64933419118)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:56:27.3478397Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:56:27.3486999Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:56:27.3843562Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:56:27.7637800Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:56:27.3947368Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:56:27.6087321Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:56:27.7637800Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:56:18.0369167Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:56:18.0383263Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:56:19.3089306Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:56:18.1851000Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:56:18.1865192Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:56:19.0044959Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 18. Run Unlock Integration E2E / Mock Tests - security-checks

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-26T02:54:57Z
**Completed**: 2026-02-26T02:55:39Z
**Duration**: 42 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425760002/job/64933421002)

#### Failed Steps

- **Step 6**: Run Mock Tests

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 2
  - Sample matches:
    - Line 39: `2026-02-26T02:55:37.3086349Z 2026-02-26 02:55:37,308 - __main__ - ERROR - ❌ 1 test(s) failed`
    - Line 56: `2026-02-26T02:55:37.3209951Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 39: `2026-02-26T02:55:37.3086349Z 2026-02-26 02:55:37,308 - __main__ - ERROR - ❌ 1 test(s) failed`
    - Line 48: `2026-02-26T02:55:37.3095700Z ❌ Failed: 1`
    - Line 97: `2026-02-26T02:55:38.1941686Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 62: `2026-02-26T02:55:37.3281751Z   if-no-files-found: warn`
    - Line 97: `2026-02-26T02:55:38.1941686Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 19. Run Unlock Integration E2E / Generate Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-26T02:56:13Z
**Completed**: 2026-02-26T02:56:18Z
**Duration**: 5 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425760002/job/64933612023)

#### Failed Steps

- **Step 4**: Check Test Status

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:56:16.2206942Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 13
  - Sample matches:
    - Line 39: `2026-02-26T02:56:16.0097198Z [36;1mTOTAL_FAILED=0[0m`
    - Line 44: `2026-02-26T02:56:16.0104756Z [36;1m    FAILED=$(jq -r '.summary.failed' "$report")[0m`
    - Line 46: `2026-02-26T02:56:16.0108068Z [36;1m    TOTAL_FAILED=$((TOTAL_FAILED + FAILED))[0m`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 4: `2026-02-26T02:56:15.8888524Z (node:2195) [DEP0005] DeprecationWarning: Buffer() is deprecated due to`
    - Line 5: `2026-02-26T02:56:15.8892057Z (Use `node --trace-deprecation ...` to show where the warning was creat`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 20. Run Biometric Voice E2E / Generate Biometric Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-26T02:56:36Z
**Completed**: 2026-02-26T02:56:40Z
**Duration**: 4 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425760002/job/64933642888)

#### Failed Steps

- **Step 4**: Check Test Status

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:56:38.8374824Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 11
  - Sample matches:
    - Line 32: `2026-02-26T02:56:38.6518150Z [36;1mTOTAL_FAILED=0[0m`
    - Line 37: `2026-02-26T02:56:38.6521900Z [36;1m    FAILED=$(jq -r '.summary.failed' "$report")[0m`
    - Line 39: `2026-02-26T02:56:38.6523500Z [36;1m    TOTAL_FAILED=$((TOTAL_FAILED + FAILED))[0m`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 21. Generate Combined Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-26T02:56:42Z
**Completed**: 2026-02-26T02:56:47Z
**Duration**: 5 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425760002/job/64933650756)

#### Failed Steps

- **Step 2**: Generate Combined Summary

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 61: `2026-02-26T02:56:45.3176611Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 9
  - Sample matches:
    - Line 25: `2026-02-26T02:56:45.2934000Z [36;1mif [ "failure" = "success" ]; then[0m`
    - Line 28: `2026-02-26T02:56:45.2936126Z [36;1m  echo "- ❌ **Unlock Integration E2E:** failure" >> $GITHUB_STEP`
    - Line 32: `2026-02-26T02:56:45.2938297Z [36;1mif [ "failure" = "success" ]; then[0m`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 22. Notify Test Status

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-26T02:56:49Z
**Completed**: 2026-02-26T02:56:52Z
**Duration**: 3 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425760002/job/64933659844)

#### Failed Steps

- **Step 3**: Failure Notification

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line -23: `2026-02-26T02:56:50.6413721Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line -31: `2026-02-26T02:56:50.5218146Z ##[group]Run echo "❌ Unlock tests failed - 'unlock my screen' may be br`
    - Line -30: `2026-02-26T02:56:50.5219698Z [36;1mecho "❌ Unlock tests failed - 'unlock my screen' may be broken!"`
    - Line -25: `2026-02-26T02:56:50.6397089Z ❌ Unlock tests failed - 'unlock my screen' may be broken!`

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

📊 *Report generated on 2026-02-26T02:58:14.679761*
🤖 *JARVIS CI/CD Auto-PR Manager*
