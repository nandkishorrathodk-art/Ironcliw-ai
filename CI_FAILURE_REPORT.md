# CI/CD Failure Analysis Report

## Executive Summary

- **Workflow**: Complete Unlock Test Suite (Master)
- **Run Number**: #10
- **Branch**: `main`
- **Commit**: `32b10ccc7538b2bbcd77c2680b593ee37c9e14a3`
- **Status**: ❌ FAILED
- **Timestamp**: 2026-02-26T02:36:18Z
- **Triggered By**: @nandkishorrathodk-art
- **Workflow URL**: [View Run](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425390707)

## Failure Overview

Total Failed Jobs: **22**

| # | Job Name | Category | Severity | Duration |
|---|----------|----------|----------|----------|
| 1 | Run Unlock Integration E2E / Mock Tests - security-checks | test_failure | high | 46s |
| 2 | Run Biometric Voice E2E / Mock Biometric Tests - wake-word-detection | timeout | high | 57s |
| 3 | Run Biometric Voice E2E / Mock Biometric Tests - embedding-validation | timeout | high | 51s |
| 4 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-voice-drift | timeout | high | 55s |
| 5 | Run Biometric Voice E2E / Mock Biometric Tests - voice-verification | timeout | high | 53s |
| 6 | Run Biometric Voice E2E / Mock Biometric Tests - dimension-adaptation | timeout | high | 50s |
| 7 | Run Biometric Voice E2E / Mock Biometric Tests - adaptive-thresholds | timeout | high | 57s |
| 8 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-cold-start | timeout | high | 55s |
| 9 | Run Biometric Voice E2E / Mock Biometric Tests - anti-spoofing | timeout | high | 42s |
| 10 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-database-failure | test_failure | high | 41s |
| 11 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-noise | timeout | high | 45s |
| 12 | Run Biometric Voice E2E / Mock Biometric Tests - stt-transcription | timeout | high | 38s |
| 13 | Run Biometric Voice E2E / Mock Biometric Tests - profile-quality-assessment | timeout | high | 54s |
| 14 | Run Biometric Voice E2E / Mock Biometric Tests - end-to-end-flow | timeout | high | 40s |
| 15 | Run Biometric Voice E2E / Mock Biometric Tests - voice-synthesis-detection | timeout | high | 42s |
| 16 | Run Biometric Voice E2E / Mock Biometric Tests - replay-attack-detection | timeout | high | 61s |
| 17 | Run Biometric Voice E2E / Mock Biometric Tests - security-validation | timeout | high | 50s |
| 18 | Run Biometric Voice E2E / Mock Biometric Tests - performance-baseline | timeout | high | 49s |
| 19 | Run Unlock Integration E2E / Generate Test Summary | test_failure | high | 7s |
| 20 | Run Biometric Voice E2E / Generate Biometric Test Summary | test_failure | high | 8s |
| 21 | Generate Combined Test Summary | test_failure | high | 3s |
| 22 | Notify Test Status | test_failure | high | 3s |

## Detailed Analysis

### 1. Run Unlock Integration E2E / Mock Tests - security-checks

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-26T02:39:13Z
**Completed**: 2026-02-26T02:39:59Z
**Duration**: 46 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425390707/job/64932286632)

#### Failed Steps

- **Step 6**: Run Mock Tests

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 2
  - Sample matches:
    - Line 39: `2026-02-26T02:39:55.4404724Z 2026-02-26 02:39:55,440 - __main__ - ERROR - ❌ 1 test(s) failed`
    - Line 56: `2026-02-26T02:39:55.4535647Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 39: `2026-02-26T02:39:55.4404724Z 2026-02-26 02:39:55,440 - __main__ - ERROR - ❌ 1 test(s) failed`
    - Line 48: `2026-02-26T02:39:55.4413125Z ❌ Failed: 1`
    - Line 97: `2026-02-26T02:39:56.7287937Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 62: `2026-02-26T02:39:55.4630439Z   if-no-files-found: warn`
    - Line 97: `2026-02-26T02:39:56.7287937Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 2. Run Biometric Voice E2E / Mock Biometric Tests - wake-word-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:39:51Z
**Completed**: 2026-02-26T02:40:48Z
**Duration**: 57 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425390707/job/64932289334)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:40:45.3632463Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:40:45.3642464Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:40:45.4009365Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:40:45.7875318Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:40:45.4117223Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:40:45.6286285Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:40:45.7875318Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:40:36.4658967Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:40:36.4672719Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:40:37.8038412Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:40:36.6216044Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:40:36.6229402Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:40:37.4489917Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 3. Run Biometric Voice E2E / Mock Biometric Tests - embedding-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:40:13Z
**Completed**: 2026-02-26T02:41:04Z
**Duration**: 51 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425390707/job/64932289337)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:41:01.9645094Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:41:01.9654509Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:41:02.0022793Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:41:02.3965746Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:41:02.0129452Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:41:02.2347474Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:41:02.3965746Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:40:51.4349334Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:40:51.4364237Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:40:54.0638850Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:40:51.7704509Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:40:51.7718332Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:40:52.7049217Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 4. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-voice-drift

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:39:01Z
**Completed**: 2026-02-26T02:39:56Z
**Duration**: 55 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425390707/job/64932289338)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:39:54.2273760Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:39:54.2282827Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:39:54.2650741Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:39:54.6002791Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:39:54.2748288Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:39:54.4699723Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:39:54.6002791Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:39:44.3550210Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:39:44.3563327Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:39:47.1675718Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:39:44.6611641Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:39:44.6625814Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:39:45.4443415Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 5. Run Biometric Voice E2E / Mock Biometric Tests - voice-verification

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:39:34Z
**Completed**: 2026-02-26T02:40:27Z
**Duration**: 53 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425390707/job/64932289340)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:40:25.1274730Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:40:25.1284314Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:40:25.1681404Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:40:25.5611525Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:40:25.1788691Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:40:25.4042678Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:40:25.5611525Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:40:14.8576492Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:40:14.8591520Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:40:16.9082657Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:40:15.1265180Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:40:15.1281190Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:40:16.0328674Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 6. Run Biometric Voice E2E / Mock Biometric Tests - dimension-adaptation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:40:05Z
**Completed**: 2026-02-26T02:40:55Z
**Duration**: 50 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425390707/job/64932289343)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:40:53.2456072Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:40:53.2466296Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:40:53.3012100Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:40:53.7153032Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:40:53.3125039Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:40:53.5506162Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:40:53.7153032Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:40:43.0325114Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:40:43.0340559Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:40:44.8556969Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:40:43.3330871Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:40:43.3346987Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:40:44.2547234Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 7. Run Biometric Voice E2E / Mock Biometric Tests - adaptive-thresholds

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:39:47Z
**Completed**: 2026-02-26T02:40:44Z
**Duration**: 57 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425390707/job/64932289347)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:40:41.6351022Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:40:41.6361454Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:40:41.6755304Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:40:42.0572389Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:40:41.6864259Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:40:41.9006860Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:40:42.0572389Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:40:32.0590707Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:40:32.0604895Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:40:33.3596248Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:40:32.2188001Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:40:32.2202911Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:40:33.0451233Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 8. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-cold-start

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:39:53Z
**Completed**: 2026-02-26T02:40:48Z
**Duration**: 55 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425390707/job/64932289348)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:40:45.6219847Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:40:45.6229067Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:40:45.6610473Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:40:46.0489615Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:40:45.6718685Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:40:45.8898121Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:40:46.0489615Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:40:36.0310558Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:40:36.0325943Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:40:37.9776801Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:40:36.2755588Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:40:36.2769612Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:40:37.1530006Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 9. Run Biometric Voice E2E / Mock Biometric Tests - anti-spoofing

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:40:29Z
**Completed**: 2026-02-26T02:41:11Z
**Duration**: 42 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425390707/job/64932289350)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:41:10.6786829Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:41:10.6795361Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:41:10.7169459Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:41:11.0979188Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:41:10.7273642Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:41:10.9394125Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:41:11.0979188Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:41:01.4471245Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:41:01.4485347Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:41:02.8397391Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:41:01.6321520Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:41:01.6335954Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:41:02.4525820Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 10. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-database-failure

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-26T02:40:20Z
**Completed**: 2026-02-26T02:41:01Z
**Duration**: 41 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425390707/job/64932289355)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:40:59.7754548Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:40:59.7765141Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:40:59.8254844Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 2
  - Sample matches:
    - Line 69: `2026-02-26T02:40:59.8364585Z   name: test-results-biometric-mock-edge-case-database-failure`
    - Line 97: `2026-02-26T02:41:00.2225788Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:40:59.8365457Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:41:00.0604090Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:41:00.2225788Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:40:49.6754412Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:40:49.6770669Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:40:51.3429266Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:40:49.8893845Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:40:49.8908079Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:40:50.7467115Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 11. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-noise

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:40:34Z
**Completed**: 2026-02-26T02:41:19Z
**Duration**: 45 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425390707/job/64932289356)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:41:16.9327606Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:41:16.9336767Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:41:16.9746226Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:41:17.3584436Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:41:16.9853519Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:41:17.2032908Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:41:17.3584436Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:41:06.8493737Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:41:06.8507522Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:41:09.1571215Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:41:07.1522270Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:41:07.1535466Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:41:08.0386018Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 12. Run Biometric Voice E2E / Mock Biometric Tests - stt-transcription

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:40:29Z
**Completed**: 2026-02-26T02:41:07Z
**Duration**: 38 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425390707/job/64932289357)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:41:05.6150074Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:41:05.6158449Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:41:05.6519650Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:41:05.9901380Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:41:05.6623996Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:41:05.8573304Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:41:05.9901380Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:40:57.4125049Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:40:57.4138022Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:40:59.2584376Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:40:57.5652311Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:40:57.5664407Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:40:58.2535255Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 13. Run Biometric Voice E2E / Mock Biometric Tests - profile-quality-assessment

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:40:28Z
**Completed**: 2026-02-26T02:41:22Z
**Duration**: 54 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425390707/job/64932289364)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:41:19.6166680Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:41:19.6175360Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:41:19.6539942Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:41:20.0344653Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:41:19.6645468Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:41:19.8799615Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:41:20.0344653Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:41:08.9309709Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:41:08.9330107Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:41:11.6560402Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:41:09.1885239Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:41:09.1898803Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:41:10.0444714Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 14. Run Biometric Voice E2E / Mock Biometric Tests - end-to-end-flow

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:40:08Z
**Completed**: 2026-02-26T02:40:48Z
**Duration**: 40 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425390707/job/64932289377)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:40:45.9646775Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:40:45.9657906Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:40:46.0053997Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:40:46.3916829Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:40:46.0163008Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:40:46.2341314Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:40:46.3916829Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:40:35.8851902Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:40:35.8867613Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:40:37.6408983Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:40:36.0875009Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:40:36.0890616Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:40:36.9374747Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 15. Run Biometric Voice E2E / Mock Biometric Tests - voice-synthesis-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:39:50Z
**Completed**: 2026-02-26T02:40:32Z
**Duration**: 42 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425390707/job/64932289382)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:40:30.9506018Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:40:30.9514475Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:40:30.9875938Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:40:31.3641342Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:40:30.9981943Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:40:31.2113042Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:40:31.3641342Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:40:22.3711699Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:40:22.3725529Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:40:23.5823751Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:40:22.5300549Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:40:22.5314807Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:40:23.3317725Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 16. Run Biometric Voice E2E / Mock Biometric Tests - replay-attack-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:40:16Z
**Completed**: 2026-02-26T02:41:17Z
**Duration**: 61 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425390707/job/64932289393)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:41:14.8891166Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:41:14.8905339Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:41:14.9839342Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:41:15.3946972Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:41:14.9964772Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:41:15.2357500Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:41:15.3946972Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:41:02.2464834Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:41:02.2487790Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:41:05.1700992Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:41:02.6099210Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:41:02.6121473Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:41:03.7829027Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 17. Run Biometric Voice E2E / Mock Biometric Tests - security-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:39:52Z
**Completed**: 2026-02-26T02:40:42Z
**Duration**: 50 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425390707/job/64932289402)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:40:40.1056506Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:40:40.1065404Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:40:40.1421048Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:40:40.5227664Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:40:40.1526217Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:40:40.3681103Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:40:40.5227664Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:40:30.2768992Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:40:30.2784225Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:40:32.1286014Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:40:30.5816739Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:40:30.5831156Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:40:31.4938311Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 18. Run Biometric Voice E2E / Mock Biometric Tests - performance-baseline

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:40:01Z
**Completed**: 2026-02-26T02:40:50Z
**Duration**: 49 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425390707/job/64932289410)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:40:47.6129376Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:40:47.6138712Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:40:47.6517300Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:40:47.9993225Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:40:47.6618092Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:40:47.8640530Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:40:47.9993225Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:40:38.1117855Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:40:38.1131678Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:40:41.0993471Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:40:38.4244627Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:40:38.4258768Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:40:39.2165491Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 19. Run Unlock Integration E2E / Generate Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-26T02:40:20Z
**Completed**: 2026-02-26T02:40:27Z
**Duration**: 7 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425390707/job/64932523000)

#### Failed Steps

- **Step 4**: Check Test Status

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:40:24.8822471Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 13
  - Sample matches:
    - Line 39: `2026-02-26T02:40:24.6867763Z [36;1mTOTAL_FAILED=0[0m`
    - Line 44: `2026-02-26T02:40:24.6875248Z [36;1m    FAILED=$(jq -r '.summary.failed' "$report")[0m`
    - Line 46: `2026-02-26T02:40:24.6878484Z [36;1m    TOTAL_FAILED=$((TOTAL_FAILED + FAILED))[0m`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 4: `2026-02-26T02:40:24.4268303Z (node:2117) [DEP0005] DeprecationWarning: Buffer() is deprecated due to`
    - Line 5: `2026-02-26T02:40:24.4275294Z (Use `node --trace-deprecation ...` to show where the warning was creat`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 20. Run Biometric Voice E2E / Generate Biometric Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-26T02:41:24Z
**Completed**: 2026-02-26T02:41:32Z
**Duration**: 8 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425390707/job/64932605184)

#### Failed Steps

- **Step 4**: Check Test Status

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:41:28.9272891Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 11
  - Sample matches:
    - Line 32: `2026-02-26T02:41:28.7297066Z [36;1mTOTAL_FAILED=0[0m`
    - Line 37: `2026-02-26T02:41:28.7305153Z [36;1m    FAILED=$(jq -r '.summary.failed' "$report")[0m`
    - Line 39: `2026-02-26T02:41:28.7308719Z [36;1m    TOTAL_FAILED=$((TOTAL_FAILED + FAILED))[0m`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 21. Generate Combined Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-26T02:41:34Z
**Completed**: 2026-02-26T02:41:37Z
**Duration**: 3 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425390707/job/64932616235)

#### Failed Steps

- **Step 2**: Generate Combined Summary

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 61: `2026-02-26T02:41:35.6701103Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 9
  - Sample matches:
    - Line 25: `2026-02-26T02:41:35.6468661Z [36;1mif [ "failure" = "success" ]; then[0m`
    - Line 28: `2026-02-26T02:41:35.6470539Z [36;1m  echo "- ❌ **Unlock Integration E2E:** failure" >> $GITHUB_STEP`
    - Line 32: `2026-02-26T02:41:35.6472480Z [36;1mif [ "failure" = "success" ]; then[0m`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 22. Notify Test Status

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-26T02:41:40Z
**Completed**: 2026-02-26T02:41:43Z
**Duration**: 3 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425390707/job/64932621900)

#### Failed Steps

- **Step 3**: Failure Notification

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line -23: `2026-02-26T02:41:41.3502094Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line -31: `2026-02-26T02:41:41.2218687Z ##[group]Run echo "❌ Unlock tests failed - 'unlock my screen' may be br`
    - Line -30: `2026-02-26T02:41:41.2220286Z [36;1mecho "❌ Unlock tests failed - 'unlock my screen' may be broken!"`
    - Line -25: `2026-02-26T02:41:41.3471065Z ❌ Unlock tests failed - 'unlock my screen' may be broken!`

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

📊 *Report generated on 2026-02-26T02:42:45.665358*
🤖 *JARVIS CI/CD Auto-PR Manager*
