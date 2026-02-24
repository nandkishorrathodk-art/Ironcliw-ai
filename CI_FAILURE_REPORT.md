# CI/CD Failure Analysis Report

## Executive Summary

- **Workflow**: Complete Unlock Test Suite (Master)
- **Run Number**: #5
- **Branch**: `main`
- **Commit**: `a77933aa5be857b416eefd479bb682c798e0a972`
- **Status**: ❌ FAILED
- **Timestamp**: 2026-02-24T06:17:33Z
- **Triggered By**: @nandkishorrathodk-art
- **Workflow URL**: [View Run](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339125502)

## Failure Overview

Total Failed Jobs: **22**

| # | Job Name | Category | Severity | Duration |
|---|----------|----------|----------|----------|
| 1 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-voice-drift | timeout | high | 57s |
| 2 | Run Biometric Voice E2E / Mock Biometric Tests - stt-transcription | timeout | high | 60s |
| 3 | Run Biometric Voice E2E / Mock Biometric Tests - profile-quality-assessment | timeout | high | 53s |
| 4 | Run Biometric Voice E2E / Mock Biometric Tests - embedding-validation | timeout | high | 43s |
| 5 | Run Biometric Voice E2E / Mock Biometric Tests - voice-verification | timeout | high | 55s |
| 6 | Run Biometric Voice E2E / Mock Biometric Tests - replay-attack-detection | timeout | high | 46s |
| 7 | Run Biometric Voice E2E / Mock Biometric Tests - voice-synthesis-detection | timeout | high | 42s |
| 8 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-database-failure | test_failure | high | 47s |
| 9 | Run Biometric Voice E2E / Mock Biometric Tests - adaptive-thresholds | timeout | high | 49s |
| 10 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-cold-start | timeout | high | 56s |
| 11 | Run Biometric Voice E2E / Mock Biometric Tests - wake-word-detection | timeout | high | 41s |
| 12 | Run Biometric Voice E2E / Mock Biometric Tests - anti-spoofing | timeout | high | 56s |
| 13 | Run Biometric Voice E2E / Mock Biometric Tests - dimension-adaptation | timeout | high | 56s |
| 14 | Run Biometric Voice E2E / Mock Biometric Tests - end-to-end-flow | timeout | high | 47s |
| 15 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-noise | timeout | high | 54s |
| 16 | Run Biometric Voice E2E / Mock Biometric Tests - performance-baseline | timeout | high | 48s |
| 17 | Run Biometric Voice E2E / Mock Biometric Tests - security-validation | timeout | high | 46s |
| 18 | Run Unlock Integration E2E / Mock Tests - security-checks | test_failure | high | 37s |
| 19 | Run Biometric Voice E2E / Generate Biometric Test Summary | test_failure | high | 3s |
| 20 | Run Unlock Integration E2E / Generate Test Summary | test_failure | high | 7s |
| 21 | Generate Combined Test Summary | test_failure | high | 4s |
| 22 | Notify Test Status | test_failure | high | 4s |

## Detailed Analysis

### 1. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-voice-drift

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:20:09Z
**Completed**: 2026-02-24T06:21:06Z
**Duration**: 57 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339125502/job/64638311376)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:21:02.7013403Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:21:02.7022906Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:21:02.7445260Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:21:03.1206407Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:21:02.7549647Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:21:02.9694882Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:21:03.1206407Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:20:52.9454811Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:20:52.9468509Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:20:54.7146240Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:20:53.1010527Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:20:53.1025145Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:20:54.3807029Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 2. Run Biometric Voice E2E / Mock Biometric Tests - stt-transcription

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:20:58Z
**Completed**: 2026-02-24T06:21:58Z
**Duration**: 60 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339125502/job/64638311377)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:21:55.7455231Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:21:55.7464890Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:21:55.7939031Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:21:56.1788586Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:21:55.8052715Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:21:56.0283572Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:21:56.1788586Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:21:46.4147308Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:21:46.4162882Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:21:48.0371141Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:21:46.5745401Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:21:46.5762858Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:21:47.6669371Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 3. Run Biometric Voice E2E / Mock Biometric Tests - profile-quality-assessment

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:20:41Z
**Completed**: 2026-02-24T06:21:34Z
**Duration**: 53 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339125502/job/64638311378)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:21:32.4702728Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:21:32.4711949Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:21:32.5083090Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:21:32.8810370Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:21:32.5189752Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:21:32.7304867Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:21:32.8810370Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:21:23.4170094Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:21:23.4183169Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:21:24.7251102Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:21:23.5754461Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:21:23.5768390Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:21:24.4022043Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 4. Run Biometric Voice E2E / Mock Biometric Tests - embedding-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:21:12Z
**Completed**: 2026-02-24T06:21:55Z
**Duration**: 43 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339125502/job/64638311380)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:21:53.7399005Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:21:53.7409595Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:21:53.7799734Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:21:54.1541056Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:21:53.7902974Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:21:54.0027651Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:21:54.1541056Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:21:43.8147969Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:21:43.8162000Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:21:45.6156273Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:21:44.0105209Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:21:44.0118835Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:21:45.0300216Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 5. Run Biometric Voice E2E / Mock Biometric Tests - voice-verification

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:20:39Z
**Completed**: 2026-02-24T06:21:34Z
**Duration**: 55 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339125502/job/64638311385)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:21:31.6704669Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:21:31.6714703Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:21:31.7113755Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:21:32.0867199Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:21:31.7221339Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:21:31.9346618Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:21:32.0867199Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:21:22.4422253Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:21:22.4435987Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:21:24.1857782Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:21:22.5907558Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:21:22.5921442Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:21:23.4512277Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 6. Run Biometric Voice E2E / Mock Biometric Tests - replay-attack-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:20:10Z
**Completed**: 2026-02-24T06:20:56Z
**Duration**: 46 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339125502/job/64638311386)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:20:54.4961603Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:20:54.4971110Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:20:54.5416988Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:20:54.9395156Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:20:54.5526643Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:20:54.7828994Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:20:54.9395156Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:20:45.1479666Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:20:45.1493277Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:20:46.4177351Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:20:45.3118387Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:20:45.3133979Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:20:46.1417573Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 7. Run Biometric Voice E2E / Mock Biometric Tests - voice-synthesis-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:19:56Z
**Completed**: 2026-02-24T06:20:38Z
**Duration**: 42 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339125502/job/64638311389)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:20:36.2738427Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:20:36.2748208Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:20:36.3132598Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:20:36.6875123Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:20:36.3236869Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:20:36.5374594Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:20:36.6875123Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:20:27.1220522Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:20:27.1235379Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:20:28.7357909Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:20:27.3122957Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:20:27.3135696Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:20:28.1648595Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 8. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-database-failure

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-24T06:20:34Z
**Completed**: 2026-02-24T06:21:21Z
**Duration**: 47 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339125502/job/64638311391)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:21:19.6104636Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:21:19.6114802Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:21:19.6584548Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 2
  - Sample matches:
    - Line 69: `2026-02-24T06:21:19.6691409Z   name: test-results-biometric-mock-edge-case-database-failure`
    - Line 97: `2026-02-24T06:21:20.0484358Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:21:19.6692236Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:21:19.8948821Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:21:20.0484358Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:21:08.4441516Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:21:08.4456455Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:21:11.8202426Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:21:08.6030411Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:21:08.6043308Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:21:09.6582551Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 9. Run Biometric Voice E2E / Mock Biometric Tests - adaptive-thresholds

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:20:59Z
**Completed**: 2026-02-24T06:21:48Z
**Duration**: 49 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339125502/job/64638311394)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:21:45.8417931Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:21:45.8427424Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:21:45.8851030Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:21:46.2724110Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:21:45.8956459Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:21:46.1156500Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:21:46.2724110Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:21:35.9491357Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:21:35.9505065Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:21:38.2008321Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:21:36.2400955Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:21:36.2415330Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:21:37.1458466Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 10. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-cold-start

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:20:38Z
**Completed**: 2026-02-24T06:21:34Z
**Duration**: 56 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339125502/job/64638311396)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:21:31.6770308Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:21:31.6778537Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:21:31.7130725Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:21:32.0881647Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:21:31.7238435Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:21:31.9393429Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:21:32.0881647Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:21:22.8507838Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:21:22.8521391Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:21:24.2294462Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:21:23.0250367Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:21:23.0263768Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:21:23.8624107Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 11. Run Biometric Voice E2E / Mock Biometric Tests - wake-word-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:20:08Z
**Completed**: 2026-02-24T06:20:49Z
**Duration**: 41 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339125502/job/64638311397)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:20:48.5986799Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:20:48.5995540Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:20:48.6333187Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:20:49.0085826Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:20:48.6441573Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:20:48.8568187Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:20:49.0085826Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:20:39.0714951Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:20:39.0727841Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:20:40.8098763Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:20:39.2227874Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:20:39.2241533Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:20:40.5401607Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 12. Run Biometric Voice E2E / Mock Biometric Tests - anti-spoofing

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:20:18Z
**Completed**: 2026-02-24T06:21:14Z
**Duration**: 56 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339125502/job/64638311398)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:21:11.2041141Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:21:11.2050062Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:21:11.2401911Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:21:11.6182633Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:21:11.2506583Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:21:11.4621252Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:21:11.6182633Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:21:01.4555832Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:21:01.4574516Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:21:03.1077655Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:21:01.6396698Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:21:01.6412639Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:21:02.5020560Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 13. Run Biometric Voice E2E / Mock Biometric Tests - dimension-adaptation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:20:22Z
**Completed**: 2026-02-24T06:21:18Z
**Duration**: 56 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339125502/job/64638311403)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:21:16.5150739Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:21:16.5159559Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:21:16.5526491Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:21:16.9394477Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:21:16.5630833Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:21:16.7880431Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:21:16.9394477Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:21:02.0918895Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:21:02.0933938Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:21:08.6540919Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:21:02.7157549Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:21:02.7171818Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:21:03.5253658Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 14. Run Biometric Voice E2E / Mock Biometric Tests - end-to-end-flow

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:21:15Z
**Completed**: 2026-02-24T06:22:02Z
**Duration**: 47 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339125502/job/64638311407)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:22:00.6924873Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:22:00.6934959Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:22:00.7379114Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:22:01.1243546Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:22:00.7488294Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:22:00.9687549Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:22:01.1243546Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:21:51.0287978Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:21:51.0303291Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:21:52.5194906Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:21:51.2688738Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:21:51.2704509Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:21:52.1535792Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 15. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-noise

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:20:23Z
**Completed**: 2026-02-24T06:21:17Z
**Duration**: 54 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339125502/job/64638311409)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:21:15.3933936Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:21:15.3944338Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:21:15.4390630Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:21:15.8255240Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:21:15.4498124Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:21:15.6700942Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:21:15.8255240Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:21:05.4709590Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:21:05.4723106Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:21:07.1093974Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:21:05.6296095Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:21:05.6311584Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:21:06.4608936Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 16. Run Biometric Voice E2E / Mock Biometric Tests - performance-baseline

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:20:18Z
**Completed**: 2026-02-24T06:21:06Z
**Duration**: 48 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339125502/job/64638311411)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:21:04.7816029Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:21:04.7826559Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:21:04.8346536Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:21:05.2300663Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:21:04.8456353Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:21:05.0683072Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:21:05.2300663Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:20:53.5084025Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:20:53.5099307Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:20:56.3057566Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:20:53.6766922Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:20:53.6783907Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:20:54.5115277Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 17. Run Biometric Voice E2E / Mock Biometric Tests - security-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:20:24Z
**Completed**: 2026-02-24T06:21:10Z
**Duration**: 46 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339125502/job/64638311416)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:21:08.2226473Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:21:08.2235853Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:21:08.2633463Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:21:08.6398619Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:21:08.2739010Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:21:08.4873054Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:21:08.6398619Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:20:58.1070855Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:20:58.1084201Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:21:00.4138764Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:20:58.3403133Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:20:58.3417512Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:20:59.1977013Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 18. Run Unlock Integration E2E / Mock Tests - security-checks

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-24T06:21:17Z
**Completed**: 2026-02-24T06:21:54Z
**Duration**: 37 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339125502/job/64638317578)

#### Failed Steps

- **Step 6**: Run Mock Tests

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 2
  - Sample matches:
    - Line 39: `2026-02-24T06:21:51.6555849Z 2026-02-24 06:21:51,655 - __main__ - ERROR - ❌ 1 test(s) failed`
    - Line 56: `2026-02-24T06:21:51.6679649Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 39: `2026-02-24T06:21:51.6555849Z 2026-02-24 06:21:51,655 - __main__ - ERROR - ❌ 1 test(s) failed`
    - Line 48: `2026-02-24T06:21:51.6563391Z ❌ Failed: 1`
    - Line 97: `2026-02-24T06:21:52.4355482Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 62: `2026-02-24T06:21:51.6754927Z   if-no-files-found: warn`
    - Line 97: `2026-02-24T06:21:52.4355482Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 19. Run Biometric Voice E2E / Generate Biometric Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-24T06:22:04Z
**Completed**: 2026-02-24T06:22:07Z
**Duration**: 3 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339125502/job/64638643605)

#### Failed Steps

- **Step 4**: Check Test Status

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:22:06.4905135Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 11
  - Sample matches:
    - Line 32: `2026-02-24T06:22:06.3086347Z [36;1mTOTAL_FAILED=0[0m`
    - Line 37: `2026-02-24T06:22:06.3089742Z [36;1m    FAILED=$(jq -r '.summary.failed' "$report")[0m`
    - Line 39: `2026-02-24T06:22:06.3091283Z [36;1m    TOTAL_FAILED=$((TOTAL_FAILED + FAILED))[0m`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 20. Run Unlock Integration E2E / Generate Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-24T06:22:09Z
**Completed**: 2026-02-24T06:22:16Z
**Duration**: 7 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339125502/job/64638649158)

#### Failed Steps

- **Step 4**: Check Test Status

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 2
  - Sample matches:
    - Line 3: `2026-02-24T06:22:12.8934836Z Starting download of artifact to: /home/runner/work/Ironcliw-ai/Ironcli`
    - Line 97: `2026-02-24T06:22:13.4266100Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 13
  - Sample matches:
    - Line 39: `2026-02-24T06:22:13.2400446Z [36;1mTOTAL_FAILED=0[0m`
    - Line 44: `2026-02-24T06:22:13.2407642Z [36;1m    FAILED=$(jq -r '.summary.failed' "$report")[0m`
    - Line 46: `2026-02-24T06:22:13.2410928Z [36;1m    TOTAL_FAILED=$((TOTAL_FAILED + FAILED))[0m`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 4: `2026-02-24T06:22:13.0133442Z (node:2108) [DEP0005] DeprecationWarning: Buffer() is deprecated due to`
    - Line 5: `2026-02-24T06:22:13.0135694Z (Use `node --trace-deprecation ...` to show where the warning was creat`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 21. Generate Combined Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-24T06:22:19Z
**Completed**: 2026-02-24T06:22:23Z
**Duration**: 4 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339125502/job/64638661836)

#### Failed Steps

- **Step 2**: Generate Combined Summary

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 61: `2026-02-24T06:22:21.6316297Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 9
  - Sample matches:
    - Line 25: `2026-02-24T06:22:21.6085929Z [36;1mif [ "failure" = "success" ]; then[0m`
    - Line 28: `2026-02-24T06:22:21.6090486Z [36;1m  echo "- ❌ **Unlock Integration E2E:** failure" >> $GITHUB_STEP`
    - Line 32: `2026-02-24T06:22:21.6094809Z [36;1mif [ "failure" = "success" ]; then[0m`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 22. Notify Test Status

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-24T06:22:26Z
**Completed**: 2026-02-24T06:22:30Z
**Duration**: 4 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339125502/job/64638670669)

#### Failed Steps

- **Step 3**: Failure Notification

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line -23: `2026-02-24T06:22:27.4541873Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line -31: `2026-02-24T06:22:27.3679327Z ##[group]Run echo "❌ Unlock tests failed - 'unlock my screen' may be br`
    - Line -30: `2026-02-24T06:22:27.3680430Z [36;1mecho "❌ Unlock tests failed - 'unlock my screen' may be broken!"`
    - Line -25: `2026-02-24T06:22:27.4522823Z ❌ Unlock tests failed - 'unlock my screen' may be broken!`

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

📊 *Report generated on 2026-02-24T06:23:50.310446*
🤖 *JARVIS CI/CD Auto-PR Manager*
