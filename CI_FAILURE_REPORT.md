# CI/CD Failure Analysis Report

## Executive Summary

- **Workflow**: Priority 2 - Biometric Voice Unlock E2E Testing
- **Run Number**: #13
- **Branch**: `main`
- **Commit**: `41b1fb38c756c5ecab9f500d4f623f76a57269d1`
- **Status**: ❌ FAILED
- **Timestamp**: 2026-02-28T17:41:17Z
- **Triggered By**: @nandkishorrathodk-art
- **Workflow URL**: [View Run](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682786)

## Failure Overview

Total Failed Jobs: **17**

| # | Job Name | Category | Severity | Duration |
|---|----------|----------|----------|----------|
| 1 | Mock Biometric Tests - wake-word-detection | timeout | high | 56s |
| 2 | Mock Biometric Tests - edge-case-database-failure | test_failure | high | 38s |
| 3 | Mock Biometric Tests - voice-verification | timeout | high | 53s |
| 4 | Mock Biometric Tests - embedding-validation | timeout | high | 50s |
| 5 | Mock Biometric Tests - profile-quality-assessment | timeout | high | 41s |
| 6 | Mock Biometric Tests - edge-case-voice-drift | timeout | high | 54s |
| 7 | Mock Biometric Tests - dimension-adaptation | timeout | high | 40s |
| 8 | Mock Biometric Tests - stt-transcription | timeout | high | 46s |
| 9 | Mock Biometric Tests - edge-case-cold-start | timeout | high | 58s |
| 10 | Mock Biometric Tests - adaptive-thresholds | timeout | high | 41s |
| 11 | Mock Biometric Tests - anti-spoofing | timeout | high | 40s |
| 12 | Mock Biometric Tests - replay-attack-detection | timeout | high | 45s |
| 13 | Mock Biometric Tests - voice-synthesis-detection | timeout | high | 41s |
| 14 | Mock Biometric Tests - edge-case-noise | timeout | high | 47s |
| 15 | Mock Biometric Tests - end-to-end-flow | timeout | high | 50s |
| 16 | Mock Biometric Tests - security-validation | timeout | high | 48s |
| 17 | Mock Biometric Tests - performance-baseline | timeout | high | 57s |

## Detailed Analysis

### 1. Mock Biometric Tests - wake-word-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-28T17:41:33Z
**Completed**: 2026-02-28T17:42:29Z
**Duration**: 56 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682786/job/65257241461)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-28T17:42:27.2941439Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-28T17:42:27.2950833Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-28T17:42:27.3327094Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T17:42:27.7113240Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T17:42:27.3434658Z   if-no-files-found: warn`
    - Line 87: `2026-02-28T17:42:27.5574684Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-28T17:42:27.7113240Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-28T17:42:17.5643348Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-28T17:42:17.5656419Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-28T17:42:18.7395409Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-28T17:42:17.7064409Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-28T17:42:17.7077818Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-28T17:42:18.5046493Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 2. Mock Biometric Tests - edge-case-database-failure

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-28T17:41:30Z
**Completed**: 2026-02-28T17:42:08Z
**Duration**: 38 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682786/job/65257241464)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-28T17:42:07.1911183Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-28T17:42:07.1919477Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-28T17:42:07.2306852Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 2
  - Sample matches:
    - Line 69: `2026-02-28T17:42:07.2411942Z   name: test-results-biometric-mock-edge-case-database-failure`
    - Line 97: `2026-02-28T17:42:07.6181907Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T17:42:07.2412761Z   if-no-files-found: warn`
    - Line 87: `2026-02-28T17:42:07.4619145Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-28T17:42:07.6181907Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-28T17:41:57.9890561Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-28T17:41:57.9903920Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-28T17:41:59.3347650Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-28T17:41:58.1618794Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-28T17:41:58.1631726Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-28T17:41:58.9752561Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 3. Mock Biometric Tests - voice-verification

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-28T17:42:15Z
**Completed**: 2026-02-28T17:43:08Z
**Duration**: 53 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682786/job/65257241466)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-28T17:43:06.4416894Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-28T17:43:06.4426333Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-28T17:43:06.4783987Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T17:43:06.8591576Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T17:43:06.4888143Z   if-no-files-found: warn`
    - Line 87: `2026-02-28T17:43:06.7043793Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-28T17:43:06.8591576Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-28T17:42:55.5529644Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-28T17:42:55.5544105Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-28T17:42:58.9285766Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-28T17:42:55.8615373Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-28T17:42:55.8629765Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-28T17:42:56.7692711Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 4. Mock Biometric Tests - embedding-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-28T17:41:36Z
**Completed**: 2026-02-28T17:42:26Z
**Duration**: 50 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682786/job/65257241467)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-28T17:42:24.0057791Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-28T17:42:24.0066647Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-28T17:42:24.0465030Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T17:42:24.3958623Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T17:42:24.0562761Z   if-no-files-found: warn`
    - Line 87: `2026-02-28T17:42:24.2557482Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-28T17:42:24.3958623Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-28T17:42:15.4758098Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-28T17:42:15.4772356Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-28T17:42:17.1992181Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-28T17:42:15.7051726Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-28T17:42:15.7068791Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-28T17:42:16.4729199Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 5. Mock Biometric Tests - profile-quality-assessment

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-28T17:41:50Z
**Completed**: 2026-02-28T17:42:31Z
**Duration**: 41 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682786/job/65257241470)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-28T17:42:29.3346306Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-28T17:42:29.3354944Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-28T17:42:29.3762318Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T17:42:29.7762358Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T17:42:29.3869390Z   if-no-files-found: warn`
    - Line 87: `2026-02-28T17:42:29.6188173Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-28T17:42:29.7762358Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-28T17:42:20.3775329Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-28T17:42:20.3789520Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-28T17:42:21.7008075Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-28T17:42:20.5429633Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-28T17:42:20.5443551Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-28T17:42:21.3648779Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 6. Mock Biometric Tests - edge-case-voice-drift

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-28T17:41:56Z
**Completed**: 2026-02-28T17:42:50Z
**Duration**: 54 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682786/job/65257241471)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-28T17:42:47.5766513Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-28T17:42:47.5775639Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-28T17:42:47.6129332Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T17:42:47.9968549Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T17:42:47.6234241Z   if-no-files-found: warn`
    - Line 87: `2026-02-28T17:42:47.8393662Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-28T17:42:47.9968549Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-28T17:42:38.9197132Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-28T17:42:38.9210910Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-28T17:42:40.1423812Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-28T17:42:39.0767117Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-28T17:42:39.0781281Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-28T17:42:39.9002073Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 7. Mock Biometric Tests - dimension-adaptation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-28T17:43:05Z
**Completed**: 2026-02-28T17:43:45Z
**Duration**: 40 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682786/job/65257241473)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-28T17:43:44.1468574Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-28T17:43:44.1478291Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-28T17:43:44.1854839Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T17:43:44.5727471Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T17:43:44.1961146Z   if-no-files-found: warn`
    - Line 87: `2026-02-28T17:43:44.4164530Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-28T17:43:44.5727471Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-28T17:43:34.5581371Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-28T17:43:34.5594791Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-28T17:43:35.9372901Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-28T17:43:34.7145762Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-28T17:43:34.7160530Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-28T17:43:35.5499862Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 8. Mock Biometric Tests - stt-transcription

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-28T17:43:21Z
**Completed**: 2026-02-28T17:44:07Z
**Duration**: 46 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682786/job/65257241476)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-28T17:44:05.8718927Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-28T17:44:05.8728839Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-28T17:44:05.9095415Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T17:44:06.2924059Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T17:44:05.9204842Z   if-no-files-found: warn`
    - Line 87: `2026-02-28T17:44:06.1364840Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-28T17:44:06.2924059Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-28T17:43:55.4552445Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-28T17:43:55.4566071Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-28T17:43:57.7317339Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-28T17:43:55.7229904Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-28T17:43:55.7243108Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-28T17:43:56.5975998Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 9. Mock Biometric Tests - edge-case-cold-start

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-28T17:41:38Z
**Completed**: 2026-02-28T17:42:36Z
**Duration**: 58 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682786/job/65257241477)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-28T17:42:33.6479020Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-28T17:42:33.6487939Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-28T17:42:33.6850082Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T17:42:34.0661657Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T17:42:33.6954806Z   if-no-files-found: warn`
    - Line 87: `2026-02-28T17:42:33.9093626Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-28T17:42:34.0661657Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-28T17:42:22.9374062Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-28T17:42:22.9388016Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-28T17:42:25.3739423Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-28T17:42:23.1710043Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-28T17:42:23.1724270Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-28T17:42:24.0255618Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 10. Mock Biometric Tests - adaptive-thresholds

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-28T17:41:35Z
**Completed**: 2026-02-28T17:42:16Z
**Duration**: 41 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682786/job/65257241479)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-28T17:42:14.4235971Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-28T17:42:14.4245029Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-28T17:42:14.4609633Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T17:42:14.8555806Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T17:42:14.4717942Z   if-no-files-found: warn`
    - Line 87: `2026-02-28T17:42:14.7007912Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-28T17:42:14.8555806Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-28T17:42:04.7072123Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-28T17:42:04.7086178Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-28T17:42:06.3505861Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-28T17:42:04.9060138Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-28T17:42:04.9074456Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-28T17:42:05.7436736Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 11. Mock Biometric Tests - anti-spoofing

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-28T17:41:33Z
**Completed**: 2026-02-28T17:42:13Z
**Duration**: 40 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682786/job/65257241483)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-28T17:42:11.4485183Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-28T17:42:11.4493972Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-28T17:42:11.4834636Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T17:42:11.8628543Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T17:42:11.4942059Z   if-no-files-found: warn`
    - Line 87: `2026-02-28T17:42:11.7084260Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-28T17:42:11.8628543Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-28T17:42:02.4301206Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-28T17:42:02.4315219Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-28T17:42:04.0229791Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-28T17:42:02.6191819Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-28T17:42:02.6205732Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-28T17:42:03.4496061Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 12. Mock Biometric Tests - replay-attack-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-28T17:43:31Z
**Completed**: 2026-02-28T17:44:16Z
**Duration**: 45 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682786/job/65257241490)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-28T17:44:14.4180016Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-28T17:44:14.4188702Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-28T17:44:14.4552659Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T17:44:14.8347587Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T17:44:14.4661035Z   if-no-files-found: warn`
    - Line 87: `2026-02-28T17:44:14.6792751Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-28T17:44:14.8347587Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-28T17:44:04.4413998Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-28T17:44:04.4429990Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-28T17:44:06.4974387Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-28T17:44:04.6115038Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-28T17:44:04.6130086Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-28T17:44:05.4556845Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 13. Mock Biometric Tests - voice-synthesis-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-28T17:42:38Z
**Completed**: 2026-02-28T17:43:19Z
**Duration**: 41 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682786/job/65257241491)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-28T17:43:17.5811901Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-28T17:43:17.5821129Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-28T17:43:17.6190375Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T17:43:18.0182641Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T17:43:17.6299795Z   if-no-files-found: warn`
    - Line 87: `2026-02-28T17:43:17.8597435Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-28T17:43:18.0182641Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-28T17:43:08.6269253Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-28T17:43:08.6283264Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-28T17:43:09.9055911Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-28T17:43:08.7984074Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-28T17:43:08.7997768Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-28T17:43:09.6343495Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 14. Mock Biometric Tests - edge-case-noise

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-28T17:43:23Z
**Completed**: 2026-02-28T17:44:10Z
**Duration**: 47 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682786/job/65257241493)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-28T17:44:09.0126556Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-28T17:44:09.0135759Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-28T17:44:09.0482435Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T17:44:09.4316077Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T17:44:09.0586835Z   if-no-files-found: warn`
    - Line 87: `2026-02-28T17:44:09.2790486Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-28T17:44:09.4316077Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-28T17:43:59.3189575Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-28T17:43:59.3203500Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-28T17:44:01.6712248Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-28T17:43:59.5962039Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-28T17:43:59.5975801Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-28T17:44:00.4924256Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 15. Mock Biometric Tests - end-to-end-flow

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-28T17:41:36Z
**Completed**: 2026-02-28T17:42:26Z
**Duration**: 50 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682786/job/65257241494)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-28T17:42:23.8295535Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-28T17:42:23.8304597Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-28T17:42:23.8651020Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T17:42:24.2419023Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T17:42:23.8755677Z   if-no-files-found: warn`
    - Line 87: `2026-02-28T17:42:24.0869818Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-28T17:42:24.2419023Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-28T17:42:13.6229942Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-28T17:42:13.6243914Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-28T17:42:15.9119454Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-28T17:42:13.9075485Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-28T17:42:13.9088320Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-28T17:42:14.7856466Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 16. Mock Biometric Tests - security-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-28T17:43:16Z
**Completed**: 2026-02-28T17:44:04Z
**Duration**: 48 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682786/job/65257241496)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-28T17:44:02.2721776Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-28T17:44:02.2732303Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-28T17:44:02.3119544Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T17:44:02.7033036Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T17:44:02.3225232Z   if-no-files-found: warn`
    - Line 87: `2026-02-28T17:44:02.5413686Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-28T17:44:02.7033036Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-28T17:43:52.5689551Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-28T17:43:52.5702979Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-28T17:43:54.9168499Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-28T17:43:52.8453660Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-28T17:43:52.8467041Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-28T17:43:53.7403564Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 17. Mock Biometric Tests - performance-baseline

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-28T17:43:14Z
**Completed**: 2026-02-28T17:44:11Z
**Duration**: 57 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22525682786/job/65257241498)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-28T17:44:08.8279390Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-28T17:44:08.8289320Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-28T17:44:08.8650892Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-28T17:44:09.2584481Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-28T17:44:08.8759790Z   if-no-files-found: warn`
    - Line 87: `2026-02-28T17:44:09.1008693Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-28T17:44:09.2584481Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-28T17:43:59.4878536Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-28T17:43:59.4894528Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-28T17:44:00.8041939Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-28T17:43:59.6508728Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-28T17:43:59.6524563Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-28T17:44:00.4950642Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

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

📊 *Report generated on 2026-02-28T17:45:42.739891*
🤖 *Ironcliw CI/CD Auto-PR Manager*
