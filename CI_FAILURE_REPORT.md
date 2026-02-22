# CI/CD Failure Analysis Report

## Executive Summary

- **Workflow**: Priority 2 - Biometric Voice Unlock E2E Testing
- **Run Number**: #1
- **Branch**: `main`
- **Commit**: `f0315234349df624ef29f63fdcb8bb7da520a6e6`
- **Status**: ❌ FAILED
- **Timestamp**: 2026-02-22T17:46:42Z
- **Triggered By**: @nandkishorrathodk-art
- **Workflow URL**: [View Run](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140258)

## Failure Overview

Total Failed Jobs: **17**

| # | Job Name | Category | Severity | Duration |
|---|----------|----------|----------|----------|
| 1 | Mock Biometric Tests - adaptive-thresholds | timeout | high | 20s |
| 2 | Mock Biometric Tests - stt-transcription | timeout | high | 23s |
| 3 | Mock Biometric Tests - wake-word-detection | timeout | high | 21s |
| 4 | Mock Biometric Tests - embedding-validation | timeout | high | 21s |
| 5 | Mock Biometric Tests - voice-verification | timeout | high | 21s |
| 6 | Mock Biometric Tests - profile-quality-assessment | timeout | high | 19s |
| 7 | Mock Biometric Tests - edge-case-voice-drift | timeout | high | 26s |
| 8 | Mock Biometric Tests - dimension-adaptation | timeout | high | 21s |
| 9 | Mock Biometric Tests - end-to-end-flow | timeout | high | 24s |
| 10 | Mock Biometric Tests - voice-synthesis-detection | timeout | high | 21s |
| 11 | Mock Biometric Tests - replay-attack-detection | timeout | high | 21s |
| 12 | Mock Biometric Tests - edge-case-noise | timeout | high | 20s |
| 13 | Mock Biometric Tests - security-validation | timeout | high | 27s |
| 14 | Mock Biometric Tests - performance-baseline | timeout | high | 25s |
| 15 | Mock Biometric Tests - anti-spoofing | timeout | high | 22s |
| 16 | Mock Biometric Tests - edge-case-database-failure | test_failure | high | 22s |
| 17 | Mock Biometric Tests - edge-case-cold-start | timeout | high | 23s |

## Detailed Analysis

### 1. Mock Biometric Tests - adaptive-thresholds

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T17:47:33Z
**Completed**: 2026-02-22T17:47:53Z
**Duration**: 20 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140258/job/64454359470)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T17:47:51.7054420Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T17:47:51.7063190Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T17:47:51.7436007Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T17:47:52.1215708Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T17:47:51.7539212Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T17:47:51.9719880Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T17:47:52.1215708Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T17:47:42.5704168Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T17:47:42.5718732Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T17:47:44.3351523Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T17:47:42.8493072Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T17:47:42.8512137Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T17:47:43.7225563Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 2. Mock Biometric Tests - stt-transcription

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T17:47:48Z
**Completed**: 2026-02-22T17:48:11Z
**Duration**: 23 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140258/job/64454359474)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T17:48:09.1327663Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T17:48:09.1336680Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T17:48:09.1690616Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T17:48:09.5390792Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T17:48:09.1793943Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T17:48:09.3896992Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T17:48:09.5390792Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T17:47:59.8604035Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T17:47:59.8617091Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T17:48:01.7145176Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T17:48:00.0993785Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T17:48:00.1007016Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T17:48:00.9349085Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 3. Mock Biometric Tests - wake-word-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T17:58:48Z
**Completed**: 2026-02-22T17:59:09Z
**Duration**: 21 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140258/job/64454359476)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T17:59:07.1042194Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T17:59:07.1051681Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T17:59:07.1461736Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T17:59:07.5225291Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T17:59:07.1571156Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T17:59:07.3695934Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T17:59:07.5225291Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T17:58:57.8032284Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T17:58:57.8046372Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T17:58:59.1172003Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T17:58:57.9516426Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T17:58:57.9530000Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T17:58:58.6450328Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 4. Mock Biometric Tests - embedding-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T17:48:49Z
**Completed**: 2026-02-22T17:49:10Z
**Duration**: 21 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140258/job/64454359478)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T17:49:08.5520249Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T17:49:08.5529336Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T17:49:08.5892694Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T17:49:08.9830587Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T17:49:08.6002158Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T17:49:08.8175309Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T17:49:08.9830587Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T17:48:58.8045903Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T17:48:58.8059209Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T17:49:01.1942769Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T17:48:58.9626361Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T17:48:58.9639121Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T17:48:59.7697721Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 5. Mock Biometric Tests - voice-verification

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T18:00:15Z
**Completed**: 2026-02-22T18:00:36Z
**Duration**: 21 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140258/job/64454359479)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T18:00:34.9899702Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T18:00:34.9909143Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T18:00:35.0289915Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T18:00:35.4072044Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T18:00:35.0402964Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T18:00:35.2556070Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T18:00:35.4072044Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T18:00:26.0671801Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T18:00:26.0685327Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T18:00:27.4614220Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T18:00:26.2221141Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T18:00:26.2241907Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T18:00:27.0304824Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 6. Mock Biometric Tests - profile-quality-assessment

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T18:04:42Z
**Completed**: 2026-02-22T18:05:01Z
**Duration**: 19 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140258/job/64454359480)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T18:05:00.7903381Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T18:05:00.7913479Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T18:05:00.8279629Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T18:05:01.1988148Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T18:05:00.8384315Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T18:05:01.0500054Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T18:05:01.1988148Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T18:04:51.4181693Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T18:04:51.4195219Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T18:04:53.1644534Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T18:04:51.6068791Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T18:04:51.6082217Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T18:04:52.4395086Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 7. Mock Biometric Tests - edge-case-voice-drift

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T18:01:12Z
**Completed**: 2026-02-22T18:01:38Z
**Duration**: 26 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140258/job/64454359483)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T18:01:35.9338887Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T18:01:35.9348279Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T18:01:35.9701268Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T18:01:36.3602089Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T18:01:35.9806201Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T18:01:36.2022975Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T18:01:36.3602089Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T18:01:26.5940030Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T18:01:26.5952952Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T18:01:27.9263260Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T18:01:26.7492793Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T18:01:26.7506061Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T18:01:27.5680198Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 8. Mock Biometric Tests - dimension-adaptation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T18:01:51Z
**Completed**: 2026-02-22T18:02:12Z
**Duration**: 21 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140258/job/64454359484)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T18:02:10.5067730Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T18:02:10.5077037Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T18:02:10.5496353Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T18:02:11.1049404Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T18:02:10.5603672Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T18:02:10.7951044Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T18:02:11.1049404Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T18:02:01.3583120Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T18:02:01.3598279Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T18:02:02.7382697Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T18:02:01.5306955Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T18:02:01.5322033Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T18:02:02.3635651Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 9. Mock Biometric Tests - end-to-end-flow

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T18:00:57Z
**Completed**: 2026-02-22T18:01:21Z
**Duration**: 24 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140258/job/64454359485)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T18:01:18.6360582Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T18:01:18.6368641Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T18:01:18.6720915Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T18:01:19.0415424Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T18:01:18.6829517Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T18:01:18.8920675Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T18:01:19.0415424Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T18:01:09.5326809Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T18:01:09.5340250Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T18:01:11.1998309Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T18:01:09.8410339Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T18:01:09.8423668Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T18:01:10.7232632Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 10. Mock Biometric Tests - voice-synthesis-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T18:08:16Z
**Completed**: 2026-02-22T18:08:37Z
**Duration**: 21 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140258/job/64454359489)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T18:08:35.8062486Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T18:08:35.8071892Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T18:08:35.8508294Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T18:08:36.2312494Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T18:08:35.8619182Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T18:08:36.0776135Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T18:08:36.2312494Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T18:08:26.3584014Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T18:08:26.3598877Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T18:08:27.8096925Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T18:08:26.5535692Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T18:08:26.5550147Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T18:08:27.2778748Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 11. Mock Biometric Tests - replay-attack-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T17:48:59Z
**Completed**: 2026-02-22T17:49:20Z
**Duration**: 21 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140258/job/64454359490)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T17:49:18.4832392Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T17:49:18.4847311Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T17:49:18.5289089Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T17:49:18.9357914Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T17:49:18.5405239Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T17:49:18.7774167Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T17:49:18.9357914Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T17:49:08.6369061Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T17:49:08.6384610Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T17:49:10.4427864Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T17:49:08.9206736Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T17:49:08.9223572Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T17:49:09.8056398Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 12. Mock Biometric Tests - edge-case-noise

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T18:09:36Z
**Completed**: 2026-02-22T18:09:56Z
**Duration**: 20 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140258/job/64454359491)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T18:09:53.9757535Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T18:09:53.9767037Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T18:09:54.0131219Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T18:09:54.3537149Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T18:09:54.0247913Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T18:09:54.2228819Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T18:09:54.3537149Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T18:09:46.5826344Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T18:09:46.5838929Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T18:09:47.6644484Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T18:09:46.7364617Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T18:09:46.7377106Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T18:09:47.4186990Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 13. Mock Biometric Tests - security-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T18:10:36Z
**Completed**: 2026-02-22T18:11:03Z
**Duration**: 27 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140258/job/64454359492)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T18:11:01.2211932Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T18:11:01.2222177Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T18:11:01.2603957Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T18:11:01.6327740Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T18:11:01.2712894Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T18:11:01.4822116Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T18:11:01.6327740Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T18:10:50.7727562Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T18:10:50.7746058Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T18:10:53.4107736Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T18:10:51.1227135Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T18:10:51.1240356Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T18:10:52.0391497Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 14. Mock Biometric Tests - performance-baseline

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T18:11:52Z
**Completed**: 2026-02-22T18:12:17Z
**Duration**: 25 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140258/job/64454359498)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T18:12:15.7834032Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T18:12:15.7843401Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T18:12:15.8216858Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T18:12:16.1937289Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T18:12:15.8324260Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T18:12:16.0441851Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T18:12:16.1937289Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T18:12:06.2507424Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T18:12:06.2520135Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T18:12:07.8899733Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T18:12:06.5095506Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T18:12:06.5108207Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T18:12:07.3360450Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 15. Mock Biometric Tests - anti-spoofing

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T18:13:42Z
**Completed**: 2026-02-22T18:14:04Z
**Duration**: 22 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140258/job/64454359500)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T18:14:01.9546201Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T18:14:01.9555721Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T18:14:01.9939016Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T18:14:02.3625524Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T18:14:02.0045995Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T18:14:02.2151946Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T18:14:02.3625524Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T18:13:52.8146778Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T18:13:52.8160563Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T18:13:54.6133876Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T18:13:53.0361662Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T18:13:53.0374813Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T18:13:53.8671845Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 16. Mock Biometric Tests - edge-case-database-failure

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-22T18:12:40Z
**Completed**: 2026-02-22T18:13:02Z
**Duration**: 22 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140258/job/64454359502)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T18:13:00.3168544Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T18:13:00.3177547Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T18:13:00.3563412Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 2
  - Sample matches:
    - Line 69: `2026-02-22T18:13:00.3682895Z   name: test-results-biometric-mock-edge-case-database-failure`
    - Line 97: `2026-02-22T18:13:00.7481730Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T18:13:00.3683907Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T18:13:00.5978438Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T18:13:00.7481730Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T18:12:50.5239231Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T18:12:50.5253700Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T18:12:52.0614324Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T18:12:50.7043482Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T18:12:50.7056738Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T18:12:51.4236710Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 17. Mock Biometric Tests - edge-case-cold-start

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-22T18:05:32Z
**Completed**: 2026-02-22T18:05:55Z
**Duration**: 23 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22282140258/job/64454359504)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-22T18:05:52.9630802Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-22T18:05:52.9639355Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-22T18:05:52.9990395Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-22T18:05:53.3851811Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-22T18:05:53.0096524Z   if-no-files-found: warn`
    - Line 87: `2026-02-22T18:05:53.2341241Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-22T18:05:53.3851811Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-22T18:05:44.1788326Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-22T18:05:44.1802213Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-22T18:05:45.4035555Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-22T18:05:44.3362682Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-22T18:05:44.3376093Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-22T18:05:45.1436531Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

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

📊 *Report generated on 2026-02-22T19:53:34.457214*
🤖 *JARVIS CI/CD Auto-PR Manager*
