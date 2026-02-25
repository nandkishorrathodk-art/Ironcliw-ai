# CI/CD Failure Analysis Report

## Executive Summary

- **Workflow**: Complete Unlock Test Suite (Master)
- **Run Number**: #9
- **Branch**: `main`
- **Commit**: `5dbd1853f217393961719f27e725f3791461618e`
- **Status**: ❌ FAILED
- **Timestamp**: 2026-02-25T12:13:07Z
- **Triggered By**: @nandkishorrathodk-art
- **Workflow URL**: [View Run](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22396340652)

## Failure Overview

Total Failed Jobs: **22**

| # | Job Name | Category | Severity | Duration |
|---|----------|----------|----------|----------|
| 1 | Run Unlock Integration E2E / Mock Tests - security-checks | test_failure | high | 37s |
| 2 | Run Biometric Voice E2E / Mock Biometric Tests - voice-verification | timeout | high | 51s |
| 3 | Run Biometric Voice E2E / Mock Biometric Tests - embedding-validation | timeout | high | 53s |
| 4 | Run Biometric Voice E2E / Mock Biometric Tests - profile-quality-assessment | timeout | high | 41s |
| 5 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-voice-drift | timeout | high | 42s |
| 6 | Run Biometric Voice E2E / Mock Biometric Tests - adaptive-thresholds | timeout | high | 40s |
| 7 | Run Biometric Voice E2E / Mock Biometric Tests - anti-spoofing | timeout | high | 53s |
| 8 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-cold-start | timeout | high | 52s |
| 9 | Run Biometric Voice E2E / Mock Biometric Tests - dimension-adaptation | timeout | high | 51s |
| 10 | Run Biometric Voice E2E / Mock Biometric Tests - security-validation | timeout | high | 49s |
| 11 | Run Biometric Voice E2E / Mock Biometric Tests - end-to-end-flow | timeout | high | 55s |
| 12 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-database-failure | test_failure | high | 44s |
| 13 | Run Biometric Voice E2E / Mock Biometric Tests - replay-attack-detection | timeout | high | 50s |
| 14 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-noise | timeout | high | 43s |
| 15 | Run Biometric Voice E2E / Mock Biometric Tests - wake-word-detection | timeout | high | 54s |
| 16 | Run Biometric Voice E2E / Mock Biometric Tests - stt-transcription | timeout | high | 47s |
| 17 | Run Biometric Voice E2E / Mock Biometric Tests - voice-synthesis-detection | timeout | high | 43s |
| 18 | Run Biometric Voice E2E / Mock Biometric Tests - performance-baseline | timeout | high | 44s |
| 19 | Run Unlock Integration E2E / Generate Test Summary | test_failure | high | 4s |
| 20 | Run Biometric Voice E2E / Generate Biometric Test Summary | test_failure | high | 6s |
| 21 | Generate Combined Test Summary | test_failure | high | 2s |
| 22 | Notify Test Status | test_failure | high | 4s |

## Detailed Analysis

### 1. Run Unlock Integration E2E / Mock Tests - security-checks

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-25T12:14:41Z
**Completed**: 2026-02-25T12:15:18Z
**Duration**: 37 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22396340652/job/64830885940)

#### Failed Steps

- **Step 6**: Run Mock Tests

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 2
  - Sample matches:
    - Line 39: `2026-02-25T12:15:15.9103806Z 2026-02-25 12:15:15,910 - __main__ - ERROR - ❌ 1 test(s) failed`
    - Line 56: `2026-02-25T12:15:15.9215276Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 39: `2026-02-25T12:15:15.9103806Z 2026-02-25 12:15:15,910 - __main__ - ERROR - ❌ 1 test(s) failed`
    - Line 48: `2026-02-25T12:15:15.9111537Z ❌ Failed: 1`
    - Line 97: `2026-02-25T12:15:16.5640817Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 62: `2026-02-25T12:15:15.9292218Z   if-no-files-found: warn`
    - Line 97: `2026-02-25T12:15:16.5640817Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 2. Run Biometric Voice E2E / Mock Biometric Tests - voice-verification

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-25T12:13:33Z
**Completed**: 2026-02-25T12:14:24Z
**Duration**: 51 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22396340652/job/64830886415)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-25T12:14:22.6427427Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-25T12:14:22.6435714Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-25T12:14:22.6788675Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-25T12:14:23.0485322Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-25T12:14:22.6896400Z   if-no-files-found: warn`
    - Line 87: `2026-02-25T12:14:22.8994910Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-25T12:14:23.0485322Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-25T12:14:12.8749041Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-25T12:14:12.8763242Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-25T12:14:15.2315949Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-25T12:14:13.1531707Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-25T12:14:13.1545442Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-25T12:14:14.0489058Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 3. Run Biometric Voice E2E / Mock Biometric Tests - embedding-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-25T12:15:31Z
**Completed**: 2026-02-25T12:16:24Z
**Duration**: 53 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22396340652/job/64830886421)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-25T12:16:21.8148664Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-25T12:16:21.8157722Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-25T12:16:21.8532337Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-25T12:16:22.2401053Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-25T12:16:21.8638278Z   if-no-files-found: warn`
    - Line 87: `2026-02-25T12:16:22.0787680Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-25T12:16:22.2401053Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-25T12:16:11.7001177Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-25T12:16:11.7015733Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-25T12:16:14.4717278Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-25T12:16:12.0173387Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-25T12:16:12.0186603Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-25T12:16:12.9305690Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 4. Run Biometric Voice E2E / Mock Biometric Tests - profile-quality-assessment

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-25T12:14:51Z
**Completed**: 2026-02-25T12:15:32Z
**Duration**: 41 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22396340652/job/64830886424)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-25T12:15:31.1209825Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-25T12:15:31.1218000Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-25T12:15:31.1573981Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-25T12:15:31.5281065Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-25T12:15:31.1682015Z   if-no-files-found: warn`
    - Line 87: `2026-02-25T12:15:31.3792064Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-25T12:15:31.5281065Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-25T12:15:21.6786221Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-25T12:15:21.6800525Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-25T12:15:23.7487584Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-25T12:15:21.8705429Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-25T12:15:21.8718328Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-25T12:15:22.6974457Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 5. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-voice-drift

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-25T12:13:57Z
**Completed**: 2026-02-25T12:14:39Z
**Duration**: 42 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22396340652/job/64830886435)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-25T12:14:37.2865281Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-25T12:14:37.2874155Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-25T12:14:37.3273072Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-25T12:14:37.7049001Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-25T12:14:37.3378366Z   if-no-files-found: warn`
    - Line 87: `2026-02-25T12:14:37.5530626Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-25T12:14:37.7049001Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-25T12:14:28.3385514Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-25T12:14:28.3404468Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-25T12:14:29.7858218Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-25T12:14:28.5240594Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-25T12:14:28.5254081Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-25T12:14:29.2403566Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 6. Run Biometric Voice E2E / Mock Biometric Tests - adaptive-thresholds

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-25T12:13:38Z
**Completed**: 2026-02-25T12:14:18Z
**Duration**: 40 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22396340652/job/64830886441)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-25T12:14:16.8134472Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-25T12:14:16.8143956Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-25T12:14:16.8576705Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-25T12:14:17.2411977Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-25T12:14:16.8681695Z   if-no-files-found: warn`
    - Line 87: `2026-02-25T12:14:17.0909626Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-25T12:14:17.2411977Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-25T12:14:07.7107238Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-25T12:14:07.7122712Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-25T12:14:08.9481421Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-25T12:14:07.8747247Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-25T12:14:07.8761663Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-25T12:14:08.6929148Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 7. Run Biometric Voice E2E / Mock Biometric Tests - anti-spoofing

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-25T12:14:52Z
**Completed**: 2026-02-25T12:15:45Z
**Duration**: 53 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22396340652/job/64830886447)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-25T12:15:43.2735369Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-25T12:15:43.2744496Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-25T12:15:43.3125053Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-25T12:15:43.6816497Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-25T12:15:43.3229090Z   if-no-files-found: warn`
    - Line 87: `2026-02-25T12:15:43.5340738Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-25T12:15:43.6816497Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-25T12:15:33.5544728Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-25T12:15:33.5558314Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-25T12:15:35.8803729Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-25T12:15:33.8030858Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-25T12:15:33.8044202Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-25T12:15:34.6734055Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 8. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-cold-start

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-25T12:14:49Z
**Completed**: 2026-02-25T12:15:41Z
**Duration**: 52 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22396340652/job/64830886460)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-25T12:15:39.3440284Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-25T12:15:39.3448247Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-25T12:15:39.3797642Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-25T12:15:39.7480860Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-25T12:15:39.3899913Z   if-no-files-found: warn`
    - Line 87: `2026-02-25T12:15:39.6002267Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-25T12:15:39.7480860Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-25T12:15:30.3450741Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-25T12:15:30.3463730Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-25T12:15:31.5524059Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-25T12:15:30.4973355Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-25T12:15:30.4986847Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-25T12:15:31.2857244Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 9. Run Biometric Voice E2E / Mock Biometric Tests - dimension-adaptation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-25T12:14:40Z
**Completed**: 2026-02-25T12:15:31Z
**Duration**: 51 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22396340652/job/64830886468)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-25T12:15:29.1283685Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-25T12:15:29.1294330Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-25T12:15:29.1697755Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-25T12:15:29.5458797Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-25T12:15:29.1804961Z   if-no-files-found: warn`
    - Line 87: `2026-02-25T12:15:29.3934060Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-25T12:15:29.5458797Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-25T12:15:19.9174592Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-25T12:15:19.9187898Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-25T12:15:21.0967777Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-25T12:15:20.0778447Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-25T12:15:20.0792157Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-25T12:15:20.8733203Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 10. Run Biometric Voice E2E / Mock Biometric Tests - security-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-25T12:13:37Z
**Completed**: 2026-02-25T12:14:26Z
**Duration**: 49 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22396340652/job/64830886471)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-25T12:14:24.4381393Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-25T12:14:24.4390810Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-25T12:14:24.4784759Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-25T12:14:24.8190835Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-25T12:14:24.4884613Z   if-no-files-found: warn`
    - Line 87: `2026-02-25T12:14:24.6862650Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-25T12:14:24.8190835Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-25T12:14:15.8989038Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-25T12:14:15.9002379Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-25T12:14:17.9211926Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-25T12:14:16.1199931Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-25T12:14:16.1213217Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-25T12:14:16.8530570Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 11. Run Biometric Voice E2E / Mock Biometric Tests - end-to-end-flow

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-25T12:13:47Z
**Completed**: 2026-02-25T12:14:42Z
**Duration**: 55 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22396340652/job/64830886479)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-25T12:14:39.9622263Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-25T12:14:39.9630674Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-25T12:14:39.9997005Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-25T12:14:40.3710765Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-25T12:14:40.0100889Z   if-no-files-found: warn`
    - Line 87: `2026-02-25T12:14:40.2207477Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-25T12:14:40.3710765Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-25T12:14:30.1250085Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-25T12:14:30.1266517Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-25T12:14:32.0339165Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-25T12:14:30.3597741Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-25T12:14:30.3613744Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-25T12:14:31.3050331Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 12. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-database-failure

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-25T12:15:24Z
**Completed**: 2026-02-25T12:16:08Z
**Duration**: 44 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22396340652/job/64830886485)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-25T12:16:07.1583084Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-25T12:16:07.1593794Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-25T12:16:07.2042720Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 2
  - Sample matches:
    - Line 69: `2026-02-25T12:16:07.2148846Z   name: test-results-biometric-mock-edge-case-database-failure`
    - Line 97: `2026-02-25T12:16:07.5935407Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-25T12:16:07.2149698Z   if-no-files-found: warn`
    - Line 87: `2026-02-25T12:16:07.4397479Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-25T12:16:07.5935407Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-25T12:15:57.9558819Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-25T12:15:57.9573699Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-25T12:15:59.2122390Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-25T12:15:58.1275451Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-25T12:15:58.1289603Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-25T12:15:58.9607435Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 13. Run Biometric Voice E2E / Mock Biometric Tests - replay-attack-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-25T12:14:22Z
**Completed**: 2026-02-25T12:15:12Z
**Duration**: 50 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22396340652/job/64830886491)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-25T12:15:09.9594914Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-25T12:15:09.9604688Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-25T12:15:09.9987922Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-25T12:15:10.3460240Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-25T12:15:10.0086309Z   if-no-files-found: warn`
    - Line 87: `2026-02-25T12:15:10.2080526Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-25T12:15:10.3460240Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-25T12:15:01.5681887Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-25T12:15:01.5695917Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-25T12:15:03.3636866Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-25T12:15:01.7920361Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-25T12:15:01.7934405Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-25T12:15:02.5353755Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 14. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-noise

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-25T12:15:19Z
**Completed**: 2026-02-25T12:16:02Z
**Duration**: 43 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22396340652/job/64830886494)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-25T12:16:00.8227367Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-25T12:16:00.8236771Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-25T12:16:00.8688604Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-25T12:16:01.2507406Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-25T12:16:00.8794226Z   if-no-files-found: warn`
    - Line 87: `2026-02-25T12:16:01.0986224Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-25T12:16:01.2507406Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-25T12:15:51.2378712Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-25T12:15:51.2393604Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-25T12:15:52.8402259Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-25T12:15:51.4297399Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-25T12:15:51.4312008Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-25T12:15:52.2718776Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 15. Run Biometric Voice E2E / Mock Biometric Tests - wake-word-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-25T12:15:02Z
**Completed**: 2026-02-25T12:15:56Z
**Duration**: 54 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22396340652/job/64830886502)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-25T12:15:53.8559794Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-25T12:15:53.8568978Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-25T12:15:53.8939102Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-25T12:15:54.2690828Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-25T12:15:53.9046787Z   if-no-files-found: warn`
    - Line 87: `2026-02-25T12:15:54.1173052Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-25T12:15:54.2690828Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-25T12:15:44.6404895Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-25T12:15:44.6418021Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-25T12:15:46.1506690Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-25T12:15:44.7929459Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-25T12:15:44.7943207Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-25T12:15:45.5906690Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 16. Run Biometric Voice E2E / Mock Biometric Tests - stt-transcription

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-25T12:15:21Z
**Completed**: 2026-02-25T12:16:08Z
**Duration**: 47 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22396340652/job/64830886515)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-25T12:16:06.2680472Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-25T12:16:06.2688744Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-25T12:16:06.3039719Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-25T12:16:06.6779245Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-25T12:16:06.3144357Z   if-no-files-found: warn`
    - Line 87: `2026-02-25T12:16:06.5296664Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-25T12:16:06.6779245Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-25T12:15:56.9081114Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-25T12:15:56.9094590Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-25T12:15:59.0563606Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-25T12:15:57.1843792Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-25T12:15:57.1856577Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-25T12:15:58.0852638Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 17. Run Biometric Voice E2E / Mock Biometric Tests - voice-synthesis-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-25T12:15:14Z
**Completed**: 2026-02-25T12:15:57Z
**Duration**: 43 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22396340652/job/64830886534)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-25T12:15:55.6323651Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-25T12:15:55.6332404Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-25T12:15:55.6700874Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-25T12:15:56.0672565Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-25T12:15:55.6811410Z   if-no-files-found: warn`
    - Line 87: `2026-02-25T12:15:55.9187179Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-25T12:15:56.0672565Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-25T12:15:46.0218516Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-25T12:15:46.0231280Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-25T12:15:47.5929396Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-25T12:15:46.2085126Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-25T12:15:46.2098698Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-25T12:15:47.0249411Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 18. Run Biometric Voice E2E / Mock Biometric Tests - performance-baseline

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-25T12:15:22Z
**Completed**: 2026-02-25T12:16:06Z
**Duration**: 44 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22396340652/job/64830886563)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-25T12:16:04.2409625Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-25T12:16:04.2418323Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-25T12:16:04.2803675Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-25T12:16:04.6679818Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-25T12:16:04.2908078Z   if-no-files-found: warn`
    - Line 87: `2026-02-25T12:16:04.5155203Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-25T12:16:04.6679818Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-25T12:15:54.8924916Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-25T12:15:54.8938413Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-25T12:15:56.4462868Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-25T12:15:55.0555870Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-25T12:15:55.0569356Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-25T12:15:56.1746390Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 19. Run Unlock Integration E2E / Generate Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-25T12:15:33Z
**Completed**: 2026-02-25T12:15:37Z
**Duration**: 4 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22396340652/job/64831133873)

#### Failed Steps

- **Step 4**: Check Test Status

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-25T12:15:35.8297987Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 13
  - Sample matches:
    - Line 39: `2026-02-25T12:15:35.6523411Z [36;1mTOTAL_FAILED=0[0m`
    - Line 44: `2026-02-25T12:15:35.6531060Z [36;1m    FAILED=$(jq -r '.summary.failed' "$report")[0m`
    - Line 46: `2026-02-25T12:15:35.6534511Z [36;1m    TOTAL_FAILED=$((TOTAL_FAILED + FAILED))[0m`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 2: `2026-02-25T12:15:35.5623169Z (node:2124) [DEP0005] DeprecationWarning: Buffer() is deprecated due to`
    - Line 3: `2026-02-25T12:15:35.5625625Z (Use `node --trace-deprecation ...` to show where the warning was creat`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 20. Run Biometric Voice E2E / Generate Biometric Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-25T12:16:27Z
**Completed**: 2026-02-25T12:16:33Z
**Duration**: 6 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22396340652/job/64831242196)

#### Failed Steps

- **Step 4**: Check Test Status

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-25T12:16:30.6630818Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 11
  - Sample matches:
    - Line 32: `2026-02-25T12:16:30.4939543Z [36;1mTOTAL_FAILED=0[0m`
    - Line 37: `2026-02-25T12:16:30.4941608Z [36;1m    FAILED=$(jq -r '.summary.failed' "$report")[0m`
    - Line 39: `2026-02-25T12:16:30.4942554Z [36;1m    TOTAL_FAILED=$((TOTAL_FAILED + FAILED))[0m`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 21. Generate Combined Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-25T12:16:36Z
**Completed**: 2026-02-25T12:16:38Z
**Duration**: 2 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22396340652/job/64831260903)

#### Failed Steps

- **Step 2**: Generate Combined Summary

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 61: `2026-02-25T12:16:37.5426718Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 9
  - Sample matches:
    - Line 25: `2026-02-25T12:16:37.5220058Z [36;1mif [ "failure" = "success" ]; then[0m`
    - Line 28: `2026-02-25T12:16:37.5221954Z [36;1m  echo "- ❌ **Unlock Integration E2E:** failure" >> $GITHUB_STEP`
    - Line 32: `2026-02-25T12:16:37.5224042Z [36;1mif [ "failure" = "success" ]; then[0m`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 22. Notify Test Status

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-25T12:16:40Z
**Completed**: 2026-02-25T12:16:44Z
**Duration**: 4 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22396340652/job/64831271106)

#### Failed Steps

- **Step 3**: Failure Notification

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line -23: `2026-02-25T12:16:41.8754796Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line -31: `2026-02-25T12:16:41.7758697Z ##[group]Run echo "❌ Unlock tests failed - 'unlock my screen' may be br`
    - Line -30: `2026-02-25T12:16:41.7759786Z [36;1mecho "❌ Unlock tests failed - 'unlock my screen' may be broken!"`
    - Line -25: `2026-02-25T12:16:41.8728182Z ❌ Unlock tests failed - 'unlock my screen' may be broken!`

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

📊 *Report generated on 2026-02-25T12:17:43.434786*
🤖 *JARVIS CI/CD Auto-PR Manager*
