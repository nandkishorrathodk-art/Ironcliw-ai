# CI/CD Failure Analysis Report

## Executive Summary

- **Workflow**: Complete Unlock Test Suite (Master)
- **Run Number**: #11
- **Branch**: `main`
- **Commit**: `157fdbe3258bd820a1758d278b7912cea5523a54`
- **Status**: ❌ FAILED
- **Timestamp**: 2026-02-26T02:48:44Z
- **Triggered By**: @nandkishorrathodk-art
- **Workflow URL**: [View Run](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425669419)

## Failure Overview

Total Failed Jobs: **22**

| # | Job Name | Category | Severity | Duration |
|---|----------|----------|----------|----------|
| 1 | Run Unlock Integration E2E / Mock Tests - security-checks | test_failure | high | 43s |
| 2 | Run Biometric Voice E2E / Mock Biometric Tests - stt-transcription | timeout | high | 43s |
| 3 | Run Biometric Voice E2E / Mock Biometric Tests - voice-verification | timeout | high | 53s |
| 4 | Run Biometric Voice E2E / Mock Biometric Tests - embedding-validation | timeout | high | 51s |
| 5 | Run Biometric Voice E2E / Mock Biometric Tests - wake-word-detection | timeout | high | 41s |
| 6 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-voice-drift | timeout | high | 43s |
| 7 | Run Biometric Voice E2E / Mock Biometric Tests - profile-quality-assessment | timeout | high | 43s |
| 8 | Run Biometric Voice E2E / Mock Biometric Tests - adaptive-thresholds | timeout | high | 53s |
| 9 | Run Biometric Voice E2E / Mock Biometric Tests - anti-spoofing | timeout | high | 52s |
| 10 | Run Biometric Voice E2E / Mock Biometric Tests - dimension-adaptation | timeout | high | 49s |
| 11 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-noise | timeout | high | 46s |
| 12 | Run Biometric Voice E2E / Mock Biometric Tests - voice-synthesis-detection | timeout | high | 37s |
| 13 | Run Biometric Voice E2E / Mock Biometric Tests - end-to-end-flow | timeout | high | 41s |
| 14 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-database-failure | test_failure | high | 51s |
| 15 | Run Biometric Voice E2E / Mock Biometric Tests - security-validation | timeout | high | 44s |
| 16 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-cold-start | timeout | high | 43s |
| 17 | Run Biometric Voice E2E / Mock Biometric Tests - replay-attack-detection | timeout | high | 54s |
| 18 | Run Biometric Voice E2E / Mock Biometric Tests - performance-baseline | timeout | high | 38s |
| 19 | Run Unlock Integration E2E / Generate Test Summary | test_failure | high | 6s |
| 20 | Run Biometric Voice E2E / Generate Biometric Test Summary | test_failure | high | 5s |
| 21 | Generate Combined Test Summary | test_failure | high | 2s |
| 22 | Notify Test Status | test_failure | high | 3s |

## Detailed Analysis

### 1. Run Unlock Integration E2E / Mock Tests - security-checks

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-26T02:49:25Z
**Completed**: 2026-02-26T02:50:08Z
**Duration**: 43 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425669419/job/64933121779)

#### Failed Steps

- **Step 6**: Run Mock Tests

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 2
  - Sample matches:
    - Line 39: `2026-02-26T02:50:05.5094138Z 2026-02-26 02:50:05,509 - __main__ - ERROR - ❌ 1 test(s) failed`
    - Line 56: `2026-02-26T02:50:05.5217396Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 39: `2026-02-26T02:50:05.5094138Z 2026-02-26 02:50:05,509 - __main__ - ERROR - ❌ 1 test(s) failed`
    - Line 48: `2026-02-26T02:50:05.5103109Z ❌ Failed: 1`
    - Line 97: `2026-02-26T02:50:06.5372845Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 62: `2026-02-26T02:50:05.5293834Z   if-no-files-found: warn`
    - Line 97: `2026-02-26T02:50:06.5372845Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 2. Run Biometric Voice E2E / Mock Biometric Tests - stt-transcription

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:51:19Z
**Completed**: 2026-02-26T02:52:02Z
**Duration**: 43 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425669419/job/64933122906)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:52:00.8039726Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:52:00.8050388Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:52:00.8569676Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:52:01.2211724Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:52:00.8675054Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:52:01.0787064Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:52:01.2211724Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:51:52.1487367Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:51:52.1510664Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:51:53.5022539Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:51:52.3235858Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:51:52.3250505Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:51:53.0470265Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 3. Run Biometric Voice E2E / Mock Biometric Tests - voice-verification

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:51:27Z
**Completed**: 2026-02-26T02:52:20Z
**Duration**: 53 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425669419/job/64933122908)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:52:16.9574838Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:52:16.9583040Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:52:16.9937542Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:52:17.3785204Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:52:17.0040478Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:52:17.2232888Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:52:17.3785204Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:52:07.6919639Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:52:07.6933416Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:52:09.3188684Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:52:07.8624649Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:52:07.8637792Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:52:08.6897782Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 4. Run Biometric Voice E2E / Mock Biometric Tests - embedding-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:51:30Z
**Completed**: 2026-02-26T02:52:21Z
**Duration**: 51 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425669419/job/64933122912)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:52:18.9350171Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:52:18.9358863Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:52:18.9743234Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:52:19.3619317Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:52:18.9861134Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:52:19.2058706Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:52:19.3619317Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:52:09.2271567Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:52:09.2286874Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:52:11.0904632Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:52:09.4597031Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:52:09.4611230Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:52:10.3379988Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 5. Run Biometric Voice E2E / Mock Biometric Tests - wake-word-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:51:32Z
**Completed**: 2026-02-26T02:52:13Z
**Duration**: 41 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425669419/job/64933122913)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:52:11.5405252Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:52:11.5415126Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:52:11.5836159Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:52:11.9774924Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:52:11.5952927Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:52:11.8164252Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:52:11.9774924Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:52:02.0483703Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:52:02.0497885Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:52:03.4438796Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:52:02.2182610Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:52:02.2196964Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:52:03.0470908Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 6. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-voice-drift

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:52:00Z
**Completed**: 2026-02-26T02:52:43Z
**Duration**: 43 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425669419/job/64933122914)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:52:41.3343979Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:52:41.3353697Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:52:41.3751718Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:52:41.7522893Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:52:41.3856615Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:52:41.5984042Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:52:41.7522893Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:52:32.1877314Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:52:32.1891079Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:52:33.7027708Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:52:32.4716488Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:52:32.4730421Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:52:33.3724551Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 7. Run Biometric Voice E2E / Mock Biometric Tests - profile-quality-assessment

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:51:59Z
**Completed**: 2026-02-26T02:52:42Z
**Duration**: 43 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425669419/job/64933122915)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:52:40.5654729Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:52:40.5665648Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:52:40.6071702Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:52:40.9978699Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:52:40.6180735Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:52:40.8399956Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:52:40.9978699Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:52:31.3878657Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:52:31.3893687Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:52:32.7149188Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:52:31.5559779Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:52:31.5573902Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:52:32.4007807Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 8. Run Biometric Voice E2E / Mock Biometric Tests - adaptive-thresholds

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:51:51Z
**Completed**: 2026-02-26T02:52:44Z
**Duration**: 53 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425669419/job/64933122919)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:52:42.9659229Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:52:42.9669031Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:52:43.0120644Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:52:43.3756040Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:52:43.0227779Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:52:43.2347537Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:52:43.3756040Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:52:33.4128298Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:52:33.4142860Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:52:35.9446906Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:52:33.7349774Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:52:33.7365377Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:52:34.5639379Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 9. Run Biometric Voice E2E / Mock Biometric Tests - anti-spoofing

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:52:22Z
**Completed**: 2026-02-26T02:53:14Z
**Duration**: 52 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425669419/job/64933122920)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:53:11.8976142Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:53:11.8985700Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:53:11.9360641Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:53:12.3237795Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:53:11.9470033Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:53:12.1662642Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:53:12.3237795Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:53:02.0397250Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:53:02.0412316Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:53:04.0278331Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:53:02.2910425Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:53:02.2924128Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:53:03.1714317Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 10. Run Biometric Voice E2E / Mock Biometric Tests - dimension-adaptation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:52:20Z
**Completed**: 2026-02-26T02:53:09Z
**Duration**: 49 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425669419/job/64933122923)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:53:07.0496020Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:53:07.0505446Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:53:07.0886130Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:53:07.4700891Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:53:07.0993762Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:53:07.3154030Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:53:07.4700891Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:52:57.0960053Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:52:57.0973079Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:52:59.0842985Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:52:57.3927274Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:52:57.3940541Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:52:58.2814886Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 11. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-noise

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:52:23Z
**Completed**: 2026-02-26T02:53:09Z
**Duration**: 46 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425669419/job/64933122925)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:53:07.5556263Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:53:07.5565953Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:53:07.5925037Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:53:07.9720340Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:53:07.6029363Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:53:07.8177137Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:53:07.9720340Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:52:57.3593003Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:52:57.3607275Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:52:59.6721040Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:52:57.6373672Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:52:57.6387487Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:52:58.5168338Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 12. Run Biometric Voice E2E / Mock Biometric Tests - voice-synthesis-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:51:55Z
**Completed**: 2026-02-26T02:52:32Z
**Duration**: 37 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425669419/job/64933122928)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:52:30.4924697Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:52:30.4934823Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:52:30.5313194Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:52:30.9127946Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:52:30.5422559Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:52:30.7572698Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:52:30.9127946Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:52:21.3872129Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:52:21.3885457Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:52:23.0402309Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:52:21.5914081Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:52:21.5927813Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:52:22.4351735Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 13. Run Biometric Voice E2E / Mock Biometric Tests - end-to-end-flow

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:52:04Z
**Completed**: 2026-02-26T02:52:45Z
**Duration**: 41 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425669419/job/64933122929)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:52:43.7868431Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:52:43.7877853Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:52:43.8240754Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:52:44.2124738Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:52:43.8353917Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:52:44.0553297Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:52:44.2124738Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:52:34.5789600Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:52:34.5802831Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:52:35.9838355Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:52:34.7393431Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:52:34.7407624Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:52:35.5524132Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 14. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-database-failure

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-26T02:52:04Z
**Completed**: 2026-02-26T02:52:55Z
**Duration**: 51 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425669419/job/64933122932)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:52:52.8660172Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:52:52.8669127Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:52:52.9013793Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 2
  - Sample matches:
    - Line 69: `2026-02-26T02:52:52.9118381Z   name: test-results-biometric-mock-edge-case-database-failure`
    - Line 97: `2026-02-26T02:52:53.2795358Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:52:52.9119525Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:52:53.1247783Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:52:53.2795358Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:52:42.4647663Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:52:42.4662646Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:52:45.3631852Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:52:42.7769855Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:52:42.7783640Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:52:43.6979862Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 15. Run Biometric Voice E2E / Mock Biometric Tests - security-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:52:00Z
**Completed**: 2026-02-26T02:52:44Z
**Duration**: 44 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425669419/job/64933122933)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:52:42.7472156Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:52:42.7481353Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:52:42.7845982Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:52:43.1682228Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:52:42.7952964Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:52:43.0135401Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:52:43.1682228Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:52:33.0461211Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:52:33.0474849Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:52:34.8328532Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:52:33.2792626Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:52:33.2806122Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:52:34.1454381Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 16. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-cold-start

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:52:34Z
**Completed**: 2026-02-26T02:53:17Z
**Duration**: 43 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425669419/job/64933122956)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:53:15.3198965Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:53:15.3210119Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:53:15.3709934Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:53:15.7681819Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:53:15.3823184Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:53:15.6074971Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:53:15.7681819Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:53:05.6006016Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:53:05.6022738Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:53:06.9550217Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:53:05.7769125Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:53:05.7784489Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:53:06.6229722Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 17. Run Biometric Voice E2E / Mock Biometric Tests - replay-attack-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:52:31Z
**Completed**: 2026-02-26T02:53:25Z
**Duration**: 54 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425669419/job/64933122958)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:53:23.5012171Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:53:23.5021012Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:53:23.5376072Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:53:23.9150748Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:53:23.5482582Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:53:23.7618298Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:53:23.9150748Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:53:13.9520814Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:53:13.9534453Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:53:15.2873655Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:53:14.1135091Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:53:14.1148021Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:53:14.9112751Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 18. Run Biometric Voice E2E / Mock Biometric Tests - performance-baseline

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-26T02:52:24Z
**Completed**: 2026-02-26T02:53:02Z
**Duration**: 38 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425669419/job/64933122959)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-26T02:53:00.7429426Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-26T02:53:00.7438888Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-26T02:53:00.7801710Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:53:01.1587478Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-26T02:53:00.7906742Z   if-no-files-found: warn`
    - Line 87: `2026-02-26T02:53:01.0056622Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-26T02:53:01.1587478Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-26T02:52:51.4092029Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-26T02:52:51.4106082Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-26T02:52:53.0518126Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-26T02:52:51.6014264Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-26T02:52:51.6028691Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-26T02:52:52.4329017Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 19. Run Unlock Integration E2E / Generate Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-26T02:52:44Z
**Completed**: 2026-02-26T02:52:50Z
**Duration**: 6 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425669419/job/64933344379)

#### Failed Steps

- **Step 4**: Check Test Status

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:52:47.7783698Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 13
  - Sample matches:
    - Line 39: `2026-02-26T02:52:47.5574564Z [36;1mTOTAL_FAILED=0[0m`
    - Line 44: `2026-02-26T02:52:47.5582775Z [36;1m    FAILED=$(jq -r '.summary.failed' "$report")[0m`
    - Line 46: `2026-02-26T02:52:47.5586313Z [36;1m    TOTAL_FAILED=$((TOTAL_FAILED + FAILED))[0m`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 4: `2026-02-26T02:52:47.4306407Z (node:2169) [DEP0005] DeprecationWarning: Buffer() is deprecated due to`
    - Line 5: `2026-02-26T02:52:47.4310632Z (Use `node --trace-deprecation ...` to show where the warning was creat`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 20. Run Biometric Voice E2E / Generate Biometric Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-26T02:53:38Z
**Completed**: 2026-02-26T02:53:43Z
**Duration**: 5 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425669419/job/64933408110)

#### Failed Steps

- **Step 4**: Check Test Status

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-26T02:53:40.8914140Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 11
  - Sample matches:
    - Line 32: `2026-02-26T02:53:40.6736601Z [36;1mTOTAL_FAILED=0[0m`
    - Line 37: `2026-02-26T02:53:40.6743973Z [36;1m    FAILED=$(jq -r '.summary.failed' "$report")[0m`
    - Line 39: `2026-02-26T02:53:40.6747480Z [36;1m    TOTAL_FAILED=$((TOTAL_FAILED + FAILED))[0m`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 21. Generate Combined Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-26T02:55:35Z
**Completed**: 2026-02-26T02:55:37Z
**Duration**: 2 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425669419/job/64933428105)

#### Failed Steps

- **Step 2**: Generate Combined Summary

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 61: `2026-02-26T02:55:36.5501238Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 9
  - Sample matches:
    - Line 25: `2026-02-26T02:55:36.5270257Z [36;1mif [ "failure" = "success" ]; then[0m`
    - Line 28: `2026-02-26T02:55:36.5272311Z [36;1m  echo "- ❌ **Unlock Integration E2E:** failure" >> $GITHUB_STEP`
    - Line 32: `2026-02-26T02:55:36.5274300Z [36;1mif [ "failure" = "success" ]; then[0m`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 22. Notify Test Status

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-26T02:55:47Z
**Completed**: 2026-02-26T02:55:50Z
**Duration**: 3 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22425669419/job/64933572020)

#### Failed Steps

- **Step 3**: Failure Notification

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line -23: `2026-02-26T02:55:48.6106732Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line -31: `2026-02-26T02:55:48.4991558Z ##[group]Run echo "❌ Unlock tests failed - 'unlock my screen' may be br`
    - Line -30: `2026-02-26T02:55:48.4992739Z [36;1mecho "❌ Unlock tests failed - 'unlock my screen' may be broken!"`
    - Line -25: `2026-02-26T02:55:48.6087543Z ❌ Unlock tests failed - 'unlock my screen' may be broken!`

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

📊 *Report generated on 2026-02-26T02:57:01.384394*
🤖 *JARVIS CI/CD Auto-PR Manager*
