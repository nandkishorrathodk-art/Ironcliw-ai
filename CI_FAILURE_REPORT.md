# CI/CD Failure Analysis Report

## Executive Summary

- **Workflow**: Priority 2 - Biometric Voice Unlock E2E Testing
- **Run Number**: #5
- **Branch**: `main`
- **Commit**: `2c22880fc1e06a4b544e0fc3acfc9517b37c77f1`
- **Status**: ❌ FAILED
- **Timestamp**: 2026-02-24T06:26:16Z
- **Triggered By**: @nandkishorrathodk-art
- **Workflow URL**: [View Run](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347398)

## Failure Overview

Total Failed Jobs: **17**

| # | Job Name | Category | Severity | Duration |
|---|----------|----------|----------|----------|
| 1 | Mock Biometric Tests - embedding-validation | timeout | high | 45s |
| 2 | Mock Biometric Tests - voice-verification | timeout | high | 50s |
| 3 | Mock Biometric Tests - profile-quality-assessment | timeout | high | 48s |
| 4 | Mock Biometric Tests - dimension-adaptation | timeout | high | 39s |
| 5 | Mock Biometric Tests - edge-case-noise | timeout | high | 48s |
| 6 | Mock Biometric Tests - stt-transcription | timeout | high | 47s |
| 7 | Mock Biometric Tests - adaptive-thresholds | timeout | high | 49s |
| 8 | Mock Biometric Tests - wake-word-detection | timeout | high | 50s |
| 9 | Mock Biometric Tests - performance-baseline | timeout | high | 49s |
| 10 | Mock Biometric Tests - replay-attack-detection | timeout | high | 38s |
| 11 | Mock Biometric Tests - edge-case-database-failure | test_failure | high | 41s |
| 12 | Mock Biometric Tests - anti-spoofing | timeout | high | 42s |
| 13 | Mock Biometric Tests - security-validation | timeout | high | 44s |
| 14 | Mock Biometric Tests - edge-case-cold-start | timeout | high | 51s |
| 15 | Mock Biometric Tests - edge-case-voice-drift | timeout | high | 40s |
| 16 | Mock Biometric Tests - end-to-end-flow | timeout | high | 49s |
| 17 | Mock Biometric Tests - voice-synthesis-detection | timeout | high | 42s |

## Detailed Analysis

### 1. Mock Biometric Tests - embedding-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:26:30Z
**Completed**: 2026-02-24T06:27:15Z
**Duration**: 45 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347398/job/64638994937)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:27:13.3955466Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:27:13.3963997Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:27:13.4367481Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:27:13.8115511Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:27:13.4475719Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:27:13.6575447Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:27:13.8115511Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:27:04.3745586Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:27:04.3759279Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:27:05.4689210Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:27:04.5222362Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:27:04.5236055Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:27:05.2192772Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 2. Mock Biometric Tests - voice-verification

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:26:31Z
**Completed**: 2026-02-24T06:27:21Z
**Duration**: 50 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347398/job/64638994946)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:27:19.1456071Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:27:19.1465066Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:27:19.1835141Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:27:19.5603279Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:27:19.1938987Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:27:19.4090446Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:27:19.5603279Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:27:08.6788861Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:27:08.6801619Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:27:11.3314374Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:27:08.9505848Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:27:08.9519524Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:27:10.3372603Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 3. Mock Biometric Tests - profile-quality-assessment

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:26:32Z
**Completed**: 2026-02-24T06:27:20Z
**Duration**: 48 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347398/job/64638994951)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:27:18.5219669Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:27:18.5227997Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:27:18.5616029Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:27:18.9381950Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:27:18.5719300Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:27:18.7869125Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:27:18.9381950Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:27:08.1197577Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:27:08.1211197Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:27:10.8030161Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:27:08.4072614Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:27:08.4085562Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:27:09.2844767Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 4. Mock Biometric Tests - dimension-adaptation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:27:44Z
**Completed**: 2026-02-24T06:28:23Z
**Duration**: 39 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347398/job/64638994956)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:28:21.3936706Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:28:21.3946342Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:28:21.4335516Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:28:21.8105278Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:28:21.4443384Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:28:21.6581598Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:28:21.8105278Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:28:11.9921467Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:28:11.9935898Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:28:13.1876912Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:28:12.1410376Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:28:12.1424833Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:28:12.9501740Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 5. Mock Biometric Tests - edge-case-noise

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:26:31Z
**Completed**: 2026-02-24T06:27:19Z
**Duration**: 48 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347398/job/64638994957)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:27:17.2964140Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:27:17.2972603Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:27:17.3332753Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:27:17.7038059Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:27:17.3434745Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:27:17.5551882Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:27:17.7038059Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:27:08.6553162Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:27:08.6567246Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:27:09.8854696Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:27:08.8052290Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:27:08.8065845Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:27:09.6119887Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 6. Mock Biometric Tests - stt-transcription

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:27:47Z
**Completed**: 2026-02-24T06:28:34Z
**Duration**: 47 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347398/job/64638994960)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:28:32.2334794Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:28:32.2343231Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:28:32.2748778Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:28:32.6582634Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:28:32.2851680Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:28:32.5062662Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:28:32.6582634Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:28:22.5072877Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:28:22.5087214Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:28:24.6701068Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:28:22.6636213Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:28:22.6650455Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:28:23.4803933Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 7. Mock Biometric Tests - adaptive-thresholds

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:26:56Z
**Completed**: 2026-02-24T06:27:45Z
**Duration**: 49 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347398/job/64638994968)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:27:42.7410801Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:27:42.7418366Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:27:42.7812182Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:27:43.1255848Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:27:42.7909248Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:27:42.9917926Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:27:43.1255848Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:27:32.5865950Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:27:32.5879978Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:27:35.5811143Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:27:32.8920023Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:27:32.8933523Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:27:34.2475552Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 8. Mock Biometric Tests - wake-word-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:27:44Z
**Completed**: 2026-02-24T06:28:34Z
**Duration**: 50 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347398/job/64638994977)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:28:32.4492348Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:28:32.4502378Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:28:32.4906007Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:28:32.8718785Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:28:32.5016595Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:28:32.7197415Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:28:32.8718785Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:28:22.7201426Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:28:22.7218871Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:28:24.8994530Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:28:22.9625543Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:28:22.9640407Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:28:23.8269966Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 9. Mock Biometric Tests - performance-baseline

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:26:36Z
**Completed**: 2026-02-24T06:27:25Z
**Duration**: 49 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347398/job/64638994979)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:27:23.1963062Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:27:23.1973624Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:27:23.2430970Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:27:23.6278595Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:27:23.2542756Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:27:23.4738722Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:27:23.6278595Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:27:10.9342876Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:27:10.9357043Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:27:15.0098404Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:27:11.0873194Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:27:11.0886199Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:27:12.3324428Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 10. Mock Biometric Tests - replay-attack-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:26:57Z
**Completed**: 2026-02-24T06:27:35Z
**Duration**: 38 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347398/job/64638994984)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:27:34.0854945Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:27:34.0864344Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:27:34.1260670Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:27:34.5011727Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:27:34.1369121Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:27:34.3491603Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:27:34.5011727Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:27:24.9842655Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:27:24.9856264Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:27:26.5463804Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:27:25.1765429Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:27:25.1778231Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:27:26.0014024Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 11. Mock Biometric Tests - edge-case-database-failure

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2026-02-24T06:27:27Z
**Completed**: 2026-02-24T06:28:08Z
**Duration**: 41 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347398/job/64638994988)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:28:06.7416690Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:28:06.7425260Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:28:06.7777932Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 2
  - Sample matches:
    - Line 69: `2026-02-24T06:28:06.7879476Z   name: test-results-biometric-mock-edge-case-database-failure`
    - Line 97: `2026-02-24T06:28:07.1463366Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:28:06.7880303Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:28:06.9972743Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:28:07.1463366Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:27:57.0818672Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:27:57.0831626Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:27:58.9093192Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:27:57.2671396Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:27:57.2685059Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:27:58.0840251Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 12. Mock Biometric Tests - anti-spoofing

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:27:41Z
**Completed**: 2026-02-24T06:28:23Z
**Duration**: 42 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347398/job/64638994990)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:28:21.8322801Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:28:21.8331651Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:28:21.8720728Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:28:22.2812253Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:28:21.8825684Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:28:22.1080265Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:28:22.2812253Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:28:11.6331028Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:28:11.6344628Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:28:13.1623757Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:28:11.7912574Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:28:11.7926565Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:28:12.5124914Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 13. Mock Biometric Tests - security-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:26:49Z
**Completed**: 2026-02-24T06:27:33Z
**Duration**: 44 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347398/job/64638994992)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:27:31.4436388Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:27:31.4446433Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:27:31.4827405Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:27:31.8592582Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:27:31.4929534Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:27:31.7094302Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:27:31.8592582Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:27:21.4639702Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:27:21.4654309Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:27:23.4855805Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:27:21.6538277Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:27:21.6551438Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:27:22.9547672Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 14. Mock Biometric Tests - edge-case-cold-start

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:28:23Z
**Completed**: 2026-02-24T06:29:14Z
**Duration**: 51 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347398/job/64638995002)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:29:10.7865178Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:29:10.7873678Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:29:10.8228305Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:29:11.1995695Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:29:10.8334035Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:29:11.0478831Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:29:11.1995695Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:29:01.8449594Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:29:01.8463371Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:29:03.0648567Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:29:01.9946116Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:29:01.9959240Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:29:02.7982698Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 15. Mock Biometric Tests - edge-case-voice-drift

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:28:25Z
**Completed**: 2026-02-24T06:29:05Z
**Duration**: 40 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347398/job/64638995004)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:29:03.8463305Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:29:03.8471990Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:29:03.8844059Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:29:04.2603847Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:29:03.8951712Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:29:04.1085953Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:29:04.2603847Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:28:54.8640440Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:28:54.8655438Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:28:56.4264428Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:28:55.0631327Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:28:55.0645571Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:28:55.9003674Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 16. Mock Biometric Tests - end-to-end-flow

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:28:19Z
**Completed**: 2026-02-24T06:29:08Z
**Duration**: 49 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347398/job/64638995006)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:29:06.7070755Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:29:06.7079598Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:29:06.7464759Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:29:07.1183806Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:29:06.7568970Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:29:06.9692756Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:29:07.1183806Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:28:57.1738678Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:28:57.1751622Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:28:59.5279032Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:28:57.4030931Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:28:57.4044134Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:28:58.2410132Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 17. Mock Biometric Tests - voice-synthesis-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2026-02-24T06:28:10Z
**Completed**: 2026-02-24T06:28:52Z
**Duration**: 42 seconds
**Job URL**: [View Logs](https://github.com/nandkishorrathodk-art/Ironcliw-ai/actions/runs/22339347398/job/64638995015)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2026-02-24T06:28:50.3736271Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2026-02-24T06:28:50.3745151Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2026-02-24T06:28:50.4122266Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2026-02-24T06:28:50.7838342Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2026-02-24T06:28:50.4224534Z   if-no-files-found: warn`
    - Line 87: `2026-02-24T06:28:50.6328577Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2026-02-24T06:28:50.7838342Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 5
  - Sample matches:
    - Line 0: `2026-02-24T06:28:41.3530148Z Collecting exceptiongroup>=1 (from pytest)`
    - Line 1: `2026-02-24T06:28:41.3544081Z   Using cached exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2026-02-24T06:28:42.4111172Z Using cached exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 20: `2026-02-24T06:28:41.4999097Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 21: `2026-02-24T06:28:41.5013383Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 38: `2026-02-24T06:28:42.1850477Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

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

📊 *Report generated on 2026-02-24T06:30:55.116984*
🤖 *JARVIS CI/CD Auto-PR Manager*
