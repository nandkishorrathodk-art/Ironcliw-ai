#!/usr/bin/env python3
import asyncio
import json
import sys
import time
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
except ImportError:
    print('Installing websockets...')
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'websockets'])
    import websockets
    from websockets.exceptions import ConnectionClosed

class TestResult(Enum):
    SUCCESS = '?'
    FAILURE = '?'
    WARNING = '??'
    INFO = '??'

@dataclass
class ConnectionMetrics:
    connection_id: str
    connected: bool = False
    connection_time: float = 0.0
    disconnection_time: Optional[float] = None
    messages_sent: int = 0
    messages_received: int = 0
    errors: int = 0
    reconnections: int = 0
    latencies: List[float] = field(default_factory=list)
    heartbeat_responses: int = 0
    heartbeat_failures: int = 0

@dataclass
class TestReport:
    suite_name: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0
    metrics: Dict[str, ConnectionMetrics] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_result(self, passed: bool, message: str):
        if passed:
            self.tests_passed += 1
            print(f'{TestResult.SUCCESS.value} {message}')
        else:
            self.tests_failed += 1
            self.errors.append(message)
            print(f'{TestResult.FAILURE.value} {message}')

    def add_warning(self, message: str):
        self.warnings.append(message)
        print(f'{TestResult.WARNING.value} {message}')

    def finalize(self):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print('\n' + '='*80)
        print(f'?? Test Suite: {self.suite_name}')
        print(f'??  Duration: {duration:.2f}s')
        print(f'{TestResult.SUCCESS.value} Passed: {self.tests_passed}')
        print(f'{TestResult.FAILURE.value} Failed: {self.tests_failed}')
        return self.tests_failed == 0

class WebSocketHealthTester:
    def __init__(self, ws_url: str, config: Dict):
        self.ws_url = ws_url
        self.config = config
        self.connections: Dict[str, websockets.WebSocketClientProtocol] = {}

    async def test_connection_lifecycle(self) -> TestReport:
        report = TestReport('Connection Lifecycle')
        try:
            ws = await asyncio.wait_for(websockets.connect(self.ws_url), timeout=10.0)
            report.add_result(True, 'Successfully established WebSocket connection')
            report.add_result(not ws.closed, 'Connection is active')
            await ws.close()
            await asyncio.sleep(0.5)
            report.add_result(ws.closed, 'Connection closed gracefully')
        except Exception as e:
            report.add_result(False, f'Connection failed: {e}')
        report.finalize()
        return report

async def main():
    import os
    config = {'test_duration': 60, 'connection_count': 5}
    ws_port = os.getenv('WS_ROUTER_PORT', '8001')
    ws_urls = [f'ws://localhost:{ws_port}/ws/unified', 'ws://localhost:8000/ws/unified', 'ws://localhost:8000/ws']
    ws_url = None
    for url in ws_urls:
        try:
            async with websockets.connect(url) as ws:
                await ws.close()
                ws_url = url
                break
        except:
            continue

    if not ws_url:
        print('??  No WebSocket server found, tests will fail if meant to be live.')
        return 1
    
    tester = WebSocketHealthTester(ws_url, config)
    report = await tester.test_connection_lifecycle()
    return 0 if report.tests_failed == 0 else 1

if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
