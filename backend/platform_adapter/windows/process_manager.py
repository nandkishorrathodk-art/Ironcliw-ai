"""
JARVIS Windows Process Manager
═══════════════════════════════════════════════════════════════════════════════

Windows process management using Task Scheduler and WMI.

Features:
    - Start/stop processes
    - Process monitoring
    - Task Scheduler integration for startup tasks
    - Process info retrieval

Windows Process Management:
    - subprocess for process launching
    - psutil for process monitoring
    - Task Scheduler (schtasks) for startup tasks
    - WMI for advanced process info

Replaces macOS launchd with Windows Task Scheduler.

Author: JARVIS System
Version: 1.0.0 (Windows Port)
"""
from __future__ import annotations

import os
import sys
import subprocess
import signal
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None
    print("Warning: psutil not installed. Some process management features unavailable.")

from ..base import BaseProcessManager


class WindowsProcessManager(BaseProcessManager):
    """Windows implementation of process management"""
    
    def __init__(self):
        """Initialize Windows process manager"""
        self._processes: Dict[int, subprocess.Popen] = {}
    
    def start_process(self, 
                     command: str, 
                     args: List[str] = None,
                     env: Dict[str, str] = None,
                     background: bool = False) -> int:
        """Start a new process, returns PID"""
        try:
            cmd_args = [command]
            if args:
                cmd_args.extend(args)
            
            process_env = os.environ.copy()
            if env:
                process_env.update(env)
            
            if background:
                creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
                
                process = subprocess.Popen(
                    cmd_args,
                    env=process_env,
                    creationflags=creation_flags,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                )
            else:
                process = subprocess.Popen(
                    cmd_args,
                    env=process_env,
                )
            
            pid = process.pid
            self._processes[pid] = process
            
            return pid
        except Exception as e:
            print(f"Warning: Failed to start process '{command}': {e}")
            return -1
    
    def stop_process(self, pid: int, graceful: bool = True) -> bool:
        """Stop a running process"""
        try:
            if psutil:
                try:
                    process = psutil.Process(pid)
                    
                    if graceful:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except psutil.TimeoutExpired:
                            process.kill()
                    else:
                        process.kill()
                    
                    if pid in self._processes:
                        del self._processes[pid]
                    
                    return True
                except psutil.NoSuchProcess:
                    if pid in self._processes:
                        del self._processes[pid]
                    return True
            else:
                subprocess.run(['taskkill', '/F', '/PID', str(pid)], 
                             capture_output=True, 
                             timeout=10)
                
                if pid in self._processes:
                    del self._processes[pid]
                
                return True
        except Exception as e:
            print(f"Warning: Failed to stop process {pid}: {e}")
            return False
    
    def is_process_running(self, pid: int) -> bool:
        """Check if process is running"""
        try:
            if psutil:
                return psutil.pid_exists(pid)
            else:
                result = subprocess.run(
                    ['tasklist', '/FI', f'PID eq {pid}'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                return str(pid) in result.stdout
        except Exception as e:
            print(f"Warning: Failed to check process {pid}: {e}")
            return False
    
    def get_process_info(self, pid: int) -> Optional[Dict[str, Any]]:
        """Get information about a process"""
        try:
            if psutil:
                try:
                    process = psutil.Process(pid)
                    
                    return {
                        'pid': pid,
                        'name': process.name(),
                        'exe': process.exe(),
                        'cmdline': process.cmdline(),
                        'status': process.status(),
                        'cpu_percent': process.cpu_percent(interval=0.1),
                        'memory_mb': process.memory_info().rss / 1024 / 1024,
                        'create_time': process.create_time(),
                        'num_threads': process.num_threads(),
                    }
                except psutil.NoSuchProcess:
                    return None
            else:
                result = subprocess.run(
                    ['tasklist', '/FI', f'PID eq {pid}', '/FO', 'CSV', '/NH'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    parts = result.stdout.strip().strip('"').split('","')
                    if len(parts) >= 2:
                        return {
                            'pid': pid,
                            'name': parts[0],
                            'memory_mb': float(parts[4].replace(' K', '')) / 1024 if len(parts) >= 5 else 0,
                        }
                
                return None
        except Exception as e:
            print(f"Warning: Failed to get process info for {pid}: {e}")
            return None
    
    def list_processes(self, filter_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all processes, optionally filtered by name"""
        try:
            processes = []
            
            if psutil:
                for proc in psutil.process_iter(['pid', 'name', 'exe', 'status', 'memory_info']):
                    try:
                        info = proc.info
                        
                        if filter_name and filter_name.lower() not in info['name'].lower():
                            continue
                        
                        processes.append({
                            'pid': info['pid'],
                            'name': info['name'],
                            'exe': info.get('exe', ''),
                            'status': info.get('status', ''),
                            'memory_mb': info['memory_info'].rss / 1024 / 1024 if info.get('memory_info') else 0,
                        })
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            else:
                cmd = ['tasklist', '/FO', 'CSV', '/NH']
                if filter_name:
                    cmd.extend(['/FI', f'IMAGENAME eq {filter_name}*'])
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.strip().strip('"').split('","')
                        if len(parts) >= 2:
                            processes.append({
                                'pid': int(parts[1]),
                                'name': parts[0],
                                'memory_mb': float(parts[4].replace(' K', '')) / 1024 if len(parts) >= 5 else 0,
                            })
            
            return processes
        except Exception as e:
            print(f"Warning: Failed to list processes: {e}")
            return []
    
    def schedule_startup(self, command: str, name: str) -> bool:
        """Schedule process to start on system boot using Task Scheduler"""
        try:
            task_name = f"JARVIS_{name}"
            
            xml_content = f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>JARVIS startup task: {name}</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
    </LogonTrigger>
  </Triggers>
  <Principals>
    <Principal>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
  </Settings>
  <Actions>
    <Exec>
      <Command>{command}</Command>
    </Exec>
  </Actions>
</Task>"""
            
            temp_xml = Path(os.environ.get('TEMP', 'C:\\Temp')) / f"{task_name}.xml"
            temp_xml.write_text(xml_content, encoding='utf-16')
            
            result = subprocess.run(
                ['schtasks', '/Create', '/TN', task_name, '/XML', str(temp_xml), '/F'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            temp_xml.unlink(missing_ok=True)
            
            return result.returncode == 0
        except Exception as e:
            print(f"Warning: Failed to schedule startup task '{name}': {e}")
            return False
    
    def unschedule_startup(self, name: str) -> bool:
        """Remove from startup schedule"""
        try:
            task_name = f"JARVIS_{name}"
            
            result = subprocess.run(
                ['schtasks', '/Delete', '/TN', task_name, '/F'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            return result.returncode == 0
        except Exception as e:
            print(f"Warning: Failed to unschedule startup task '{name}': {e}")
            return False
