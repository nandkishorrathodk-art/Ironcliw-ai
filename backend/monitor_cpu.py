#!/usr/bin/env python3
"""Monitor CPU usage of Ironcliw backend"""

import psutil
import time
import sys

def monitor_cpu(duration=30):
    print("📊 Monitoring Ironcliw CPU Usage...")
    print("=" * 50)
    
    # Find the backend process
    backend_pid = None
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] in ['python', 'python3']:
                cmdline = proc.cmdline() if hasattr(proc, 'cmdline') else []
                cmdline_str = ' '.join(cmdline) if cmdline else ''
                if 'main.py' in cmdline_str and '8000' in cmdline_str:
                    backend_pid = proc.info['pid']
                    print(f"Found backend process: PID {backend_pid}")
                    break
        except:
            continue
    
    if not backend_pid:
        print("❌ Backend process not found!")
        return
    
    # Monitor
    proc = psutil.Process(backend_pid)
    cpu_samples = []
    mem_samples = []
    
    print("\nTime  | CPU % | Memory MB | Status")
    print("-" * 40)
    
    for i in range(duration):
        try:
            cpu = proc.cpu_percent(interval=1.0)
            mem = proc.memory_info().rss / 1024 / 1024
            
            cpu_samples.append(cpu)
            mem_samples.append(mem)
            
            status = "✅ GOOD" if cpu < 30 else "⚡ OK" if cpu < 50 else "🚨 HIGH"
            
            print(f"{i+1:3d}s  | {cpu:5.1f} | {mem:9.0f} | {status}")
            
        except psutil.NoSuchProcess:
            print("Process terminated!")
            break
    
    # Summary
    if cpu_samples:
        avg_cpu = sum(cpu_samples) / len(cpu_samples)
        max_cpu = max(cpu_samples)
        min_cpu = min(cpu_samples)
        avg_mem = sum(mem_samples) / len(mem_samples)
        
        print("\n" + "=" * 50)
        print("📈 SUMMARY:")
        print(f"   Average CPU: {avg_cpu:.1f}% (target: 25%)")
        print(f"   Min-Max CPU: {min_cpu:.1f}% - {max_cpu:.1f}%")
        print(f"   Average Mem: {avg_mem:.0f}MB")
        
        reduction = (97 - avg_cpu) / 97 * 100
        print(f"\n   🎯 CPU Reduction: {reduction:.0f}%")
        
        if avg_cpu < 30:
            print("   ✅ SUCCESS: Target achieved!")
        elif avg_cpu < 50:
            print("   ⚡ PARTIAL: Good improvement")
        else:
            print("   🚨 NEEDS WORK: Still high")

if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    monitor_cpu(duration)