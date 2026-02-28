import asyncio
import os
import sys
from pathlib import Path

# Add the project root to PYTHONPATH
sys.path.insert(0, str(Path(r"c:\Users\nandk\Ironcliw").resolve()))

from backend.core.intelligent_gcp_optimizer import get_gcp_optimizer
from backend.core.platform_memory_monitor import get_memory_monitor

async def run_test():
    optimizer = get_gcp_optimizer(
        {"cost": {"daily_budget_limit": 1.00, "cost_optimization_mode": "aggressive"}}
    )
    monitor = get_memory_monitor()
    snapshot = await monitor.get_memory_pressure()
    # Force extreme memory pressure explicitly to trigger the VM deployment path
    snapshot.platform = "linux"
    snapshot.available_gb = 0.1
    snapshot.used_gb = 15.9
    snapshot.total_gb = 16.0
    snapshot.usage_percent = 99.9
    snapshot.linux_actual_pressure_gb = 0.1
    snapshot.linux_psi_full_avg10 = 99.9
    snapshot.pressure_level = "critical"
    
    # Pass a workload process to justify the VM creation
    processes = [{"name": "python (ml_training)"}]
    should_create, reason, score = await optimizer.should_create_vm(snapshot, current_processes=processes)
    
    print("--- OPTIMIZER TEST RESULT ---")
    print(f"SHOULD CREATE VM: {should_create}")
    print(f"URGENT: {score.gcp_urgent}")
    print(f"REASONING: {reason}")
    print(f"COMPOSITE SCORE: {score.composite_score}")

if __name__ == "__main__":
    asyncio.run(run_test())
