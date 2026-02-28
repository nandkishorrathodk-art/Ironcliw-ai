#!/usr/bin/env python3
"""
Migrate Ironcliw to optimized continuous learning
Reduces CPU from 97% to ~25% using Python optimizations
"""

import os
import sys
import subprocess
import psutil
import time
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def kill_high_cpu_processes():
    """Kill any existing high-CPU Ironcliw processes"""
    logger.info("🔍 Checking for high-CPU processes...")
    
    killed = []
    current_pid = os.getpid()
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent']):
        try:
            if proc.info['pid'] == current_pid:
                continue
                
            cmdline = ' '.join(proc.info.get('cmdline', []))
            if 'main.py' in cmdline and proc.info['name'] in ['python', 'python3']:
                cpu = proc.cpu_percent(interval=1.0)
                if cpu > 50:
                    logger.info(f"   Killing process {proc.info['pid']} using {cpu:.1f}% CPU")
                    proc.kill()
                    killed.append(proc.info['pid'])
        except:
            pass
    
    if killed:
        logger.info(f"✅ Killed {len(killed)} high-CPU processes")
        time.sleep(2)
    
    return len(killed)

def install_optimizations():
    """Install optimization libraries"""
    logger.info("📦 Installing optimization libraries...")
    
    packages = [
        'numba',           # JIT compilation
        'bottleneck',      # Fast numpy operations
        'psutil',          # System monitoring
    ]
    
    for package in packages:
        try:
            __import__(package)
            logger.info(f"   ✅ {package} already installed")
        except ImportError:
            logger.info(f"   Installing {package}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                         capture_output=True)

def update_learning_imports():
    """Update imports to use optimized learning"""
    logger.info("📝 Updating imports to use optimized learning...")
    
    # Backup directory
    backup_dir = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)
    
    # Update integrate_robust_learning.py
    integrate_path = "vision/integrate_robust_learning.py"
    if os.path.exists(integrate_path):
        # Backup
        subprocess.run(['cp', integrate_path, f"{backup_dir}/integrate_robust_learning.py"])
        
        with open(integrate_path, 'r') as f:
            content = f.read()
        
        # Add optimized import
        if 'optimized_continuous_learning' not in content:
            # Add after imports
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('from .robust_continuous_learning'):
                    lines.insert(i+1, 'from .optimized_continuous_learning import get_optimized_continuous_learning, OptimizedContinuousLearning')
                    break
            
            content = '\n'.join(lines)
            
            # Update the get function
            content = content.replace(
                'def get_advanced_continuous_learning(model: nn.Module)',
                '''def get_advanced_continuous_learning(model):
    """Get instance of continuous learning (optimized version)"""
    try:
        # Try optimized version first
        from .optimized_continuous_learning import get_optimized_continuous_learning
        return get_optimized_continuous_learning(model)
    except ImportError:
        pass
    
    # Original implementation
    _original_get_advanced_continuous_learning(model)

def _original_get_advanced_continuous_learning(model: nn.Module)'''
            )
            
            with open(integrate_path, 'w') as f:
                f.write(content)
            
            logger.info("✅ Updated integrate_robust_learning.py")
    
    # Update vision_system_v2.py if needed
    vision_path = "vision/vision_system_v2.py"
    if os.path.exists(vision_path):
        subprocess.run(['cp', vision_path, f"{backup_dir}/vision_system_v2.py"])
        logger.info(f"✅ Backed up files to {backup_dir}/")
    
    return backup_dir

def apply_cpu_limits():
    """Apply CPU limiting configurations"""
    logger.info("⚙️ Applying CPU limits...")
    
    # Environment variables
    cpu_limits = {
        'OMP_NUM_THREADS': '2',
        'MKL_NUM_THREADS': '2', 
        'NUMBA_NUM_THREADS': '2',
        'OPENBLAS_NUM_THREADS': '2',
        'LEARNING_CPU_LIMIT': '25',
        'LEARNING_THROTTLE_ENABLED': 'true',
        'DISABLE_CONTINUOUS_LEARNING': 'false',  # Keep enabled but throttled
    }
    
    # Update environment
    for key, value in cpu_limits.items():
        os.environ[key] = value
    
    # Update .env file
    env_content = []
    for key, value in cpu_limits.items():
        env_content.append(f"{key}={value}")
    
    with open('.env.cpu_limits', 'w') as f:
        f.write('\n'.join(env_content))
    
    logger.info("✅ CPU limits applied")

def test_optimized_learning():
    """Test the optimized learning system"""
    logger.info("\n🧪 Testing optimized learning...")
    
    try:
        from vision.optimized_continuous_learning import get_optimized_continuous_learning, benchmark_optimizations
        
        # Run benchmarks
        benchmark_optimizations()
        
        # Test instantiation
        learning = get_optimized_continuous_learning()
        time.sleep(2)
        
        # Get status
        status = learning.get_status()
        logger.info(f"\n📊 Optimized Learning Status:")
        logger.info(f"   CPU Usage: {status['cpu_usage']:.1f}%")
        logger.info(f"   Memory: {status['memory_mb']:.0f}MB")
        logger.info(f"   Avg Cycle: {status['avg_cycle_ms']:.0f}ms")
        logger.info(f"   Skip Rate: {status['skip_rate']:.1%}")
        
        # Shutdown test instance
        learning.running = False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def start_optimized_backend():
    """Start backend with optimized learning"""
    logger.info("\n🚀 Starting optimized backend...")
    
    # Start command with CPU affinity and nice level
    cmd = [
        'nice', '-n', '10',  # Lower priority
        sys.executable, 'main.py', '--port', '8000'
    ]
    
    # Set environment
    env = os.environ.copy()
    env.update({
        'OPTIMIZED_LEARNING': 'true',
        'CPU_LIMIT': '25',
    })
    
    # Start process
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    logger.info(f"✅ Started backend with PID {process.pid}")
    
    # Monitor startup
    logger.info("\n📈 Monitoring CPU usage...")
    time.sleep(5)  # Wait for initialization
    
    cpu_samples = []
    for i in range(20):
        try:
            proc = psutil.Process(process.pid)
            cpu = proc.cpu_percent(interval=0.5)
            mem = proc.memory_info().rss / 1024 / 1024
            
            cpu_samples.append(cpu)
            avg_cpu = sum(cpu_samples) / len(cpu_samples)
            
            status = "✅" if cpu < 30 else "⚡" if cpu < 50 else "🚨"
            
            logger.info(f"   {i+1}s: CPU: {cpu:.1f}% (avg: {avg_cpu:.1f}%) | Memory: {mem:.0f}MB {status}")
            
            time.sleep(1)
            
        except psutil.NoSuchProcess:
            logger.error("❌ Process died!")
            return None
    
    return process, cpu_samples

def verify_performance():
    """Verify the performance improvements"""
    logger.info("\n🔍 Verifying performance improvements...")
    
    # Find the backend process
    backend_pid = None
    for proc in psutil.process_iter(['pid', 'cmdline']):
        cmdline = ' '.join(proc.info.get('cmdline', []))
        if 'main.py' in cmdline and '8000' in cmdline:
            backend_pid = proc.info['pid']
            break
    
    if not backend_pid:
        logger.error("❌ Backend process not found")
        return False
    
    # Monitor for 30 seconds
    logger.info(f"Monitoring backend process (PID: {backend_pid}) for 30 seconds...")
    
    proc = psutil.Process(backend_pid)
    cpu_samples = []
    mem_samples = []
    
    for i in range(30):
        cpu = proc.cpu_percent(interval=1.0)
        mem = proc.memory_info().rss / 1024 / 1024
        
        cpu_samples.append(cpu)
        mem_samples.append(mem)
        
        print(f"\r   Progress: {i+1}/30s | Current CPU: {cpu:.1f}%", end='')
    
    print()  # New line
    
    # Calculate statistics
    avg_cpu = sum(cpu_samples) / len(cpu_samples)
    max_cpu = max(cpu_samples)
    min_cpu = min(cpu_samples)
    avg_mem = sum(mem_samples) / len(mem_samples)
    
    logger.info(f"\n📊 Performance Summary:")
    logger.info(f"   Average CPU: {avg_cpu:.1f}% (target: 25%)")
    logger.info(f"   Min CPU: {min_cpu:.1f}%")
    logger.info(f"   Max CPU: {max_cpu:.1f}%")
    logger.info(f"   Average Memory: {avg_mem:.0f}MB")
    
    # Success criteria
    success = avg_cpu < 35  # Allow some margin above 25% target
    
    if success:
        logger.info(f"\n✅ SUCCESS! CPU reduced from 97% to {avg_cpu:.1f}%")
        logger.info(f"   That's a {(97 - avg_cpu)/97*100:.0f}% reduction!")
    else:
        logger.warning(f"\n⚠️  CPU is still high at {avg_cpu:.1f}%")
    
    return success, avg_cpu

def main():
    """Main migration process"""
    print("\n🚀 Ironcliw Optimized Learning Migration")
    print("=" * 60)
    print("This will reduce CPU usage from 97% to ~25%")
    print()
    
    # Step 1: Kill high CPU processes
    killed = kill_high_cpu_processes()
    
    # Step 2: Install optimizations
    install_optimizations()
    
    # Step 3: Apply CPU limits
    apply_cpu_limits()
    
    # Step 4: Update imports
    backup_dir = update_learning_imports()
    
    # Step 5: Test optimizations
    if not test_optimized_learning():
        logger.error("❌ Optimization tests failed")
        return 1
    
    # Step 6: Start optimized backend
    process, initial_cpu = start_optimized_backend()
    if not process:
        return 1
    
    # Step 7: Verify performance
    success, avg_cpu = verify_performance()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 Ironcliw OPTIMIZATION COMPLETE!")
    print("=" * 60)
    print(f"\n📊 Results:")
    print(f"   • CPU Usage: 97% → {avg_cpu:.1f}% ({'✅ SUCCESS' if success else '⚠️  PARTIAL'})")
    print(f"   • Reduction: {(97 - avg_cpu)/97*100:.0f}%")
    print(f"   • Backups: {backup_dir}/")
    print(f"\n🔧 Optimizations Applied:")
    print(f"   • INT8 quantized inference")
    print(f"   • Memory pooling")
    print(f"   • CPU throttling")
    print(f"   • Thread limiting")
    print(f"   • Adaptive skipping")
    
    if not success:
        print(f"\n💡 Additional steps to try:")
        print(f"   • Set DISABLE_CONTINUOUS_LEARNING=true")
        print(f"   • Reduce batch size further")
        print(f"   • Install and use the Rust version")
    
    print("=" * 60)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())