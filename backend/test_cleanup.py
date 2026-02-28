#!/usr/bin/env python3
"""
Test script for the enhanced process cleanup manager.
Run this to clean up stuck Ironcliw processes and verify the system is ready.
"""

import sys
import time
from process_cleanup_manager import (
    ProcessCleanupManager,
    emergency_cleanup,
    ensure_fresh_jarvis_instance,
    get_system_recommendations
)


def main():
    print("=" * 60)
    print("🧹 Ironcliw Process Cleanup Test")
    print("=" * 60)
    
    manager = ProcessCleanupManager()
    
    # Step 1: Check for crash recovery
    print("\n1️⃣ Checking for crash recovery...")
    if manager.check_for_segfault_recovery():
        print("   ✅ Performed crash recovery - system cleaned")
    else:
        print("   ✅ No crash recovery needed")
    
    # Step 2: Check system state
    print("\n2️⃣ Analyzing system state...")
    state = manager.analyze_system_state()
    print(f"   • CPU: {state['cpu_percent']:.1f}%")
    print(f"   • Memory: {state['memory_percent']:.1f}%")
    print(f"   • Ironcliw processes: {len(state['jarvis_processes'])}")
    print(f"   • Stuck processes: {len(state['stuck_processes'])}")
    print(f"   • Zombie processes: {len(state['zombie_processes'])}")
    
    # Step 3: Get recommendations
    print("\n3️⃣ System recommendations:")
    recommendations = get_system_recommendations()
    if recommendations:
        for rec in recommendations:
            print(f"   • {rec}")
    else:
        print("   ✅ System is healthy")
    
    # Step 4: Check if emergency cleanup is needed
    needs_cleanup = (
        len(state['jarvis_processes']) > 0 or
        len(state['stuck_processes']) > 0 or
        len(state['zombie_processes']) > 0 or
        state['memory_percent'] > 70
    )
    
    if needs_cleanup:
        print("\n⚠️  System needs cleanup!")
        if '--auto' in sys.argv:
            print("   🔧 Performing automatic emergency cleanup...")
            results = emergency_cleanup(force=True)
            print("   ✅ Cleanup complete!")
        else:
            print("   Run with --auto flag to perform automatic cleanup")
            print("   Example: python test_cleanup.py --auto")
    else:
        print("\n✅ System is clean and ready for Ironcliw!")
    
    # Step 5: Ensure fresh instance
    print("\n4️⃣ Checking for fresh Ironcliw instance...")
    if ensure_fresh_jarvis_instance():
        print("   ✅ Ready to start fresh Ironcliw instance")
    else:
        print("   ⚠️  Another Ironcliw instance is running")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()