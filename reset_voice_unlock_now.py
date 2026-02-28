#!/usr/bin/env python3
"""
IMMEDIATE Voice Unlock Reset
============================
Run this WHILE Ironcliw is running to reset circuit breakers.

This script connects to the running Ironcliw instance and resets
the circuit breakers that are blocking voice unlock.

Usage: python3 reset_voice_unlock_now.py
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def main():
    print("=" * 60)
    print("🔧 Ironcliw Voice Unlock - Immediate Reset")
    print("=" * 60)
    print()
    
    fixes_applied = []
    
    # 1. Reset ECAPA circuit breaker
    print("1️⃣  Resetting ECAPA-TDNN circuit breaker...")
    try:
        from cloud_services.ecapa_cloud_service import get_model_manager, CircuitState
        
        ecapa = get_model_manager()
        
        old_state = ecapa.circuit_breaker.state.name
        old_failures = ecapa.circuit_breaker.failure_count
        
        print(f"   Current state: {old_state}")
        print(f"   Failure count: {old_failures}")
        
        if ecapa.circuit_breaker.state == CircuitState.OPEN:
            ecapa.circuit_breaker.state = CircuitState.CLOSED
            ecapa.circuit_breaker.failure_count = 0
            ecapa.circuit_breaker.last_failure_time = None
            print("   ✅ Circuit breaker RESET")
            fixes_applied.append("ECAPA circuit breaker")
        else:
            print("   ✅ Already CLOSED")
        
        # Check if model is ready
        print(f"   Model ready: {ecapa.is_ready}")
        if not ecapa.is_ready:
            print("   ⚠️  Model not initialized - this is the ROOT CAUSE")
            print("   └─ Error: " + str(ecapa._error or "Unknown"))
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    
    # 2. Reset IntelligentVoiceUnlockService circuit breakers
    print("2️⃣  Resetting Voice Unlock Service circuit breakers...")
    try:
        from voice_unlock.intelligent_voice_unlock_service import IntelligentVoiceUnlockService
        
        service = IntelligentVoiceUnlockService()
        
        for name in list(service._circuit_breaker_failures.keys()):
            failures = service._circuit_breaker_failures[name]
            is_open = failures >= service.circuit_breaker_threshold
            
            if is_open:
                print(f"   {name}: OPEN ({failures} failures)")
                service._circuit_breaker_failures[name] = 0
                print(f"   ✅ {name}: RESET")
                fixes_applied.append(f"VoiceUnlock.{name}")
            else:
                print(f"   {name}: CLOSED ({failures} failures)")
        
        if not service._circuit_breaker_failures:
            print("   ✅ No circuit breakers to reset")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    
    # 3. Reset Hybrid STT Router circuit breakers
    print("3️⃣  Resetting STT Router circuit breakers...")
    try:
        from voice.hybrid_stt_router import HybridSTTRouter
        
        router = HybridSTTRouter()
        
        for engine in list(router._circuit_breaker_failures.keys()):
            failures = router._circuit_breaker_failures[engine]
            is_open = failures >= router._circuit_breaker_threshold
            
            if is_open:
                print(f"   {engine}: OPEN ({failures} failures)")
                router._circuit_breaker_failures[engine] = 0
                print(f"   ✅ {engine}: RESET")
                fixes_applied.append(f"STT.{engine}")
            else:
                print(f"   {engine}: CLOSED ({failures} failures)")
        
        if not router._circuit_breaker_failures:
            print("   ✅ No circuit breakers to reset")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    print("=" * 60)
    
    if fixes_applied:
        print(f"✅ Reset {len(fixes_applied)} circuit breaker(s):")
        for fix in fixes_applied:
            print(f"   • {fix}")
        print()
        print("🎤 Try voice unlock now: 'Ironcliw, lock my screen'")
    else:
        print("ℹ️  No circuit breakers needed resetting")
        print()
        print("If voice unlock still doesn't work, the issue is likely:")
        print("   1. ECAPA model not loaded (check model files)")
        print("   2. No voice profiles enrolled")
        print("   3. Audio issues (microphone, format)")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
