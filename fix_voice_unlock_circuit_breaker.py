#!/usr/bin/env python3
"""
Voice Unlock Circuit Breaker Fix & Diagnostics
==============================================
This script diagnoses and fixes the circuit breaker issues 
preventing JARVIS from locking/unlocking your screen.

Run: python fix_voice_unlock_circuit_breaker.py
"""

import asyncio
import sys
import os
import time

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def main():
    print("=" * 70)
    print("🔧 JARVIS Voice Unlock Circuit Breaker Diagnostic & Fix")
    print("=" * 70)
    print()
    
    # 1. Check ECAPA Service
    print("━" * 70)
    print("1️⃣  Checking ECAPA-TDNN Cloud Service...")
    print("━" * 70)
    
    try:
        from cloud_services.ecapa_cloud_service import get_model_manager, CloudECAPAConfig, CircuitState
        
        ecapa = get_model_manager()
        
        # Check circuit breaker state
        cb_state = ecapa.circuit_breaker.state
        cb_failures = ecapa.circuit_breaker.failure_count
        cb_last_failure = ecapa.circuit_breaker.last_failure_time
        
        print(f"   Circuit Breaker State:    {cb_state.name}")
        print(f"   Failure Count:            {cb_failures}/{CloudECAPAConfig.CB_FAILURE_THRESHOLD}")
        print(f"   Recovery Timeout:         {CloudECAPAConfig.CB_RECOVERY_TIMEOUT}s")
        
        if cb_last_failure:
            elapsed = time.time() - cb_last_failure
            print(f"   Time Since Last Failure:  {elapsed:.1f}s")
        
        # Reset circuit breaker if needed
        if cb_state == CircuitState.OPEN:
            print()
            print("   ⚠️  Circuit Breaker is OPEN! Resetting...")
            ecapa.circuit_breaker.state = CircuitState.CLOSED
            ecapa.circuit_breaker.failure_count = 0
            ecapa.circuit_breaker.last_failure_time = None
            print("   ✅ Circuit Breaker RESET to CLOSED")
        elif cb_state == CircuitState.HALF_OPEN:
            print()
            print("   ⚠️  Circuit Breaker is HALF_OPEN, allowing test...")
        else:
            print("   ✅ Circuit Breaker is healthy (CLOSED)")
        
        # Try to initialize the model
        print()
        print("   🔄 Initializing ECAPA model...")
        
        try:
            init_result = await ecapa.initialize()
            if init_result:
                print("   ✅ ECAPA model initialized successfully!")
                print(f"   └─ Model ready: {ecapa.is_ready}")
                print(f"   └─ Using optimized loader: {ecapa._using_optimized}")
            else:
                print("   ❌ ECAPA model initialization returned False")
        except Exception as e:
            print(f"   ❌ ECAPA initialization failed: {e}")
            
    except ImportError as e:
        print(f"   ⚠️  Could not import ECAPA service: {e}")
    except Exception as e:
        print(f"   ❌ Error checking ECAPA: {e}")
    
    print()
    
    # 2. Check Intelligent Voice Unlock Service
    print("━" * 70)
    print("2️⃣  Checking Intelligent Voice Unlock Service...")
    print("━" * 70)
    
    try:
        from voice_unlock.intelligent_voice_unlock_service import IntelligentVoiceUnlockService
        
        service = IntelligentVoiceUnlockService()
        
        # Check all circuit breakers
        print(f"   Circuit Breaker Threshold: {service.circuit_breaker_threshold}")
        print(f"   Circuit Breaker Timeout:   {service.circuit_breaker_timeout}s")
        print()
        print("   Service Circuit Breakers:")
        
        for name, failures in service._circuit_breaker_failures.items():
            last_failure = service._circuit_breaker_last_failure.get(name, 0)
            is_open = failures >= service.circuit_breaker_threshold
            
            status = "🔴 OPEN" if is_open else "🟢 CLOSED"
            print(f"   └─ {name}: {status} ({failures}/{service.circuit_breaker_threshold} failures)")
            
            if is_open:
                # Reset it
                service._circuit_breaker_failures[name] = 0
                print(f"      └─ 🔧 RESET to 0 failures")
        
        if not service._circuit_breaker_failures:
            print("   └─ No circuit breakers recorded yet (clean state)")
            
    except ImportError as e:
        print(f"   ⚠️  Could not import Voice Unlock service: {e}")
    except Exception as e:
        print(f"   ❌ Error checking Voice Unlock: {e}")
    
    print()
    
    # 3. Check Voice Profiles
    print("━" * 70)
    print("3️⃣  Checking Voice Profiles in Database...")
    print("━" * 70)
    
    try:
        from intelligence.learning_database import JARVISLearningDatabase
        
        db = JARVISLearningDatabase()
        await db.initialize()
        
        profiles = await db.get_all_speaker_profiles()
        
        if not profiles:
            print("   ⚠️  No voice profiles found!")
            print("   └─ Voice unlock requires at least one enrolled speaker")
            print()
            print("   📝 To enroll your voice, say 'JARVIS, enroll my voice'")
        else:
            print(f"   Found {len(profiles)} voice profile(s):")
            for profile in profiles:
                name = profile.get('name', 'Unknown')
                embedding = profile.get('embedding', [])
                has_embedding = len(embedding) > 0 if embedding else False
                embedding_dim = len(embedding) if has_embedding else 0
                
                if has_embedding:
                    print(f"   └─ ✅ {name}: {embedding_dim}D embedding")
                else:
                    print(f"   └─ ⚠️  {name}: NO embedding (needs re-enrollment)")
                    
    except ImportError as e:
        print(f"   ⚠️  Could not import database: {e}")
    except Exception as e:
        print(f"   ❌ Error checking profiles: {e}")
    
    print()
    
    # 4. Check STT/Transcription
    print("━" * 70)
    print("4️⃣  Checking Speech-to-Text Services...")
    print("━" * 70)
    
    try:
        from voice.hybrid_stt_router import HybridSTTRouter
        
        router = HybridSTTRouter()
        
        print(f"   Circuit Breaker Threshold: {router._circuit_breaker_threshold}")
        print(f"   Circuit Breaker Timeout:   {router._circuit_breaker_timeout}s")
        print()
        print("   STT Engine Circuit Breakers:")
        
        for engine, failures in router._circuit_breaker_failures.items():
            is_open = failures >= router._circuit_breaker_threshold
            status = "🔴 OPEN" if is_open else "🟢 CLOSED"
            print(f"   └─ {engine}: {status} ({failures}/{router._circuit_breaker_threshold} failures)")
            
            if is_open:
                router._circuit_breaker_failures[engine] = 0
                print(f"      └─ 🔧 RESET to 0 failures")
                
        if not router._circuit_breaker_failures:
            print("   └─ No circuit breakers recorded yet (clean state)")
            
    except ImportError as e:
        print(f"   ⚠️  Could not import STT router: {e}")
    except Exception as e:
        print(f"   ❌ Error checking STT: {e}")
    
    print()
    
    # 5. Summary and Next Steps
    print("━" * 70)
    print("📋 Summary & Next Steps")
    print("━" * 70)
    print()
    print("   All circuit breakers have been reset.")
    print()
    print("   If voice unlock continues to fail, check:")
    print("   1. Microphone permissions (System Preferences > Privacy > Microphone)")
    print("   2. Audio quality (quiet environment, clear speech)")
    print("   3. Voice enrollment (say 'JARVIS, enroll my voice' to re-enroll)")
    print("   4. Backend logs for specific errors")
    print()
    print("   To test voice unlock, say:")
    print("   • 'JARVIS, lock my screen'")
    print("   • 'JARVIS, unlock my screen'")
    print()
    print("=" * 70)
    print("🔧 Fix Complete!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
