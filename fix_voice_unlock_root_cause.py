#!/usr/bin/env python3
"""
Voice Unlock Root Cause Fix
===========================
This script diagnoses and fixes the ROOT CAUSE of voice unlock failures,
not just the circuit breaker symptoms.

The chain of failure:
1. ECAPA model initialization fails
2. Each embedding extraction attempt fails
3. Circuit breaker trips after 5 failures
4. Voice unlock is blocked

This script:
1. Diagnoses WHY ECAPA is failing
2. Fixes the underlying issues
3. Re-initializes the model
4. Resets circuit breakers
5. Verifies the fix worked

Run: python3 fix_voice_unlock_root_cause.py
"""

import asyncio
import sys
import os
import time
import shutil

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Suppress excessive logging during diagnosis
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def print_header(text):
    print()
    print("=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_section(text):
    print()
    print(f"━━━ {text} ━━━")

def print_ok(text):
    print(f"  ✅ {text}")

def print_warn(text):
    print(f"  ⚠️  {text}")

def print_error(text):
    print(f"  ❌ {text}")

def print_info(text):
    print(f"  ℹ️  {text}")

async def diagnose_ecapa():
    """Diagnose ECAPA model issues."""
    print_section("1. ECAPA-TDNN Model Diagnosis")
    
    issues = []
    fixes_applied = []
    
    try:
        from cloud_services.ecapa_cloud_service import (
            get_model_manager,
            CloudECAPAConfig, 
            CircuitState
        )
        
        # Check environment
        print_info(f"STRICT_OFFLINE: {CloudECAPAConfig.STRICT_OFFLINE}")
        print_info(f"Cache locations: {CloudECAPAConfig.get_cache_locations()}")
        
        # Check cache
        cache_dir, cache_msg, diagnostics = CloudECAPAConfig.find_valid_cache()
        
        if cache_dir:
            print_ok(f"Valid cache found: {cache_dir}")
            
            # Check cache contents
            required_files = ['hyperparams.yaml', 'embedding_model.ckpt']
            for req_file in required_files:
                path = os.path.join(cache_dir, req_file)
                if os.path.exists(path):
                    size_mb = os.path.getsize(path) / (1024 * 1024)
                    print_ok(f"  {req_file}: {size_mb:.2f} MB")
                else:
                    print_error(f"  {req_file}: MISSING")
                    issues.append(f"Missing required file: {req_file}")
        else:
            print_error(f"No valid cache: {cache_msg}")
            issues.append("No valid ECAPA model cache found")
            
            # Check if we can create one
            print_info("Checking if model can be downloaded...")
            if CloudECAPAConfig.STRICT_OFFLINE:
                print_error("STRICT_OFFLINE=true prevents downloading")
                issues.append("Cannot download in offline mode")
            else:
                print_warn("Model will be downloaded on first use (slow)")
        
        # Get singleton instance
        ecapa = get_model_manager()
        
        # Check current state
        print()
        print_info(f"Model ready: {ecapa.is_ready}")
        print_info(f"Model loading: {ecapa._loading}")
        print_info(f"Last error: {ecapa._error}")
        
        # Circuit breaker state
        cb = ecapa.circuit_breaker
        print_info(f"Circuit breaker: {cb.state.name}")
        print_info(f"Failure count: {cb.failure_count}/{cb.failure_threshold}")
        
        if cb.state == CircuitState.OPEN:
            issues.append("Circuit breaker is OPEN")
            
            # Reset it
            cb.state = CircuitState.CLOSED
            cb.failure_count = 0
            cb.last_failure_time = None
            print_ok("Circuit breaker RESET")
            fixes_applied.append("Reset ECAPA circuit breaker")
        
        # If model not ready, try to initialize
        if not ecapa.is_ready:
            print()
            print_info("Attempting to initialize ECAPA model...")
            
            # Clear any previous error
            ecapa._error = None
            ecapa._ready = False
            
            try:
                init_result = await ecapa.initialize()
                
                if init_result:
                    print_ok("ECAPA model initialized successfully!")
                    print_ok(f"  Load source: {ecapa.load_source}")
                    print_ok(f"  Load time: {ecapa.load_time_ms:.0f}ms")
                    fixes_applied.append("Initialized ECAPA model")
                else:
                    print_error("ECAPA initialization returned False")
                    if ecapa._error:
                        print_error(f"  Error: {ecapa._error}")
                        issues.append(f"Init failed: {ecapa._error}")
                    
            except Exception as e:
                print_error(f"ECAPA initialization exception: {e}")
                issues.append(f"Init exception: {str(e)}")
        else:
            print_ok("ECAPA model already initialized")
        
        return ecapa, issues, fixes_applied
        
    except ImportError as e:
        print_error(f"Cannot import ECAPA service: {e}")
        return None, [f"Import error: {e}"], []
    except Exception as e:
        print_error(f"Diagnosis error: {e}")
        import traceback
        traceback.print_exc()
        return None, [f"Diagnosis error: {e}"], []


async def diagnose_voice_unlock_service():
    """Diagnose IntelligentVoiceUnlockService."""
    print_section("2. Voice Unlock Service Diagnosis")
    
    issues = []
    fixes_applied = []
    
    try:
        from voice_unlock.intelligent_voice_unlock_service import IntelligentVoiceUnlockService
        
        service = IntelligentVoiceUnlockService()
        
        print_info(f"Circuit breaker threshold: {service.circuit_breaker_threshold}")
        print_info(f"Circuit breaker timeout: {service.circuit_breaker_timeout}s")
        
        # Check circuit breakers
        open_breakers = []
        for name, failures in service._circuit_breaker_failures.items():
            is_open = failures >= service.circuit_breaker_threshold
            if is_open:
                open_breakers.append(name)
                print_warn(f"  {name}: OPEN ({failures} failures)")
                
                # Reset
                service._circuit_breaker_failures[name] = 0
                print_ok(f"  {name}: RESET")
                fixes_applied.append(f"Reset {name} circuit breaker")
            else:
                print_ok(f"  {name}: CLOSED ({failures} failures)")
        
        if not service._circuit_breaker_failures:
            print_ok("No circuit breakers recorded (clean state)")
        
        if open_breakers:
            issues.append(f"Open circuit breakers: {', '.join(open_breakers)}")
        
        return service, issues, fixes_applied
        
    except ImportError as e:
        print_error(f"Cannot import service: {e}")
        return None, [f"Import error: {e}"], []
    except Exception as e:
        print_error(f"Diagnosis error: {e}")
        return None, [f"Diagnosis error: {e}"], []


async def diagnose_voice_profiles():
    """Check voice profiles in database."""
    print_section("3. Voice Profile Diagnosis")
    
    issues = []
    
    try:
        from intelligence.learning_database import JARVISLearningDatabase
        
        db = JARVISLearningDatabase()
        await db.initialize()
        
        profiles = await db.get_all_speaker_profiles()
        
        if not profiles:
            print_error("No voice profiles found!")
            print_info("Voice unlock requires enrolled voice profiles")
            print_info("Say 'JARVIS, enroll my voice' to create one")
            issues.append("No voice profiles enrolled")
            return issues
        
        print_ok(f"Found {len(profiles)} voice profile(s)")
        
        for profile in profiles:
            name = profile.get('name', 'Unknown')
            speaker_id = profile.get('speaker_id', 'N/A')
            embedding = profile.get('embedding', [])
            
            if embedding and len(embedding) > 0:
                dim = len(embedding)
                expected = 192  # ECAPA-TDNN standard
                
                if dim == expected:
                    print_ok(f"  {name} (ID: {speaker_id}): {dim}D embedding ✓")
                else:
                    print_warn(f"  {name} (ID: {speaker_id}): {dim}D (expected {expected}D)")
                    issues.append(f"Profile {name} has wrong embedding dimension")
            else:
                print_error(f"  {name} (ID: {speaker_id}): NO EMBEDDING")
                print_info(f"    Re-enroll this speaker's voice")
                issues.append(f"Profile {name} has no embedding")
        
        return issues
        
    except ImportError as e:
        print_error(f"Cannot import database: {e}")
        return [f"Database import error: {e}"]
    except Exception as e:
        print_error(f"Profile check error: {e}")
        return [f"Profile check error: {e}"]


async def test_embedding_extraction(ecapa):
    """Test that embedding extraction works."""
    print_section("4. Embedding Extraction Test")
    
    if not ecapa or not ecapa.is_ready:
        print_error("ECAPA not ready, skipping test")
        return False
    
    try:
        import numpy as np
        
        # Create synthetic audio (1 second of random noise - just for testing)
        # Real audio would come from microphone
        sample_rate = 16000
        duration = 1.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1
        
        print_info("Extracting embedding from test audio...")
        
        start = time.time()
        embedding = await ecapa.extract_embedding(audio, use_cache=False)
        elapsed = (time.time() - start) * 1000
        
        if embedding is not None:
            print_ok(f"Embedding extracted in {elapsed:.0f}ms")
            print_ok(f"  Shape: {embedding.shape}")
            print_ok(f"  Dimension: {embedding.shape[0]}")
            
            # Verify dimension
            if embedding.shape[0] == 192:
                print_ok("  Dimension correct (192D)")
                return True
            else:
                print_warn(f"  Unexpected dimension (expected 192)")
                return False
        else:
            print_error("Embedding extraction returned None")
            return False
            
    except Exception as e:
        print_error(f"Embedding extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    print_header("JARVIS Voice Unlock Root Cause Fix")
    print()
    print("This script diagnoses and fixes why voice unlock isn't working.")
    print("It goes beyond resetting circuit breakers to fix the underlying issues.")
    
    all_issues = []
    all_fixes = []
    
    # 1. Diagnose ECAPA
    ecapa, issues, fixes = await diagnose_ecapa()
    all_issues.extend(issues)
    all_fixes.extend(fixes)
    
    # 2. Diagnose Voice Unlock Service
    service, issues, fixes = await diagnose_voice_unlock_service()
    all_issues.extend(issues)
    all_fixes.extend(fixes)
    
    # 3. Diagnose Voice Profiles
    issues = await diagnose_voice_profiles()
    all_issues.extend(issues)
    
    # 4. Test embedding extraction (only if ECAPA is ready)
    if ecapa and ecapa.is_ready:
        test_passed = await test_embedding_extraction(ecapa)
        if not test_passed:
            all_issues.append("Embedding extraction test failed")
    else:
        print_section("4. Embedding Extraction Test")
        print_warn("Skipped - ECAPA not ready")
        all_issues.append("Cannot test embedding - ECAPA not ready")
    
    # Summary
    print_header("SUMMARY")
    
    if all_fixes:
        print()
        print("Fixes Applied:")
        for fix in all_fixes:
            print(f"  ✅ {fix}")
    
    if all_issues:
        print()
        print("Remaining Issues:")
        for issue in all_issues:
            print(f"  ⚠️  {issue}")
        
        print()
        print("Recommended Actions:")
        
        if any("No voice profiles" in i for i in all_issues):
            print("  1. Enroll your voice: Say 'JARVIS, enroll my voice'")
        
        if any("no embedding" in i.lower() for i in all_issues):
            print("  2. Re-enroll speakers with missing embeddings")
        
        if any("ECAPA" in i or "Init" in i for i in all_issues):
            print("  3. Check ECAPA model installation:")
            print("     - Ensure SpeechBrain is installed: pip install speechbrain")
            print("     - Check model cache at ~/.cache/huggingface/")
            print("     - Try: python -c 'from speechbrain.pretrained import EncoderClassifier; print(\"OK\")'")
        
        if any("circuit breaker" in i.lower() for i in all_issues):
            print("  4. Circuit breakers were reset - retry voice unlock")
    else:
        print()
        print_ok("All checks passed! Voice unlock should work.")
    
    print()
    print_header("NEXT STEPS")
    print()
    print("  1. Restart JARVIS backend:  ./start_jarvis.sh")
    print("  2. Test voice unlock:       Say 'JARVIS, lock my screen'")
    print("  3. Check logs for errors:   tail -f logs/jarvis.log")
    print()
    
    return len(all_issues) == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
