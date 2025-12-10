#!/usr/bin/env python3
"""
🔍 VOICE PROFILE DIAGNOSTIC TOOL
═══════════════════════════════════════════════════════════════════════
Diagnoses voice profile issues in the VBI system.

Usage:
    python scripts/diagnose_voice_profiles.py

This script checks:
1. Profile count and validity
2. Embedding dimensions and norms
3. Embedding conversion from bytes to numpy
4. Cosine similarity between profiles
"""

import asyncio
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def print_header(title: str):
    """Print a header."""
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}\n")


def print_section(title: str):
    """Print a section."""
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}\n")


async def diagnose_profiles():
    """Run comprehensive profile diagnostics."""
    
    print_header("🔍 VOICE PROFILE DIAGNOSTIC TOOL")
    
    # ════════════════════════════════════════════════════════════════════
    # STEP 1: Check Learning Database
    # ════════════════════════════════════════════════════════════════════
    
    print_section("📚 Learning Database Check")
    
    try:
        from intelligence.learning_database import get_learning_database
        db = await get_learning_database()
        
        if not db:
            print("❌ Learning database not available!")
            return
        
        profiles = await db.get_all_speaker_profiles()
        print(f"✅ Found {len(profiles)} profile(s) in database")
        
        if not profiles:
            print("⚠️  No profiles found! Voice verification will fail.")
            print("    Solution: Enroll a voice profile using 'Learn my voice as [name]'")
            return
        
        # Analyze each profile
        for i, profile in enumerate(profiles):
            print(f"\n  Profile {i+1}:")
            print(f"    • Speaker Name: {profile.get('speaker_name', 'Unknown')}")
            print(f"    • Speaker ID: {profile.get('speaker_id', 'N/A')}")
            print(f"    • Is Primary User: {profile.get('is_primary_user', False)}")
            print(f"    • Total Samples: {profile.get('total_samples', 0)}")
            
            # Check embedding field
            embedding_raw = profile.get("embedding")  # Should be list (converted)
            voiceprint_raw = profile.get("voiceprint_embedding")  # Raw bytes
            
            print(f"\n    📊 Embedding Analysis:")
            print(f"       • 'embedding' field type: {type(embedding_raw).__name__}")
            print(f"       • 'voiceprint_embedding' field type: {type(voiceprint_raw).__name__}")
            
            # Test the converted embedding (what parallel_vbi_orchestrator uses)
            if embedding_raw is not None:
                if isinstance(embedding_raw, (list, tuple)):
                    emb_arr = np.array(embedding_raw, dtype=np.float32)
                    print(f"       ✅ 'embedding' is LIST with {len(embedding_raw)} elements")
                    print(f"       ✅ Converted to numpy: shape={emb_arr.shape}, dtype={emb_arr.dtype}")
                    print(f"       ✅ Norm: {np.linalg.norm(emb_arr):.6f}")
                    print(f"       ✅ Min/Max: {emb_arr.min():.6f} / {emb_arr.max():.6f}")
                    print(f"       ✅ Mean: {emb_arr.mean():.6f}")
                    
                    # Check for NaN/Inf
                    if np.any(np.isnan(emb_arr)):
                        print(f"       ❌ CONTAINS NaN VALUES!")
                    if np.any(np.isinf(emb_arr)):
                        print(f"       ❌ CONTAINS Inf VALUES!")
                        
                elif isinstance(embedding_raw, (bytes, bytearray)):
                    print(f"       ⚠️  'embedding' is still BYTES ({len(embedding_raw)} bytes)")
                    print(f"          This indicates the conversion in learning_database.py failed!")
                else:
                    print(f"       ⚠️  'embedding' is unexpected type: {type(embedding_raw)}")
            else:
                print(f"       ❌ 'embedding' field is None!")
            
            # Test raw voiceprint_embedding
            if voiceprint_raw is not None:
                if isinstance(voiceprint_raw, (bytes, bytearray)):
                    try:
                        raw_arr = np.frombuffer(voiceprint_raw, dtype=np.float32).copy()
                        print(f"       • 'voiceprint_embedding': {len(voiceprint_raw)} bytes -> {len(raw_arr)} floats")
                    except Exception as e:
                        print(f"       ❌ Failed to convert voiceprint_embedding: {e}")
                else:
                    print(f"       ⚠️  'voiceprint_embedding' is not bytes: {type(voiceprint_raw)}")
            else:
                print(f"       ⚠️  'voiceprint_embedding' is None")
    
    except Exception as e:
        print(f"❌ Learning database error: {e}")
        import traceback
        traceback.print_exc()
    
    # ════════════════════════════════════════════════════════════════════
    # STEP 2: Check Direct SQLite
    # ════════════════════════════════════════════════════════════════════
    
    print_section("💾 Direct SQLite Check")
    
    try:
        import aiosqlite
        
        db_paths = [
            os.path.expanduser("~/.jarvis/learning/jarvis_learning.db"),  # Primary location
            os.path.expanduser("~/.jarvis/jarvis_learning.db"),  # Legacy
            os.path.expanduser("~/Library/Application Support/JARVIS/jarvis_learning.db"),
            "data/jarvis_learning.db",
        ]
        
        db_path = None
        for path in db_paths:
            if os.path.exists(path):
                db_path = path
                break
        
        if db_path:
            print(f"✅ Found SQLite database: {db_path}")
            print(f"   Size: {os.path.getsize(db_path) / 1024:.1f} KB")
            
            async with aiosqlite.connect(db_path) as db:
                db.row_factory = aiosqlite.Row
                
                # Check speaker_profiles table
                async with db.execute(
                    "SELECT speaker_name, voiceprint_embedding, embedding_dimension FROM speaker_profiles"
                ) as cursor:
                    rows = await cursor.fetchall()
                    print(f"\n   📋 speaker_profiles table: {len(rows)} row(s)")
                    
                    for row in rows:
                        name = row["speaker_name"]
                        emb_bytes = row["voiceprint_embedding"]
                        dim = row["embedding_dimension"]
                        
                        if emb_bytes:
                            arr = np.frombuffer(emb_bytes, dtype=np.float32)
                            print(f"      • {name}: {len(arr)} dims, norm={np.linalg.norm(arr):.4f}, stored_dim={dim}")
                        else:
                            print(f"      • {name}: NO EMBEDDING")
        else:
            print("⚠️  SQLite database not found at expected paths")
            
    except Exception as e:
        print(f"❌ SQLite check error: {e}")
    
    # ════════════════════════════════════════════════════════════════════
    # STEP 3: Check Unified Voice Cache
    # ════════════════════════════════════════════════════════════════════
    
    print_section("📦 Unified Voice Cache Check")
    
    try:
        from voice_unlock.unified_voice_cache_manager import get_unified_voice_cache
        
        cache = await get_unified_voice_cache()
        if cache and cache.is_ready:
            profiles = cache.get_preloaded_profiles()
            print(f"✅ Unified cache ready with {len(profiles)} profile(s)")
            
            for name, profile in profiles.items():
                if profile.embedding is not None:
                    emb = np.array(profile.embedding, dtype=np.float32)
                    print(f"   • {name}: {len(emb)} dims, norm={np.linalg.norm(emb):.4f}")
                else:
                    print(f"   • {name}: NO EMBEDDING")
        else:
            print("⚠️  Unified voice cache not ready")
            
    except Exception as e:
        print(f"⚠️  Unified cache check: {e}")
    
    # ════════════════════════════════════════════════════════════════════
    # STEP 4: Test Similarity Calculation
    # ════════════════════════════════════════════════════════════════════
    
    print_section("🧮 Similarity Calculation Test")
    
    # Create a test embedding (random 192-dim vector, normalized)
    test_dim = 192
    test_emb = np.random.randn(test_dim).astype(np.float32)
    test_emb = test_emb / np.linalg.norm(test_emb)
    
    print(f"   Test embedding: {test_dim} dims, norm={np.linalg.norm(test_emb):.6f}")
    
    try:
        from intelligence.learning_database import get_learning_database
        db = await get_learning_database()
        
        if db:
            profiles = await db.get_all_speaker_profiles()
            
            for profile in profiles:
                speaker_name = profile.get("speaker_name", "Unknown")
                
                # Use the CORRECT field (embedding, not voiceprint_embedding)
                emb_raw = profile.get("embedding")
                
                if emb_raw is None:
                    print(f"   ❌ {speaker_name}: No 'embedding' field")
                    continue
                
                if isinstance(emb_raw, (list, tuple)):
                    profile_emb = np.array(emb_raw, dtype=np.float32)
                elif isinstance(emb_raw, (bytes, bytearray)):
                    profile_emb = np.frombuffer(emb_raw, dtype=np.float32)
                else:
                    print(f"   ⚠️  {speaker_name}: Unexpected type {type(emb_raw)}")
                    continue
                
                if len(profile_emb) != test_dim:
                    print(f"   ⚠️  {speaker_name}: Dimension mismatch ({len(profile_emb)} vs {test_dim})")
                    continue
                
                profile_emb = profile_emb / (np.linalg.norm(profile_emb) + 1e-10)
                similarity = float(np.dot(test_emb, profile_emb))
                
                print(f"   • {speaker_name}: similarity={similarity:.4f} (with random test)")
                
    except Exception as e:
        print(f"   ❌ Similarity test error: {e}")
    
    # ════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════════════════════
    
    print_header("📋 DIAGNOSTIC SUMMARY")
    
    print("Key points to check:")
    print("  1. Profiles should have 'embedding' as a LIST (not bytes)")
    print("  2. Embedding dimension should be 192 (ECAPA-TDNN)")
    print("  3. Embedding norm should be > 0")
    print("  4. No NaN or Inf values in embeddings")
    print("")
    print("If voice verification still fails:")
    print("  • Check the logs for '[VERIFY]' messages")
    print("  • Ensure the test audio embedding dimension matches profile dimension")
    print("  • Try re-enrolling the voice profile")
    print("")


if __name__ == "__main__":
    asyncio.run(diagnose_profiles())
