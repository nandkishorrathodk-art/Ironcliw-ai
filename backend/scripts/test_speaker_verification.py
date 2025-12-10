#!/usr/bin/env python3
"""
🔐 SPEAKER VERIFICATION TEST
═══════════════════════════════════════════════════════════════════════
Tests the full speaker verification flow to identify issues.

Usage:
    python scripts/test_speaker_verification.py
"""

import asyncio
import os
import sys
import struct
import sqlite3

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_header(title: str):
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}\n")


def print_section(title: str):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


def get_profile_from_sqlite():
    """Get profile directly from SQLite."""
    db_path = os.path.expanduser("~/.jarvis/learning/jarvis_learning.db")
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found at {db_path}")
        return None
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT speaker_name, voiceprint_embedding, embedding_dimension FROM speaker_profiles")
    rows = cursor.fetchall()
    
    conn.close()
    
    if not rows:
        print("❌ No profiles in database!")
        return None
    
    row = rows[0]
    emb_bytes = row["voiceprint_embedding"]
    
    if not emb_bytes:
        print("❌ Profile has no embedding!")
        return None
    
    # Convert bytes to list of floats
    num_floats = len(emb_bytes) // 4
    floats = list(struct.unpack(f'{num_floats}f', emb_bytes))
    
    return {
        "speaker_name": row["speaker_name"],
        "embedding": floats,
        "dimension": row["embedding_dimension"],
    }


async def test_learning_database():
    """Test the learning database get_all_speaker_profiles."""
    print_section("Testing get_all_speaker_profiles()")
    
    try:
        from intelligence.learning_database import get_learning_database
        
        db = await get_learning_database()
        print(f"✅ Learning database initialized")
        print(f"   Path: {db.sqlite_path}")
        
        profiles = await db.get_all_speaker_profiles()
        print(f"✅ get_all_speaker_profiles() returned {len(profiles)} profiles")
        
        if not profiles:
            print("❌ PROBLEM: No profiles returned!")
            print("   This is why verification fails!")
            return None
        
        for profile in profiles:
            name = profile.get("speaker_name", "Unknown")
            emb = profile.get("embedding")
            vp = profile.get("voiceprint_embedding")
            
            print(f"\n   Profile: {name}")
            print(f"   - 'embedding' type: {type(emb).__name__}")
            print(f"   - 'voiceprint_embedding' type: {type(vp).__name__}")
            
            if isinstance(emb, list):
                print(f"   - 'embedding' length: {len(emb)}")
                # Calculate norm
                norm = sum(x*x for x in emb) ** 0.5
                print(f"   - 'embedding' norm: {norm:.6f}")
                print(f"   ✅ 'embedding' is correctly converted to list!")
            elif isinstance(emb, bytes):
                print(f"   ❌ 'embedding' is still bytes! Conversion failed!")
            elif emb is None:
                print(f"   ❌ 'embedding' is None!")
            
            return profile
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_cosine_similarity(profile):
    """Test cosine similarity calculation."""
    print_section("Testing Cosine Similarity")
    
    if not profile:
        print("❌ No profile to test")
        return
    
    emb = profile.get("embedding")
    if not emb or not isinstance(emb, list):
        print("❌ Profile embedding is not a list")
        return
    
    # Create a test embedding (same as profile - should give 1.0 similarity)
    test_emb = emb.copy()
    
    # Calculate norm
    def norm(vec):
        return sum(x*x for x in vec) ** 0.5
    
    def cosine_sim(a, b):
        dot = sum(x*y for x, y in zip(a, b))
        norm_a = norm(a)
        norm_b = norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return dot / (norm_a * norm_b)
    
    # Test 1: Same embedding should give ~1.0
    sim = cosine_sim(test_emb, emb)
    print(f"   Test 1 - Same embedding: similarity = {sim:.6f}")
    if sim > 0.99:
        print("   ✅ Correct! Same embedding gives ~1.0")
    else:
        print("   ❌ Problem! Same embedding should give ~1.0")
    
    # Test 2: Slightly perturbed embedding
    perturbed = [x + 0.01 for x in emb]
    sim = cosine_sim(perturbed, emb)
    print(f"   Test 2 - Slightly perturbed: similarity = {sim:.6f}")
    if sim > 0.9:
        print("   ✅ Correct! Similar embedding gives high similarity")
    
    # Test 3: Random embedding (should give low similarity)
    import random
    random_emb = [random.gauss(0, 0.1) for _ in range(len(emb))]
    sim = cosine_sim(random_emb, emb)
    print(f"   Test 3 - Random embedding: similarity = {sim:.6f}")
    if sim < 0.5:
        print("   ✅ Correct! Random embedding gives low similarity")


async def main():
    print_header("🔐 SPEAKER VERIFICATION TEST")
    
    # Test 1: Direct SQLite access
    print_section("Direct SQLite Check")
    sqlite_profile = get_profile_from_sqlite()
    
    if sqlite_profile:
        print(f"✅ Found profile: {sqlite_profile['speaker_name']}")
        print(f"   Dimension: {sqlite_profile['dimension']}")
        print(f"   Embedding length: {len(sqlite_profile['embedding'])}")
        
        # Calculate norm
        emb = sqlite_profile['embedding']
        norm = sum(x*x for x in emb) ** 0.5
        print(f"   Embedding norm: {norm:.6f}")
    
    # Test 2: Learning database
    db_profile = await test_learning_database()
    
    # Test 3: Cosine similarity
    test_cosine_similarity(db_profile or sqlite_profile)
    
    # Summary
    print_header("📋 SUMMARY")
    
    if sqlite_profile:
        print("✅ SQLite profile exists and is valid")
    else:
        print("❌ SQLite profile missing or invalid")
    
    if db_profile:
        emb = db_profile.get("embedding")
        if isinstance(emb, list) and len(emb) > 0:
            print("✅ Learning database returns profile with correct 'embedding' list")
        else:
            print("❌ Learning database 'embedding' field is not a list!")
            print("   This is likely the root cause of 0% confidence!")
    else:
        print("❌ Learning database returned no profiles!")
        print("   This is likely the root cause of 0% confidence!")
    
    print("\n" + "═" * 70)
    print("  If issues found, restart JARVIS and try again:")
    print("    python3 start_system.py --restart")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
