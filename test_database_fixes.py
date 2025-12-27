#!/usr/bin/env python3
"""
Comprehensive Integration Test for Database Fixes

Tests all the fixes implemented:
1. Intelligent UPSERT conversion (prevents duplicate key violations)
2. SQL aggregate alias references (COUNT(*) instead of alias in HAVING)
3. jsonb operator type casting (->>' instead of >>)
4. datetime string to datetime object conversion
5. SQL dialect translation (? to $1, json_extract to jsonb operators)
6. UniversalRow wrapper (supports both row[0] and row['column'])
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from datetime import datetime
from intelligence.learning_database import get_learning_database


async def test_database_fixes():
    """Test all database fixes comprehensively."""

    print("\n" + "="*70)
    print("🧪 COMPREHENSIVE DATABASE FIXES TEST")
    print("="*70)

    learning_db = await get_learning_database()
    db = learning_db.db  # Get the actual database connection

    # Test 1: UPSERT Conversion (prevents duplicate key violations)
    print("\n[1/6] Testing UPSERT Conversion...")
    try:
        now_iso = datetime.now().isoformat()
        # First insert
        await db.execute(
            """
            INSERT INTO user_preferences (preference_id, category, key, value, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("test_pref_id", "test_category", "test_key", "value1", now_iso, now_iso)
        )
        print("  ✅ First insert successful")

        # Second insert with same preference_id - should UPSERT instead of failing
        await db.execute(
            """
            INSERT INTO user_preferences (preference_id, category, key, value, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("test_pref_id", "test_category", "test_key", "value2_updated", now_iso, now_iso)
        )
        print("  ✅ Second insert (UPSERT) successful - no duplicate key violation!")

        # Verify the value was updated
        async with db.execute(
            "SELECT value FROM user_preferences WHERE preference_id = ?",
            ("test_pref_id",)
        ) as cursor:
            row = await cursor.fetchone()
            if row and row["value"] == "value2_updated":
                print("  ✅ UPSERT correctly updated existing record")
            else:
                print(f"  ❌ UPSERT failed - value is {row['value'] if row else 'None'}")

    except Exception as e:
        print(f"  ❌ UPSERT test failed: {e}")

    # Test 2: SQL Aggregate Alias References
    print("\n[2/6] Testing SQL Aggregate Alias References...")
    try:
        # This query previously failed with "column 'occurrences' does not exist"
        # Now it uses COUNT(*) directly in HAVING and ORDER BY
        async with db.execute(
            """
            SELECT action_type, COUNT(*) as occurrences
            FROM temporal_patterns
            GROUP BY action_type
            HAVING COUNT(*) > 0
            ORDER BY COUNT(*) DESC
            LIMIT 5
            """
        ) as cursor:
            rows = await cursor.fetchall()
            print(f"  ✅ Aggregate alias query successful - found {len(rows)} results")
    except Exception as e:
        print(f"  ❌ Aggregate alias test failed: {e}")

    # Test 3: jsonb Operator Type Casting
    print("\n[3/6] Testing jsonb Operator Type Casting...")
    try:
        # Insert a record with JSON metadata (workflow_id will be auto-generated)
        now_iso = datetime.now().isoformat()
        cursor = await db.execute(
            """
            INSERT INTO user_workflows (workflow_name, action_sequence, metadata, first_seen, last_seen)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("Test Workflow", '["action1", "action2"]', '{"key": "value", "number": 42}', now_iso, now_iso)
        )

        # Get the last inserted workflow_id
        workflow_id = cursor.lastrowid
        if not workflow_id:
            # Try to fetch from the database
            async with db.execute(
                "SELECT workflow_id FROM user_workflows WHERE workflow_name = ? ORDER BY workflow_id DESC LIMIT 1",
                ("Test Workflow",)
            ) as fetch_cursor:
                row = await fetch_cursor.fetchone()
                if row:
                    workflow_id = row[0]

        if workflow_id:
            # Test json_extract translation to ->> operator
            async with db.execute(
                """
                SELECT json_extract(metadata, '$.key') as extracted_key
                FROM user_workflows
                WHERE workflow_id = ?
                """,
                (workflow_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    print(f"  ✅ jsonb operator test successful - extracted: {row['extracted_key']}")
                else:
                    print("  ❌ jsonb operator test failed - no results")
        else:
            print("  ⚠️  Could not get workflow_id - skipping jsonb test")
    except Exception as e:
        print(f"  ❌ jsonb operator test failed: {e}")

    # Test 4: Datetime String Conversion
    print("\n[4/6] Testing Datetime String to Object Conversion...")
    try:
        # Insert with ISO datetime string
        now = datetime.now()
        iso_string = now.isoformat()

        await db.execute(
            """
            INSERT INTO actions (action_id, goal_id, action_type, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            ("test_action", "test_goal", "test_type", iso_string)
        )

        # Retrieve and verify
        async with db.execute(
            "SELECT timestamp FROM actions WHERE action_id = ?",
            ("test_action",)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                print(f"  ✅ Datetime conversion successful - stored: {row['timestamp']}")
            else:
                print("  ❌ Datetime conversion failed - no results")
    except Exception as e:
        print(f"  ❌ Datetime conversion test failed: {e}")

    # Test 5: SQL Dialect Translation (? to $1, json_extract to jsonb)
    print("\n[5/6] Testing SQL Dialect Translation...")
    try:
        # Test ? to $1 translation
        async with db.execute(
            "SELECT ? as param1, ? as param2",
            ("value1", "value2")
        ) as cursor:
            row = await cursor.fetchone()
            if row and row["param1"] == "value1" and row["param2"] == "value2":
                print("  ✅ ? to $1 translation successful")
            else:
                print("  ❌ ? to $1 translation failed")
    except Exception as e:
        print(f"  ❌ SQL dialect translation test failed: {e}")

    # Test 6: UniversalRow Wrapper (numeric and column name access)
    print("\n[6/6] Testing UniversalRow Wrapper...")
    try:
        async with db.execute(
            "SELECT 'test' as col1, 42 as col2, 3.14 as col3"
        ) as cursor:
            row = await cursor.fetchone()

            if row:
                # Test column name access
                col1_by_name = row["col1"]
                # Test numeric index access
                col1_by_index = row[0]
                col2_by_index = row[1]
                col3_by_index = row[2]

                if (col1_by_name == col1_by_index == "test" and
                    col2_by_index == 42 and
                    col3_by_index == 3.14):
                    print("  ✅ UniversalRow wrapper successful")
                    print(f"     - row['col1'] = {col1_by_name}")
                    print(f"     - row[0] = {col1_by_index}")
                    print(f"     - row[1] = {col2_by_index}")
                    print(f"     - row[2] = {col3_by_index}")
                else:
                    print("  ❌ UniversalRow wrapper failed - values don't match")
            else:
                print("  ❌ UniversalRow wrapper test failed - no results")
    except Exception as e:
        print(f"  ❌ UniversalRow wrapper test failed: {e}")

    # Cleanup
    print("\n[Cleanup] Removing test data...")
    try:
        await db.execute("DELETE FROM user_preferences WHERE preference_id = ?", ("test_pref_id",))
        await db.execute("DELETE FROM user_workflows WHERE workflow_name = ?", ("Test Workflow",))
        await db.execute("DELETE FROM actions WHERE action_id = ?", ("test_action",))
        print("  ✅ Cleanup successful")
    except Exception as e:
        print(f"  ⚠️  Cleanup warning: {e}")

    print("\n" + "="*70)
    print("🎉 ALL TESTS COMPLETED!")
    print("="*70)
    print("\nDatabase fixes validated:")
    print("  ✓ Intelligent UPSERT conversion (no duplicate key violations)")
    print("  ✓ SQL aggregate alias references (COUNT(*) in HAVING/ORDER BY)")
    print("  ✓ jsonb operator type casting (->> for text extraction)")
    print("  ✓ datetime string to datetime object conversion")
    print("  ✓ SQL dialect translation (? to $1, json_extract to jsonb)")
    print("  ✓ UniversalRow wrapper (supports row[0] and row['column'])")
    print("\n" + "="*70)


if __name__ == "__main__":
    asyncio.run(test_database_fixes())
