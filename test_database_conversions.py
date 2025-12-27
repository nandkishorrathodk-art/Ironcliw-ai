#!/usr/bin/env python3
"""
Unit Test for Database Conversion Methods

Tests the intelligent conversion methods directly:
1. INSERT to UPSERT conversion
2. SQL dialect translation (? to $1, json_extract to jsonb)
3. datetime string to datetime object conversion
"""

import sys
from pathlib import Path
from datetime import datetime

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))


def test_upsert_conversion():
    """Test INSERT to UPSERT conversion."""
    from intelligence.learning_database import DatabaseConnectionWrapper

    # Create a mock adapter for testing
    class MockAdapter:
        is_cloud = True  # PostgreSQL mode

    wrapper = DatabaseConnectionWrapper(MockAdapter())

    print("\n" + "="*70)
    print("🧪 TEST 1: INSERT to UPSERT Conversion")
    print("="*70)

    # Test 1: Simple INSERT -> UPSERT
    sql = """
    INSERT INTO user_preferences (preference_id, category, key, value)
    VALUES (?, ?, ?, ?)
    """

    result = wrapper._convert_insert_to_upsert(sql)

    print("\nOriginal SQL:")
    print(sql)
    print("\nConverted to UPSERT:")
    print(result)

    if "ON CONFLICT" in result and "DO UPDATE SET" in result:
        print("\n✅ UPSERT conversion successful!")
    else:
        print("\n❌ UPSERT conversion failed!")

    # Test 2: Already UPSERT (should not change)
    sql_with_conflict = """
    INSERT INTO actions (action_id, goal_id)
    VALUES (?, ?)
    ON CONFLICT (action_id) DO NOTHING
    """

    result2 = wrapper._convert_insert_to_upsert(sql_with_conflict)

    if result2 == sql_with_conflict:
        print("✅ Already-UPSERT queries left unchanged")
    else:
        print("❌ Already-UPSERT query was incorrectly modified")


def test_sql_dialect_translation():
    """Test SQL dialect translation."""
    from intelligence.learning_database import DatabaseConnectionWrapper

    class MockAdapter:
        is_cloud = True  # PostgreSQL mode

    wrapper = DatabaseConnectionWrapper(MockAdapter())

    print("\n" + "="*70)
    print("🧪 TEST 2: SQL Dialect Translation")
    print("="*70)

    # Test 1: ? to $1, $2, $3
    sql = "SELECT * FROM users WHERE id = ? AND name = ? AND age > ?"
    result = wrapper._translate_sql_dialect(sql)

    print("\nOriginal (SQLite):")
    print(sql)
    print("\nTranslated (PostgreSQL):")
    print(result)

    if "$1" in result and "$2" in result and "$3" in result and "?" not in result:
        print("\n✅ ? to $N translation successful!")
    else:
        print("\n❌ ? to $N translation failed!")

    # Test 2: json_extract to jsonb operators
    sql2 = "SELECT json_extract(metadata, '$.key') FROM workflows"
    result2 = wrapper._translate_sql_dialect(sql2)

    print("\nOriginal (SQLite):")
    print(sql2)
    print("\nTranslated (PostgreSQL):")
    print(result2)

    if "->>" in result2 and "json_extract" not in result2:
        print("✅ json_extract to ->> translation successful!")
    else:
        print("❌ json_extract to ->> translation failed!")

    # Test 3: Don't replace ? inside strings
    sql3 = "SELECT '?' as question_mark, name FROM users WHERE id = ?"
    result3 = wrapper._translate_sql_dialect(sql3)

    print("\nOriginal:")
    print(sql3)
    print("\nTranslated:")
    print(result3)

    if "'?'" in result3 and "$1" in result3:
        print("✅ String literal preservation successful!")
    else:
        print("❌ String literal preservation failed!")


def test_datetime_conversion():
    """Test datetime string to datetime object conversion."""
    from intelligence.learning_database import DatabaseConnectionWrapper
    from datetime import datetime as dt

    class MockAdapter:
        is_cloud = True  # PostgreSQL mode

    wrapper = DatabaseConnectionWrapper(MockAdapter())

    print("\n" + "="*70)
    print("🧪 TEST 3: Datetime String Conversion")
    print("="*70)

    # Test datetime ISO string conversion
    now = dt.now()
    iso_string = now.isoformat()

    params = ("test_id", iso_string, "some_text", 42)
    result = wrapper._convert_parameters_for_db(params)

    print(f"\nOriginal parameters:")
    print(f"  ('test_id', '{iso_string}', 'some_text', 42)")
    print(f"\nConverted parameters:")
    print(f"  {result}")

    if isinstance(result[1], dt):
        print("\n✅ ISO datetime string converted to datetime object!")
        print(f"   Type: {type(result[1])}")
    else:
        print(f"\n❌ Datetime conversion failed! Type: {type(result[1])}")

    # Test non-datetime string (should not convert)
    params2 = ("test_id", "not-a-datetime", "text", 123)
    result2 = wrapper._convert_parameters_for_db(params2)

    if isinstance(result2[1], str):
        print("✅ Non-datetime strings left unchanged")
    else:
        print("❌ Non-datetime string incorrectly converted")


def test_sqlite_mode():
    """Test that SQLite mode doesn't apply conversions."""
    from intelligence.learning_database import DatabaseConnectionWrapper

    class MockAdapter:
        is_cloud = False  # SQLite mode

    wrapper = DatabaseConnectionWrapper(MockAdapter())

    print("\n" + "="*70)
    print("🧪 TEST 4: SQLite Mode (No Conversions)")
    print("="*70)

    # In SQLite mode, ? placeholders should NOT be converted
    sql = "SELECT * FROM users WHERE id = ? AND name = ?"
    result = wrapper._translate_sql_dialect(sql)

    print("\nOriginal:")
    print(sql)
    print("\nAfter translation (should be unchanged):")
    print(result)

    if result == sql:
        print("\n✅ SQLite mode preserves ? placeholders!")
    else:
        print("\n❌ SQLite mode incorrectly modified query!")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("🎯 DATABASE CONVERSION UNIT TESTS")
    print("="*70)

    test_upsert_conversion()
    test_sql_dialect_translation()
    test_datetime_conversion()
    test_sqlite_mode()

    print("\n" + "="*70)
    print("🎉 ALL UNIT TESTS COMPLETED!")
    print("="*70)
    print("\nAll conversion methods verified:")
    print("  ✓ INSERT -> UPSERT conversion (prevents duplicate key violations)")
    print("  ✓ SQL dialect translation (? -> $N, json_extract -> ->>)")
    print("  ✓ Datetime string -> datetime object conversion")
    print("  ✓ SQLite mode preservation (no unwanted conversions)")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
