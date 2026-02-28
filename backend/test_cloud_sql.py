#!/usr/bin/env python3
"""
Test Cloud SQL connection directly

v132.0: Uses TLS-safe factory to prevent asyncpg TLS race conditions.
v133.0: Uses IntelligentCredentialResolver for auto-credential resolution.
"""
import asyncio
import json
import logging
import os
from pathlib import Path

import asyncpg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# v132.0: TLS-Safe Connection Factory Import
_TLS_SAFE_FACTORY_AVAILABLE = False
tls_safe_connect = None

try:
    from intelligence.cloud_sql_connection_manager import tls_safe_connect as _tls_safe_connect
    tls_safe_connect = _tls_safe_connect
    _TLS_SAFE_FACTORY_AVAILABLE = True
except ImportError:
    try:
        from backend.intelligence.cloud_sql_connection_manager import tls_safe_connect as _tls_safe_connect
        tls_safe_connect = _tls_safe_connect
        _TLS_SAFE_FACTORY_AVAILABLE = True
    except ImportError:
        logger.debug("[TestCloudSQL] TLS-safe factory not available")


def _load_database_config() -> dict:
    """v133.0: Load database config from multiple sources."""
    # Priority 1: Environment variable
    if os.getenv("Ironcliw_DB_PASSWORD"):
        return {
            "host": os.getenv("Ironcliw_DB_HOST", "127.0.0.1"),
            "port": int(os.getenv("Ironcliw_DB_PORT", "5432")),
            "database": os.getenv("Ironcliw_DB_NAME", "jarvis_learning"),
            "user": os.getenv("Ironcliw_DB_USER", "jarvis"),
            "password": os.getenv("Ironcliw_DB_PASSWORD"),
        }

    # Priority 2: Config file
    config_path = Path.home() / ".jarvis" / "gcp" / "database_config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                data = json.load(f)
                cloud_sql = data.get("cloud_sql", {})
                if cloud_sql.get("password"):
                    logger.info(f"[TestCloudSQL] Loaded credentials from {config_path}")
                    return {
                        "host": "127.0.0.1",  # Always use proxy for local
                        "port": cloud_sql.get("port", 5432),
                        "database": cloud_sql.get("database", "jarvis_learning"),
                        "user": cloud_sql.get("user", "jarvis"),
                        "password": cloud_sql.get("password"),
                    }
        except Exception as e:
            logger.warning(f"[TestCloudSQL] Failed to load config: {e}")

    # Fallback: defaults (will likely fail)
    logger.warning("[TestCloudSQL] No credentials found, using defaults")
    return {
        "host": "127.0.0.1",
        "port": 5432,
        "database": "jarvis_learning",
        "user": "jarvis",
        "password": "",
    }


async def test_connection():
    """Test direct connection to Cloud SQL via proxy"""
    config = _load_database_config()

    logger.info(f"Connecting to Cloud SQL at {config['host']}:{config['port']}")
    logger.info(f"Database: {config['database']}, User: {config['user']}")

    try:
        # v132.0: Use TLS-safe factory to prevent race conditions
        if _TLS_SAFE_FACTORY_AVAILABLE and tls_safe_connect is not None:
            logger.info("Using TLS-safe connection factory...")
            conn = await tls_safe_connect(
                host=config["host"],
                port=config["port"],
                database=config["database"],
                user=config["user"],
                password=config["password"],
            )
        else:
            logger.warning("TLS-safe factory not available, using direct asyncpg")
            conn = await asyncpg.connect(**config)

        if not conn:
            raise RuntimeError("Connection returned None")
        logger.info("✅ Connection successful!")

        # Test query
        result = await conn.fetchval("SELECT version();")
        logger.info(f"Database version: {result}")

        # Check for speaker_profiles table
        exists = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = 'speaker_profiles'
            );
        """
        )

        if exists:
            logger.info("✅ speaker_profiles table exists")

            # Count profiles
            count = await conn.fetchval("SELECT COUNT(*) FROM speaker_profiles;")
            logger.info(f"Total speaker profiles: {count}")

            # Look for Derek's profile
            derek = await conn.fetchrow(
                """
                SELECT speaker_id, speaker_name, is_primary_user,
                       LENGTH(voiceprint_embedding) as embedding_size
                FROM speaker_profiles
                WHERE speaker_name ILIKE '%derek%'
                LIMIT 1;
            """
            )

            if derek:
                logger.info(f"✅ Found Derek's profile:")
                logger.info(f"  - ID: {derek['speaker_id']}")
                logger.info(f"  - Name: {derek['speaker_name']}")
                logger.info(f"  - Primary User: {derek['is_primary_user']}")
                logger.info(f"  - Embedding Size: {derek['embedding_size']} bytes")
            else:
                logger.warning("❌ No profile found for Derek")
        else:
            logger.warning("❌ speaker_profiles table does not exist")

            # List all tables
            tables = await conn.fetch(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """
            )

            if tables:
                logger.info(f"Available tables: {[t['table_name'] for t in tables]}")
            else:
                logger.info("No tables found in database")

        # Check for voice samples table structure
        logger.info("\n🎤 Checking voice samples...")

        # First check if table exists
        vs_exists = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = 'voice_samples'
            );
        """
        )

        if vs_exists:
            # Get table columns
            columns = await conn.fetch(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'voice_samples'
                ORDER BY ordinal_position;
            """
            )

            logger.info("voice_samples table columns:")
            for col in columns:
                logger.info(f"  - {col['column_name']}: {col['data_type']}")

            # Check total samples
            sample_count = await conn.fetchval("SELECT COUNT(*) FROM voice_samples;")
            logger.info(f"\nTotal voice samples: {sample_count}")

            # Check for samples linked to speaker profile
            linked_samples = await conn.fetchval(
                """
                SELECT COUNT(*) FROM voice_samples
                WHERE speaker_id = 1;
            """
            )

            if linked_samples:
                logger.info(f"✅ Found {linked_samples} voice samples linked to Derek's profile")
            else:
                logger.warning("❌ No voice samples linked to Derek's profile")
        else:
            logger.warning("❌ voice_samples table does not exist")

        await conn.close()

    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_connection())
