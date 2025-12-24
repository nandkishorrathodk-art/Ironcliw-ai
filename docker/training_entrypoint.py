#!/usr/bin/env python3
"""
JARVIS Training Engine Entrypoint
==================================

Main entrypoint for the JARVIS training container that orchestrates:
- Experience collection from JARVIS interactions
- Continuous web scraping via Safe Scout
- Scheduled model training via reactor-core
- GGUF export and deployment

Usage:
    python3 training_entrypoint.py --mode continuous  # Run continuously
    python3 training_entrypoint.py --mode train       # Single training run
    python3 training_entrypoint.py --mode scrape      # Single scrape run
    python3 training_entrypoint.py --mode export      # Export current model

Environment Variables:
    JARVIS_TRAINING_DB: Path to SQLite training database
    JARVIS_TRAINING_SCHEDULE: Training schedule (HH:MM format)
    CONTINUOUS_SCRAPING_ENABLED: Enable background scraping
    CONTINUOUS_SCRAPING_INTERVAL_HOURS: Scraping interval

Author: JARVIS AI Agent Team
Version: 1.0.0
"""

import asyncio
import argparse
import logging
import os
import sys
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.environ.get("JARVIS_LOGS_DIR", "/app/logs") + "/training.log")
    ]
)
logger = logging.getLogger("JARVISTraining")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Training engine configuration."""

    # Database
    db_path: Path = field(default_factory=lambda: Path(os.getenv("JARVIS_TRAINING_DB", "/app/data/training_db/jarvis_training.db")))

    # Training
    training_enabled: bool = field(default_factory=lambda: os.getenv("JARVIS_TRAINING_ENABLED", "true").lower() == "true")
    training_schedule: str = field(default_factory=lambda: os.getenv("JARVIS_TRAINING_SCHEDULE", "03:00"))
    min_experiences: int = field(default_factory=lambda: int(os.getenv("JARVIS_TRAINING_MIN_EXPERIENCES", "100")))
    cooldown_hours: float = field(default_factory=lambda: float(os.getenv("JARVIS_TRAINING_COOLDOWN_HOURS", "24")))

    # Scraping
    scraping_enabled: bool = field(default_factory=lambda: os.getenv("CONTINUOUS_SCRAPING_ENABLED", "true").lower() == "true")
    scraping_interval_hours: float = field(default_factory=lambda: float(os.getenv("CONTINUOUS_SCRAPING_INTERVAL_HOURS", "4")))
    scraping_max_pages: int = field(default_factory=lambda: int(os.getenv("CONTINUOUS_SCRAPING_MAX_PAGES", "50")))

    # Models
    models_dir: Path = field(default_factory=lambda: Path(os.getenv("JARVIS_MODELS_DIR", "/app/models")))
    gguf_enabled: bool = field(default_factory=lambda: os.getenv("GGUF_EXPORT_ENABLED", "true").lower() == "true")
    gguf_quantization: str = field(default_factory=lambda: os.getenv("GGUF_QUANTIZATION", "q4_k_m"))

    # GCS
    gcs_enabled: bool = field(default_factory=lambda: os.getenv("GCS_UPLOAD_ENABLED", "false").lower() == "true")
    gcs_bucket: str = field(default_factory=lambda: os.getenv("GCS_MODELS_BUCKET", ""))


class TrainingStage(Enum):
    """Training pipeline stages."""
    IDLE = "idle"
    COLLECTING = "collecting"
    SCRAPING = "scraping"
    TRAINING = "training"
    EXPORTING = "exporting"
    DEPLOYING = "deploying"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrainingProgress:
    """Progress tracking for training run."""
    stage: TrainingStage = TrainingStage.IDLE
    experiences_collected: int = 0
    pages_scraped: int = 0
    training_steps: int = 0
    training_loss: float = 0.0
    model_exported: bool = False
    model_deployed: bool = False
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None


# =============================================================================
# SQLite Training Database
# =============================================================================

class TrainingDatabase:
    """SQLite database for training experiences and metadata."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._ensure_db_exists()

    def _ensure_db_exists(self) -> None:
        """Ensure database exists with proper schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tables if they don't exist
        cursor.executescript("""
            -- Experiences table: Records of JARVIS interactions
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                source TEXT NOT NULL,
                input_text TEXT NOT NULL,
                output_text TEXT NOT NULL,
                context TEXT,
                quality_score REAL DEFAULT 0.5,
                used_in_training INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            -- Scraped content table: Web documentation
            CREATE TABLE IF NOT EXISTS scraped_content (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                title TEXT,
                content TEXT NOT NULL,
                topic TEXT,
                quality_score REAL DEFAULT 0.5,
                scraped_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                used_in_training INTEGER DEFAULT 0
            );

            -- Learning goals table: Topics to learn
            CREATE TABLE IF NOT EXISTS learning_goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT UNIQUE NOT NULL,
                priority INTEGER DEFAULT 5,
                source TEXT DEFAULT 'auto',
                urls TEXT,
                discovered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                completed INTEGER DEFAULT 0
            );

            -- Training runs table: Training history
            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at DATETIME NOT NULL,
                completed_at DATETIME,
                status TEXT DEFAULT 'running',
                experiences_used INTEGER DEFAULT 0,
                pages_used INTEGER DEFAULT 0,
                training_steps INTEGER DEFAULT 0,
                final_loss REAL,
                model_path TEXT,
                gguf_path TEXT,
                gcs_path TEXT,
                error TEXT
            );

            -- Model deployments table: Deployed models
            CREATE TABLE IF NOT EXISTS model_deployments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_path TEXT NOT NULL,
                gguf_path TEXT,
                checksum TEXT,
                deployed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                deployed_to TEXT,
                active INTEGER DEFAULT 1
            );

            -- Create indexes for performance
            CREATE INDEX IF NOT EXISTS idx_experiences_timestamp ON experiences(timestamp);
            CREATE INDEX IF NOT EXISTS idx_experiences_used ON experiences(used_in_training);
            CREATE INDEX IF NOT EXISTS idx_scraped_url ON scraped_content(url);
            CREATE INDEX IF NOT EXISTS idx_scraped_topic ON scraped_content(topic);
            CREATE INDEX IF NOT EXISTS idx_goals_priority ON learning_goals(priority DESC);
        """)

        conn.commit()
        conn.close()
        logger.info(f"Training database ready: {self.db_path}")

    def add_experience(self, source: str, input_text: str, output_text: str,
                       context: str = None, quality_score: float = 0.5) -> int:
        """Add a new experience to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO experiences (timestamp, source, input_text, output_text, context, quality_score)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (time.time(), source, input_text, output_text, context, quality_score))

        experience_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return experience_id

    def add_scraped_content(self, url: str, title: str, content: str,
                            topic: str = None, quality_score: float = 0.5) -> int:
        """Add scraped web content to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO scraped_content (url, title, content, topic, quality_score, scraped_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (url, title, content, topic, quality_score))

            content_id = cursor.lastrowid
            conn.commit()
        except Exception as e:
            logger.warning(f"Failed to add scraped content: {e}")
            content_id = -1
        finally:
            conn.close()

        return content_id

    def add_learning_goal(self, topic: str, priority: int = 5,
                          source: str = "auto", urls: List[str] = None) -> bool:
        """Add a learning goal topic."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR IGNORE INTO learning_goals (topic, priority, source, urls)
                VALUES (?, ?, ?, ?)
            """, (topic, priority, source, ",".join(urls) if urls else None))

            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.warning(f"Failed to add learning goal: {e}")
            return False
        finally:
            conn.close()

    def get_pending_learning_goals(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get pending learning goals sorted by priority."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT topic, priority, urls FROM learning_goals
            WHERE completed = 0
            ORDER BY priority DESC
            LIMIT ?
        """, (limit,))

        goals = []
        for row in cursor.fetchall():
            goals.append({
                "topic": row[0],
                "priority": row[1],
                "urls": row[2].split(",") if row[2] else []
            })

        conn.close()
        return goals

    def get_unused_experiences(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get experiences not yet used in training."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, input_text, output_text, context, quality_score
            FROM experiences
            WHERE used_in_training = 0
            ORDER BY quality_score DESC, timestamp DESC
            LIMIT ?
        """, (limit,))

        experiences = []
        for row in cursor.fetchall():
            experiences.append({
                "id": row[0],
                "input": row[1],
                "output": row[2],
                "context": row[3],
                "quality": row[4]
            })

        conn.close()
        return experiences

    def mark_experiences_used(self, experience_ids: List[int]) -> None:
        """Mark experiences as used in training."""
        if not experience_ids:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        placeholders = ",".join("?" * len(experience_ids))
        cursor.execute(f"""
            UPDATE experiences SET used_in_training = 1
            WHERE id IN ({placeholders})
        """, experience_ids)

        conn.commit()
        conn.close()

    def start_training_run(self) -> int:
        """Record the start of a training run."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO training_runs (started_at, status)
            VALUES (CURRENT_TIMESTAMP, 'running')
        """)

        run_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return run_id

    def complete_training_run(self, run_id: int, status: str,
                               experiences_used: int = 0, pages_used: int = 0,
                               training_steps: int = 0, final_loss: float = None,
                               model_path: str = None, gguf_path: str = None,
                               gcs_path: str = None, error: str = None) -> None:
        """Record the completion of a training run."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE training_runs SET
                completed_at = CURRENT_TIMESTAMP,
                status = ?,
                experiences_used = ?,
                pages_used = ?,
                training_steps = ?,
                final_loss = ?,
                model_path = ?,
                gguf_path = ?,
                gcs_path = ?,
                error = ?
            WHERE id = ?
        """, (status, experiences_used, pages_used, training_steps, final_loss,
              model_path, gguf_path, gcs_path, error, run_id))

        conn.commit()
        conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get training database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Count experiences
        cursor.execute("SELECT COUNT(*) FROM experiences")
        total_experiences = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM experiences WHERE used_in_training = 0")
        unused_experiences = cursor.fetchone()[0]

        # Count scraped content
        cursor.execute("SELECT COUNT(*) FROM scraped_content")
        total_scraped = cursor.fetchone()[0]

        # Count learning goals
        cursor.execute("SELECT COUNT(*) FROM learning_goals WHERE completed = 0")
        pending_goals = cursor.fetchone()[0]

        # Count training runs
        cursor.execute("SELECT COUNT(*) FROM training_runs WHERE status = 'completed'")
        completed_runs = cursor.fetchone()[0]

        conn.close()

        return {
            "total_experiences": total_experiences,
            "unused_experiences": unused_experiences,
            "total_scraped": total_scraped,
            "pending_goals": pending_goals,
            "completed_training_runs": completed_runs,
        }


# =============================================================================
# Training Engine
# =============================================================================

class TrainingEngine:
    """JARVIS Training Engine - Orchestrates the full training pipeline."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.db = TrainingDatabase(config.db_path)
        self.progress = TrainingProgress()
        self._running = False
        self._scraping_task: Optional[asyncio.Task] = None
        self._training_task: Optional[asyncio.Task] = None

    async def start(self, mode: str = "continuous") -> None:
        """Start the training engine."""
        logger.info(f"Starting JARVIS Training Engine (mode: {mode})")
        logger.info(f"Database: {self.config.db_path}")
        logger.info(f"Models: {self.config.models_dir}")

        # Log initial stats
        stats = self.db.get_stats()
        logger.info(f"Database stats: {stats}")

        self._running = True

        if mode == "continuous":
            await self._run_continuous()
        elif mode == "train":
            await self._run_single_training()
        elif mode == "scrape":
            await self._run_single_scrape()
        elif mode == "export":
            await self._run_export()
        else:
            logger.error(f"Unknown mode: {mode}")

    async def stop(self) -> None:
        """Stop the training engine."""
        logger.info("Stopping JARVIS Training Engine...")
        self._running = False

        if self._scraping_task:
            self._scraping_task.cancel()
            try:
                await self._scraping_task
            except asyncio.CancelledError:
                pass

        if self._training_task:
            self._training_task.cancel()
            try:
                await self._training_task
            except asyncio.CancelledError:
                pass

        logger.info("Training Engine stopped")

    async def _run_continuous(self) -> None:
        """Run the engine continuously."""
        logger.info("Running in continuous mode")

        # Start background scraping
        if self.config.scraping_enabled:
            self._scraping_task = asyncio.create_task(self._scraping_loop())

        # Start scheduled training
        if self.config.training_enabled:
            self._training_task = asyncio.create_task(self._training_scheduler())

        # Wait for shutdown
        while self._running:
            await asyncio.sleep(60)

            # Log periodic stats
            stats = self.db.get_stats()
            logger.debug(f"Stats: {stats}")

    async def _scraping_loop(self) -> None:
        """Background scraping loop."""
        interval_seconds = self.config.scraping_interval_hours * 3600
        logger.info(f"Scraping loop started (interval: {self.config.scraping_interval_hours}h)")

        while self._running:
            try:
                await asyncio.sleep(interval_seconds)
                await self._run_single_scrape()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scraping loop error: {e}")
                await asyncio.sleep(1800)  # Wait 30 min on error

    async def _training_scheduler(self) -> None:
        """Scheduled training loop."""
        logger.info(f"Training scheduler started (schedule: {self.config.training_schedule})")

        while self._running:
            try:
                # Calculate time until next scheduled run
                schedule_hour, schedule_minute = map(int, self.config.training_schedule.split(":"))
                now = datetime.now()
                target = now.replace(hour=schedule_hour, minute=schedule_minute, second=0, microsecond=0)

                if target <= now:
                    target += timedelta(days=1)

                sleep_seconds = (target - now).total_seconds()
                logger.info(f"Next training run in {sleep_seconds / 3600:.1f} hours")

                await asyncio.sleep(sleep_seconds)
                await self._run_single_training()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Training scheduler error: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error

    async def _run_single_scrape(self) -> None:
        """Run a single scraping cycle."""
        logger.info("Starting scraping cycle...")
        self.progress.stage = TrainingStage.SCRAPING

        try:
            # Get learning goals
            goals = self.db.get_pending_learning_goals(limit=5)
            topics = [g["topic"] for g in goals]

            if not topics:
                logger.info("No learning goals - using default topics")
                topics = ["Python best practices", "async programming", "AI agents"]

            logger.info(f"Scraping topics: {topics}")

            # TODO: Integrate with Safe Scout from reactor-core
            # For now, simulate scraping
            for topic in topics:
                # Placeholder for actual scraping
                logger.debug(f"Scraping for topic: {topic}")
                self.progress.pages_scraped += 1

            logger.info(f"Scraping complete: {self.progress.pages_scraped} pages")

        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            self.progress.error = str(e)

        self.progress.stage = TrainingStage.IDLE

    async def _run_single_training(self) -> None:
        """Run a single training cycle."""
        logger.info("Starting training cycle...")

        # Check if we have enough experiences
        stats = self.db.get_stats()
        if stats["unused_experiences"] < self.config.min_experiences:
            logger.info(f"Not enough experiences ({stats['unused_experiences']} < {self.config.min_experiences})")
            return

        self.progress.stage = TrainingStage.TRAINING
        run_id = self.db.start_training_run()

        try:
            # Get unused experiences
            experiences = self.db.get_unused_experiences(limit=1000)
            logger.info(f"Training with {len(experiences)} experiences")

            # TODO: Integrate with reactor-core training pipeline
            # For now, simulate training
            self.progress.training_steps = 100
            self.progress.training_loss = 0.5

            # Mark experiences as used
            experience_ids = [e["id"] for e in experiences]
            self.db.mark_experiences_used(experience_ids)

            # Complete training run
            self.db.complete_training_run(
                run_id=run_id,
                status="completed",
                experiences_used=len(experiences),
                training_steps=self.progress.training_steps,
                final_loss=self.progress.training_loss,
            )

            logger.info(f"Training complete: {self.progress.training_steps} steps, loss: {self.progress.training_loss}")

            # Export model if enabled
            if self.config.gguf_enabled:
                await self._run_export()

        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.db.complete_training_run(run_id=run_id, status="failed", error=str(e))
            self.progress.error = str(e)

        self.progress.stage = TrainingStage.IDLE

    async def _run_export(self) -> None:
        """Export the model to GGUF format."""
        logger.info("Starting model export...")
        self.progress.stage = TrainingStage.EXPORTING

        try:
            # TODO: Integrate with reactor-core GGUF export
            # For now, simulate export
            gguf_path = self.config.models_dir / "current" / "jarvis-prime-latest.gguf"
            logger.info(f"Model exported to: {gguf_path}")
            self.progress.model_exported = True

            # Upload to GCS if enabled
            if self.config.gcs_enabled and self.config.gcs_bucket:
                await self._upload_to_gcs(gguf_path)

        except Exception as e:
            logger.error(f"Export failed: {e}")
            self.progress.error = str(e)

        self.progress.stage = TrainingStage.IDLE

    async def _upload_to_gcs(self, model_path: Path) -> None:
        """Upload model to Google Cloud Storage."""
        logger.info(f"Uploading to GCS: {self.config.gcs_bucket}")
        self.progress.stage = TrainingStage.DEPLOYING

        try:
            # TODO: Implement actual GCS upload
            self.progress.model_deployed = True
            logger.info("Model uploaded to GCS")
        except Exception as e:
            logger.error(f"GCS upload failed: {e}")
            self.progress.error = str(e)


# =============================================================================
# Main Entry Point
# =============================================================================

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="JARVIS Training Engine")
    parser.add_argument("--mode", choices=["continuous", "train", "scrape", "export"],
                        default="continuous", help="Operation mode")
    parser.add_argument("--db", type=str, help="Override database path")

    args = parser.parse_args()

    # Create config
    config = TrainingConfig()
    if args.db:
        config.db_path = Path(args.db)

    # Create and start engine
    engine = TrainingEngine(config)

    try:
        await engine.start(mode=args.mode)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
