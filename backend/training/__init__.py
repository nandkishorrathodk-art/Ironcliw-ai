"""
JARVIS Training Package
========================

Automated training pipeline for JARVIS-Prime models.

Components:
- ReactorCorePipeline: Complete training pipeline
- DatasetBuilder: Build datasets from telemetry
- LoRATrainer: LoRA fine-tuning
- GGUFExporter: Export to GGUF format

Usage:
    from training import get_training_pipeline, trigger_training

    # Trigger training
    job = await trigger_training(
        output_name="jarvis-prime-v2",
    )

    # Check status
    status = get_training_status()
"""

from .reactor_core_pipeline import (
    # Pipeline
    ReactorCorePipeline,
    get_training_pipeline,

    # Configuration
    TrainingConfig,
    TrainingStatus,
    TrainingJob,

    # Components
    DatasetBuilder,
    LoRATrainer,
    GGUFExporter,

    # Convenience functions
    trigger_training,
    get_training_status,
)

__all__ = [
    # Pipeline
    "ReactorCorePipeline",
    "get_training_pipeline",

    # Configuration
    "TrainingConfig",
    "TrainingStatus",
    "TrainingJob",

    # Components
    "DatasetBuilder",
    "LoRATrainer",
    "GGUFExporter",

    # Convenience functions
    "trigger_training",
    "get_training_status",
]
