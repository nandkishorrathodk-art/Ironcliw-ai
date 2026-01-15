"""
Unified Data Management System v1.0
====================================

Enterprise-grade data management for the JARVIS Trinity ecosystem.
Handles all aspects of data lifecycle across JARVIS (Body), JARVIS Prime (Mind),
and Reactor Core (Learning).

Implements 8 critical data management patterns:
1. Training Data Collection - Automatic collection, batching, forwarding
2. Data Versioning - Git-like versioning with content-addressable storage
3. Data Validation - Schema validation, quality checks, anomaly detection
4. Data Privacy - PII detection, anonymization, encryption, compliance
5. Data Retention Policies - TTL-based archival and deletion
6. Data Deduplication - Content-hash based dedup with similarity matching
7. Data Sampling - Stratified, importance, and active learning sampling
8. Data Lineage - DAG-based provenance tracking with full audit trail

Author: Trinity Data System
Version: 1.0.0
"""

from backend.core.data_management.unified_engine import (
    # Main Engine
    UnifiedDataManagementEngine,
    get_data_management_engine,
    initialize_data_management,
    shutdown_data_management,
    # Configuration
    DataManagementConfig,
    # Data Types
    DataRecord,
    DataBatch,
    DataVersion,
    DataLineageNode,
    DataLineageEdge,
    ValidationResult,
    PrivacyReport,
    SamplingStrategy,
    # Enums
    DataType,
    DataQuality,
    PrivacyLevel,
    RetentionPolicy,
    SamplingMethod,
    LineageEventType,
    # Collectors
    TrainingDataCollector,
    # Versioning
    DataVersionManager,
    # Validation
    DataValidator,
    SchemaRegistry,
    # Privacy
    DataPrivacyManager,
    PIIDetector,
    DataAnonymizer,
    # Retention
    DataRetentionManager,
    # Deduplication
    DataDeduplicator,
    # Sampling
    IntelligentDataSampler,
    # Lineage
    DataLineageTracker,
)

from backend.core.data_management.cross_repo_data import (
    # Cross-Repo Integration
    CrossRepoDataBridge,
    get_cross_repo_data_bridge,
    initialize_cross_repo_data,
    shutdown_cross_repo_data,
    # Data Types
    CrossRepoDataPacket,
    DataSyncState,
    # Enums
    DataFlowDirection,
    SyncMode,
)

from backend.core.data_management.supervisor_integration import (
    # Enums
    DataSystemState,
    DataComponentState,
    # Configuration
    DataManagementSupervisorConfig,
    # Data Structures
    DataComponentHealth,
    DataManagementInitResult,
    DataManagementHealthReport,
    # Coordinator
    DataManagementSupervisorCoordinator,
    # Global Functions
    get_data_management_supervisor,
    initialize_data_management_supervisor,
    shutdown_data_management_supervisor,
    get_data_management_status,
    get_data_management_health,
    # Context Manager
    DataManagementContext,
)

__all__ = [
    # Main Engine
    "UnifiedDataManagementEngine",
    "get_data_management_engine",
    "initialize_data_management",
    "shutdown_data_management",
    # Configuration
    "DataManagementConfig",
    # Data Types
    "DataRecord",
    "DataBatch",
    "DataVersion",
    "DataLineageNode",
    "DataLineageEdge",
    "ValidationResult",
    "PrivacyReport",
    "SamplingStrategy",
    # Enums
    "DataType",
    "DataQuality",
    "PrivacyLevel",
    "RetentionPolicy",
    "SamplingMethod",
    "LineageEventType",
    # Collectors
    "TrainingDataCollector",
    # Versioning
    "DataVersionManager",
    # Validation
    "DataValidator",
    "SchemaRegistry",
    # Privacy
    "DataPrivacyManager",
    "PIIDetector",
    "DataAnonymizer",
    # Retention
    "DataRetentionManager",
    # Deduplication
    "DataDeduplicator",
    # Sampling
    "IntelligentDataSampler",
    # Lineage
    "DataLineageTracker",
    # Cross-Repo
    "CrossRepoDataBridge",
    "get_cross_repo_data_bridge",
    "initialize_cross_repo_data",
    "shutdown_cross_repo_data",
    "CrossRepoDataPacket",
    "DataSyncState",
    "DataFlowDirection",
    "SyncMode",
    # =========================================================================
    # SUPERVISOR INTEGRATION
    # =========================================================================
    # Enums
    "DataSystemState",
    "DataComponentState",
    # Configuration
    "DataManagementSupervisorConfig",
    # Data Structures
    "DataComponentHealth",
    "DataManagementInitResult",
    "DataManagementHealthReport",
    # Coordinator
    "DataManagementSupervisorCoordinator",
    # Global Functions
    "get_data_management_supervisor",
    "initialize_data_management_supervisor",
    "shutdown_data_management_supervisor",
    "get_data_management_status",
    "get_data_management_health",
    # Context Manager
    "DataManagementContext",
]
