"""
Ironcliw Governance Module
========================

Provides clinical-grade governance for Ironcliw task execution:
- SOP Enforcer: Requires design plans before code changes
- Thinking Protocol: Structured reasoning framework
- Complexity Analyzer: Task complexity assessment

Author: Ironcliw AI System
"""

from backend.core.governance.sop_enforcer import (
    # Configuration
    SOPEnforcerConfig,

    # Enums
    TaskComplexity,
    PlanStatus,
    EnforcementAction,

    # Pydantic Models
    ProposedChange,
    RiskAssessment,
    TestPlan,
    DesignPlan,

    # Core Classes
    ThinkingProtocol,
    IroncliwThinkingProtocol,
    ComplexityAnalyzer,
    SOPEnforcer,

    # Integration
    enforce_sop_before_execution,

    # Convenience Functions
    get_sop_enforcer,
    require_design_plan,
)

__all__ = [
    # Configuration
    "SOPEnforcerConfig",

    # Enums
    "TaskComplexity",
    "PlanStatus",
    "EnforcementAction",

    # Pydantic Models
    "ProposedChange",
    "RiskAssessment",
    "TestPlan",
    "DesignPlan",

    # Core Classes
    "ThinkingProtocol",
    "IroncliwThinkingProtocol",
    "ComplexityAnalyzer",
    "SOPEnforcer",

    # Integration
    "enforce_sop_before_execution",

    # Convenience Functions
    "get_sop_enforcer",
    "require_design_plan",
]
