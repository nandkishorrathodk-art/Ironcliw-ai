"""
v77.0: Safety Module - Gaps #16-22
===================================

Comprehensive safety validation for code evolution:
- Gap #16: Staging environment with hot-swap
- Gap #17: Circuit breaker (in orchestrator)
- Gap #18: AST validation before commit
- Gap #19: Import resolution checking
- Gap #20: Security vulnerability scanning
- Gap #21: Type checking integration
- Gap #22: Resource exhaustion prevention

Author: JARVIS v77.0
"""

from .ast_validator import ASTValidator, ASTValidationResult
from .security_scanner import SecurityScanner, SecurityScanResult, VulnerabilitySeverity
from .type_checker import TypeChecker, TypeCheckResult
from .staging_environment import StagingEnvironment, StagingResult

__all__ = [
    "ASTValidator",
    "ASTValidationResult",
    "SecurityScanner",
    "SecurityScanResult",
    "VulnerabilitySeverity",
    "TypeChecker",
    "TypeCheckResult",
    "StagingEnvironment",
    "StagingResult",
]
