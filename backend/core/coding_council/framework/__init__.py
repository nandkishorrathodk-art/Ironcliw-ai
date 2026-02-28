"""
v77.0: Framework Module - Gaps #8-15
=====================================

Core framework patterns:
- Gap #8: Comprehensive timeout management
- Gap #9: Circuit breaker pattern
- Gap #10: Rate limiting
- Gap #11: Resource coordination
- Gap #12: Retry with exponential backoff
- Gap #13: Bulkhead isolation
- Gap #14: Graceful degradation
- Gap #15: Health checks

Author: Ironcliw v77.0
"""

from .timeout_wrapper import (
    TimeoutWrapper,
    TimeoutConfig,
    timeout,
    with_timeout,
)
from .circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitBreakerError,
    circuit_breaker,
)
from .rate_limiter import (
    RateLimiter,
    TokenBucket,
    SlidingWindow,
    rate_limit,
)
from .resource_coordinator import (
    ResourceCoordinator,
    Resource,
    ResourcePool,
    ResourceAllocation,
)
from .retry import (
    RetryPolicy,
    retry,
    exponential_backoff,
)
from .bulkhead import (
    Bulkhead,
    BulkheadFull,
    bulkhead,
)

__all__ = [
    # Timeout
    "TimeoutWrapper",
    "TimeoutConfig",
    "timeout",
    "with_timeout",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitState",
    "CircuitBreakerError",
    "circuit_breaker",
    # Rate Limiter
    "RateLimiter",
    "TokenBucket",
    "SlidingWindow",
    "rate_limit",
    # Resource Coordinator
    "ResourceCoordinator",
    "Resource",
    "ResourcePool",
    "ResourceAllocation",
    # Retry
    "RetryPolicy",
    "retry",
    "exponential_backoff",
    # Bulkhead
    "Bulkhead",
    "BulkheadFull",
    "bulkhead",
]
