"""
Trinity Cost Client v1.0 - Cross-Repo Cost Management
======================================================

This client allows jarvis-prime and reactor-core to report and track costs
through JARVIS's centralized Cost Sync system.

Features:
- Report API/inference costs to central budget tracker
- Check budget before expensive operations  
- Get real-time remaining budget
- Receive budget alerts

Usage in jarvis-prime/reactor-core:
    from trinity_clients.trinity_cost_client import get_cost_client
    
    client = await get_cost_client()
    
    # Check budget before expensive operation
    if await client.check_budget(0.05):
        result = await call_expensive_api()
        await client.report_api_cost(0.05)

Author: JARVIS System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# File-based communication for cross-repo cost reporting
COST_REPORT_DIR = Path.home() / ".jarvis" / "cross_repo" / "costs"

# HTTP endpoint for live cost sync (when JARVIS API is available)  
COST_SYNC_API_URL = os.getenv("JARVIS_COST_SYNC_URL", "http://127.0.0.1:8010/api/cost-sync")

# Default repo name detection
def _detect_repo_name() -> str:
    """Auto-detect which repo we're running in."""
    cwd = str(Path.cwd())
    script = str(Path(__file__).resolve())
    
    if "jarvis-prime" in cwd or "jarvis-prime" in script:
        return "jarvis-prime"
    elif "reactor-core" in cwd or "reactor-core" in script:
        return "reactor-core"
    else:
        return "jarvis"


# =============================================================================
# TRINITY COST CLIENT
# =============================================================================

class TrinityCostClient:
    """
    Cross-repo cost management client.
    
    Enables jarvis-prime and reactor-core to:
    1. Report costs to JARVIS's central budget tracker
    2. Check remaining budget before expensive operations
    3. Receive budget alerts
    
    Communication methods (in priority order):
    1. Direct import (if running in JARVIS process)
    2. HTTP API (when JARVIS API server is available)
    3. File-based (always available fallback)
    """
    
    def __init__(self, repo_name: Optional[str] = None):
        self.repo_name = repo_name or _detect_repo_name()
        self._direct_sync = None
        self._tried_direct = False
        self._session = None
        
        # Local cost tracking (for file-based mode)
        self._local_costs = {
            "api_cost_usd": 0.0,
            "inference_cost_usd": 0.0,
            "compute_cost_usd": 0.0,
            "tokens_in": 0,
            "tokens_out": 0,
            "local_inference_count": 0,
        }
        
        # Budget callbacks
        self._budget_callbacks: list = []
        
        # Ensure cost dir exists
        COST_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[TrinityCostClient] Initialized for {self.repo_name}")
    
    async def _get_direct_sync(self):
        """Try to get direct access to JARVIS cost sync."""
        if self._tried_direct:
            return self._direct_sync
        
        self._tried_direct = True
        
        try:
            from backend.core.cross_repo_cost_sync import get_cross_repo_cost_sync
            self._direct_sync = await get_cross_repo_cost_sync(self.repo_name)
            logger.info("[TrinityCostClient] Using direct JARVIS Cost Sync")
        except ImportError:
            logger.debug("[TrinityCostClient] JARVIS Cost Sync not available directly")
        
        return self._direct_sync
    
    async def report_api_cost(self, cost_usd: float) -> bool:
        """
        Report an API call cost.
        
        Args:
            cost_usd: Cost in USD
            
        Returns:
            True if reported successfully
        """
        self._local_costs["api_cost_usd"] += cost_usd
        
        # Try direct sync first
        sync = await self._get_direct_sync()
        if sync:
            sync.record_api_call(cost_usd)
            return True
        
        # Fall back to file
        return self._write_cost_report()
    
    async def report_inference_cost(
        self,
        cost_usd: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
        is_local: bool = False,
    ) -> bool:
        """
        Report model inference cost.
        
        Args:
            cost_usd: Cost in USD
            tokens_in: Input tokens
            tokens_out: Output tokens
            is_local: Whether this was local inference (saves cloud cost)
        """
        self._local_costs["inference_cost_usd"] += cost_usd
        self._local_costs["tokens_in"] += tokens_in
        self._local_costs["tokens_out"] += tokens_out
        
        if is_local:
            self._local_costs["local_inference_count"] += 1
        
        # Try direct sync first
        sync = await self._get_direct_sync()
        if sync:
            sync.record_inference(tokens_in, tokens_out, cost_usd, is_local)
            return True
        
        return self._write_cost_report()
    
    async def report_compute_cost(self, cost_usd: float, runtime_hours: float = 0) -> bool:
        """Report compute/VM cost."""
        self._local_costs["compute_cost_usd"] += cost_usd
        
        sync = await self._get_direct_sync()
        if sync:
            sync.record_vm_usage(runtime_hours, cost_usd)
            return True
        
        return self._write_cost_report()
    
    async def check_budget(self, required_cost: float) -> bool:
        """
        Check if budget allows this cost.
        
        Call this BEFORE making expensive API calls or starting VMs.
        
        Args:
            required_cost: Amount to check
            
        Returns:
            True if cost can be incurred within budget
        """
        sync = await self._get_direct_sync()
        if sync:
            return sync.can_incur_cost(required_cost)
        
        # File-based check
        remaining = await self.get_remaining_budget()
        return required_cost <= remaining
    
    async def get_remaining_budget(self) -> float:
        """Get remaining daily budget in USD."""
        sync = await self._get_direct_sync()
        if sync:
            return sync.get_remaining_budget()
        
        # File-based: read from unified state file
        try:
            unified_file = COST_REPORT_DIR / "_unified_state.json"
            if unified_file.exists():
                data = json.loads(unified_file.read_text())
                budget = data.get("daily_budget", 1.0)
                total = data.get("total_daily_cost", 0.0)
                return max(0, budget - total)
        except Exception:
            pass
        
        return 1.0  # Default $1 if no data
    
    async def get_total_cost_today(self) -> float:
        """Get total cost incurred today across all repos."""
        sync = await self._get_direct_sync()
        if sync:
            state = sync.get_unified_state()
            return state.total_daily_cost
        
        # File-based
        try:
            unified_file = COST_REPORT_DIR / "_unified_state.json"
            if unified_file.exists():
                data = json.loads(unified_file.read_text())
                return data.get("total_daily_cost", 0.0)
        except Exception:
            pass
        
        return 0.0
    
    def _write_cost_report(self) -> bool:
        """Write cost report to file for file-based sync."""
        try:
            report = {
                "repo_name": self.repo_name,
                "instance_id": f"{self.repo_name}-{os.getpid()}",
                "timestamp": time.time(),
                "daily_cost": sum([
                    self._local_costs["api_cost_usd"],
                    self._local_costs["inference_cost_usd"],
                    self._local_costs["compute_cost_usd"],
                ]),
                **self._local_costs,
            }
            
            report_file = COST_REPORT_DIR / f"{self.repo_name}.json"
            tmp_file = report_file.with_suffix(".tmp")
            tmp_file.write_text(json.dumps(report, indent=2))
            tmp_file.replace(report_file)
            return True
            
        except Exception as e:
            logger.warning(f"[TrinityCostClient] Failed to write cost report: {e}")
            return False
    
    def on_budget_alert(self, callback: Callable) -> None:
        """Register a callback for budget alerts."""
        self._budget_callbacks.append(callback)
    
    def get_local_costs(self) -> Dict[str, Any]:
        """Get this client's local cost tracking."""
        return {
            **self._local_costs,
            "total_cost": sum([
                self._local_costs["api_cost_usd"],
                self._local_costs["inference_cost_usd"],
                self._local_costs["compute_cost_usd"],
            ]),
        }
    
    async def close(self) -> None:
        """Clean up resources."""
        if self._session:
            await self._session.close()


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_client: Optional[TrinityCostClient] = None


async def get_cost_client(repo_name: Optional[str] = None) -> TrinityCostClient:
    """Get or create the global cost client."""
    global _client
    
    if _client is None:
        _client = TrinityCostClient(repo_name)
    
    return _client


async def report_cost(
    api_cost: float = 0.0,
    inference_cost: float = 0.0,
    compute_cost: float = 0.0,
    tokens_in: int = 0,
    tokens_out: int = 0,
) -> bool:
    """
    Convenience function to report costs.
    
    Usage:
        await report_cost(api_cost=0.05)
        await report_cost(inference_cost=0.01, tokens_in=100, tokens_out=50)
    """
    client = await get_cost_client()
    
    success = True
    if api_cost > 0:
        success &= await client.report_api_cost(api_cost)
    if inference_cost > 0 or tokens_in > 0:
        success &= await client.report_inference_cost(inference_cost, tokens_in, tokens_out)
    if compute_cost > 0:
        success &= await client.report_compute_cost(compute_cost)
    
    return success


async def check_budget(required_cost: float) -> bool:
    """Convenience function to check budget."""
    client = await get_cost_client()
    return await client.check_budget(required_cost)


async def get_remaining_budget() -> float:
    """Convenience function to get remaining budget."""
    client = await get_cost_client()
    return await client.get_remaining_budget()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "TrinityCostClient",
    "get_cost_client",
    "report_cost",
    "check_budget",
    "get_remaining_budget",
]
