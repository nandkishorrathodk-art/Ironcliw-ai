import pytest
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

from backend.core.hybrid_orchestrator import IntelligenceMode, get_intelligence_mode

@pytest.mark.asyncio
async def test_orchestrator_initializes_in_unified_mode():
    """Verify that the intelligence mode defaults to UNIFIED."""
    mode = get_intelligence_mode()
    assert mode == IntelligenceMode.UNIFIED

@pytest.mark.asyncio
async def test_proactive_engine_receives_signals():
    """Verify that Proactive Intelligence Engine connects to UAE."""
    from backend.intelligence.proactive_intelligence_engine import initialize_proactive_intelligence, ProactiveIntelligenceEngine
    
    mock_uae = MagicMock()
    mock_uae.get_context.return_value = {"mock_context": "test_data"}
    
    pie = await initialize_proactive_intelligence(uae_engine=mock_uae)
    assert pie is not None
    assert pie.uae == mock_uae
    
    # Trigger context update manually
    await pie._update_context()
    assert pie.current_context.uae_context == {"mock_context": "test_data"}
    await pie.stop()

@pytest.mark.asyncio
async def test_enhanced_sai_cross_repo():
    """Verify Enhanced SAI tracks cross repo status."""
    from backend.intelligence.enhanced_sai_orchestrator import get_enhanced_sai
    sai = get_enhanced_sai()
    assert sai is not None
    assert sai.cross_repo_engine is not None
    
    # We won't test external connections in unit tests, just object initialization
    assert hasattr(sai.cross_repo_engine, 'get_insights')
