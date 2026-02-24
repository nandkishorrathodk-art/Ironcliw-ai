import pytest
import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from backend.core.hybrid_orchestrator import IntelligenceMode, set_intelligence_mode, get_intelligence_mode, _get_agi_orchestrator
from intelligence.agi_orchestrator import AGIOrchestrator, CognitiveInput

@pytest.mark.asyncio
async def test_agi_orchestrator_initialization():
    """Test AGI Orchestrator properly triggers initialization of all its subcomponents"""
    set_intelligence_mode(IntelligenceMode.AGI)
    assert get_intelligence_mode() == IntelligenceMode.AGI

    agi = await _get_agi_orchestrator()
    assert agi is not None
    assert isinstance(agi, AGIOrchestrator)

    # Check component load status by inspecting the _state
    # Wait for the async initialization to settle if necessary
    await asyncio.sleep(1)

    assert "MetaCognitive" in agi._state.component_statuses
    assert "EmotionalIntelligence" in agi._state.component_statuses
    assert "ContinuousImprovement" in agi._state.component_statuses
    assert "PerceptionFusion" in agi._state.component_statuses
    assert "LongTermMemory" in agi._state.component_statuses

@pytest.mark.asyncio
async def test_agi_processing_lifecycle():
    """Test processing an input through the AGI engine."""
    agi = await _get_agi_orchestrator()
    
    input_data = CognitiveInput(
        modality="text",
        content="Hello, JARVIS. Please analyze this text.",
        context={"source": "test_script"}
    )
    
    output = await agi.process(input_data)
    
    assert output is not None
    assert output.confidence > 0.0
    # It should have run through perception, cognition and meta cognitive layers
    assert len(output.components_used) > 0

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
