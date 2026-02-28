"""
Minimal Computer Use Test Server
================================

A lightweight FastAPI server to test the Computer Use integration
without the heavy Ironcliw initialization.

Run with: python tests/minimal_computer_use_server.py
Test with: POST http://localhost:8000/api/computer-use/execute

Author: Ironcliw AI System
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ironcliw Computer Use Test Server",
    version="1.0.0",
    description="Minimal server for testing Computer Use integration"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ComputerUseRequest(BaseModel):
    """Request to execute a computer use action."""
    goal: str = Field(..., description="What to do (e.g., 'Open Calculator')")
    use_safe_code: bool = Field(True, description="Use safe code execution sandbox")
    timeout_seconds: float = Field(60.0, description="Max execution time")


class ComputerUseResponse(BaseModel):
    """Response from computer use execution."""
    success: bool
    goal: str
    actions: list = []
    message: str
    execution_time_ms: float = 0.0
    error: Optional[str] = None


# Global connector instance
_connector = None


async def get_connector():
    """Lazy-load the computer use connector."""
    global _connector
    if _connector is None:
        try:
            from backend.display.computer_use_connector import ClaudeComputerUseConnector
            _connector = ClaudeComputerUseConnector()
            logger.info("✅ ClaudeComputerUseConnector initialized")
        except ImportError as e:
            logger.error(f"Failed to import connector: {e}")
            raise HTTPException(status_code=500, detail=f"Connector not available: {e}")
    return _connector


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Ironcliw Computer Use Test Server",
        "version": "1.0.0",
        "endpoints": [
            "GET /health - Health check",
            "POST /api/computer-use/execute - Execute computer use action",
            "GET /api/computer-use/status - Get connector status",
        ]
    }


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy", "service": "computer-use-test"}


@app.get("/api/computer-use/status")
async def get_status():
    """Get computer use connector status."""
    try:
        connector = await get_connector()
        return {
            "initialized": connector is not None,
            "has_client": hasattr(connector, 'client') and connector.client is not None,
            "refinements_initialized": getattr(connector, '_refinements_initialized', False),
        }
    except Exception as e:
        return {"error": str(e), "initialized": False}


@app.post("/api/computer-use/execute", response_model=ComputerUseResponse)
async def execute_computer_use(request: ComputerUseRequest):
    """
    Execute a computer use action.

    This endpoint triggers Claude Computer Use to control the screen.

    Example request:
    {
        "goal": "Open the Calculator app",
        "use_safe_code": true,
        "timeout_seconds": 60
    }
    """
    import time
    start_time = time.time()

    logger.info(f"📱 Computer Use request: {request.goal}")

    try:
        connector = await get_connector()

        # Check if connector has a client
        if not hasattr(connector, 'client') or connector.client is None:
            return ComputerUseResponse(
                success=False,
                goal=request.goal,
                message="Anthropic client not configured. Set ANTHROPIC_API_KEY.",
                error="No API key configured"
            )

        # Execute the task
        logger.info(f"🎯 Executing: {request.goal}")

        result = await asyncio.wait_for(
            connector.execute_task(request.goal),
            timeout=request.timeout_seconds
        )

        execution_time = (time.time() - start_time) * 1000

        # Handle TaskResult object
        if hasattr(result, 'status'):
            from backend.display.computer_use_connector import TaskStatus
            success = result.status == TaskStatus.SUCCESS
            message = result.final_message if hasattr(result, 'final_message') else "Task completed"
            actions = [str(a) for a in result.actions_executed] if hasattr(result, 'actions_executed') else []
            error = None if success else message
        else:
            # Fallback for dict response
            success = result.get("success", False)
            message = result.get("message", "Task completed")
            actions = result.get("actions", [])
            error = result.get("error")

        return ComputerUseResponse(
            success=success,
            goal=request.goal,
            actions=actions,
            message=message,
            execution_time_ms=execution_time,
            error=error
        )

    except asyncio.TimeoutError:
        execution_time = (time.time() - start_time) * 1000
        return ComputerUseResponse(
            success=False,
            goal=request.goal,
            message="Execution timed out",
            execution_time_ms=execution_time,
            error=f"Timeout after {request.timeout_seconds}s"
        )

    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        logger.error(f"❌ Computer Use error: {e}")
        return ComputerUseResponse(
            success=False,
            goal=request.goal,
            message=str(e),
            execution_time_ms=execution_time,
            error=str(e)
        )


@app.post("/api/safe-code/execute")
async def execute_safe_code(code: str):
    """
    Execute code through the safe code executor.

    This tests the Open Interpreter sandbox integration.
    """
    try:
        from backend.intelligence.computer_use_refinements import (
            SafeCodeExecutor,
            ComputerUseConfig,
        )

        config = ComputerUseConfig()
        executor = SafeCodeExecutor(config)

        result = await executor.execute(code)

        return {
            "success": result.success,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "blocked_reason": result.blocked_reason,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    print("=" * 60)
    print("Ironcliw Computer Use Test Server")
    print("=" * 60)
    print("")
    print("Starting on http://localhost:8000")
    print("")
    print("Test endpoints:")
    print("  GET  http://localhost:8000/health")
    print("  GET  http://localhost:8000/api/computer-use/status")
    print("  POST http://localhost:8000/api/computer-use/execute")
    print("")
    print("Postman test body:")
    print('  {"goal": "Open the Calculator app", "use_safe_code": true}')
    print("")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000)
