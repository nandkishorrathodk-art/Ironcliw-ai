#!/usr/bin/env python3
"""
Simple server runner that handles imports properly.

v238.0: Reads port from JARVIS_BACKEND_PORT / TRINITY_PORT env vars
        instead of hardcoding 8000. Supports --port CLI arg.
"""
import argparse
import sys
import os

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(backend_dir)
sys.path.insert(0, backend_dir)
sys.path.insert(0, parent_dir)

# Now run uvicorn
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JARVIS Backend Server")
    parser.add_argument(
        "--port", type=int,
        default=int(os.getenv(
            "JARVIS_BACKEND_PORT",
            os.getenv("TRINITY_PORT", "8000"),
        )),
        help="Port to listen on (default: $JARVIS_BACKEND_PORT or $TRINITY_PORT or 8000)",
    )
    parser.add_argument(
        "--host", type=str,
        default=os.getenv("JARVIS_BACKEND_HOST", "0.0.0.0"),
        help="Host to bind to (default: $JARVIS_BACKEND_HOST or 0.0.0.0)",
    )
    args = parser.parse_args()

    import uvicorn
    uvicorn.run("main:app", host=args.host, port=args.port, reload=False)
