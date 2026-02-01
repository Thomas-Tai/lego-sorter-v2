"""
Entry point to start the LEGO Part Recognition API server.

Usage:
    python run_api.py [--host HOST] [--port PORT]

Default:
    Host: 0.0.0.0
    Port: 8000
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Start LEGO Part Recognition API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev)")
    args = parser.parse_args()
    
    print(f"Starting API server on {args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    
    uvicorn.run(
        "sorter_app.services.inference_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
