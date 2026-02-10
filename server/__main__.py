"""Entry point for ``python -m server``.

Starts the LEGO Part Recognition inference API via uvicorn.
"""

import argparse

import uvicorn


def main() -> None:
    """Parse CLI args and launch the inference server."""
    parser = argparse.ArgumentParser(description="Start LEGO Part Recognition API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload (dev)"
    )
    args = parser.parse_args()

    uvicorn.run(
        "server.inference_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
