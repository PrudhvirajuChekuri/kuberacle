"""Run the RAG API server locally.

Usage:
    python scripts/serve.py [--host 0.0.0.0] [--port 8000] [--reload]
"""

import argparse

import uvicorn


def parse_args() -> argparse.Namespace:
    """Parse server CLI arguments."""
    parser = argparse.ArgumentParser(description="Serve the kuberacle API")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for local development",
    )
    return parser.parse_args()


def main() -> None:
    """Start the uvicorn server for the RAG API app."""
    args = parse_args()
    uvicorn.run(
        "kuberacle.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
