"""Serve the Transit desk API and built UI on one port."""

from __future__ import annotations

import argparse

import uvicorn

from tool.server import create_app


def main():
    parser = argparse.ArgumentParser(description="Transit desk")
    parser.add_argument("--data", default="var/desk", help="Queue, runs, and candidates")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    uvicorn.run(create_app(args.data), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
