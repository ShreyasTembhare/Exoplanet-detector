"""Serve the Transit desk API and built UI on one port."""

from __future__ import annotations

import argparse
import ipaddress
import os
import sys

import uvicorn

from tool.server import create_app


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def desk_options(host: str, env: dict, *, insecure: bool = False, auto_start: bool | None = None) -> dict:
    """Security defaults: a network-reachable desk needs a token and jails path inputs."""
    token = (env.get("DESK_TOKEN") or "").strip() or None
    exposed = not is_loopback(host)
    if exposed and not token and not insecure:
        raise SystemExit(
            f"Refusing to serve on {host} without DESK_TOKEN. Set DESK_TOKEN, bind 127.0.0.1, or pass --insecure."
        )
    raw_dirs = env.get("DESK_ALLOWED_DIRS")
    if raw_dirs is not None:
        allowed = [p for p in raw_dirs.split(os.pathsep) if p]
    else:
        allowed = [] if exposed else None
    return {
        "token": token,
        "allowed_dirs": allowed,
        "enable_auto_start": _truthy(env.get("AUTO_START")) if auto_start is None else auto_start,
        "checkpoint_minutes": float(env.get("DESK_CHECKPOINT_MINUTES") or 0),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Transit desk")
    parser.add_argument("--data", default=os.environ.get("DESK_DATA", "var/desk"), help="Queue, runs, and candidates")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--auto-start", dest="auto_start", action="store_true", default=None,
                        help="Start (or resume) continuous search on boot; same as AUTO_START=1")
    parser.add_argument("--no-auto-start", dest="auto_start", action="store_false")
    parser.add_argument("--insecure", action="store_true", help="Allow a non-loopback bind without DESK_TOKEN")
    args = parser.parse_args(argv)
    options = desk_options(args.host, os.environ, insecure=args.insecure, auto_start=args.auto_start)
    if options["token"] is None and not is_loopback(args.host):
        print("WARNING: desk is reachable from the network without a token.", file=sys.stderr)
    uvicorn.run(create_app(args.data, **options), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
