"""Run the local AuthentiText FastAPI service."""

from __future__ import annotations

import argparse

import uvicorn


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    uvicorn.run(
        "authentitext.api:app",
        host=args.host,
        port=args.port,
        access_log=True,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
