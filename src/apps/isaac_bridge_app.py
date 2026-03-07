from __future__ import annotations

import argparse

from runtime.isaac_runtime import main as isaac_runtime_main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct Isaac bridge launcher.")
    parser.add_argument("--command", type=str, default="")
    parser.add_argument("--memory-db-path", type=str, default="state/memory/memory.sqlite")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    forwarded = []
    if str(args.command).strip() != "":
        forwarded.extend(["--command", str(args.command)])
    forwarded.extend(["--memory-db-path", str(args.memory_db_path)])
    return isaac_runtime_main(forwarded)
