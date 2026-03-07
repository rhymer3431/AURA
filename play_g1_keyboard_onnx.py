from __future__ import annotations

import os
import sys


def _ensure_src_on_path() -> None:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(repo_root, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


_ensure_src_on_path()


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        from locomotion.args import build_arg_parser

        build_arg_parser().parse_args(["--help"])
        return 0

    from locomotion.entrypoint import main as locomotion_main

    return int(locomotion_main())


if __name__ == "__main__":
    raise SystemExit(main())
