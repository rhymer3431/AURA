"""Public entrypoint for the standalone runtime service."""

from runtime.service import build_arg_parser, main

__all__ = ["build_arg_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
