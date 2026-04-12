"""Public entrypoint for the managed inference system."""

from systems.inference.stack.server import build_arg_parser, main

__all__ = ["build_arg_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
