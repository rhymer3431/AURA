"""Public entrypoint for the standalone navigation system."""

from systems.navigation.service import build_arg_parser, main

__all__ = ["build_arg_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
