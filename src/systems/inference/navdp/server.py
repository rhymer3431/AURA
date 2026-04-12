"""NavDP server entrypoint owned by the inference subsystem."""

from systems.inference.navdp.backend.navdp_server import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
