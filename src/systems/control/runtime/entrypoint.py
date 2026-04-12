"""Runtime entrypoint owned by the control subsystem."""

from importlib import import_module


def run(*args, **kwargs):
    runtime_module = import_module("simulation.api.runtime")
    return runtime_module.run(*args, **kwargs)


__all__ = ["run"]
