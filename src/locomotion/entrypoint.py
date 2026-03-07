"""Entrypoint that preserves Isaac Sim bootstrap ordering."""

from __future__ import annotations

from .args import BOOTSTRAP_ARGS, build_arg_parser


def main():
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": BOOTSTRAP_ARGS.headless})
    try:
        args = build_arg_parser().parse_args()

        from .runtime import run

        return run(args, simulation_app)
    finally:
        simulation_app.close()
