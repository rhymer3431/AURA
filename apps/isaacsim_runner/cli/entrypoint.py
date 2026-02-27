from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from apps.isaacsim_runner.bridges.mock import run_mock_loop
from apps.isaacsim_runner.cli.args import build_argument_parser
from apps.isaacsim_runner.config.base import configure_logging
from apps.isaacsim_runner.runtime.orchestrator import _run_native_isaac

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from telemetry_runtime import JsonlTelemetryLogger, now_perf
except Exception:  # pragma: no cover - optional telemetry dependency
    JsonlTelemetryLogger = None  # type: ignore

    def now_perf() -> float:
        return time.perf_counter()


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)

    usd_path = Path(args.usd).resolve()
    if not usd_path.exists():
        raise FileNotFoundError(f"USD file not found: {usd_path}")
    logging.info("Using USD: %s", usd_path)

    if args.mock:
        run_mock_loop(args)
        return
    _run_native_isaac(args, jsonl_logger_cls=JsonlTelemetryLogger, perf_now=now_perf)
