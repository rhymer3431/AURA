#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tests.pipeline.pipeline_test_utils import (
    DEFAULT_CONFIG,
    DEFAULT_JOINT_MAP,
    GrootManipulator,
    action_scale_and_defaults,
    build_manip_cfg,
    extract_actions_at_abort,
    extract_pd_gain_rows,
    find_latest_abort_window,
    init_run_dir,
    load_joint_map_summary,
    load_phase_records,
    render_report,
    summarize_abort_window,
    summarize_phase_metrics,
    write_report,
)


async def _run(args: argparse.Namespace) -> int:
    phase = "walk_ramp"
    run_dir = init_run_dir(phase=phase, run_dir=args.run_dir)
    manip_cfg = build_manip_cfg(config_path=Path(args.config), run_dir=run_dir, phase=phase)
    manip = GrootManipulator(manip_cfg)
    aborted = False
    abort_step = None
    abort_vx = None
    err: str | None = None
    try:
        await manip.warmup()
        if not bool(getattr(manip, "locomotion_enabled", False)):
            raise RuntimeError("No locomotion backend is enabled after warmup.")
        manip.set_telemetry_phase(phase)
        steps = max(1, int(round(3.0 * 50.0)))
        dt = 1.0 / 50.0
        for i in range(steps):
            vx = 0.5 * (float(i) / float(max(1, steps - 1)))
            try:
                await manip.execute_locomotion(
                    linear_x=vx,
                    linear_y=0.0,
                    angular_z=0.0,
                    duration_s=dt,
                    source="test_walk_ramp_pipeline",
                )
            except Exception as exc:
                aborted = True
                abort_step = i
                abort_vx = vx
                err = str(exc)
                logging.exception("Walk ramp aborted")
                break
    except Exception as exc:
        err = str(exc)
        logging.exception("Walk ramp setup failed")
    finally:
        await manip.stop()

    records = load_phase_records(run_dir, phase)
    metrics = summarize_phase_metrics(records)
    if bool(metrics.get("fall_flag")):
        aborted = True
        err = err or "fall_flag detected"
    if metrics.get("max_abs_action") is not None and float(metrics.get("max_abs_action") or 0.0) > 2.5:
        aborted = True
        err = err or f"resp_max_abs_action {metrics.get('max_abs_action')} > 2.5"
    abort_window_path = find_latest_abort_window(run_dir)
    abort_window_stats = summarize_abort_window(abort_window_path)
    actions_at_abort = extract_actions_at_abort(records)

    mapping = load_joint_map_summary(DEFAULT_JOINT_MAP)
    action_scale = action_scale_and_defaults(records)
    standing = {"pass": "n/a", "max_abs_action": "n/a", "max_roll": "n/a", "max_pitch": "n/a"}
    walking = {
        "completed": not aborted,
        "aborted": aborted,
        "abort_step": abort_step,
        "abort_vx": abort_vx,
        "actions_at_abort": actions_at_abort,
        "last_1s_stats": abort_window_stats or "n/a",
    }
    closed_loop = {
        "pass": "n/a",
        "max_roll": "n/a",
        "max_pitch": "n/a",
        "min_base_height": "n/a",
        "tracking_err_rms": "n/a",
        "server_latency_p95": "n/a",
        "loop_jitter_std": "n/a",
        "overrun_count": "n/a",
    }
    pd_rows = extract_pd_gain_rows(records)
    report = render_report(
        run_dir=run_dir,
        phase=phase,
        mapping=mapping,
        action_scale=action_scale,
        control_metrics=metrics,
        standing=standing,
        walking=walking,
        closed_loop=closed_loop,
        pd_gain_rows=pd_rows,
    )
    report_path = write_report(run_dir, report)
    print(f"run_dir: {run_dir}")
    print(f"report: {report_path}")
    if abort_window_path is not None:
        print(f"abort_window: {abort_window_path}")
    if aborted:
        print(f"result: ABORTED step={abort_step} vx={abort_vx} reason={err}")
        return 1
    print("result: COMPLETED")
    print(f"max_abs_action: {metrics.get('max_abs_action')}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3-second walk-ramp pipeline test with abort capture.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    rc = asyncio.run(_run(args))
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
