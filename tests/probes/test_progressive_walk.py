#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import msgpack
import msgpack_numpy
import numpy as np
import zmq

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

msgpack_numpy.patch()


def _flatten_actions(payload: Any) -> np.ndarray:
    arr = np.asarray(payload, dtype=np.float32).reshape(-1)
    if arr.shape[0] < 29:
        raise ValueError(f"joint_actions length is {arr.shape[0]}, expected >=29")
    return arr[:29]


def main() -> None:
    parser = argparse.ArgumentParser(description="Progressive vx ramp test for SONIC policy server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--ramp-s", type=float, default=3.0)
    parser.add_argument("--vx-max", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=2.5, help="Absolute joint threshold for abort.")
    parser.add_argument("--timeout-ms", type=int, default=5000)
    args = parser.parse_args()

    steps = max(1, int(round(args.ramp_s * args.hz)))
    dt = 1.0 / max(args.hz, 1.0)
    velocities = np.linspace(0.0, float(args.vx_max), steps, dtype=np.float32)

    try:
        from apps.sonic_policy_server import server as sonic

        current_joint_pos = np.asarray(sonic.DEFAULT_ANGLES_ISAAC, dtype=np.float32).reshape(29)
    except Exception:
        current_joint_pos = np.zeros((29,), dtype=np.float32)
    current_joint_vel = np.zeros((29,), dtype=np.float32)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, int(args.timeout_ms))
    sock.setsockopt(zmq.SNDTIMEO, int(args.timeout_ms))
    sock.connect(f"tcp://{args.host}:{args.port}")

    max_abs = 0.0
    abort_step = -1
    abort_vx = 0.0
    abort_actions: np.ndarray | None = None
    try:
        for i, vx in enumerate(velocities):
            t0 = time.perf_counter()
            req = {
                "vx": float(vx),
                "vy": 0.0,
                "yaw_rate": 0.0,
                "style": "normal",
                "joint_pos": current_joint_pos.astype(np.float32),
                "joint_vel": current_joint_vel.astype(np.float32),
            }
            sock.send(msgpack.packb(req, default=msgpack_numpy.encode, use_bin_type=True))
            resp = msgpack.unpackb(sock.recv(), object_hook=msgpack_numpy.decode, raw=False, strict_map_key=False)
            if not isinstance(resp, dict):
                raise RuntimeError(f"Unexpected response type: {type(resp).__name__}")
            if resp.get("error"):
                raise RuntimeError(f"SONIC error: {resp['error']}")

            actions = _flatten_actions(resp.get("joint_actions"))
            step_abs = float(np.max(np.abs(actions)))
            if step_abs > max_abs:
                max_abs = step_abs
            if step_abs > args.threshold:
                abort_step = i
                abort_vx = float(vx)
                abort_actions = actions.copy()
                break

            current_joint_vel = (actions - current_joint_pos) / max(dt, 1e-4)
            current_joint_pos = actions

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, dt - elapsed))
    finally:
        sock.close(linger=0)
        ctx.term()

    if abort_step >= 0:
        print(
            f"RESULT: ABORTED at step={abort_step}, vx={abort_vx:.3f} "
            f"(threshold {args.threshold:.2f})"
        )
        print(f"max_abs_action: {max_abs:.4f}")
        print(f"actions_at_abort: {np.array2string(abort_actions, precision=4, separator=',')}")
    else:
        print(f"RESULT: COMPLETED ramp to vx={float(args.vx_max):.3f}")
        print(f"max_abs_action: {max_abs:.4f}")


if __name__ == "__main__":
    main()
