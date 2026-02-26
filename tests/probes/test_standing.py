#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from typing import Any

import msgpack
import msgpack_numpy
import numpy as np
import zmq

msgpack_numpy.patch()


def _flatten_actions(payload: Any) -> np.ndarray:
    arr = np.asarray(payload, dtype=np.float32).reshape(-1)
    if arr.shape[0] < 29:
        raise ValueError(f"joint_actions length is {arr.shape[0]}, expected >=29")
    return arr[:29]


def main() -> None:
    parser = argparse.ArgumentParser(description="Standing-stability probe for SONIC policy server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--threshold", type=float, default=2.0, help="Absolute joint threshold for instability.")
    parser.add_argument("--timeout-ms", type=int, default=5000)
    args = parser.parse_args()

    dt = 1.0 / max(args.hz, 1.0)
    steps = max(1, int(round(args.duration_s * args.hz)))
    print(f"Sending {steps} zero-velocity commands ({args.duration_s:.2f}s at {args.hz:.1f}Hz)...")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, int(args.timeout_ms))
    sock.setsockopt(zmq.SNDTIMEO, int(args.timeout_ms))
    sock.connect(f"tcp://{args.host}:{args.port}")

    max_abs = 0.0
    unstable_step = -1
    unstable_values: np.ndarray | None = None
    try:
        for i in range(steps):
            t0 = time.perf_counter()
            req = {
                "vx": 0.0,
                "vy": 0.0,
                "yaw_rate": 0.0,
                "style": "normal",
                "joint_pos": np.zeros(29, dtype=np.float32),
                "joint_vel": np.zeros(29, dtype=np.float32),
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
            if unstable_step < 0 and step_abs > args.threshold:
                unstable_step = i
                unstable_values = actions.copy()
                break

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, dt - elapsed))
    finally:
        sock.close(linger=0)
        ctx.term()

    if unstable_step >= 0:
        print(f"RESULT: UNSTABLE (threshold {args.threshold:.2f} exceeded at step {unstable_step})")
        print(f"max_abs_action: {max_abs:.4f}")
        print(f"actions_at_failure: {np.array2string(unstable_values, precision=4, separator=',')}")
    else:
        print("RESULT: STABLE")
        print(f"max_abs_action: {max_abs:.4f}")


if __name__ == "__main__":
    main()
