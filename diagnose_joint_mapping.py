#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import msgpack
import msgpack_numpy
import numpy as np
import zmq

msgpack_numpy.patch()


def _default_map_path(root: Path) -> Path:
    candidate = root / "apps" / "agent_runtime" / "modules" / "g1_joint_map.json"
    if candidate.exists():
        return candidate
    return root / "g1_joint_map.json"


def _load_isaac_names(probe_path: Path) -> list[str]:
    if not probe_path.exists():
        return []
    payload = json.loads(probe_path.read_text(encoding="utf-8"))
    names = payload.get("body29_joint_names")
    if isinstance(names, list) and names:
        return [str(v) for v in names]
    all_names = payload.get("joint_names")
    if isinstance(all_names, list):
        return [str(v) for v in all_names[:29]]
    return []


def _flatten_actions(payload: Any) -> np.ndarray:
    arr = np.asarray(payload, dtype=np.float32).reshape(-1)
    if arr.shape[0] < 29:
        raise ValueError(f"joint_actions length is {arr.shape[0]}, expected >=29")
    return arr[:29]


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Diagnose SONIC joint index mapping against Isaac Sim DOF order.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--timeout-ms", type=int, default=10000)
    parser.add_argument("--map", dest="map_path", default=str(_default_map_path(root)))
    parser.add_argument("--isaac-probe", default=str(root / "tmp" / "inspect_g1_joints_data.json"))
    args = parser.parse_args()

    map_path = Path(args.map_path).resolve()
    if not map_path.exists():
        raise FileNotFoundError(f"Joint map not found: {map_path}")
    isaac_probe_path = Path(args.isaac_probe).resolve()

    with map_path.open("r", encoding="utf-8") as f:
        jmap = json.load(f)
    joints = sorted(jmap.get("joints", []), key=lambda item: int(item["sonic_idx"]))
    if len(joints) < 29:
        raise ValueError(f"Joint map has {len(joints)} entries, expected >=29")

    isaac_names = _load_isaac_names(isaac_probe_path)
    try:
        import sonic_policy_server as sonic  # local module

        default_isaac = np.asarray(sonic.DEFAULT_ANGLES_ISAAC, dtype=np.float32).reshape(-1)[:29]
    except Exception:
        default_isaac = np.zeros((29,), dtype=np.float32)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, int(args.timeout_ms))
    sock.setsockopt(zmq.SNDTIMEO, int(args.timeout_ms))
    sock.connect(f"tcp://{args.host}:{args.port}")
    try:
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
    finally:
        sock.close(linger=0)
        ctx.term()

    if not isinstance(resp, dict):
        raise RuntimeError(f"Unexpected response type: {type(resp).__name__}")
    if resp.get("error"):
        raise RuntimeError(f"SONIC error: {resp['error']}")

    actions = _flatten_actions(resp.get("joint_actions"))
    print(
        f"{'SONIC':>5} {'ISAAC':>5} {'joint name':>35} {'isaac dof':>35} {'group':>12} "
        f"{'value':>10} {'default':>10} {'delta':>10}"
    )
    print("-" * 150)
    mismatch_count = 0
    for entry in joints[:29]:
        sonic_idx = int(entry["sonic_idx"])
        isaac_idx = int(entry.get("isaac_idx", sonic_idx))
        joint_name = str(entry["name"])
        group = str(entry.get("group", ""))
        isaac_name = ""
        if 0 <= isaac_idx < len(isaac_names):
            isaac_name = isaac_names[isaac_idx]
            if isaac_name != joint_name:
                mismatch_count += 1
        value = float(actions[sonic_idx])
        default_value = float(default_isaac[sonic_idx]) if sonic_idx < default_isaac.shape[0] else 0.0
        delta = value - default_value
        print(
            f"{sonic_idx:5d} {isaac_idx:5d} {joint_name:>35} {isaac_name:>35} {group:>12} "
            f"{value:10.4f} {default_value:10.4f} {delta:10.4f}"
        )

    max_abs = float(np.max(np.abs(actions)))
    print("")
    print(f"map_file: {map_path}")
    print(f"isaac_probe: {isaac_probe_path if isaac_probe_path.exists() else 'not found'}")
    print(f"max_abs_action: {max_abs:.4f}")
    print(f"name_mismatches_vs_isaac_probe: {mismatch_count}")
    if max_abs > 1.5:
        print("status: WARNING (large neutral action magnitude)")
    elif mismatch_count > 0:
        print("status: WARNING (map names differ from isaac probe)")
    else:
        print("status: OK")


if __name__ == "__main__":
    main()
