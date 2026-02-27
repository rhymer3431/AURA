from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.agent_runtime.modules.manipulation_groot_trt import GrootManipulator  # noqa: E402
from apps.sonic_policy_server.telemetry_runtime import (  # noqa: E402
    compute_stats,
    file_sha256,
    load_jsonl,
    resolve_run_dir,
    write_report_markdown,
)


DEFAULT_CONFIG = ROOT / "apps" / "agent_runtime" / "config.yaml"
DEFAULT_JOINT_MAP = ROOT / "apps" / "agent_runtime" / "modules" / "g1_joint_map.json"

FILE_DIFF_SUMMARY = [
    "apps/sonic_policy_server/telemetry_runtime.py",
    "apps/agent_runtime/modules/g1_action_adapter.py",
    "apps/agent_runtime/modules/manipulation_groot_trt.py",
    "apps/sonic_policy_server/server.py",
    "apps/isaacsim_runner/isaac_runner.py",
    "tests/pipeline/pipeline_test_utils.py",
    "tests/pipeline/test_standing_pipeline.py",
    "tests/pipeline/test_walk_ramp_pipeline.py",
    "tests/pipeline/test_closed_loop_60s.py",
]


def load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(text)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    data = json.loads(text)
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid config file structure: {path}")
    return data


def build_manip_cfg(config_path: Path, run_dir: Path, phase: str) -> Dict[str, Any]:
    cfg = load_config(config_path)
    manip_cfg = dict(cfg.get("manipulation", {}))
    manip_cfg["backend"] = "mock"
    manip_cfg["mock_mode"] = True
    manip_cfg["fallback_to_mock"] = True
    manip_cfg.setdefault("sonic_server", {})
    manip_cfg["sonic_server"]["enabled"] = True
    telemetry_cfg = dict(manip_cfg.get("telemetry", {}))
    telemetry_cfg["enabled"] = True
    telemetry_cfg["phase"] = str(phase)
    telemetry_cfg["run_dir"] = str(run_dir)
    telemetry_cfg.setdefault("flush_every", 64)
    telemetry_cfg.setdefault("flush_interval_s", 0.5)
    telemetry_cfg.setdefault("full_dump_every_steps", 25)
    manip_cfg["telemetry"] = telemetry_cfg
    return manip_cfg


def init_run_dir(phase: str, run_dir: Optional[str]) -> Path:
    out = resolve_run_dir(run_dir=run_dir) if run_dir else resolve_run_dir()
    os.environ["AURA_TELEMETRY_RUN_DIR"] = str(out)
    os.environ["AURA_TELEMETRY_PHASE"] = str(phase)
    return out


def load_phase_records(run_dir: Path, phase: str) -> List[Dict[str, Any]]:
    return load_jsonl(run_dir / f"{phase}.jsonl")


def _float_values(records: List[Dict[str, Any]], key: str) -> List[float]:
    out: List[float] = []
    for rec in records:
        if key not in rec:
            continue
        try:
            val = float(rec[key])
        except Exception:
            continue
        if math.isfinite(val):
            out.append(val)
    return out


def _extract_action_scale_multiplier(records: List[Dict[str, Any]], default: float = 0.10) -> float:
    for rec in reversed(records):
        try:
            if "action_scale_multiplier" in rec:
                return float(rec["action_scale_multiplier"])
        except Exception:
            continue
    return float(default)


def _build_action_scale_base() -> List[float]:
    armature_5020 = 0.003609725
    armature_7520_14 = 0.010177520
    armature_7520_22 = 0.025101925
    armature_4010 = 0.00425
    natural_freq = 10.0 * 2.0 * math.pi
    stiffness_5020 = armature_5020 * natural_freq * natural_freq
    stiffness_7520_14 = armature_7520_14 * natural_freq * natural_freq
    stiffness_7520_22 = armature_7520_22 * natural_freq * natural_freq
    stiffness_4010 = armature_4010 * natural_freq * natural_freq
    s_5020 = 0.25 * 25.0 / stiffness_5020
    s_7520_14 = 0.25 * 88.0 / stiffness_7520_14
    s_7520_22 = 0.25 * 139.0 / stiffness_7520_22
    s_4010 = 0.25 * 5.0 / stiffness_4010
    return [
        s_7520_22,
        s_7520_22,
        s_7520_14,
        s_7520_22,
        s_5020,
        s_5020,
        s_7520_22,
        s_7520_22,
        s_7520_14,
        s_7520_22,
        s_5020,
        s_5020,
        s_7520_14,
        s_5020,
        s_5020,
        s_5020,
        s_5020,
        s_5020,
        s_5020,
        s_5020,
        s_4010,
        s_4010,
        s_5020,
        s_5020,
        s_5020,
        s_5020,
        s_5020,
        s_4010,
        s_4010,
    ]


def _default_angles_isaac() -> List[float]:
    default_mujo = [
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    ]
    isaac_to_mujo = [
        0,
        6,
        12,
        1,
        7,
        13,
        2,
        8,
        14,
        3,
        9,
        15,
        22,
        4,
        10,
        16,
        23,
        5,
        11,
        17,
        24,
        18,
        25,
        19,
        26,
        20,
        27,
        21,
        28,
    ]
    return [float(default_mujo[idx]) for idx in isaac_to_mujo]


def summarize_phase_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    loop_dt = _float_values(records, "loop_dt")
    server_latency = _float_values(records, "server_latency_ms")
    tracking_rms = _float_values(records, "tracking_err_rms")
    base_roll = [abs(v) for v in _float_values(records, "base_roll")]
    base_pitch = [abs(v) for v in _float_values(records, "base_pitch")]
    base_height = _float_values(records, "base_height")
    resp_max_abs = _float_values(records, "resp_max_abs_action")
    overrun = _float_values(records, "loop_overrun_ms")
    locomotion_zero = False
    for rec in records:
        vx = float(rec.get("vx", 0.0) or 0.0)
        vy = float(rec.get("vy", 0.0) or 0.0)
        yaw_rate = float(rec.get("yaw_rate", 0.0) or 0.0)
        if (abs(vx) + abs(vy) + abs(yaw_rate)) > 1e-3 and bool(rec.get("req_is_all_zero_pos", False)):
            locomotion_zero = True
            break
    fall_flag = any(bool(rec.get("fall_flag", False)) for rec in records)
    slip_flag = any(bool(rec.get("slip_flag", False)) for rec in records)
    jitter_std = compute_stats(loop_dt).get("std")
    hz_stats = None
    if loop_dt:
        hz_values = [1.0 / dt for dt in loop_dt if dt > 1e-6]
        hz_stats = compute_stats(hz_values)
    return {
        "record_count": len(records),
        "max_abs_action": max(resp_max_abs) if resp_max_abs else None,
        "max_roll": max(base_roll) if base_roll else None,
        "max_pitch": max(base_pitch) if base_pitch else None,
        "min_base_height": min(base_height) if base_height else None,
        "tracking_err_rms": compute_stats(tracking_rms).get("mean") if tracking_rms else None,
        "server_latency_p95": compute_stats(server_latency).get("p95") if server_latency else None,
        "loop_jitter_std": jitter_std,
        "overrun_count": sum(1 for x in overrun if x > 0.0),
        "avg_hz": (hz_stats or {}).get("mean"),
        "fall_flag": fall_flag,
        "slip_flag": slip_flag,
        "locomotion_req_all_zero": locomotion_zero,
    }


def find_latest_abort_window(run_dir: Path) -> Optional[Path]:
    matches = sorted(run_dir.glob("*abort*_last1s.jsonl"), key=lambda p: p.stat().st_mtime)
    if not matches:
        return None
    return matches[-1]


def summarize_abort_window(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    records = load_jsonl(path)
    if not records:
        return {}
    resp_abs = _float_values(records, "resp_max_abs_action")
    latency = _float_values(records, "server_latency_ms")
    return {
        "records": len(records),
        "resp_max_abs_action_max": max(resp_abs) if resp_abs else None,
        "server_latency_p95": compute_stats(latency).get("p95") if latency else None,
    }


def extract_actions_at_abort(records: List[Dict[str, Any]]) -> Optional[List[float]]:
    for rec in reversed(records):
        actions = rec.get("resp_joint_actions_29")
        if isinstance(actions, list) and actions:
            try:
                return [float(v) for v in actions[:29]]
            except Exception:
                continue
    return None


def extract_pd_gain_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for rec in reversed(records):
        if rec.get("event") != "pd_gains_applied":
            continue
        groups = rec.get("pd_groups")
        if isinstance(groups, list):
            out: List[Dict[str, Any]] = []
            for group in groups:
                if isinstance(group, dict):
                    out.append(group)
            return out
    return []


def action_scale_and_defaults(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    multiplier = _extract_action_scale_multiplier(records, default=0.10)
    scale = [v * multiplier for v in _build_action_scale_base()]
    defaults = _default_angles_isaac()
    scale_stats = compute_stats(scale)
    default_stats = compute_stats(defaults)
    return {
        "multiplier": multiplier,
        "per_joint": True,
        "action_scale": scale,
        "action_scale_stats": scale_stats,
        "default_angles_29": defaults,
        "default_angles_stats": default_stats,
    }


def load_joint_map_summary(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    joints = payload.get("joints", [])
    return {
        "joint_map_path": str(path),
        "joint_map_sha256": file_sha256(path),
        "joint_count": len(joints) if isinstance(joints, list) else 0,
    }


def render_report(
    run_dir: Path,
    phase: str,
    mapping: Dict[str, Any],
    action_scale: Dict[str, Any],
    control_metrics: Dict[str, Any],
    standing: Dict[str, Any],
    walking: Dict[str, Any],
    closed_loop: Dict[str, Any],
    pd_gain_rows: Optional[List[Dict[str, Any]]] = None,
) -> str:
    pd_gain_rows = pd_gain_rows or []
    pd_lines: List[str] = []
    if pd_gain_rows:
        pd_lines.append("| Group | Joint Count | KP Min | KP Max | KD Min | KD Max |")
        pd_lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in pd_gain_rows:
            pd_lines.append(
                f"| {row.get('label','')} | {int(row.get('joint_count',0))} | "
                f"{float(row.get('kp_min',0.0)):.2f} | {float(row.get('kp_max',0.0)):.2f} | "
                f"{float(row.get('kd_min',0.0)):.2f} | {float(row.get('kd_max',0.0)):.2f} |"
            )
    else:
        pd_lines.append("No PD gain telemetry rows were found in this run directory.")

    abort_actions = walking.get("actions_at_abort")
    abort_actions_str = "n/a"
    if isinstance(abort_actions, list):
        abort_actions_str = "[" + ", ".join(f"{float(v):.4f}" for v in abort_actions[:29]) + "]"

    report = "\n".join(
        [
            "# Runtime Stability Report",
            f"- Run dir: `{run_dir}`",
            f"- Phase file: `{phase}.jsonl`",
            "",
            "## Joint mapping",
            f"- g1_joint_map.json path: `{mapping.get('joint_map_path','')}`",
            f"- g1_joint_map.json sha256: `{mapping.get('joint_map_sha256','')}`",
            f"- req_joint_pos_29 all-zero during locomotion: `{control_metrics.get('locomotion_req_all_zero')}`",
            "",
            "## Action scale",
            f"- ACTION_SCALE multiplier: `{action_scale.get('multiplier')}`",
            f"- ACTION_SCALE per-joint: `{action_scale.get('per_joint')}`",
            (
                "- ACTION_SCALE summary: "
                f"min={float(action_scale['action_scale_stats'].get('min') or 0.0):.6f}, "
                f"max={float(action_scale['action_scale_stats'].get('max') or 0.0):.6f}, "
                f"mean={float(action_scale['action_scale_stats'].get('mean') or 0.0):.6f}"
            ),
            (
                "- default_angles_29 summary: "
                f"min={float(action_scale['default_angles_stats'].get('min') or 0.0):.4f}, "
                f"max={float(action_scale['default_angles_stats'].get('max') or 0.0):.4f}, "
                f"mean={float(action_scale['default_angles_stats'].get('mean') or 0.0):.4f}"
            ),
            "",
            "## Control frequency",
            f"- avg hz: `{control_metrics.get('avg_hz')}`",
            f"- jitter std: `{control_metrics.get('loop_jitter_std')}`",
            f"- overrun count: `{control_metrics.get('overrun_count')}`",
            "",
            "## PD gains applied",
            *pd_lines,
            "",
            "## Standing result",
            f"- pass: `{standing.get('pass')}`",
            f"- max_abs_action: `{standing.get('max_abs_action')}`",
            f"- max roll/pitch: `{standing.get('max_roll')}` / `{standing.get('max_pitch')}`",
            "",
            "## Walking result",
            f"- completed: `{walking.get('completed')}`",
            f"- aborted: `{walking.get('aborted')}`",
            f"- abort step/vx: `{walking.get('abort_step')}` / `{walking.get('abort_vx')}`",
            f"- actions_at_abort: `{abort_actions_str}`",
            f"- last 1s stats: `{walking.get('last_1s_stats')}`",
            "",
            "## Closed-loop result",
            f"- pass: `{closed_loop.get('pass')}`",
            f"- max roll/pitch: `{closed_loop.get('max_roll')}` / `{closed_loop.get('max_pitch')}`",
            f"- min base height: `{closed_loop.get('min_base_height')}`",
            f"- tracking_err_rms: `{closed_loop.get('tracking_err_rms')}`",
            f"- server latency p95: `{closed_loop.get('server_latency_p95')}`",
            f"- loop jitter std: `{closed_loop.get('loop_jitter_std')}`",
            f"- overrun count: `{closed_loop.get('overrun_count')}`",
            "",
            "## File-by-file diff summary",
            *[f"- `{path}`" for path in FILE_DIFF_SUMMARY],
            "",
        ]
    )
    return report


def write_report(run_dir: Path, content: str) -> Path:
    return write_report_markdown(run_dir=run_dir, content=content)
