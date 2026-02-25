from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import deque
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


RUN_DIR_ENV = "AURA_TELEMETRY_RUN_DIR"
PHASE_ENV = "AURA_TELEMETRY_PHASE"

TELEMETRY_SCHEMA_KEYS = {
    "phase",
    "step_idx",
    "t_req_send",
    "t_resp_recv",
    "t_apply",
    "vx",
    "vy",
    "yaw_rate",
    "style",
    "req_joint_pos_29",
    "req_joint_vel_29",
    "req_is_all_zero_pos",
    "req_is_all_zero_vel",
    "req_max_abs_pos",
    "req_max_abs_vel",
    "resp_joint_actions_29",
    "resp_max_abs_action",
    "default_angles_29",
    "action_scale",
    "target_joint_pos_29",
    "target_max_abs",
    "applied_dof_count",
    "applied_body29_dof_names",
    "applied_target_len",
    "delta_max_abs_before_clip",
    "delta_max_abs_after_clip",
    "fb_joint_pos_29",
    "fb_joint_vel_29",
    "tracking_err_29",
    "tracking_err_max_abs",
    "tracking_err_rms",
    "base_roll",
    "base_pitch",
    "base_yaw",
    "base_height",
    "fall_flag",
    "slip_flag",
    "loop_dt",
    "loop_overrun_ms",
    "publish_hz_window",
    "server_latency_ms",
    "rtf",
}


def now_perf() -> float:
    return time.perf_counter()


def _safe_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def _percentile(sorted_values: Sequence[float], q: float) -> Optional[float]:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    qq = max(0.0, min(1.0, float(q)))
    pos = qq * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def compute_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    cleaned: List[float] = []
    for raw in values:
        val = _safe_float(raw)
        if val is not None:
            cleaned.append(val)
    if not cleaned:
        return {
            "count": 0.0,
            "mean": None,
            "std": None,
            "p50": None,
            "p95": None,
            "max": None,
            "min": None,
            "rms": None,
        }
    cleaned.sort()
    mean = float(sum(cleaned) / len(cleaned))
    var = float(sum((v - mean) * (v - mean) for v in cleaned) / len(cleaned))
    rms = float(math.sqrt(sum(v * v for v in cleaned) / len(cleaned)))
    return {
        "count": float(len(cleaned)),
        "mean": mean,
        "std": float(math.sqrt(max(0.0, var))),
        "p50": _percentile(cleaned, 0.50),
        "p95": _percentile(cleaned, 0.95),
        "max": float(cleaned[-1]),
        "min": float(cleaned[0]),
        "rms": rms,
    }


def summarize_window(values: Sequence[float]) -> Dict[str, Optional[float]]:
    return compute_stats(values)


def clamp_and_count_clip(x: Iterable[float], lo: float, hi: float) -> Tuple[List[float], int]:
    low = float(lo)
    high = float(hi)
    if high < low:
        low, high = high, low
    clipped: List[float] = []
    clip_count = 0
    for raw in x:
        val = _safe_float(raw)
        if val is None:
            val = 0.0
            clip_count += 1
        if val < low:
            val = low
            clip_count += 1
        elif val > high:
            val = high
            clip_count += 1
        clipped.append(float(val))
    return clipped, clip_count


def resolve_run_dir(base_dir: str | Path = "logs", run_dir: str | Path | None = None) -> Path:
    if run_dir is not None:
        path = Path(run_dir).resolve()
        path.mkdir(parents=True, exist_ok=True)
        os.environ[RUN_DIR_ENV] = str(path)
        return path
    env_run_dir = os.environ.get(RUN_DIR_ENV, "").strip()
    if env_run_dir:
        path = Path(env_run_dir).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    root = Path(base_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    path = root / stamp
    suffix = 1
    while path.exists():
        path = root / f"{stamp}_{suffix:02d}"
        suffix += 1
    path.mkdir(parents=True, exist_ok=True)
    os.environ[RUN_DIR_ENV] = str(path)
    return path


def file_sha256(path: str | Path) -> str:
    src = Path(path)
    h = sha256()
    with src.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    src = Path(path)
    records: List[Dict[str, Any]] = []
    if not src.exists():
        return records
    with src.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


def write_report_markdown(run_dir: str | Path, content: str) -> Path:
    path = Path(run_dir) / "report.md"
    path.write_text(content, encoding="utf-8")
    return path


class JsonlTelemetryLogger:
    def __init__(
        self,
        phase: Optional[str],
        component: str,
        run_dir: str | Path | None = None,
        base_dir: str | Path = "logs",
        flush_every: int = 64,
        flush_interval_s: float = 0.5,
        recent_window_maxlen: int = 4096,
    ) -> None:
        self.component = str(component)
        self.phase = str(phase or os.environ.get(PHASE_ENV, "standing"))
        self.run_dir = resolve_run_dir(base_dir=base_dir, run_dir=run_dir)
        self.file_path = self.run_dir / f"{self.phase}.jsonl"
        self.flush_every = max(1, int(flush_every))
        self.flush_interval_s = max(0.01, float(flush_interval_s))
        self._lock = threading.Lock()
        self._fp = self.file_path.open("a", encoding="utf-8")
        self._buffer: List[str] = []
        self._last_flush_t = now_perf()
        self._recent: Deque[Tuple[float, Dict[str, Any]]] = deque(maxlen=max(32, int(recent_window_maxlen)))

    def switch_phase(self, phase: str) -> None:
        new_phase = str(phase).strip() or "standing"
        with self._lock:
            self._flush_unlocked()
            self._fp.close()
            self.phase = new_phase
            self.file_path = self.run_dir / f"{self.phase}.jsonl"
            self._fp = self.file_path.open("a", encoding="utf-8")

    def log(self, record: Mapping[str, Any]) -> None:
        if not isinstance(record, Mapping):
            return
        now = now_perf()
        payload: Dict[str, Any] = dict(record)
        payload.setdefault("component", self.component)
        payload.setdefault("phase", self.phase)
        payload.setdefault("perf_time", now)
        payload.setdefault("timestamp_utc", datetime.utcnow().isoformat(timespec="milliseconds") + "Z")
        line = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        with self._lock:
            self._buffer.append(line)
            self._recent.append((now, payload))
            if len(self._buffer) >= self.flush_every or (now - self._last_flush_t) >= self.flush_interval_s:
                self._flush_unlocked()

    def recent_records(self, within_s: float = 1.0, ref_time: Optional[float] = None) -> List[Dict[str, Any]]:
        horizon = max(0.0, float(within_s))
        anchor = now_perf() if ref_time is None else float(ref_time)
        out: List[Dict[str, Any]] = []
        with self._lock:
            for ts, rec in self._recent:
                if (anchor - ts) <= horizon:
                    out.append(dict(rec))
        return out

    def dump_recent_window(self, filename: str, within_s: float = 1.0, ref_time: Optional[float] = None) -> Path:
        target = self.run_dir / str(filename)
        records = self.recent_records(within_s=within_s, ref_time=ref_time)
        with target.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=True, separators=(",", ":")) + "\n")
        return target

    def flush(self) -> None:
        with self._lock:
            self._flush_unlocked()

    def _flush_unlocked(self) -> None:
        if not self._buffer:
            self._last_flush_t = now_perf()
            return
        self._fp.write("\n".join(self._buffer) + "\n")
        self._fp.flush()
        self._buffer.clear()
        self._last_flush_t = now_perf()

    def close(self) -> None:
        with self._lock:
            self._flush_unlocked()
            self._fp.close()

    def __enter__(self) -> "JsonlTelemetryLogger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

