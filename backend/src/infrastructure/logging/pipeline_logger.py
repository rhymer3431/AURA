from __future__ import annotations

import csv
import json
import time
import math
from dataclasses import dataclass, asdict
from typing import Any, List, Optional
import numpy as np

@dataclass
class PipelineEvent:
    ts: float
    module: str
    brain_module: Optional[str]
    event: str


class PipelineLogger:
    """
    Structured logger with active-time weighting.
    Output schema: ts, module, brain_module, event, weight
    """

    def __init__(self, enabled: bool = True) -> None:
        self._events: List[PipelineEvent] = []
        self.enabled = enabled

    def log(
        self,
        module: str,
        event: str,
        matched_brain: Optional[str] = None,
        **payload: Any,
    ) -> None:
        if not self.enabled:
            return
        ts = time.time()
        brain = matched_brain or payload.get("matched_brain") if payload else None
        ev = PipelineEvent(
            ts=ts,
            module=module,
            brain_module=brain,
            event=event,
        )
        self._events.append(ev)

    @property
    def events(self) -> List[PipelineEvent]:
        return self._events

    def to_csv(self, path: str) -> None:
        if not self.enabled:
            return
        events = sorted(self._events, key=lambda ev: ev.ts)
        weights = self._compute_weights(events)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["ts", "module", "brain_module", "event", "weight"],
            )
            writer.writeheader()
            for ev, w in zip(events, weights):
                writer.writerow(
                    {
                        "ts": ev.ts,
                        "module": ev.module,
                        "brain_module": ev.brain_module or "",
                        "event": ev.event,
                        "weight": w,
                    }
                )

    def to_jsonl(self, path: str) -> None:
        if not self.enabled:
            return
        events = sorted(self._events, key=lambda ev: ev.ts)
        weights = self._compute_weights(events)
        with open(path, "w", encoding="utf-8") as f:
            for ev, w in zip(events, weights):
                obj = asdict(ev)
                obj["weight"] = w
                f.write(json.dumps(obj, ensure_ascii=False))
                f.write("\n")


    @staticmethod
    def _compute_weights(events: List[PipelineEvent]) -> List[float]:
        n = len(events)
        if n == 0:
            return []

        # 1) deltas (다음 이벤트와의 시간 차)
        deltas = []
        for i, ev in enumerate(events):
            next_ts = events[i+1].ts if i+1 < n else None
            d = float(next_ts - ev.ts) if next_ts else 0.0
            deltas.append(max(0.0, d))

        # 모든 delta가 0이면 그대로 0 반환
        if all(d == 0.0 for d in deltas):
            return [0.0] * n

        # 2) log 압축
        logs = np.log1p(np.array(deltas))

        # 3) Robust scaling: median / MAD
        median = np.median(logs)
        mad = np.median(np.abs(logs - median)) + 1e-6
        robust = (logs - median) / mad

        # 4) Outlier 클램프
        robust = np.clip(robust, -3, 3)

        # 5) Sigmoid → [0,1]
        weights = 1 / (1 + np.exp(-robust))

        # 소숫점 2자리까지 반올림
        weights = [round(float(w), 2) for w in weights]

        return weights
