from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from perception.speaker_events import SpeakerEvent


@dataclass(frozen=True)
class FrameSourceReport:
    source_name: str
    status: str
    live_available: bool = False
    fallback_used: bool = False
    notice: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameSample:
    frame_id: int
    source_name: str
    rgb: np.ndarray
    depth: np.ndarray
    camera_pose_xyz: tuple[float, float, float]
    camera_quat_wxyz: tuple[float, float, float, float]
    robot_pose_xyz: tuple[float, float, float]
    robot_yaw_rad: float
    camera_intrinsic: np.ndarray
    sim_time_s: float
    metadata: dict[str, Any] = field(default_factory=dict)
    speaker_events: list[SpeakerEvent] = field(default_factory=list)


class FrameSource(Protocol):
    def start(self) -> FrameSourceReport:
        raise NotImplementedError

    def read(self) -> FrameSample | None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def report(self) -> FrameSourceReport:
        raise NotImplementedError


def build_synthetic_frame_sample(
    *,
    frame_id: int,
    scene: str,
    source_name: str,
    room_id: str = "",
    width: int = 96,
    height: int = 96,
) -> FrameSample:
    scene_name = str(scene).strip().lower()
    rgb = np.zeros((int(height), int(width), 3), dtype=np.uint8)
    depth = np.full((int(height), int(width)), 1.5, dtype=np.float32)
    metadata: dict[str, Any] = {"room_id": room_id, "frame_source": source_name}
    speaker_events: list[SpeakerEvent] = []

    if scene_name in {"apple", "cube"}:
        rgb[24:72, 28:68, 0] = 255
        metadata["target_class_hint"] = "apple" if scene_name == "apple" else "cube"
        metadata["color_hint"] = "red"
    elif scene_name == "person":
        rgb[18:78, 34:62, 2] = 255
        metadata["target_class_hint"] = "person"
        metadata["color_hint"] = "blue"
        metadata["embedding_id"] = "synthetic_person_blue"
        metadata["appearance_signature"] = "blue"
        speaker_events.append(
            SpeakerEvent(
                timestamp=time.time(),
                direction_yaw_rad=0.45,
                speaker_id="person_0001",
                confidence=0.95,
                metadata={"source": source_name},
            )
        )
    else:
        rgb[24:72, 28:68, 1] = 255
        metadata["target_class_hint"] = scene_name or "object"
        metadata["color_hint"] = "green"

    intrinsic = np.asarray(
        [
            [float(width), 0.0, float(width) * 0.5],
            [0.0, float(height), float(height) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return FrameSample(
        frame_id=int(frame_id),
        source_name=str(source_name),
        rgb=rgb,
        depth=depth,
        camera_pose_xyz=(0.0, 0.0, 1.2),
        camera_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        robot_pose_xyz=(0.0, 0.0, 0.0),
        robot_yaw_rad=0.0,
        camera_intrinsic=intrinsic,
        sim_time_s=float(time.time()),
        metadata=metadata,
        speaker_events=speaker_events,
    )


class SyntheticFrameSource:
    def __init__(self, *, scene: str, source_name: str, room_id: str = "", width: int = 96, height: int = 96) -> None:
        self._scene = str(scene)
        self._source_name = str(source_name)
        self._room_id = str(room_id)
        self._width = int(width)
        self._height = int(height)
        self._frame_id = 0
        self._report = FrameSourceReport(
            source_name=self._source_name,
            status="ready",
            live_available=False,
            fallback_used=False,
            notice="synthetic frame source ready",
        )

    def start(self) -> FrameSourceReport:
        return self._report

    def read(self) -> FrameSample:
        self._frame_id += 1
        return build_synthetic_frame_sample(
            frame_id=self._frame_id,
            scene=self._scene,
            source_name=self._source_name,
            room_id=self._room_id,
            width=self._width,
            height=self._height,
        )

    def close(self) -> None:
        return None

    def report(self) -> FrameSourceReport:
        return self._report


class AutoFrameSource:
    def __init__(self, live_source: FrameSource, fallback_source: FrameSource, *, strict_live: bool = False) -> None:
        self._live_source = live_source
        self._fallback_source = fallback_source
        self._strict_live = bool(strict_live)
        self._active_source: FrameSource | None = None
        self._report = FrameSourceReport(source_name="auto", status="pending")

    def start(self) -> FrameSourceReport:
        live_report = self._live_source.start()
        if live_report.status == "ready":
            self._active_source = self._live_source
            self._report = FrameSourceReport(
                source_name=live_report.source_name,
                status="ready",
                live_available=True,
                fallback_used=False,
                notice=live_report.notice or "live frame source selected",
                details={"live_report": live_report.details},
            )
            return self._report
        if self._strict_live:
            self._active_source = self._live_source
            self._report = FrameSourceReport(
                source_name=live_report.source_name,
                status="unavailable",
                live_available=False,
                fallback_used=False,
                notice=live_report.notice or "live frame source unavailable",
                details={"live_report": live_report.details},
            )
            return self._report
        fallback_report = self._fallback_source.start()
        self._active_source = self._fallback_source
        self._report = FrameSourceReport(
            source_name=fallback_report.source_name,
            status=fallback_report.status,
            live_available=False,
            fallback_used=True,
            notice=live_report.notice or "live frame source unavailable; synthetic fallback selected",
            details={
                "live_report": live_report.details,
                "fallback_report": fallback_report.details,
            },
        )
        return self._report

    def read(self) -> FrameSample | None:
        if self._active_source is None:
            self.start()
        if self._active_source is None:
            return None
        return self._active_source.read()

    def close(self) -> None:
        self._live_source.close()
        self._fallback_source.close()

    def report(self) -> FrameSourceReport:
        return self._report
