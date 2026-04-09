from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from systems.transport.messages import CapabilityReport, HealthPing, RuntimeNotice
from server.snapshot_adapter import SnapshotAdapter


@dataclass(slots=True)
class RuntimePublisher:
    runtime_io_provider: Callable[[], object | None]
    bridge_provider: Callable[[], object | None]
    detector_report_provider: Callable[[], object | None]
    detector_backend_provider: Callable[[], str]
    snapshot_provider: Callable[[], object | None]
    snapshot_interval_provider: Callable[[], int]
    last_snapshot_frame_getter: Callable[[], int]
    last_snapshot_frame_setter: Callable[[int], None]

    def publish_tick(self, *, frame_idx: int, notices: tuple[object, ...], status) -> None:  # noqa: ANN001
        self.publish_notices(notices)
        self.publish_status(status)
        self.publish_runtime_snapshot(frame_idx=frame_idx)

    def publish_notices(self, notices: tuple[object, ...]) -> None:
        for notice in notices:
            self.publish_notice(
                level=str(getattr(notice, "level", "info")),
                notice=str(getattr(notice, "notice", "")),
                details=getattr(notice, "details", {}),
                component=str(getattr(notice, "component", "aura_runtime")),
            )

    def publish_status(self, status) -> None:  # noqa: ANN001
        if status is None or self.runtime_io_provider() is None:
            return
        bridge = self.bridge_provider()
        if bridge is None:
            return
        bridge.publish_status(status)

    def publish_detector_capability(self) -> None:
        if self.runtime_io_provider() is None:
            return
        detector_report = self.detector_report_provider()
        bridge = self.bridge_provider()
        if detector_report is None or bridge is None:
            return
        bridge.publish_capability(
            CapabilityReport(
                component="detector",
                status="ready" if detector_report.ready_for_inference else "fallback",
                backend_name=self.detector_backend_provider(),
                details=detector_report.as_dict(),
                warnings=list(getattr(detector_report, "warnings", [])),
                errors=list(getattr(detector_report, "errors", [])),
            )
        )

    def publish_notice(
        self,
        *,
        level: str,
        notice: str,
        details: dict[str, object] | None = None,
        component: str = "aura_runtime",
    ) -> None:
        if self.runtime_io_provider() is None:
            return
        bridge = self.bridge_provider()
        if bridge is None:
            return
        bridge.publish_notice(
            RuntimeNotice(component=component, level=level, notice=notice, details=dict(details or {}))
        )

    def publish_runtime_snapshot(self, *, frame_idx: int) -> None:
        if self.runtime_io_provider() is None:
            return
        interval = max(int(self.snapshot_interval_provider()), 1)
        last_snapshot_frame = int(self.last_snapshot_frame_getter())
        if last_snapshot_frame >= 0 and (int(frame_idx) - last_snapshot_frame) < interval:
            return
        self.last_snapshot_frame_setter(int(frame_idx))
        bridge = self.bridge_provider()
        if bridge is None:
            return
        snapshot = self.snapshot_provider()
        bridge.publish_health(
            HealthPing(
                component="aura_runtime",
                details={
                    "worldState": {} if snapshot is None else snapshot.to_dict(),
                    "snapshot": SnapshotAdapter.to_legacy_runtime_payload(snapshot),
                },
            )
        )

