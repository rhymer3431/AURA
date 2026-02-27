from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from apps.agent_runtime.modules.look_at_controller import LookAtController
from apps.agent_runtime.modules.perception_yoloe_trt import YOLOEPerception


@dataclass
class _FakeAim:
    yaw: float = 0.0
    pitch: float = 0.0

    def get_camera_aim(self):
        return self.yaw, self.pitch

    def command_camera_aim(self, yaw: float, pitch: float, source: str):
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        return self.yaw, self.pitch


def main() -> None:
    perception = YOLOEPerception({"mock_mode": True, "base_interval_s": 0.02, "input_size": [480, 640]})
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    payload = {"image": frame, "timestamp": time.time(), "frame_id": "validate", "color_format": "rgb"}
    dets = perception._infer_once(payload)
    if not dets:
        raise RuntimeError("No detections produced by mock perception pipeline.")
    det = dets[0]
    if det.bbox_xyxy is None or det.bbox_cxcy is None:
        raise RuntimeError("Detection struct is missing bbox_xyxy/bbox_cxcy.")
    if not isinstance(det.frame_id, str):
        raise RuntimeError("Detection struct is missing frame_id.")

    perception._class_cycle = [det.class_name]
    perception._update_detection_state(
        detections=dets,
        image_shape=frame.shape[:2],
        frame_id="validate",
        timestamp=time.time(),
    )

    aim = _FakeAim()
    controller = LookAtController(
        get_target=lambda label, max_age: perception.get_tracked_target(label, max_age),
        get_camera_aim=aim.get_camera_aim,
        command_camera_aim=aim.command_camera_aim,
    )
    controller.activate(det.class_name, {"smoothing": 1.0, "max_rate_deg_s": 20.0, "deadband_px": 2.0})

    for _ in range(8):
        payload = {"image": frame, "timestamp": time.time(), "frame_id": "validate", "color_format": "rgb"}
        dets = perception._infer_once(payload)
        perception._update_detection_state(
            detections=dets,
            image_shape=frame.shape[:2],
            frame_id="validate",
            timestamp=time.time(),
        )
        controller._last_step_mono -= 0.1
        status = controller.step()
        if status.state == "target_lost":
            break

    print(
        f"look_at status={status.state} object={status.object_label} "
        f"score={status.target_score:.2f} yaw={aim.yaw:.4f} pitch={aim.pitch:.4f}"
    )
    if abs(aim.yaw) < 1e-8 and abs(aim.pitch) < 1e-8:
        raise RuntimeError("look_at did not change yaw/pitch from zero.")


if __name__ == "__main__":
    main()
