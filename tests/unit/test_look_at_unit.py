import math
import time

import pytest

np = pytest.importorskip("numpy")

from apps.agent_runtime.modules.contracts import Detection2D3D
from apps.agent_runtime.modules.look_at_controller import LookAtController
from apps.agent_runtime.modules.perception_yoloe_trt import YOLOEPerception, nms_xyxy


def _det(label: str, score: float, x1: float, y1: float, x2: float, y2: float) -> Detection2D3D:
    w = x2 - x1
    h = y2 - y1
    return Detection2D3D(
        object_id=f"{label}-0",
        class_name=label,
        score=score,
        bbox_xywh=(x1, y1, w, h),
        bbox_xyxy=(x1, y1, x2, y2),
        bbox_cxcy=(x1 + 0.5 * w, y1 + 0.5 * h),
        image_size=(640, 480),
        frame_id="camera",
        timestamp=time.time(),
    )


def test_decode_and_nms_shapes() -> None:
    perception = YOLOEPerception({"mock_mode": True, "input_size": [640, 640]})
    pred = np.array(
        [
            [
                [100.0, 100.0, 220.0, 220.0, 0.92, 0.0],
                [110.0, 110.0, 230.0, 230.0, 0.86, 0.0],
                [320.0, 300.0, 360.0, 360.0, 0.97, 1.0],
            ]
        ],
        dtype=np.float32,
    )
    boxes, scores, cls_ids = perception._decode_model_outputs([pred])
    assert boxes.shape == (3, 4)
    assert scores.shape == (3,)
    assert cls_ids.shape == (3,)

    keep = nms_xyxy(boxes, scores, iou_threshold=0.5, max_dets=10)
    assert len(keep) == 2


def test_tracker_prefers_center_and_smooths() -> None:
    perception = YOLOEPerception(
        {
            "mock_mode": True,
            "input_size": [640, 640],
            "tracker": {"prefer_center": True, "center_weight": 0.4, "ema_alpha": 0.5},
        }
    )
    t0 = time.time()
    dets = [
        _det("cup", 0.93, 560.0, 50.0, 620.0, 140.0),  # high score, far from center
        _det("cup", 0.82, 280.0, 200.0, 360.0, 320.0),  # lower score, near center
    ]
    perception._update_detection_state(dets, image_shape=(480, 640), frame_id="cam", timestamp=t0)
    target = perception.get_tracked_target("cup")
    assert target is not None
    assert abs(target.cx - 320.0) < 20.0
    assert abs(target.cy - 260.0) < 20.0

    dets2 = [_det("cup", 0.95, 300.0, 210.0, 380.0, 330.0)]
    perception._update_detection_state(dets2, image_shape=(480, 640), frame_id="cam", timestamp=t0 + 0.1)
    target2 = perception.get_tracked_target("cup")
    assert target2 is not None
    assert target2.cx > target.cx
    assert target2.cy > target.cy


def test_look_at_controller_deadband_and_rate_limit() -> None:
    yaw_pitch = {"yaw": 0.0, "pitch": 0.0}
    target = {"obj": None}

    class _Target:
        def __init__(self, cx: float, cy: float, score: float) -> None:
            self.cx = cx
            self.cy = cy
            self.score = score
            self.image_width = 640
            self.image_height = 480

    def _get_target(label: str, max_age: float):
        return target["obj"]

    def _get_aim():
        return yaw_pitch["yaw"], yaw_pitch["pitch"]

    def _cmd_aim(yaw: float, pitch: float, source: str):
        yaw_pitch["yaw"] = yaw
        yaw_pitch["pitch"] = pitch
        return yaw, pitch

    ctrl = LookAtController(_get_target, _get_aim, _cmd_aim)
    ctrl.activate(
        "person",
        {
            "deadband_px": 10.0,
            "max_rate_deg_s": 10.0,
            "smoothing": 1.0,
            "kx": 0.02,
            "ky": 0.02,
        },
    )

    # Deadband: no command change.
    target["obj"] = _Target(324.0, 238.0, 0.8)
    ctrl._last_step_mono -= 1.0
    status = ctrl.step()
    assert status.state == "tracking"
    assert abs(yaw_pitch["yaw"]) < 1e-9
    assert abs(yaw_pitch["pitch"]) < 1e-9

    # Large error: command is rate limited.
    target["obj"] = _Target(640.0, 0.0, 0.9)
    ctrl._last_step_mono -= 1.0
    ctrl.step()
    max_step = math.radians(10.0)
    assert abs(yaw_pitch["yaw"]) <= max_step + 5e-6
    assert abs(yaw_pitch["pitch"]) <= max_step + 5e-6
