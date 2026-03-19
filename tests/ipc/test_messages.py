from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ipc.codec import decode_message, encode_message
from ipc.messages import ActionCommand, ActionStatus, CapabilityReport, FrameHeader, HealthPing, RuntimeControlRequest, RuntimeNotice, TaskRequest


def test_ipc_message_roundtrip_encode_decode() -> None:
    samples = [
        FrameHeader(frame_id=1, timestamp_ns=123, source="isaac", width=640, height=480),
        ActionCommand(action_type="NAV_TO_PLACE", target_place_id="place_0001", target_pose_xyz=(1.0, 2.0, 0.0)),
        ActionStatus(command_id="cmd_1", state="running", robot_pose_xyz=(0.0, 0.0, 0.0)),
        TaskRequest(command_text="아까 봤던 사과를 찾아가", target_json={"target_class": "apple"}),
        RuntimeControlRequest(action="set_idle"),
        CapabilityReport(component="detector", status="fallback", backend_name="color_seg_fallback"),
        RuntimeNotice(component="bridge", level="warning", notice="fallback"),
        HealthPing(component="memory_agent"),
    ]

    for sample in samples:
        decoded = decode_message(encode_message(sample))
        assert type(decoded) is type(sample)
        assert decoded == sample
