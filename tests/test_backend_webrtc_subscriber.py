from __future__ import annotations

import asyncio
import time
import uuid
from types import SimpleNamespace

import numpy as np

from backend.webrtc.config import WebRTCServiceConfig
from backend.webrtc.models import FrameCache, build_frame_meta_message
from backend.webrtc.subscriber import ObservationSubscriber
from systems.transport import FrameHeader, SharedMemoryRing, encode_ndarray, ref_to_dict


class _NoopBus:
    def poll(self, _topic: str, *, max_items: int = 0):  # noqa: ARG002
        return []

    def close(self) -> None:
        return None


class _SingleRecordBus:
    def __init__(self, record) -> None:  # noqa: ANN001
        self._record = record
        self._sent = False

    def poll(self, topic: str, *, max_items: int = 0):  # noqa: ARG002
        if topic != "isaac.observation" or self._sent:
            return []
        self._sent = True
        return [self._record]

    def close(self) -> None:
        return None


class _OverwriteShm:
    def read(self, ref):  # noqa: ANN001, ARG002
        raise RuntimeError("Shared memory slot was overwritten before it was read.")

    def close(self) -> None:
        return None


def _make_frame(*, last_frame_monotonic: float) -> FrameCache:
    metadata = {
        "viewer_overlay": {
            "trajectory_pixels": [[10, 20], [30, 40]],
            "trajectoryPixels": [[10, 20], [30, 40]],
            "system2_pixel_goal": [111, 222],
            "system2PixelGoal": [111, 222],
            "active_target": {
                "className": "Navigation Goal",
                "source": "navigation",
                "nav_goal_pixel": [111, 222],
                "world_pose_xyz": [1.0, 2.0, 0.0],
            },
            "activeTarget": {
                "className": "Navigation Goal",
                "source": "navigation",
                "nav_goal_pixel": [111, 222],
                "world_pose_xyz": [1.0, 2.0, 0.0],
            },
        }
    }
    return FrameCache(
        seq=7,
        frame_header=FrameHeader(
            frame_id=12,
            timestamp_ns=123456789,
            source="perception_runtime",
            width=320,
            height=180,
            rgb_encoding="rgb8",
            depth_encoding="",
            camera_pose_xyz=(0.0, 0.0, 0.0),
            camera_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            robot_pose_xyz=(1.0, 2.0, 3.0),
            robot_yaw_rad=0.25,
            sim_time_s=4.5,
            metadata=metadata,
        ),
        rgb_image=np.zeros((180, 320, 3), dtype=np.uint8),
        depth_image_m=None,
        viewer_overlay=metadata["viewer_overlay"],
        last_frame_monotonic=last_frame_monotonic,
    )


def test_webrtc_subscriber_decodes_shared_memory_refs_only() -> None:
    shm_name = f"aura_viewer_test_{uuid.uuid4().hex[:12]}"
    writer = SharedMemoryRing(name=shm_name, slot_size=4096, capacity=2, create=True)
    reader = SharedMemoryRing(name=shm_name, slot_size=4096, capacity=2, create=False)
    subscriber = ObservationSubscriber(
        WebRTCServiceConfig(shm_name=shm_name, shm_slot_size=4096, shm_capacity=2),
        bus=_NoopBus(),
        shm_ring=reader,
    )
    rgb = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    depth = np.linspace(0.0, 1.0, 4, dtype=np.float32).reshape(2, 2)
    try:
        rgb_ref = writer.write(encode_ndarray(rgb))
        depth_ref = writer.write(encode_ndarray(depth))
        inline_rgb = encode_ndarray(rgb).hex()
        inline_depth = encode_ndarray(depth).hex()

        decoded_rgb = subscriber._decode_rgb({"rgb_ref": ref_to_dict(rgb_ref), "rgb_inline": inline_rgb})
        decoded_depth = subscriber._decode_depth({"depth_ref": ref_to_dict(depth_ref), "depth_inline": inline_depth})

        assert decoded_rgb is not None
        assert decoded_depth is not None
        assert np.array_equal(decoded_rgb, rgb)
        assert np.array_equal(decoded_depth, depth)
        assert subscriber._decode_rgb({"rgb_inline": inline_rgb}) is None
        assert subscriber._decode_depth({"depth_inline": inline_depth}) is None
    finally:
        reader.close()
        writer.close(unlink=True)


def test_webrtc_frame_meta_preserves_overlay_contract_keys() -> None:
    overlay = {
        "trajectory_pixels": [[10, 20], [30, 40]],
        "system2_pixel_goal": [111, 222],
        "active_target": {
            "className": "Navigation Goal",
            "source": "navigation",
            "nav_goal_pixel": [111, 222],
            "world_pose_xyz": [1.0, 2.0, 0.0],
        },
    }
    metadata = {
        **overlay,
        "viewer_overlay": dict(overlay),
    }
    frame = FrameCache(
        seq=7,
        frame_header=FrameHeader(
            frame_id=12,
            timestamp_ns=123456789,
            source="perception_runtime",
            width=320,
            height=180,
            rgb_encoding="rgb8",
            depth_encoding="",
            camera_pose_xyz=(0.0, 0.0, 0.0),
            camera_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            robot_pose_xyz=(1.0, 2.0, 3.0),
            robot_yaw_rad=0.25,
            sim_time_s=4.5,
            metadata=metadata,
        ),
        rgb_image=np.zeros((180, 320, 3), dtype=np.uint8),
        depth_image_m=None,
        viewer_overlay=metadata["viewer_overlay"],
        last_frame_monotonic=time.monotonic(),
    )

    payload = build_frame_meta_message(frame)

    assert payload["trajectory_pixels"] == [[10, 20], [30, 40]]
    assert payload["trajectoryPixels"] == [[10, 20], [30, 40]]
    assert payload["system2_pixel_goal"] == [111, 222]
    assert payload["system2PixelGoal"] == [111, 222]
    assert payload["active_target"]["nav_goal_pixel"] == [111, 222]
    assert payload["activeTarget"]["world_pose_xyz"] == [1.0, 2.0, 0.0]


def test_webrtc_subscriber_keeps_last_good_payload_and_marks_stream_stalled() -> None:
    subscriber = ObservationSubscriber(
        WebRTCServiceConfig(stale_frame_timeout_sec=0.05),
        bus=_NoopBus(),
        shm_ring=None,
    )
    now = time.monotonic()
    subscriber._frame = _make_frame(last_frame_monotonic=now)

    fresh_state = subscriber.build_state_snapshot()
    fresh_meta = subscriber.build_frame_meta()

    assert fresh_state["type"] == "frame_state"
    assert fresh_meta is not None
    assert fresh_state["streamStalled"] is False
    assert fresh_meta["streamStalled"] is False

    subscriber._frame = _make_frame(last_frame_monotonic=time.monotonic() - 0.10)

    stale_state = subscriber.build_state_snapshot()
    stale_meta = subscriber.build_frame_meta()

    assert stale_state["type"] == "frame_state"
    assert stale_state["frame_id"] == fresh_state["frame_id"]
    assert stale_state["streamStalled"] is True
    assert stale_state["lastGoodFrameAgeMs"] is not None
    assert stale_meta is not None
    assert stale_meta["frame_id"] == fresh_meta["frame_id"]
    assert stale_meta["streamStalled"] is True
    assert subscriber.debug_counters["staleTransitions"] >= 1


def test_webrtc_subscriber_ignores_transient_shm_overwrite_errors() -> None:
    async def scenario() -> None:
        header = FrameHeader(
            frame_id=1,
            timestamp_ns=123,
            source="runtime",
            width=2,
            height=2,
            rgb_encoding="rgb8",
            depth_encoding="",
            camera_pose_xyz=(0.0, 0.0, 0.0),
            camera_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            robot_pose_xyz=(0.0, 0.0, 0.0),
            robot_yaw_rad=0.0,
            sim_time_s=0.0,
            metadata={
                "rgb_ref": {
                    "name": "aura_viewer_shm_01",
                    "slot_index": 0,
                    "payload_size": 16,
                    "sequence": 1,
                }
            },
        )
        subscriber = ObservationSubscriber(
            WebRTCServiceConfig(poll_interval_ms=1),
            bus=_SingleRecordBus(SimpleNamespace(message=header)),
            shm_ring=_OverwriteShm(),
        )
        await subscriber.start()
        await asyncio.sleep(0.03)
        assert subscriber._task is not None
        assert subscriber._task.done() is False
        assert subscriber.current_frame is None
        assert subscriber.debug_counters["decodeDrops"] == 1
        assert subscriber.debug_counters["shmOverwriteDrops"] == 1
        await subscriber.close()

    asyncio.run(scenario())
