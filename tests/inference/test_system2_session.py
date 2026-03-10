from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference.vlm import System2Session, System2SessionConfig, parse_system2_output


def test_parse_system2_output_handles_pixel_goal() -> None:
    decision = parse_system2_output("321, 123", width=640, height=480)

    assert decision.mode == "pixel_goal"
    assert decision.pixel_goal == (123, 321)
    assert decision.needs_requery is False


def test_parse_system2_output_handles_stop() -> None:
    decision = parse_system2_output("STOP", width=640, height=480)

    assert decision.mode == "stop"
    assert decision.pixel_goal is None
    assert decision.needs_requery is False


def test_parse_system2_output_handles_left_turn() -> None:
    decision = parse_system2_output("←", width=640, height=480)

    assert decision.mode == "yaw_left"
    assert decision.needs_requery is False


def test_parse_system2_output_handles_right_turn() -> None:
    decision = parse_system2_output("→", width=640, height=480)

    assert decision.mode == "yaw_right"
    assert decision.needs_requery is False


def test_parse_system2_output_handles_wait() -> None:
    decision = parse_system2_output("↓", width=640, height=480)

    assert decision.mode == "wait"
    assert decision.needs_requery is True


def test_parse_system2_output_prefers_coordinates_from_mixed_text() -> None:
    decision = parse_system2_output("Waypoint: 200, 150 near the ramp", width=640, height=480)

    assert decision.mode == "pixel_goal"
    assert decision.pixel_goal == (150, 200)


def test_system2_session_resets_history_and_samples_frames() -> None:
    session = System2Session(
        System2SessionConfig(
            endpoint="http://127.0.0.1:8080/v1/chat/completions",
            model="mock-model",
            mode="mock",
            num_history=8,
            max_images_per_request=9,
        )
    )
    session.reset("find the loading dock")

    for frame_id in range(10):
        session.observe(frame_id, np.full((8, 8, 3), frame_id, dtype=np.uint8))

    request = session.prepare_request(events={"force_s2": True})

    assert request.frame_id == 9
    assert request.history_frame_ids == (0, 1, 2, 3, 4, 5, 6, 8)

    session.reset("inspect the pallet")
    session.observe(50, np.zeros((8, 8, 3), dtype=np.uint8))
    request_after_reset = session.prepare_request()

    assert request_after_reset.frame_id == 50
    assert request_after_reset.history_frame_ids == ()


def test_system2_session_records_raw_text_and_history_ids() -> None:
    session = System2Session(
        System2SessionConfig(
            endpoint="http://127.0.0.1:8080/v1/chat/completions",
            model="mock-model",
            mode="mock",
            num_history=8,
            max_images_per_request=9,
        )
    )
    session.reset("move to the center aisle")
    for frame_id in range(4):
        session.observe(frame_id, np.full((8, 8, 3), frame_id, dtype=np.uint8))

    request = session.prepare_request()
    result = session.execute_request(request)
    assert result.ok is True
    session.record_result(result)
    debug_state = session.debug_state()

    assert debug_state["last_output"] != ""
    assert debug_state["last_reason"] == "mock_forward"
    assert debug_state["last_history_frame_ids"] == [0, 1, 2]
    assert debug_state["last_decision_mode"] == "pixel_goal"


def test_system2_session_caps_total_images_per_request() -> None:
    session = System2Session(
        System2SessionConfig(
            endpoint="http://127.0.0.1:8080/v1/chat/completions",
            model="mock-model",
            mode="mock",
            num_history=8,
            max_images_per_request=3,
        )
    )
    session.reset("find the dock")
    for frame_id in range(10):
        session.observe(frame_id, np.full((8, 8, 3), frame_id, dtype=np.uint8))

    request = session.prepare_request()

    assert request.frame_id == 9
    assert request.history_frame_ids == (0, 8)
