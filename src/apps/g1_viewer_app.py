from __future__ import annotations

import argparse
import time

import numpy as np

from adapters.sensors.isaac_bridge_adapter import IsaacBridgeAdapter
from common.cv2_compat import cv2
from ipc.shm_ring import SharedMemoryRing
from ipc.zmq_bus import ZmqBus
from runtime.g1_bridge_args import (
    DEFAULT_VIEWER_CONTROL_ENDPOINT,
    DEFAULT_VIEWER_SHM_CAPACITY,
    DEFAULT_VIEWER_SHM_NAME,
    DEFAULT_VIEWER_SHM_SLOT_SIZE,
    DEFAULT_VIEWER_TELEMETRY_ENDPOINT,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenCV viewer for G1 shared-memory camera frames with YOLO overlay.")
    parser.add_argument("--control-endpoint", type=str, default=DEFAULT_VIEWER_CONTROL_ENDPOINT)
    parser.add_argument("--telemetry-endpoint", type=str, default=DEFAULT_VIEWER_TELEMETRY_ENDPOINT)
    parser.add_argument("--shm-name", type=str, default=DEFAULT_VIEWER_SHM_NAME)
    parser.add_argument("--shm-slot-size", type=int, default=DEFAULT_VIEWER_SHM_SLOT_SIZE)
    parser.add_argument("--shm-capacity", type=int, default=DEFAULT_VIEWER_SHM_CAPACITY)
    parser.add_argument("--poll-interval-ms", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--agent-id", type=str, default="g1_viewer")
    return parser


def _require_gui_support() -> None:
    missing = [
        name
        for name in ("imshow", "waitKey", "destroyAllWindows", "rectangle", "putText")
        if not hasattr(cv2, name)
    ]
    if missing:
        raise RuntimeError(f"OpenCV GUI support is required for viewer mode, missing: {', '.join(missing)}")


def _attach_shm(args: argparse.Namespace) -> SharedMemoryRing | None:
    try:
        return SharedMemoryRing(
            name=str(args.shm_name),
            slot_size=int(args.shm_slot_size),
            capacity=int(args.shm_capacity),
            create=False,
        )
    except FileNotFoundError:
        return None


def _to_bgr(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"rgb_image must be [H,W,3+] for viewing, got {array.shape}")
    if array.shape[-1] == 4:
        rgba_to_bgr = getattr(cv2, "COLOR_RGBA2BGR", None)
        if rgba_to_bgr is not None:
            return cv2.cvtColor(array, rgba_to_bgr)
        array = array[..., :3]
    return cv2.cvtColor(array[..., :3], cv2.COLOR_RGB2BGR)


def _color_for_track(track_id: str) -> tuple[int, int, int]:
    seed = sum(ord(char) for char in str(track_id))
    return (
        int(64 + (seed * 53) % 160),
        int(64 + (seed * 97) % 160),
        int(64 + (seed * 193) % 160),
    )


def _draw_overlay(frame_bgr: np.ndarray, overlay: dict[str, object], *, frame_id: int, source: str) -> np.ndarray:
    canvas = np.asarray(frame_bgr, dtype=np.uint8).copy()
    detections = overlay.get("detections", [])
    if not isinstance(detections, list):
        detections = []
    font = getattr(cv2, "FONT_HERSHEY_SIMPLEX", 0)
    line_type = getattr(cv2, "LINE_AA", 16)

    for item in detections:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = (int(value) for value in bbox)
        track_id = str(item.get("track_id", ""))
        color = _color_for_track(track_id)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        label = f"{str(item.get('class_name', 'object'))} {float(item.get('confidence', 0.0)):.2f}"
        if track_id != "":
            label = f"{label} {track_id}"
        depth_m = item.get("depth_m")
        if isinstance(depth_m, (int, float)):
            label = f"{label} {float(depth_m):.2f}m"
        label_y = y0 - 8 if y0 > 20 else y0 + 18
        cv2.putText(canvas, label, (x0, label_y), font, 0.5, color, 1, line_type)

    hud = (
        f"frame={int(frame_id)} "
        f"source={source} "
        f"backend={str(overlay.get('detector_backend', ''))} "
        f"detections={len(detections)}"
    )
    cv2.putText(canvas, hud, (10, 22), font, 0.6, (0, 255, 255), 2, line_type)
    return canvas


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.no_gui:
        _require_gui_support()

    bus = ZmqBus(
        control_endpoint=str(args.control_endpoint),
        telemetry_endpoint=str(args.telemetry_endpoint),
        role="agent",
        identity=str(args.agent_id),
    )
    shm_ring: SharedMemoryRing | None = None
    frames_shown = 0

    try:
        while True:
            records = bus.poll("isaac.observation", max_items=32)
            if not records:
                time.sleep(max(float(args.poll_interval_ms), 1.0) / 1000.0)
                continue

            frame_header = records[-1].message
            if shm_ring is None:
                shm_ring = _attach_shm(args)
                if shm_ring is None:
                    time.sleep(max(float(args.poll_interval_ms), 1.0) / 1000.0)
                    continue

            try:
                batch = IsaacBridgeAdapter(bus, shm_ring=shm_ring).reconstruct_batch(frame_header)
            except FileNotFoundError:
                if shm_ring is not None:
                    shm_ring.close()
                    shm_ring = None
                time.sleep(max(float(args.poll_interval_ms), 1.0) / 1000.0)
                continue
            except RuntimeError as exc:
                print(f"[G1_VIEWER] dropped frame_id={frame_header.frame_id}: {exc}")
                continue

            if batch.rgb_image is None:
                continue

            frames_shown += 1
            overlay = frame_header.metadata.get("viewer_overlay", {})
            if not isinstance(overlay, dict):
                overlay = {}

            if args.no_gui:
                detections = overlay.get("detections", [])
                detection_count = len(detections) if isinstance(detections, list) else 0
                print(
                    "[G1_VIEWER] "
                    f"frame_id={frame_header.frame_id} shape={batch.rgb_image.shape} "
                    f"source={frame_header.source} detections={detection_count}"
                )
            else:
                canvas = _draw_overlay(
                    _to_bgr(batch.rgb_image),
                    overlay,
                    frame_id=int(frame_header.frame_id),
                    source=str(frame_header.source),
                )
                cv2.imshow("G1 View", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            if int(args.max_frames) > 0 and frames_shown >= int(args.max_frames):
                break
    finally:
        if shm_ring is not None:
            shm_ring.close()
        bus.close()
        if not args.no_gui and hasattr(cv2, "destroyAllWindows"):
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
