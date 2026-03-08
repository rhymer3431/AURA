from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from adapters.sensors.isaac_live_source import IsaacLiveFrameSource, IsaacLiveSourceConfig
from common.cv2_compat import cv2
from common.depth_visualization import build_rgb_depth_panel, sanitize_depth_image


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="In-editor Isaac Sim depth viewer with OpenCV visualization.")
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=640)
    parser.add_argument("--depth-max-m", type=float, default=5.0)
    parser.add_argument("--poll-interval-ms", type=int, default=30)
    parser.add_argument("--wait-key-ms", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--window-name", type=str, default="Isaac Depth Viewer")
    parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--strict-live", action="store_true")
    parser.add_argument("--force-runtime-camera", action="store_true")
    return parser


def _format_gui_error(exc: Exception) -> str:
    return (
        "OpenCV GUI support is unavailable in the Isaac Python environment. "
        "Use a desktop OpenCV build such as `opencv-python`, not `opencv-python-headless`, "
        "or run with `--no-gui --save-dir <dir>`. "
        f"detail={type(exc).__name__}: {exc}"
    )


def _require_gui_support() -> None:
    missing = [name for name in ("imshow", "waitKey", "namedWindow", "destroyWindow") if not hasattr(cv2, name)]
    if missing:
        raise RuntimeError(f"OpenCV GUI support is required for viewer mode, missing: {', '.join(missing)}")
    probe_name = "__isaac_depth_view_probe__"
    probe_image = np.zeros((1, 1, 3), dtype=np.uint8)
    try:
        cv2.namedWindow(probe_name)
        cv2.imshow(probe_name, probe_image)
        cv2.waitKey(1)
        cv2.destroyWindow(probe_name)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(_format_gui_error(exc)) from exc


def _close_windows_safely() -> None:
    destroy_all = getattr(cv2, "destroyAllWindows", None)
    if not callable(destroy_all):
        return
    try:
        destroy_all()
    except Exception as exc:  # noqa: BLE001
        print(f"[Depth Viewer] skipped OpenCV window teardown: {_format_gui_error(exc)}")


def _resolve_stage(stage=None):  # noqa: ANN001
    if stage is not None:
        return stage
    import omni.usd

    resolved_stage = omni.usd.get_context().get_stage()
    if resolved_stage is None:
        raise RuntimeError("No active USD stage found. Start Isaac Sim first, then rerun the depth viewer.")
    return resolved_stage


def _resolve_simulation_app(simulation_app=None):  # noqa: ANN001
    if simulation_app is not None:
        return simulation_app
    try:
        import omni.kit.app
    except Exception:  # noqa: BLE001
        return SimpleNamespace(update=lambda: None)
    app = omni.kit.app.get_app()
    if app is None:
        return SimpleNamespace(update=lambda: None)
    return app


def _call_update(simulation_app) -> None:  # noqa: ANN001
    update = getattr(simulation_app, "update", None)
    if callable(update):
        update()


def _depth_stats(depth_m: np.ndarray) -> dict[str, float]:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        return {"valid_pixels": 0.0, "min_m": 0.0, "max_m": 0.0, "mean_m": 0.0}
    valid_depth = depth[valid]
    return {
        "valid_pixels": float(valid_depth.size),
        "min_m": float(np.min(valid_depth)),
        "max_m": float(np.max(valid_depth)),
        "mean_m": float(np.mean(valid_depth)),
    }


def _annotate_panel(panel_bgr: np.ndarray, *, sample, depth_max_m: float) -> np.ndarray:  # noqa: ANN001
    if not hasattr(cv2, "putText"):
        return np.asarray(panel_bgr, dtype=np.uint8)
    canvas = np.asarray(panel_bgr, dtype=np.uint8).copy()
    stats = _depth_stats(np.asarray(sample.depth, dtype=np.float32))
    capture_report = sample.metadata.get("capture_report", {})
    font = getattr(cv2, "FONT_HERSHEY_SIMPLEX", 0)
    line_type = getattr(cv2, "LINE_AA", 16)
    lines = [
        f"frame={int(sample.frame_id)} source={sample.source_name}",
        (
            f"depth_source={capture_report.get('depth_source', 'unknown')} "
            f"min={stats['min_m']:.3f}m max={stats['max_m']:.3f}m mean={stats['mean_m']:.3f}m"
        ),
        (
            f"camera={capture_report.get('camera_prim_path', '') or 'n/a'} "
            f"depth_prim={capture_report.get('depth_camera_prim_path', '') or 'n/a'}"
        ),
        f"display_range=0.0..{float(depth_max_m):.2f}m valid_pixels={int(stats['valid_pixels'])}",
    ]
    for index, text in enumerate(lines):
        y = 24 + index * 22
        cv2.putText(canvas, text, (12, y), font, 0.55, (0, 255, 255), 2, line_type)
    return canvas


def _write_panel_png(path: Path, panel_bgr: np.ndarray) -> None:
    ok, buffer = cv2.imencode(".png", np.asarray(panel_bgr, dtype=np.uint8))
    if not ok:
        raise RuntimeError(f"Failed to encode PNG for {path}")
    path.write_bytes(bytes(np.asarray(buffer, dtype=np.uint8).tolist()))


def run_depth_viewer(
    argv: list[str] | None = None,
    *,
    simulation_app=None,
    stage=None,
) -> int:
    args = build_arg_parser().parse_args(argv or [])
    if not args.no_gui:
        _require_gui_support()

    resolved_stage = _resolve_stage(stage)
    resolved_app = _resolve_simulation_app(simulation_app)
    save_dir = Path(args.save_dir).expanduser() if str(args.save_dir).strip() != "" else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    frame_source = IsaacLiveFrameSource(
        simulation_app=resolved_app,
        stage=resolved_stage,
        robot_pose_provider=lambda: (0.0, 0.0, 0.0),
        robot_yaw_provider=lambda: 0.0,
        config=IsaacLiveSourceConfig(
            source_name="isaac_depth_viewer",
            strict_live=bool(args.strict_live),
            image_width=int(args.image_width),
            image_height=int(args.image_height),
            depth_max_m=float(args.depth_max_m),
            force_runtime_mount=bool(args.force_runtime_camera),
        ),
    )
    report = frame_source.start()
    if report.status != "ready":
        raise RuntimeError(report.notice or "Isaac live frame source did not become ready.")

    frames_processed = 0
    try:
        while True:
            _call_update(resolved_app)
            sample = frame_source.read()
            if sample is None:
                time.sleep(max(float(args.poll_interval_ms), 1.0) / 1000.0)
                continue

            sanitized_depth, _ = sanitize_depth_image(np.asarray(sample.depth, dtype=np.float32), float(args.depth_max_m))
            panel = build_rgb_depth_panel(
                np.asarray(sample.rgb, dtype=np.uint8),
                sanitized_depth,
                float(args.depth_max_m),
            )
            panel = _annotate_panel(panel, sample=sample, depth_max_m=float(args.depth_max_m))
            if save_dir is not None:
                _write_panel_png(save_dir / f"depth_view_{int(sample.frame_id):06d}.png", panel)

            frames_processed += 1
            if args.no_gui:
                stats = _depth_stats(sanitized_depth)
                capture_report = sample.metadata.get("capture_report", {})
                print(
                    "[Depth Viewer] "
                    f"frame_id={sample.frame_id} "
                    f"depth_source={capture_report.get('depth_source', 'unknown')} "
                    f"valid_pixels={int(stats['valid_pixels'])} "
                    f"min_m={stats['min_m']:.3f} "
                    f"max_m={stats['max_m']:.3f} "
                    f"mean_m={stats['mean_m']:.3f}"
                )
            else:
                try:
                    cv2.imshow(str(args.window_name), panel)
                    key = cv2.waitKey(max(int(args.wait_key_ms), 1)) & 0xFF
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(_format_gui_error(exc)) from exc
                if key in (27, ord("q")):
                    break

            if int(args.max_frames) > 0 and frames_processed >= int(args.max_frames):
                break
    finally:
        frame_source.close()
        if not args.no_gui:
            _close_windows_safely()

    return 0


def main(argv: list[str] | None = None) -> int:
    return run_depth_viewer(argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
