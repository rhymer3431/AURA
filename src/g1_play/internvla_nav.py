"""System 2 HTTP client and pixel-goal projection helpers."""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
import time
from typing import Any

import numpy as np


def _require_dependencies():
    missing = []
    try:
        import requests
    except Exception:  # pragma: no cover - runtime dependency validation
        missing.append("requests")
        requests = None
    try:
        from PIL import Image
    except Exception:  # pragma: no cover - runtime dependency validation
        missing.append("Pillow")
        Image = None
    if missing:
        raise RuntimeError(
            "InternVLA navigation integration requires "
            + ", ".join(missing)
            + " in the Isaac Sim Python environment."
        )
    return requests, Image


def _image_bytes(rgb: np.ndarray) -> bytes:
    _, Image = _require_dependencies()
    rgb = np.asarray(rgb)
    if rgb.ndim != 3:
        raise ValueError(f"Expected an HxWxC RGB array, got shape {rgb.shape}.")
    if rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    if np.issubdtype(rgb.dtype, np.floating):
        if rgb.max(initial=0.0) <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    else:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _depth_bytes(depth: np.ndarray) -> bytes:
    _, Image = _require_dependencies()
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected an HxW depth array, got shape {depth.shape}.")
    depth_1e4 = np.rint(np.clip(depth, 0.0, 6.5535) * 10000.0).astype(np.uint16)
    buffer = io.BytesIO()
    Image.fromarray(depth_1e4).save(buffer, format="PNG")
    return buffer.getvalue()


@dataclass(slots=True)
class System2Result:
    """Normalized System 2 result returned by the HTTP grounding server."""

    status: str
    uv_norm: np.ndarray | None
    text: str
    latency_ms: float
    stamp_s: float
    pixel_xy: np.ndarray | None = None
    decision_mode: str | None = None
    action_sequence: tuple[str, ...] | None = None
    needs_requery: bool = False
    raw_payload: dict[str, Any] | None = None


def _normalize_eval_dual_response(
    payload: dict[str, Any],
    *,
    image_width: int,
    image_height: int,
    stamp_s: float,
    latency_ms: float,
) -> System2Result:
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object from InternVLA server, got {type(payload)!r}.")

    pixel_goal = payload.get("pixel_goal")
    if pixel_goal is not None:
        pixel = np.asarray(pixel_goal, dtype=np.float32).reshape(2)
        pixel_xy = np.asarray(
            (
                float(np.clip(pixel[0], 0.0, max(image_width - 1, 0))),
                float(np.clip(pixel[1], 0.0, max(image_height - 1, 0))),
            ),
            dtype=np.float32,
        )
        uv_norm = pixel_xy_to_normalized_uv(
            (int(round(float(pixel_xy[0]))), int(round(float(pixel_xy[1])))),
            image_width=image_width,
            image_height=image_height,
        )
        return System2Result(
            status="goal",
            uv_norm=uv_norm,
            text=f"pixel_goal:{int(round(float(pixel_xy[0])))},{int(round(float(pixel_xy[1])))}",
            latency_ms=float(latency_ms),
            stamp_s=float(stamp_s),
            pixel_xy=pixel_xy,
            decision_mode="pixel_goal",
            needs_requery=False,
            raw_payload=dict(payload),
        )

    action_seq = payload.get("discrete_action")
    mapping = {
        0: ("stop", "stop"),
        1: ("hold", "forward"),
        2: ("hold", "yaw_left"),
        3: ("hold", "yaw_right"),
        5: ("hold", "wait"),
    }
    normalized_actions: list[str] = []
    action = None
    if isinstance(action_seq, list):
        for raw_action in action_seq:
            try:
                action_id = int(raw_action)
            except Exception:
                continue
            if action_id in mapping:
                if action is None:
                    action = action_id
                normalized_actions.append(mapping[action_id][1])
    if action in mapping:
        status, decision_mode = mapping[action]
        return System2Result(
            status=status,
            uv_norm=None,
            text=f"discrete_action:{action}",
            latency_ms=float(latency_ms),
            stamp_s=float(stamp_s),
            pixel_xy=None,
            decision_mode=decision_mode,
            action_sequence=tuple(normalized_actions),
            needs_requery=False,
            raw_payload=dict(payload),
        )

    return System2Result(
        status="hold",
        uv_norm=None,
        text=str(payload.get("message", "")).strip() or "wait",
        latency_ms=float(latency_ms),
        stamp_s=float(stamp_s),
        pixel_xy=None,
        decision_mode="wait",
        needs_requery=False,
        raw_payload=dict(payload),
    )


class InternVlaNavClient:
    """HTTP client for the official InternNav /navigation server."""

    def __init__(self, server_url: str, timeout_s: float):
        self.server_url = str(server_url).rstrip("/")
        self.timeout_s = max(0.1, float(timeout_s))
        self._instruction = ""
        self._language = "auto"
        self._needs_reset = True
        self._request_idx = 0
        self._check_session_id = ""

    @staticmethod
    def validate_runtime_dependencies():
        _require_dependencies()

    def _post(self, path: str, **kwargs: Any) -> dict[str, Any]:
        requests, _ = _require_dependencies()
        response = requests.post(f"{self.server_url}{path}", timeout=self.timeout_s, **kwargs)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = ""
            try:
                detail = response.text.strip()
            except Exception:
                detail = ""
            if detail:
                raise RuntimeError(
                    f"InternVLA server returned HTTP {response.status_code}: {detail}"
                ) from exc
            raise
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"Expected a JSON object from InternVLA server, got {type(payload)!r}.")
        return payload

    def reset_session(
        self,
        *,
        session_id: str,
        instruction: str,
        language: str,
        image_width: int,
        image_height: int,
    ) -> dict[str, Any]:
        normalized_instruction = " ".join(str(instruction).strip().split())
        if not normalized_instruction:
            raise ValueError("instruction is required.")
        self._instruction = normalized_instruction
        self._language = str(language).strip() or "auto"
        self._needs_reset = True
        self._request_idx = 0
        return {
            "status": "ok",
            "session_id": str(session_id),
            "reset_queued": True,
            "instruction": self._instruction,
            "language": self._language,
            "image_width": int(image_width),
            "image_height": int(image_height),
        }

    def step_session(
        self,
        *,
        session_id: str,
        rgb: np.ndarray,
        depth: np.ndarray,
        stamp_s: float,
        force_infer: bool = False,
    ) -> System2Result:
        del session_id
        del force_infer
        if not self._instruction:
            raise RuntimeError("reset_session() must be called before step_session().")

        reset_flag = bool(self._needs_reset)
        request_idx = int(self._request_idx)
        start = time.monotonic()
        payload = self._post(
            "/navigation",
            files={
                "image": ("rgb.jpg", _image_bytes(rgb), "image/jpeg"),
                "depth": ("depth.png", _depth_bytes(depth), "image/png"),
            },
            data={
                "json": json.dumps(
                    {
                        "reset": reset_flag,
                        "idx": request_idx,
                        "instruction": self._instruction,
                        "language": self._language,
                    }
                )
            },
        )
        self._needs_reset = False
        self._request_idx += 1
        return _normalize_eval_dual_response(
            payload,
            image_width=int(np.asarray(rgb).shape[1]),
            image_height=int(np.asarray(rgb).shape[0]),
            stamp_s=float(stamp_s),
            latency_ms=(time.monotonic() - start) * 1000.0,
        )

    @property
    def check_session_id(self) -> str:
        return self._check_session_id

    def open_check_session(
        self,
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        normalized_session_id = str(session_id or "").strip()
        if normalized_session_id:
            payload["session_id"] = normalized_session_id
        response = self._post("/check/session/open", json=payload)
        resolved_session_id = str(response.get("session_id", "")).strip()
        if resolved_session_id:
            self._check_session_id = resolved_session_id
        return response

    def check_message(
        self,
        message: Any,
        *,
        session_id: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"message": message}
        normalized_session_id = str(session_id or "").strip() or self._check_session_id
        if normalized_session_id:
            payload["session_id"] = normalized_session_id
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        response = self._post("/check/session/message", json=payload)
        resolved_session_id = str(response.get("session_id", "")).strip()
        if resolved_session_id:
            self._check_session_id = resolved_session_id
        return response

    def check_answer(
        self,
        message: Any,
        *,
        session_id: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        response = self.check_message(
            message,
            session_id=session_id,
            max_tokens=max_tokens,
        )
        answer = response.get("answer", "")
        return str(answer).strip()

    def close_check_session(self, *, session_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        normalized_session_id = str(session_id or "").strip() or self._check_session_id
        if normalized_session_id:
            payload["session_id"] = normalized_session_id
        response = self._post("/check/session/close", json=payload)
        if bool(response.get("reopened_default_session")):
            self._check_session_id = str(response.get("default_check_session_id", "")).strip()
        elif not normalized_session_id or normalized_session_id == self._check_session_id:
            self._check_session_id = ""
        return response


def normalized_uv_to_pixel_xy(uv_norm: np.ndarray, *, image_width: int, image_height: int) -> tuple[int, int]:
    """Map normalized UV coordinates onto integer pixel indices."""

    uv = np.asarray(uv_norm, dtype=np.float32).reshape(2)
    width = max(1, int(image_width))
    height = max(1, int(image_height))
    x = int(round(float(np.clip(uv[0], 0.0, 1.0)) * max(width - 1, 0)))
    y = int(round(float(np.clip(uv[1], 0.0, 1.0)) * max(height - 1, 0)))
    return x, y


def pixel_xy_to_normalized_uv(pixel_xy: tuple[int, int], *, image_width: int, image_height: int) -> np.ndarray:
    """Convert integer pixel indices into normalized UV coordinates."""

    width = max(1, int(image_width))
    height = max(1, int(image_height))
    x = int(np.clip(pixel_xy[0], 0, width - 1))
    y = int(np.clip(pixel_xy[1], 0, height - 1))
    return np.asarray(
        (
            float(x / max(width - 1, 1)),
            float(y / max(height - 1, 1)),
        ),
        dtype=np.float32,
    )


def sample_depth_window(
    depth_image: np.ndarray,
    *,
    pixel_xy: tuple[int, int],
    window_size: int,
    depth_min_m: float,
    depth_max_m: float,
) -> float | None:
    """Return the median valid depth around a target pixel."""

    depth = np.asarray(depth_image, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected an HxW depth image, got shape {depth.shape}.")

    width = depth.shape[1]
    height = depth.shape[0]
    if width == 0 or height == 0:
        return None

    x = int(np.clip(pixel_xy[0], 0, width - 1))
    y = int(np.clip(pixel_xy[1], 0, height - 1))
    radius = max(0, int(window_size) // 2)
    x0 = max(0, x - radius)
    x1 = min(width, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(height, y + radius + 1)

    window = depth[y0:y1, x0:x1]
    valid = window[np.isfinite(window)]
    valid = valid[(valid >= float(depth_min_m)) & (valid <= float(depth_max_m))]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def deproject_pixel_to_camera_point(pixel_xy: tuple[int, int], *, depth_m: float, intrinsic: np.ndarray) -> np.ndarray:
    """Back-project a pixel into the Isaac world-camera frame.

    The runtime camera uses `camera_axes="world"`, so the local axes are interpreted as:
    +X forward, +Y left, +Z up.
    """

    matrix = np.asarray(intrinsic, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] < 3:
        raise ValueError(f"Expected a 3x3 or 4x4 intrinsic matrix, got shape {matrix.shape}.")

    u = float(pixel_xy[0])
    v = float(pixel_xy[1])
    fx = float(matrix[0, 0])
    fy = float(matrix[1, 1])
    cx = float(matrix[0, 2])
    cy = float(matrix[1, 2])
    if fx == 0.0 or fy == 0.0:
        raise ValueError("Camera intrinsics fx/fy must be non-zero.")

    depth = float(depth_m)
    x_forward = depth
    y_left = -((u - cx) / fx) * depth
    z_up = -((v - cy) / fy) * depth
    return np.asarray((x_forward, y_left, z_up), dtype=np.float32)


def camera_point_to_world_xy(
    camera_point_xyz: np.ndarray,
    *,
    camera_pos_w: np.ndarray,
    camera_rot_w: np.ndarray,
) -> np.ndarray:
    """Transform a local camera-frame point into world XY."""

    point_camera = np.asarray(camera_point_xyz, dtype=np.float32).reshape(3)
    world = np.asarray(camera_pos_w, dtype=np.float32).reshape(3) + (
        np.asarray(camera_rot_w, dtype=np.float32).reshape(3, 3) @ point_camera
    )
    return np.asarray(world[:2], dtype=np.float32)


def resolve_goal_world_xy(
    *,
    uv_norm: np.ndarray,
    depth_image: np.ndarray,
    intrinsic: np.ndarray,
    camera_pos_w: np.ndarray,
    camera_rot_w: np.ndarray,
    window_size: int,
    depth_min_m: float,
    depth_max_m: float,
) -> tuple[np.ndarray, tuple[int, int], float] | None:
    """Resolve a normalized System 2 pixel goal into a world-frame XY point."""

    pixel_xy = normalized_uv_to_pixel_xy(
        uv_norm,
        image_width=np.asarray(depth_image).shape[1],
        image_height=np.asarray(depth_image).shape[0],
    )
    depth_m = sample_depth_window(
        depth_image,
        pixel_xy=pixel_xy,
        window_size=window_size,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
    )
    if depth_m is None:
        return None
    camera_point = deproject_pixel_to_camera_point(pixel_xy, depth_m=depth_m, intrinsic=intrinsic)
    world_xy = camera_point_to_world_xy(camera_point, camera_pos_w=camera_pos_w, camera_rot_w=camera_rot_w)
    return world_xy, pixel_xy, depth_m


def resolve_goal_world_xy_from_pixel(
    pixel_xy: tuple[int, int],
    *,
    depth_image: np.ndarray,
    intrinsic: np.ndarray,
    camera_pos_w: np.ndarray,
    camera_rot_w: np.ndarray,
    window_size: int,
    depth_min_m: float,
    depth_max_m: float,
) -> tuple[np.ndarray, tuple[int, int], float] | None:
    """Resolve a world goal from integer pixel coordinates."""

    depth = np.asarray(depth_image)
    if depth.ndim != 2:
        raise ValueError(f"Expected an HxW depth image, got shape {depth.shape}.")
    uv_norm = pixel_xy_to_normalized_uv(
        pixel_xy,
        image_width=depth.shape[1],
        image_height=depth.shape[0],
    )
    return resolve_goal_world_xy(
        uv_norm=uv_norm,
        depth_image=depth_image,
        intrinsic=intrinsic,
        camera_pos_w=camera_pos_w,
        camera_rot_w=camera_rot_w,
        window_size=window_size,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
    )
