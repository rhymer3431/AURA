from __future__ import annotations

from dataclasses import dataclass
import io
import json
import time
import uuid
from typing import Any

import numpy as np
import requests
from PIL import Image

from common.geometry import project_camera_point_to_world


DecisionMode = str


def build_vlm_endpoint(url: str) -> str:
    raw = str(url).strip()
    if raw == "":
        return "http://127.0.0.1:15801/navigation"
    if raw.endswith("/navigation"):
        return raw
    return raw.rstrip("/") + "/navigation"


def extract_chat_content(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) == 0:
        raise ValueError("VLM response missing choices.")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                maybe_text = item.get("text")
                if isinstance(maybe_text, str):
                    parts.append(maybe_text)
        return "\n".join(parts)
    raise ValueError("Unsupported VLM content format.")


def _image_bytes(rgb: np.ndarray) -> bytes:
    image = np.asarray(rgb)
    if image.ndim != 3:
        raise ValueError(f"Expected an HxWxC RGB array, got shape {image.shape}.")
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if np.issubdtype(image.dtype, np.floating):
        if float(image.max(initial=0.0)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    else:
        image = np.clip(image, 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _depth_bytes(depth: np.ndarray) -> bytes:
    depth_image = np.asarray(depth, dtype=np.float32)
    if depth_image.ndim != 2:
        raise ValueError(f"Expected an HxW depth array, got shape {depth_image.shape}.")
    depth_u16 = np.rint(np.clip(depth_image, 0.0, 6.5535) * 10000.0).astype(np.uint16)
    buffer = io.BytesIO()
    Image.fromarray(depth_u16).save(buffer, format="PNG")
    return buffer.getvalue()


@dataclass(frozen=True)
class System2Decision:
    mode: DecisionMode
    pixel_goal: tuple[int, int] | None = None
    reason: str = ""
    raw_text: str = ""
    history_frame_ids: tuple[int, ...] = ()
    needs_requery: bool = False


@dataclass(frozen=True)
class System2Result:
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


@dataclass(frozen=True)
class System2Request:
    frame_id: int
    width: int
    height: int
    history_frame_ids: tuple[int, ...]
    body: dict[str, Any]


@dataclass(frozen=True)
class System2SessionResult:
    ok: bool
    decision: System2Decision | None = None
    latency_ms: float = 0.0
    error: str = ""
    source: str = "internvla"
    raw_payload: dict[str, Any] | None = None


@dataclass(frozen=True)
class System2SessionConfig:
    endpoint: str
    model: str = "internvla"
    timeout_sec: float = 35.0
    language: str = "auto"


def normalized_uv_to_pixel_xy(uv_norm: np.ndarray, *, image_width: int, image_height: int) -> tuple[int, int]:
    uv = np.asarray(uv_norm, dtype=np.float32).reshape(2)
    width = max(1, int(image_width))
    height = max(1, int(image_height))
    x = int(round(float(np.clip(uv[0], 0.0, 1.0)) * max(width - 1, 0)))
    y = int(round(float(np.clip(uv[1], 0.0, 1.0)) * max(height - 1, 0)))
    return x, y


def pixel_xy_to_normalized_uv(pixel_xy: tuple[int, int], *, image_width: int, image_height: int) -> np.ndarray:
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


def parse_system2_output(
    raw_text: str,
    *,
    width: int,
    height: int,
    history_frame_ids: tuple[int, ...] = (),
) -> System2Decision:
    text = str(raw_text).strip()
    lowered = text.lower()
    numbers = [int(token) for token in __import__("re").findall(r"-?\d+", text)]
    if len(numbers) >= 2:
        pixel_y = int(np.clip(numbers[0], 0, max(int(height) - 1, 0)))
        pixel_x = int(np.clip(numbers[1], 0, max(int(width) - 1, 0)))
        return System2Decision(
            mode="pixel_goal",
            pixel_goal=(pixel_x, pixel_y),
            reason=text or "pixel_goal",
            raw_text=text,
            history_frame_ids=tuple(history_frame_ids),
            needs_requery=False,
        )
    if "stop" in lowered:
        return System2Decision("stop", reason=text or "STOP", raw_text=text, history_frame_ids=tuple(history_frame_ids))
    if "left" in lowered:
        return System2Decision("yaw_left", reason=text or "LEFT", raw_text=text, history_frame_ids=tuple(history_frame_ids))
    if "right" in lowered:
        return System2Decision("yaw_right", reason=text or "RIGHT", raw_text=text, history_frame_ids=tuple(history_frame_ids))
    if "forward" in lowered:
        return System2Decision("forward", reason=text or "FORWARD", raw_text=text, history_frame_ids=tuple(history_frame_ids))
    if "look down" in lowered or "lookdown" in lowered or "tilt down" in lowered:
        return System2Decision("look_down", reason=text or "LOOK_DOWN", raw_text=text, history_frame_ids=tuple(history_frame_ids), needs_requery=True)
    return System2Decision("wait", reason=text or "WAIT", raw_text=text, history_frame_ids=tuple(history_frame_ids), needs_requery=True)


def _normalize_eval_response(
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


def sample_depth_window(
    depth_image: np.ndarray,
    *,
    pixel_xy: tuple[int, int],
    window_size: int,
    depth_min_m: float,
    depth_max_m: float,
) -> float | None:
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


def resolve_goal_world_xy_from_pixel(
    *,
    pixel_xy: tuple[int, int],
    depth_image: np.ndarray,
    intrinsic: np.ndarray,
    camera_pos_world: np.ndarray,
    camera_quat_wxyz: np.ndarray,
    window_size: int = 11,
    depth_min_m: float = 0.1,
    depth_max_m: float = 6.0,
) -> np.ndarray | None:
    depth_m = sample_depth_window(
        depth_image,
        pixel_xy=pixel_xy,
        window_size=window_size,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
    )
    if depth_m is None:
        return None
    point_camera = deproject_pixel_to_camera_point(pixel_xy, depth_m=depth_m, intrinsic=intrinsic)
    point_world = project_camera_point_to_world(
        point_xyz_camera=point_camera,
        camera_pos_world=camera_pos_world,
        camera_quat_wxyz=camera_quat_wxyz,
    )
    return np.asarray(point_world[:2], dtype=np.float32)


class System2Session:
    def __init__(self, config: System2SessionConfig) -> None:
        self.config = config
        self._server_url = build_vlm_endpoint(config.endpoint).rsplit("/navigation", 1)[0]
        self._instruction = ""
        self._language = str(config.language).strip() or "auto"
        self._needs_reset = True
        self._request_idx = 0
        self._session_id = f"system2-{uuid.uuid4().hex}"

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def instruction(self) -> str:
        return self._instruction

    def reset(self, instruction: str, *, language: str | None = None) -> None:
        normalized_instruction = " ".join(str(instruction).strip().split())
        if normalized_instruction == "":
            raise ValueError("instruction is required.")
        self._instruction = normalized_instruction
        if language is not None and str(language).strip():
            self._language = str(language).strip()
        self._needs_reset = True
        self._request_idx = 0
        self._session_id = f"system2-{uuid.uuid4().hex}"

    def reset_session(
        self,
        *,
        session_id: str,
        instruction: str,
        language: str,
        image_width: int,
        image_height: int,
    ) -> dict[str, Any]:
        del image_width, image_height
        self._session_id = str(session_id).strip() or f"system2-{uuid.uuid4().hex}"
        self.reset(instruction, language=language)
        return {
            "status": "ok",
            "session_id": self._session_id,
            "reset_queued": True,
            "instruction": self._instruction,
            "language": self._language,
        }

    def prepare_request(self, *, frame_id: int, width: int, height: int) -> System2Request:
        if self._instruction == "":
            raise RuntimeError("System2Session.reset() must be called before prepare_request().")
        return System2Request(
            frame_id=int(frame_id),
            width=int(width),
            height=int(height),
            history_frame_ids=(),
            body={
                "reset": bool(self._needs_reset),
                "idx": int(self._request_idx),
                "instruction": self._instruction,
                "language": self._language,
                "session_id": self._session_id,
            },
        )

    def step_session(
        self,
        *,
        session_id: str,
        rgb: np.ndarray,
        depth: np.ndarray,
        stamp_s: float,
        force_infer: bool = False,
    ) -> System2Result:
        del force_infer
        self._session_id = str(session_id).strip() or self._session_id
        request = self.prepare_request(frame_id=int(self._request_idx), width=int(np.asarray(rgb).shape[1]), height=int(np.asarray(rgb).shape[0]))
        start = time.monotonic()
        response = requests.post(
            f"{self._server_url}/navigation",
            files={
                "image": ("rgb.jpg", _image_bytes(rgb), "image/jpeg"),
                "depth": ("depth.png", _depth_bytes(depth), "image/png"),
            },
            data={"json": json.dumps(request.body)},
            timeout=float(self.config.timeout_sec),
        )
        response.raise_for_status()
        payload = response.json()
        self._needs_reset = False
        self._request_idx += 1
        return _normalize_eval_response(
            payload,
            image_width=int(np.asarray(rgb).shape[1]),
            image_height=int(np.asarray(rgb).shape[0]),
            stamp_s=float(stamp_s),
            latency_ms=(time.monotonic() - start) * 1000.0,
        )

    def execute_request(self, request: System2Request, *, rgb: np.ndarray, depth: np.ndarray, stamp_s: float) -> System2SessionResult:
        start = time.perf_counter()
        try:
            response = requests.post(
                f"{self._server_url}/navigation",
                files={
                    "image": ("rgb.jpg", _image_bytes(rgb), "image/jpeg"),
                    "depth": ("depth.png", _depth_bytes(depth), "image/png"),
                },
                data={"json": json.dumps(request.body)},
                timeout=float(self.config.timeout_sec),
            )
            response.raise_for_status()
            payload = response.json()
            result = _normalize_eval_response(
                payload,
                image_width=int(request.width),
                image_height=int(request.height),
                stamp_s=float(stamp_s),
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
            self._needs_reset = False
            self._request_idx += 1
            return System2SessionResult(
                ok=True,
                decision=System2Decision(
                    mode=str(result.decision_mode or "wait"),
                    pixel_goal=None if result.pixel_xy is None else (
                        int(round(float(result.pixel_xy[0]))),
                        int(round(float(result.pixel_xy[1]))),
                    ),
                    reason=str(result.text),
                    raw_text=str(result.text),
                    needs_requery=bool(result.needs_requery),
                ),
                latency_ms=float(result.latency_ms),
                source="internvla",
                raw_payload=dict(payload) if isinstance(payload, dict) else None,
            )
        except Exception as exc:  # noqa: BLE001
            return System2SessionResult(
                ok=False,
                latency_ms=(time.perf_counter() - start) * 1000.0,
                error=f"{type(exc).__name__}: {exc}",
                source="internvla",
            )

    def debug_state(self) -> dict[str, Any]:
        return {
            "session_id": self._session_id,
            "instruction": self._instruction,
            "language": self._language,
            "pending_reset": bool(self._needs_reset),
            "request_idx": int(self._request_idx),
            "endpoint": f"{self._server_url}/navigation",
        }
