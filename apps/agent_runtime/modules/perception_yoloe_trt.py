from __future__ import annotations

import asyncio
import inspect
import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional, Sequence, Tuple

from .contracts import Detection2D3D

try:  # pragma: no cover - optional runtime dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    np = None  # type: ignore

try:  # pragma: no cover - optional runtime dependency
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    cv2 = None  # type: ignore


@dataclass
class _LetterboxMeta:
    orig_w: int
    orig_h: int
    input_w: int
    input_h: int
    scale: float
    pad_x: float
    pad_y: float


@dataclass
class TrackedTarget:
    label: str
    cx: float
    cy: float
    score: float
    timestamp: float
    frame_id: str
    image_width: int
    image_height: int
    bbox_xyxy: Tuple[float, float, float, float]


def nms_xyxy(
    boxes_xyxy: Any,
    scores: Any,
    iou_threshold: float,
    max_dets: int,
) -> List[int]:
    if np is None:
        return []
    boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    conf = np.asarray(scores, dtype=np.float32).reshape(-1)
    if boxes.size == 0 or conf.size == 0:
        return []

    boxes = boxes.reshape(-1, 4)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    order = conf.argsort()[::-1]
    keep: List[int] = []
    iou_th = float(max(0.01, min(0.99, iou_threshold)))
    max_keep = max(1, int(max_dets))

    while order.size > 0 and len(keep) < max_keep:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0.0, inter / union, 0.0)

        order = rest[iou <= iou_th]

    return keep


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _parse_input_size(raw: Any) -> Tuple[int, int]:
    if isinstance(raw, int):
        size = max(16, int(raw))
        return size, size
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        h = max(16, int(raw[0]))
        w = max(16, int(raw[1]))
        return h, w
    return 640, 640


def _msg_stamp_to_time(stamp: Any) -> float:
    if stamp is None:
        return time.time()
    sec = float(getattr(stamp, "sec", 0.0))
    nsec = float(getattr(stamp, "nanosec", 0.0))
    if sec > 0.0 or nsec > 0.0:
        return sec + nsec * 1e-9
    return time.time()


class _SimpleLabelTracker:
    def __init__(self, ema_alpha: float, center_weight: float, prefer_center: bool, max_age_s: float) -> None:
        self.ema_alpha = _clip(float(ema_alpha), 0.01, 1.0)
        self.center_weight = _clip(float(center_weight), 0.0, 1.0)
        self.prefer_center = bool(prefer_center)
        self.max_age_s = max(0.05, float(max_age_s))
        self._targets: Dict[str, TrackedTarget] = {}
        self._lock = threading.Lock()

    def update(
        self,
        detections: Sequence[Detection2D3D],
        image_size: Tuple[int, int],
        timestamp: float,
        frame_id: str,
    ) -> None:
        width, height = image_size
        if width <= 0 or height <= 0:
            return

        grouped: Dict[str, List[Detection2D3D]] = {}
        for det in detections:
            label = (det.class_name or "").strip().lower()
            if not label:
                continue
            grouped.setdefault(label, []).append(det)

        with self._lock:
            for label, candidates in grouped.items():
                chosen = self._pick_candidate(candidates, width=width, height=height)
                if chosen is None:
                    continue
                center = self._det_center(chosen)
                if center is None:
                    continue
                cx, cy = center
                bbox_xyxy = self._det_xyxy(chosen)
                prev = self._targets.get(label)
                if prev is not None:
                    alpha = self.ema_alpha
                    cx = (1.0 - alpha) * prev.cx + alpha * cx
                    cy = (1.0 - alpha) * prev.cy + alpha * cy
                    score = (1.0 - alpha) * prev.score + alpha * float(chosen.score)
                else:
                    score = float(chosen.score)
                self._targets[label] = TrackedTarget(
                    label=label,
                    cx=float(cx),
                    cy=float(cy),
                    score=_clip(score, 0.0, 1.0),
                    timestamp=float(timestamp),
                    frame_id=str(getattr(chosen, "frame_id", "") or frame_id),
                    image_width=int(width),
                    image_height=int(height),
                    bbox_xyxy=bbox_xyxy,
                )

            # Prune stale targets.
            stale = [k for k, v in self._targets.items() if (timestamp - v.timestamp) > self.max_age_s]
            for key in stale:
                self._targets.pop(key, None)

    def get_target(self, label: str, max_age_s: Optional[float] = None) -> Optional[TrackedTarget]:
        key = (label or "").strip().lower()
        if not key:
            return None
        age_limit = self.max_age_s if max_age_s is None else max(0.05, float(max_age_s))
        now = time.time()
        with self._lock:
            target = self._targets.get(key)
            if target is None:
                return None
            if (now - target.timestamp) > age_limit:
                return None
            return TrackedTarget(
                label=target.label,
                cx=target.cx,
                cy=target.cy,
                score=target.score,
                timestamp=target.timestamp,
                frame_id=target.frame_id,
                image_width=target.image_width,
                image_height=target.image_height,
                bbox_xyxy=target.bbox_xyxy,
            )

    def _pick_candidate(
        self,
        candidates: Sequence[Detection2D3D],
        width: int,
        height: int,
    ) -> Optional[Detection2D3D]:
        if not candidates:
            return None
        if not self.prefer_center:
            return max(candidates, key=lambda d: float(d.score))

        cx0 = 0.5 * float(width)
        cy0 = 0.5 * float(height)

        def rank(det: Detection2D3D) -> float:
            center = self._det_center(det)
            if center is None:
                return float(det.score)
            cx, cy = center
            dx = (cx - cx0) / max(1.0, cx0)
            dy = (cy - cy0) / max(1.0, cy0)
            dist = math.sqrt(dx * dx + dy * dy)
            return float(det.score) - self.center_weight * dist

        return max(candidates, key=rank)

    @staticmethod
    def _det_xyxy(det: Detection2D3D) -> Tuple[float, float, float, float]:
        bbox_xyxy = getattr(det, "bbox_xyxy", None)
        if bbox_xyxy is not None and len(bbox_xyxy) >= 4:
            return (
                float(bbox_xyxy[0]),
                float(bbox_xyxy[1]),
                float(bbox_xyxy[2]),
                float(bbox_xyxy[3]),
            )
        bbox_xywh = det.bbox_xywh
        if bbox_xywh is None:
            return (0.0, 0.0, 0.0, 0.0)
        x, y, w, h = [float(v) for v in bbox_xywh]
        return (x, y, x + max(0.0, w), y + max(0.0, h))

    @classmethod
    def _det_center(cls, det: Detection2D3D) -> Optional[Tuple[float, float]]:
        bbox_cxcy = getattr(det, "bbox_cxcy", None)
        if bbox_cxcy is not None and len(bbox_cxcy) >= 2:
            return float(bbox_cxcy[0]), float(bbox_cxcy[1])
        x1, y1, x2, y2 = cls._det_xyxy(det)
        if x2 <= x1 or y2 <= y1:
            return None
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


class YOLOEPerception:
    """YOLOE TensorRT perception with optional ROS2 camera subscriber and label tracker."""

    def __init__(
        self,
        cfg: Dict,
        on_detections: Optional[Callable[[List[Detection2D3D]], Awaitable[None] | None]] = None,
    ) -> None:
        self.engine_path = str(
            cfg.get("engine_path", os.environ.get("AURA_YOLOE_ENGINE_PATH", "models/yoloe_26m_fp8.engine"))
        )
        self.labels_path = str(
            cfg.get("labels_path", os.environ.get("AURA_YOLOE_LABELS_PATH", "models/yoloe_labels.txt"))
        )

        input_size_raw = cfg.get("input_size", os.environ.get("AURA_YOLOE_INPUT_SIZE", (640, 640)))
        self.input_h, self.input_w = _parse_input_size(input_size_raw)
        self.conf_threshold = float(cfg.get("conf_threshold", os.environ.get("AURA_YOLOE_CONF", 0.25)))
        self.iou_threshold = float(cfg.get("iou_threshold", os.environ.get("AURA_YOLOE_IOU", 0.45)))
        self.max_dets = int(cfg.get("max_dets", os.environ.get("AURA_YOLOE_MAX_DETS", 100)))
        self.device_index = int(cfg.get("device_index", cfg.get("gpu_id", os.environ.get("AURA_YOLOE_GPU", 0))))
        self.model_color = str(cfg.get("model_color", "rgb")).strip().lower()
        self.assume_input_color = str(cfg.get("assume_input_color", "bgr")).strip().lower()
        self.normalize_mode = str(cfg.get("normalize", "0_1")).strip().lower()

        self.queue_size = int(cfg.get("queue_size", 2))
        self.base_interval_s = float(cfg.get("base_interval_s", 0.05))
        self.warmup_iters = int(cfg.get("warmup_iters", 2))
        self.mock_mode = bool(cfg.get("mock_mode", True))
        self.fallback_to_mock = bool(cfg.get("fallback_to_mock", True))
        self.enable_fallback_variant = bool(cfg.get("enable_fallback_variant", True))
        self.camera_enabled = bool(cfg.get("camera_enabled", not self.mock_mode))
        self.camera_namespace = str(cfg.get("camera_namespace", ""))
        self.camera_topic = str(cfg.get("camera_topic", "camera/color/image_raw"))
        self.camera_compressed_topic = str(
            cfg.get("camera_compressed_topic", "camera/color/image_raw/compressed")
        )
        self.camera_use_compressed = bool(cfg.get("camera_use_compressed", False))

        tracker_cfg = dict(cfg.get("tracker", {}))
        self.tracker_ema_alpha = float(tracker_cfg.get("ema_alpha", 0.35))
        self.tracker_prefer_center = bool(tracker_cfg.get("prefer_center", True))
        self.tracker_center_weight = float(tracker_cfg.get("center_weight", 0.25))
        self.tracker_max_age_s = float(tracker_cfg.get("max_age_s", 0.8))

        self.debug_overlay_enabled = bool(cfg.get("debug_overlay_enabled", False))
        self.debug_overlay_dir = str(cfg.get("debug_overlay_dir", ""))
        self.debug_overlay_every_n = max(1, int(cfg.get("debug_overlay_every_n", 12)))

        self.on_detections = on_detections

        self._frame_queue: Deque[Any] = deque(maxlen=max(1, self.queue_size))
        self._frame_lock = threading.Lock()
        self._drop_count = 0
        self._tick = 0
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._degrade_level = 0
        self._class_cycle = ["apple", "bottle", "cup", "book"]
        self._active_variant = "M"
        self._state_lock = threading.Lock()
        self._latest_detections: List[Detection2D3D] = []
        self._latest_image_size = (self.input_w, self.input_h)
        self._last_detection_ts = 0.0
        self._labels: List[str] = []
        self._target_tracker = _SimpleLabelTracker(
            ema_alpha=self.tracker_ema_alpha,
            center_weight=self.tracker_center_weight,
            prefer_center=self.tracker_prefer_center,
            max_age_s=self.tracker_max_age_s,
        )
        self._last_no_frame_log_ts = 0.0
        self._last_overlay_save_ts = 0.0

        # TensorRT/CUDA runtime handles.
        self._trt = None
        self._cuda = None
        self._cuda_context = None
        self._engine = None
        self._context = None
        self._stream = None
        self._input_binding_idx: Optional[int] = None
        self._output_binding_indices: List[int] = []
        self._binding_shapes: Dict[int, Tuple[int, ...]] = {}
        self._binding_dtypes: Dict[int, Any] = {}
        self._host_buffers: Dict[int, Any] = {}
        self._device_buffers: Dict[int, Any] = {}
        self._bindings: List[int] = []

        # Optional ROS2 camera ingestion.
        self._rclpy = None
        self._camera_node = None
        self._camera_executor = None
        self._camera_sub = None

    def _resolve_topic(self, topic: str) -> str:
        raw = str(topic).strip()
        if raw.startswith("/"):
            return raw
        ns = self.camera_namespace.strip("/")
        if ns:
            return f"/{ns}/{raw.lstrip('/')}"
        return f"/{raw.lstrip('/')}"

    def get_latest_detections(self) -> List[Detection2D3D]:
        with self._state_lock:
            return list(self._latest_detections)

    def get_tracked_target(self, label: str, max_age_s: Optional[float] = None) -> Optional[TrackedTarget]:
        return self._target_tracker.get_target(label, max_age_s=max_age_s)

    def detections_recent(self, max_age_s: float = 1.0) -> bool:
        with self._state_lock:
            ts = self._last_detection_ts
        return (time.time() - ts) <= max(0.05, float(max_age_s))

    async def warmup(self) -> None:
        self._labels = self._load_labels(self.labels_path)
        if not self.mock_mode:
            if not Path(self.engine_path).exists():
                if self.fallback_to_mock:
                    logging.warning(
                        "YOLOE engine not found (%s). Switching to mock mode.", self.engine_path
                    )
                    self.mock_mode = True
                else:
                    raise FileNotFoundError(
                        f"YOLOE engine not found and fallback_to_mock=false: {self.engine_path}"
                    )
            else:
                try:
                    self._init_tensorrt()
                except Exception as exc:
                    if self.fallback_to_mock:
                        logging.warning(
                            "YOLOE TensorRT init failed (%s). Switching to mock mode.",
                            exc,
                        )
                        self.mock_mode = True
                    else:
                        raise
        else:
            logging.info("YOLOE mock mode enabled.")

        if self.camera_enabled:
            self._init_camera_subscriber()

        # Warmup.
        warm_frame = self._make_blank_frame(self.input_h, self.input_w)
        for i in range(self.warmup_iters):
            if not self.mock_mode:
                try:
                    payload = {"image": warm_frame, "timestamp": time.time(), "frame_id": "warmup"}
                    _ = self._infer_once(payload)
                except Exception as exc:
                    logging.warning("YOLOE warmup inference failed: %s", exc)
            await asyncio.sleep(0.05)
            logging.info("YOLOE warmup iteration %s/%s", i + 1, self.warmup_iters)

    def submit_frame(self, frame: Any) -> None:
        payload = self._normalize_frame_payload(frame)
        if payload is None:
            return
        with self._frame_lock:
            if len(self._frame_queue) >= self._frame_queue.maxlen:
                self._drop_count += 1
            self._frame_queue.append(payload)

    def set_degrade_level(self, level: int) -> None:
        self._degrade_level = max(0, int(level))
        if self._degrade_level >= 2 and self.enable_fallback_variant:
            self._active_variant = "S"
        else:
            self._active_variant = "M"
        logging.info(
            "YOLOE degrade level=%s, active_variant=%s, dropped_frames=%s",
            self._degrade_level,
            self._active_variant,
            self._drop_count,
        )

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="perception_yoloe")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            await self._task
            self._task = None
        self._shutdown_camera_subscriber()
        self._release_tensorrt()

    async def _run_loop(self) -> None:
        while self._running:
            self._spin_camera_once()
            frame_payload = self._pull_latest_frame()
            if frame_payload is None:
                if self.mock_mode:
                    frame_payload = {
                        "image": self._make_blank_frame(self.input_h, self.input_w),
                        "timestamp": time.time(),
                        "frame_id": "mock",
                        "color_format": self.assume_input_color,
                    }
                else:
                    now = time.time()
                    if now - self._last_no_frame_log_ts >= 3.0:
                        logging.info("YOLOE waiting for camera frames on %s", self._resolve_topic(self.camera_topic))
                        self._last_no_frame_log_ts = now
                    await asyncio.sleep(0.01)
                    continue

            detections = self._infer_once(frame_payload)
            self._update_detection_state(
                detections=detections,
                image_shape=frame_payload["image"].shape[:2],
                frame_id=str(frame_payload.get("frame_id", "")),
                timestamp=float(frame_payload.get("timestamp", time.time())),
            )
            if detections and self.on_detections is not None:
                out = self.on_detections(detections)
                if inspect.isawaitable(out):
                    await out
            if self.debug_overlay_enabled:
                self._emit_debug_overlay(frame_payload["image"], detections)
            await asyncio.sleep(self._effective_interval_s())

    def _effective_interval_s(self) -> float:
        if self._degrade_level <= 0:
            return self.base_interval_s
        if self._degrade_level == 1:
            return self.base_interval_s * 1.5
        if self._degrade_level == 2:
            return self.base_interval_s * 2.5
        return self.base_interval_s * 3.5

    def _infer_once(self, frame_payload: Dict[str, Any]) -> List[Detection2D3D]:
        self._tick += 1
        image = frame_payload.get("image")
        if image is None:
            return []

        if not self.mock_mode:
            try:
                tensor, meta = self._preprocess(frame_payload)
                outputs = self._run_tensorrt(tensor)
                return self._postprocess(outputs, meta, frame_payload)
            except Exception as exc:
                if self.fallback_to_mock:
                    logging.warning("YOLOE TRT inference failed (%s). Switching to mock mode.", exc)
                    self.mock_mode = True
                else:
                    logging.exception("YOLOE TRT inference failed.")
                    return []

        if self._tick % 2 != 0 and self._frame_queue:
            return []

        cls_name = self._class_cycle[(self._tick // 2) % len(self._class_cycle)]
        h, w = image.shape[:2]
        cx = w * (0.5 + 0.08 * math.sin(self._tick * 0.13))
        cy = h * (0.5 + 0.06 * math.cos(self._tick * 0.11))
        bw = max(32.0, 0.18 * float(w))
        bh = max(32.0, 0.22 * float(h))
        x1 = _clip(cx - 0.5 * bw, 0.0, float(w - 1))
        y1 = _clip(cy - 0.5 * bh, 0.0, float(h - 1))
        x2 = _clip(cx + 0.5 * bw, 0.0, float(w - 1))
        y2 = _clip(cy + 0.5 * bh, 0.0, float(h - 1))
        score = 0.75 + 0.20 * abs(math.sin(self._tick * 0.07))
        detection = self._build_detection(
            label=cls_name,
            score=score,
            box_xyxy=(x1, y1, x2, y2),
            image_size=(w, h),
            timestamp=float(frame_payload.get("timestamp", time.time())),
            frame_id=str(frame_payload.get("frame_id", "")),
            object_suffix=0,
        )
        logging.info(
            "Perception detection: class=%s score=%.2f queue_drop=%s variant=%s",
            detection.class_name,
            detection.score,
            self._drop_count,
            self._active_variant,
        )
        return [detection]

    def _normalize_frame_payload(self, frame: Any) -> Optional[Dict[str, Any]]:
        if np is None:
            return None
        if frame is None:
            return None
        if isinstance(frame, dict):
            image = frame.get("image")
            if image is None:
                return None
            arr = np.asarray(image)
            if arr.ndim < 2:
                return None
            return {
                "image": arr,
                "timestamp": float(frame.get("timestamp", time.time())),
                "frame_id": str(frame.get("frame_id", "")),
                "color_format": str(frame.get("color_format", self.assume_input_color)).lower(),
            }

        arr = np.asarray(frame)
        if arr.ndim < 2:
            return None
        return {
            "image": arr,
            "timestamp": time.time(),
            "frame_id": "",
            "color_format": self.assume_input_color,
        }

    def _pull_latest_frame(self) -> Optional[Dict[str, Any]]:
        with self._frame_lock:
            if not self._frame_queue:
                return None
            while len(self._frame_queue) > 1:
                self._frame_queue.popleft()
                self._drop_count += 1
            return self._frame_queue.pop()

    def _make_blank_frame(self, h: int, w: int) -> Any:
        if np is None:
            return None
        return np.zeros((max(2, int(h)), max(2, int(w)), 3), dtype=np.uint8)

    def _load_labels(self, labels_path: str) -> List[str]:
        path = Path(labels_path)
        if not path.exists():
            logging.warning("YOLOE labels file not found: %s. Falling back to class_<id> names.", labels_path)
            return []
        labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not labels:
            logging.warning("YOLOE labels file is empty: %s", labels_path)
        return labels

    def _init_tensorrt(self) -> None:
        if np is None:
            raise RuntimeError("numpy is required for YOLOE TensorRT inference.")
        try:
            import tensorrt as trt  # type: ignore
            import pycuda.driver as cuda  # type: ignore
        except Exception as exc:
            raise RuntimeError("Missing TensorRT/CUDA Python bindings (tensorrt + pycuda).") from exc

        cuda.init()
        if self.device_index < 0 or self.device_index >= cuda.Device.count():
            raise RuntimeError(f"Invalid CUDA device index for YOLOE: {self.device_index}")
        self._cuda_context = cuda.Device(self.device_index).make_context()

        trt_logger = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(trt_logger)
        engine_bytes = Path(self.engine_path).read_bytes()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.engine_path}")
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")

        self._trt = trt
        self._cuda = cuda
        self._engine = engine
        self._context = context
        self._stream = cuda.Stream()
        self._input_binding_idx = None
        self._output_binding_indices = []
        self._binding_shapes = {}
        self._binding_dtypes = {}
        self._host_buffers = {}
        self._device_buffers = {}
        self._bindings = [0] * int(engine.num_bindings)

        for idx in range(int(engine.num_bindings)):
            name = engine.get_binding_name(idx)
            is_input = bool(engine.binding_is_input(idx))
            dtype = trt.nptype(engine.get_binding_dtype(idx))
            self._binding_dtypes[idx] = dtype
            if is_input:
                self._input_binding_idx = idx
                # Force static input shape at configured resolution.
                target_shape = (1, 3, int(self.input_h), int(self.input_w))
                context.set_binding_shape(idx, target_shape)
            else:
                self._output_binding_indices.append(idx)
            logging.info("YOLOE binding[%s] name=%s input=%s dtype=%s", idx, name, is_input, dtype)

        if self._input_binding_idx is None:
            raise RuntimeError("TensorRT engine has no input binding.")

        for idx in range(int(engine.num_bindings)):
            shape = tuple(int(v) for v in context.get_binding_shape(idx))
            if any(v < 0 for v in shape):
                raise RuntimeError(f"Unresolved dynamic binding shape idx={idx}: {shape}")
            self._binding_shapes[idx] = shape
            size = int(np.prod(shape))
            host = cuda.pagelocked_empty(size, dtype=self._binding_dtypes[idx])
            device = cuda.mem_alloc(host.nbytes)
            self._host_buffers[idx] = host
            self._device_buffers[idx] = device
            self._bindings[idx] = int(device)
            logging.info("YOLOE alloc binding[%s] shape=%s bytes=%s", idx, shape, host.nbytes)

        logging.info(
            "YOLOE TensorRT ready: engine=%s input=(1,3,%s,%s) outputs=%s device=%s",
            self.engine_path,
            self.input_h,
            self.input_w,
            len(self._output_binding_indices),
            self.device_index,
        )

    def _release_tensorrt(self) -> None:
        if self._cuda_context is not None:
            try:
                self._cuda_context.pop()
            except Exception:
                pass
            try:
                self._cuda_context.detach()
            except Exception:
                pass
        self._cuda_context = None
        self._stream = None
        self._context = None
        self._engine = None
        self._cuda = None
        self._trt = None
        self._bindings = []
        self._host_buffers = {}
        self._device_buffers = {}

    def _preprocess(self, frame_payload: Dict[str, Any]) -> Tuple[Any, _LetterboxMeta]:
        if np is None:
            raise RuntimeError("numpy is required for preprocessing.")

        image = np.asarray(frame_payload["image"])
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
            image = np.transpose(image, (1, 2, 0))
        if image.ndim != 3 or image.shape[-1] != 3:
            raise RuntimeError(f"Unsupported frame shape for YOLOE: {image.shape}")

        if image.dtype != np.uint8:
            image = np.asarray(np.clip(image, 0, 255), dtype=np.uint8)
        else:
            image = np.ascontiguousarray(image)

        color_format = str(frame_payload.get("color_format", self.assume_input_color)).lower()
        if color_format not in {"rgb", "bgr"}:
            color_format = self.assume_input_color

        if color_format != self.model_color:
            image = image[..., ::-1]

        resized, meta = self._letterbox(image)
        tensor = resized.astype(np.float32)
        if self.normalize_mode in {"0_1", "zero_one", "01"}:
            tensor /= 255.0
        elif self.normalize_mode in {"-1_1", "neg1_1"}:
            tensor = tensor / 127.5 - 1.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)
        return tensor, meta

    def _letterbox(self, image: Any) -> Tuple[Any, _LetterboxMeta]:
        if np is None:
            raise RuntimeError("numpy is required for letterbox.")
        orig_h, orig_w = image.shape[:2]
        scale = min(self.input_w / max(1, orig_w), self.input_h / max(1, orig_h))
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))

        if cv2 is not None:
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            y_idx = np.linspace(0, orig_h - 1, new_h).astype(np.int32)
            x_idx = np.linspace(0, orig_w - 1, new_w).astype(np.int32)
            resized = image[y_idx][:, x_idx]

        canvas = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        pad_x = (self.input_w - new_w) // 2
        pad_y = (self.input_h - new_h) // 2
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        meta = _LetterboxMeta(
            orig_w=int(orig_w),
            orig_h=int(orig_h),
            input_w=int(self.input_w),
            input_h=int(self.input_h),
            scale=float(scale),
            pad_x=float(pad_x),
            pad_y=float(pad_y),
        )
        return canvas, meta

    def _run_tensorrt(self, input_tensor: Any) -> List[Any]:
        if np is None:
            raise RuntimeError("numpy is required for TensorRT inference.")
        if self._engine is None or self._context is None or self._cuda is None or self._stream is None:
            raise RuntimeError("YOLOE TensorRT runtime is not initialized.")
        if self._input_binding_idx is None:
            raise RuntimeError("YOLOE TensorRT input binding is missing.")

        input_idx = self._input_binding_idx
        flat = input_tensor.reshape(-1)
        host_in = self._host_buffers[input_idx]
        if host_in.size != flat.size:
            raise RuntimeError(
                f"YOLOE input shape mismatch. tensor={input_tensor.shape}, binding={self._binding_shapes[input_idx]}"
            )
        np.copyto(host_in, flat.astype(host_in.dtype, copy=False))

        self._cuda.memcpy_htod_async(self._device_buffers[input_idx], host_in, self._stream)
        ok = self._context.execute_async_v2(bindings=self._bindings, stream_handle=self._stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v2 returned false.")
        for idx in self._output_binding_indices:
            self._cuda.memcpy_dtoh_async(self._host_buffers[idx], self._device_buffers[idx], self._stream)
        self._stream.synchronize()

        outputs: List[Any] = []
        for idx in self._output_binding_indices:
            shape = self._binding_shapes[idx]
            view = np.asarray(self._host_buffers[idx]).reshape(shape)
            outputs.append(view)
        return outputs

    def _decode_model_outputs(self, outputs: Sequence[Any]) -> Tuple[Any, Any, Any]:
        if np is None:
            raise RuntimeError("numpy is required for YOLOE postprocessing.")
        if not outputs:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )

        pred = np.asarray(outputs[0])
        while pred.ndim > 2 and pred.shape[0] == 1:
            pred = pred[0]
        if pred.ndim == 1:
            pred = pred.reshape(1, -1)
        if pred.ndim != 2:
            pred = pred.reshape(pred.shape[0], -1)

        # Handle common TRT layouts where channels/classes are first.
        if pred.shape[1] < 6 and pred.shape[0] >= 6:
            pred = pred.T
        elif pred.shape[1] > 6 and pred.shape[0] in {84, 85, 86, 116}:
            pred = pred.T

        if pred.shape[1] == 6:
            boxes = pred[:, :4].astype(np.float32)
            scores = pred[:, 4].astype(np.float32)
            cls_ids = pred[:, 5].astype(np.int32)
            # Normalize support.
            if boxes.size and float(np.nanmax(np.abs(boxes))) <= 1.5:
                boxes[:, [0, 2]] *= float(self.input_w)
                boxes[:, [1, 3]] *= float(self.input_h)
            return boxes, scores, cls_ids

        if pred.shape[1] < 6:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )

        # YOLO-style [cx, cy, w, h, objectness, cls...].
        xywh = pred[:, :4].astype(np.float32)
        objectness = pred[:, 4].astype(np.float32)
        cls_logits = pred[:, 5:].astype(np.float32)
        if cls_logits.size == 0:
            cls_ids = np.zeros((xywh.shape[0],), dtype=np.int32)
            scores = objectness
        else:
            cls_ids = cls_logits.argmax(axis=1).astype(np.int32)
            cls_scores = cls_logits[np.arange(cls_logits.shape[0]), cls_ids]
            scores = objectness * cls_scores
        cx = xywh[:, 0]
        cy = xywh[:, 1]
        w = xywh[:, 2]
        h = xywh[:, 3]
        boxes = np.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], axis=1)
        if boxes.size and float(np.nanmax(np.abs(boxes))) <= 1.5:
            boxes[:, [0, 2]] *= float(self.input_w)
            boxes[:, [1, 3]] *= float(self.input_h)
        return boxes.astype(np.float32), scores.astype(np.float32), cls_ids.astype(np.int32)

    def _postprocess(
        self,
        outputs: Sequence[Any],
        letterbox: _LetterboxMeta,
        frame_payload: Dict[str, Any],
    ) -> List[Detection2D3D]:
        if np is None:
            return []

        boxes, scores, cls_ids = self._decode_model_outputs(outputs)
        if boxes.size == 0:
            return []

        valid = scores >= float(self.conf_threshold)
        boxes = boxes[valid]
        scores = scores[valid]
        cls_ids = cls_ids[valid]
        if boxes.size == 0:
            return []

        keep = nms_xyxy(boxes, scores, iou_threshold=self.iou_threshold, max_dets=self.max_dets)
        if not keep:
            return []
        boxes = boxes[keep]
        scores = scores[keep]
        cls_ids = cls_ids[keep]

        dets: List[Detection2D3D] = []
        frame_id = str(frame_payload.get("frame_id", ""))
        ts = float(frame_payload.get("timestamp", time.time()))
        for i in range(len(keep)):
            box = boxes[i]
            score = float(scores[i])
            cls_id = int(cls_ids[i])
            x1 = (float(box[0]) - letterbox.pad_x) / max(1e-6, letterbox.scale)
            y1 = (float(box[1]) - letterbox.pad_y) / max(1e-6, letterbox.scale)
            x2 = (float(box[2]) - letterbox.pad_x) / max(1e-6, letterbox.scale)
            y2 = (float(box[3]) - letterbox.pad_y) / max(1e-6, letterbox.scale)
            x1 = _clip(x1, 0.0, float(letterbox.orig_w - 1))
            y1 = _clip(y1, 0.0, float(letterbox.orig_h - 1))
            x2 = _clip(x2, 0.0, float(letterbox.orig_w - 1))
            y2 = _clip(y2, 0.0, float(letterbox.orig_h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            label = self._labels[cls_id] if 0 <= cls_id < len(self._labels) else f"class_{cls_id}"
            det = self._build_detection(
                label=label,
                score=score,
                box_xyxy=(x1, y1, x2, y2),
                image_size=(letterbox.orig_w, letterbox.orig_h),
                timestamp=ts,
                frame_id=frame_id,
                object_suffix=i,
            )
            dets.append(det)
        return dets

    def _build_detection(
        self,
        label: str,
        score: float,
        box_xyxy: Tuple[float, float, float, float],
        image_size: Tuple[int, int],
        timestamp: float,
        frame_id: str,
        object_suffix: int,
    ) -> Detection2D3D:
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        cx = x1 + 0.5 * w
        cy = y1 + 0.5 * h
        detection = Detection2D3D(
            object_id=f"{label}-{object_suffix}",
            class_name=str(label).strip().lower(),
            score=_clip(float(score), 0.0, 1.0),
            bbox_xywh=(x1, y1, w, h),
            bbox_xyxy=(x1, y1, x2, y2),
            bbox_cxcy=(cx, cy),
            image_size=(int(image_size[0]), int(image_size[1])),
            frame_id=str(frame_id),
            position_in_map=None,
            timestamp=float(timestamp),
        )
        return detection

    def _update_detection_state(
        self,
        detections: Sequence[Detection2D3D],
        image_shape: Tuple[int, int],
        frame_id: str,
        timestamp: float,
    ) -> None:
        height, width = int(image_shape[0]), int(image_shape[1])
        with self._state_lock:
            self._latest_detections = list(detections)
            self._latest_image_size = (width, height)
            if detections:
                self._last_detection_ts = float(timestamp)
        self._target_tracker.update(
            detections=detections,
            image_size=(width, height),
            timestamp=float(timestamp),
            frame_id=frame_id,
        )

    def _init_camera_subscriber(self) -> None:
        if not self.camera_enabled:
            return
        try:
            import rclpy  # type: ignore
            from rclpy.executors import SingleThreadedExecutor  # type: ignore
            from sensor_msgs.msg import CompressedImage, Image  # type: ignore
        except Exception as exc:
            logging.warning("ROS2 camera subscriber unavailable for YOLOE: %s", exc)
            return

        if not rclpy.ok():
            rclpy.init(args=None)

        self._rclpy = rclpy
        self._camera_node = rclpy.create_node("yoloe_perception_camera")
        self._camera_executor = SingleThreadedExecutor()
        self._camera_executor.add_node(self._camera_node)

        if self.camera_use_compressed:
            topic = self._resolve_topic(self.camera_compressed_topic)
            self._camera_sub = self._camera_node.create_subscription(
                CompressedImage,
                topic,
                self._on_compressed_image,
                4,
            )
            logging.info("YOLOE camera subscription: topic=%s type=CompressedImage", topic)
        else:
            topic = self._resolve_topic(self.camera_topic)
            self._camera_sub = self._camera_node.create_subscription(
                Image,
                topic,
                self._on_image,
                4,
            )
            logging.info("YOLOE camera subscription: topic=%s type=Image", topic)

    def _shutdown_camera_subscriber(self) -> None:
        self._camera_sub = None
        if self._camera_executor is not None and self._camera_node is not None:
            try:
                self._camera_executor.remove_node(self._camera_node)
            except Exception:
                pass
        if self._camera_node is not None:
            try:
                self._camera_node.destroy_node()
            except Exception:
                pass
        self._camera_node = None
        self._camera_executor = None
        self._rclpy = None

    def _spin_camera_once(self) -> None:
        if self._camera_executor is None:
            return
        try:
            self._camera_executor.spin_once(timeout_sec=0.0)
        except Exception as exc:
            logging.debug("YOLOE camera spin_once failed: %s", exc)

    def _on_image(self, msg: Any) -> None:
        if np is None:
            return
        try:
            h = int(getattr(msg, "height", 0))
            w = int(getattr(msg, "width", 0))
            encoding = str(getattr(msg, "encoding", "")).lower()
            raw = bytes(getattr(msg, "data", b""))
            if h <= 0 or w <= 0 or not raw:
                return

            if encoding in {"rgb8", "bgr8"}:
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
                color = "rgb" if encoding == "rgb8" else "bgr"
            elif encoding in {"mono8", "8uc1"}:
                mono = np.frombuffer(raw, dtype=np.uint8).reshape(h, w)
                arr = np.stack([mono, mono, mono], axis=-1)
                color = self.assume_input_color
            else:
                logging.debug("YOLOE unsupported camera encoding: %s", encoding)
                return

            stamp = _msg_stamp_to_time(getattr(getattr(msg, "header", None), "stamp", None))
            frame_id = str(getattr(getattr(msg, "header", None), "frame_id", ""))
            self.submit_frame({"image": arr, "timestamp": stamp, "frame_id": frame_id, "color_format": color})
        except Exception as exc:
            logging.debug("YOLOE failed to parse Image message: %s", exc)

    def _on_compressed_image(self, msg: Any) -> None:
        if np is None or cv2 is None:
            return
        try:
            raw = bytes(getattr(msg, "data", b""))
            if not raw:
                return
            arr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                return
            stamp = _msg_stamp_to_time(getattr(getattr(msg, "header", None), "stamp", None))
            frame_id = str(getattr(getattr(msg, "header", None), "frame_id", ""))
            self.submit_frame({"image": arr, "timestamp": stamp, "frame_id": frame_id, "color_format": "bgr"})
        except Exception as exc:
            logging.debug("YOLOE failed to parse CompressedImage message: %s", exc)

    def _emit_debug_overlay(self, frame: Any, detections: Sequence[Detection2D3D]) -> None:
        if cv2 is None or np is None:
            return
        if frame is None:
            return
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            return
        overlay = arr.copy()
        for det in detections:
            bbox_xyxy = getattr(det, "bbox_xyxy", None)
            if bbox_xyxy is None:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (24, 220, 70), 2)
            label = f"{det.class_name}:{det.score:.2f}"
            cv2.putText(overlay, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (24, 220, 70), 1)

        if not self.debug_overlay_dir:
            return
        if self._tick % self.debug_overlay_every_n != 0:
            return
        out_dir = Path(self.debug_overlay_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"yoloe_overlay_{self._tick:06d}.jpg"
        try:
            cv2.imwrite(str(out_path), overlay)
        except Exception as exc:
            logging.debug("YOLOE failed to save debug overlay: %s", exc)
