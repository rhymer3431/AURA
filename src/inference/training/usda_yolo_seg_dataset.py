from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from runtime.global_route import MapMeta, load_occupancy_grid

from . import usda_yolo_dataset as detection

YOLO_CLASS_NAMES = detection.YOLO_CLASS_NAMES
YOLO_CLASS_TO_ID = detection.YOLO_CLASS_TO_ID
SceneObjectRecord = detection.SceneObjectRecord
CameraSpec = detection.CameraSpec
DEFAULT_TRAIN_COUNT = detection.DEFAULT_TRAIN_COUNT
DEFAULT_VAL_COUNT = detection.DEFAULT_VAL_COUNT
DEFAULT_TEST_COUNT = detection.DEFAULT_TEST_COUNT
DEFAULT_IMAGE_WIDTH = detection.DEFAULT_IMAGE_WIDTH
DEFAULT_IMAGE_HEIGHT = detection.DEFAULT_IMAGE_HEIGHT
DEFAULT_SEED = detection.DEFAULT_SEED
MIN_BBOX_WIDTH_PX = detection.MIN_BBOX_WIDTH_PX
MIN_BBOX_HEIGHT_PX = detection.MIN_BBOX_HEIGHT_PX
MIN_BBOX_AREA_PX = detection.MIN_BBOX_AREA_PX
CONTOUR_APPROX_RATIO = 0.0025
MIN_CONTOUR_AREA_PX = 8.0
MANIFEST_SCHEMA_VERSION = "usda_yolo_seg_v1"
RGB_EXPOSURE_GAIN = 1.5


def _repo_root() -> Path:
    return detection._repo_root()


DEFAULT_OUTPUT_DIR = _repo_root() / "artifacts" / "datasets" / "kujiale_0003_yolo_seg"
DEFAULT_SCENE_USDA_PATH = detection.DEFAULT_SCENE_USDA_PATH
DEFAULT_ROOMS_JSON_PATH = detection.DEFAULT_ROOMS_JSON_PATH
DEFAULT_OCCUPANCY_IMAGE_PATH = detection.DEFAULT_OCCUPANCY_IMAGE_PATH
DEFAULT_CONFIG_PATH = detection.DEFAULT_CONFIG_PATH


@dataclass(frozen=True)
class RenderedSegmentationInstance:
    class_name: str
    prim_path: str
    instance_id: int
    bbox_xyxy: tuple[int, int, int, int]
    polygon_xy: tuple[tuple[int, int], ...]
    mask_area_px: int


@dataclass(frozen=True)
class RenderedSegmentationSample:
    rgb_image: np.ndarray
    segments: tuple[RenderedSegmentationInstance, ...]
    camera_position_xyz: tuple[float, float, float]
    look_at_xyz: tuple[float, float, float]


def load_scene_objects(scene_usda_path: str | Path, *, rooms_json_path: str | Path | None = None) -> list[SceneObjectRecord]:
    return detection.load_scene_objects(scene_usda_path, rooms_json_path=rooms_json_path)


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    return detection.load_jsonl_records(path)


def _coerce_segmentation_map(payload: Any) -> np.ndarray | None:
    array = detection._IsaacYoloRenderer._coerce_array(payload)
    if array is None:
        return None
    mask = np.asarray(array)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim != 2:
        return None
    if not np.issubdtype(mask.dtype, np.integer):
        return None
    return np.asarray(mask, dtype=np.uint32)


def _dedupe_polygon_points(points_xy: np.ndarray) -> tuple[tuple[int, int], ...]:
    unique_points: list[tuple[int, int]] = []
    for row in points_xy:
        point = (int(row[0]), int(row[1]))
        if unique_points and unique_points[-1] == point:
            continue
        unique_points.append(point)
    if len(unique_points) >= 2 and unique_points[0] == unique_points[-1]:
        unique_points.pop()
    return tuple(unique_points)


def _polygon_area_xy(points_xy: tuple[tuple[int, int], ...]) -> float:
    if len(points_xy) < 3:
        return 0.0
    coords = np.asarray(points_xy, dtype=np.float64)
    x_values = coords[:, 0]
    y_values = coords[:, 1]
    return float(abs(np.dot(x_values, np.roll(y_values, -1)) - np.dot(y_values, np.roll(x_values, -1))) * 0.5)


def _mask_to_largest_polygon(mask: np.ndarray) -> tuple[tuple[tuple[int, int], ...], tuple[int, int, int, int]] | None:
    try:
        import cv2
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("OpenCV (cv2) is required to convert segmentation masks into YOLO polygons.") from exc

    binary_mask = np.asarray(mask, dtype=np.uint8)
    if binary_mask.ndim != 2 or binary_mask.size == 0 or int(binary_mask.sum()) <= 0:
        return None
    contours_result = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_result[0] if len(contours_result) == 2 else contours_result[1]
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    if contour_area < MIN_CONTOUR_AREA_PX:
        return None
    perimeter = float(cv2.arcLength(contour, True))
    epsilon = max(1.0, perimeter * CONTOUR_APPROX_RATIO)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    points = approx.reshape(-1, 2) if approx is not None and len(approx) >= 3 else contour.reshape(-1, 2)
    polygon_xy = _dedupe_polygon_points(np.asarray(points, dtype=np.int32))
    if len(polygon_xy) < 3 or _polygon_area_xy(polygon_xy) <= 0.0:
        return None
    x_value, y_value, width_px, height_px = cv2.boundingRect(contour)
    bbox_xyxy = (
        int(x_value),
        int(y_value),
        int(x_value + width_px),
        int(y_value + height_px),
    )
    return polygon_xy, bbox_xyxy


def _tone_map_rgb_image(image: Any, *, exposure_gain: float = RGB_EXPOSURE_GAIN) -> np.ndarray:
    if float(exposure_gain) <= 0.0:
        raise ValueError(f"exposure_gain must be positive, got {exposure_gain}")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"RGB image must have at least 3 channels, got shape={array.shape}")
    rgb = np.asarray(array[..., :3], dtype=np.float32)
    max_value = float(np.nanmax(rgb)) if rgb.size > 0 else 0.0
    if max_value > 1.5:
        rgb = rgb / 255.0
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    rgb = np.clip(rgb * float(exposure_gain), 0.0, 1.0)
    threshold = 0.0031308
    rgb = np.where(
        rgb <= threshold,
        12.92 * rgb,
        1.055 * np.power(rgb, 1.0 / 2.4) - 0.055,
    )
    return np.clip(np.rint(rgb * 255.0), 0.0, 255.0).astype(np.uint8)


def _polygon_to_yolo(points_xy: tuple[tuple[int, int], ...], *, image_width: int, image_height: int) -> list[float]:
    width = float(image_width)
    height = float(image_height)
    values: list[float] = []
    for x_value, y_value in points_xy:
        values.append(float(x_value) / width)
        values.append(float(y_value) / height)
    return values


def _filter_and_normalize_segments(
    rendered_sample: RenderedSegmentationSample,
    *,
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for item in rendered_sample.segments:
        if item.class_name not in YOLO_CLASS_TO_ID:
            continue
        x_min, y_min, x_max, y_max = item.bbox_xyxy
        width_px = int(x_max - x_min)
        height_px = int(y_max - y_min)
        area_px = int(width_px * height_px)
        if width_px < MIN_BBOX_WIDTH_PX or height_px < MIN_BBOX_HEIGHT_PX or area_px < MIN_BBOX_AREA_PX:
            continue
        if len(item.polygon_xy) < 3 or _polygon_area_xy(item.polygon_xy) <= 0.0:
            continue
        polygon_yolo = _polygon_to_yolo(
            item.polygon_xy,
            image_width=image_width,
            image_height=image_height,
        )
        if any(not (0.0 <= value <= 1.0) for value in polygon_yolo):
            continue
        kept.append(
            {
                "class_id": int(YOLO_CLASS_TO_ID[item.class_name]),
                "class_name": item.class_name,
                "prim_path": item.prim_path,
                "instance_id": int(item.instance_id),
                "bbox_xyxy": [int(x_min), int(y_min), int(x_max), int(y_max)],
                "polygon_xy": [[int(x_value), int(y_value)] for x_value, y_value in item.polygon_xy],
                "polygon_yolo": [float(value) for value in polygon_yolo],
                "mask_area_px": int(item.mask_area_px),
            }
        )
    return kept


def _label_lines_from_segments(segments: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for segment in segments:
        coords = " ".join(f"{float(value):.6f}" for value in segment["polygon_yolo"])
        lines.append(f"{int(segment['class_id'])} {coords}")
    return lines


def _write_label_file(path: Path, segments: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_label_lines_from_segments(segments)) + "\n", encoding="utf-8")


def _write_readme(root: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# USDA YOLO-Seg Dataset",
        "",
        "Synthetic instance-segmentation dataset rendered from `datasets/InteriorAgent/kujiale_0003/kujiale_0003.usda`.",
        "",
        "## Splits",
        "",
        f"- train: {manifest['split_counts']['train']}",
        f"- val: {manifest['split_counts']['val']}",
        f"- test: {manifest['split_counts']['test']}",
        "",
        "## Classes",
        "",
    ]
    for index, name in enumerate(YOLO_CLASS_NAMES):
        lines.append(f"- `{index}` `{name}`")
    lines.extend(
        [
            "",
            "## Usage",
            "",
            "- Ultralytics config file: `data.yaml`",
            "- Labels use YOLO-seg polygon format: `class_id x1 y1 x2 y2 ...`.",
        ]
    )
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _segment_matches_target_prim(segment_prim_path: str, target_prim_path: str) -> bool:
    segment_path = str(segment_prim_path).rstrip("/")
    target_path = str(target_prim_path).rstrip("/")
    return segment_path == target_path or segment_path.startswith(f"{target_path}/")


class _IsaacYoloSegRenderer(detection._IsaacYoloRenderer):
    def _create_camera(self) -> None:
        assert self._stage is not None
        assert self._camera_class is not None
        camera_prim_path = "/World/YoloSegDatasetCamera"
        self._stage.DefinePrim(camera_prim_path, "Camera")
        self._camera = self._camera_class(
            prim_path=camera_prim_path,
            resolution=(self._image_width, self._image_height),
            annotator_device="cpu",
        )
        for _ in range(6):
            self._sim_app.update()
        initialize = getattr(self._camera, "initialize", None)
        if callable(initialize):
            initialize()
        set_lens_distortion_model = getattr(self._camera, "set_lens_distortion_model", None)
        if callable(set_lens_distortion_model):
            set_lens_distortion_model("pinhole")
        render_product_path = None
        get_render_product_path = getattr(self._camera, "get_render_product_path", None)
        if callable(get_render_product_path):
            for _ in range(12):
                self._sim_app.update()
                render_product_path = get_render_product_path()
                if render_product_path:
                    break
        if not render_product_path:
            raise RuntimeError("Isaac camera render product path is unavailable after initialize().")
        add_instance_segmentation = getattr(self._camera, "add_instance_segmentation_to_frame", None)
        if not callable(add_instance_segmentation):
            raise RuntimeError("Isaac Camera.add_instance_segmentation_to_frame is unavailable.")
        add_instance_segmentation(init_params={"semanticTypes": ["class"], "colorize": False})
        add_semantic_segmentation = getattr(self._camera, "add_semantic_segmentation_to_frame", None)
        if callable(add_semantic_segmentation):
            add_semantic_segmentation(init_params={"semanticTypes": ["class"], "colorize": False})
        resume = getattr(self._camera, "resume", None)
        if callable(resume):
            resume()
        if self._timeline is not None:
            self._timeline.play()
        for _ in range(6):
            self._sim_app.update()

    def _capture_single_segmentation_payload(self, name: str) -> tuple[np.ndarray | None, dict[str, Any]]:
        get_data = getattr(self._camera, "get_data", None)
        if callable(get_data):
            try:
                seg_data, seg_info = get_data(name)
                return _coerce_segmentation_map(seg_data), dict(seg_info or {})
            except Exception:  # noqa: BLE001
                pass
        current_frame_fn = getattr(self._camera, "get_current_frame", None)
        if callable(current_frame_fn):
            try:
                frame = current_frame_fn()
                if isinstance(frame, dict):
                    payload = frame.get(name)
                    if isinstance(payload, dict):
                        return _coerce_segmentation_map(payload.get("data")), dict(payload.get("info", {}))
                    if payload is not None:
                        return _coerce_segmentation_map(payload), {}
            except Exception:  # noqa: BLE001
                pass
        return None, {}

    def _capture_segmentation_payloads(
        self,
    ) -> tuple[np.ndarray | None, dict[str, Any], np.ndarray | None, dict[str, Any]]:
        instance_map, instance_info = self._capture_single_segmentation_payload("instance_segmentation")
        semantic_map, semantic_info = self._capture_single_segmentation_payload("semantic_segmentation")
        return instance_map, instance_info, semantic_map, semantic_info

    @staticmethod
    def _extract_prim_path(info: dict[str, Any], instance_id: int) -> str:
        labels = info.get("idToLabels") or info.get("id_to_labels") or {}
        candidate = None
        if isinstance(labels, dict):
            candidate = labels.get(instance_id)
            if candidate is None:
                candidate = labels.get(str(instance_id))
        if isinstance(candidate, dict):
            for key in ("primPath", "prim_path", "prim", "path"):
                value = candidate.get(key)
                if isinstance(value, str) and value.strip() != "":
                    return value.strip()
        if isinstance(candidate, list):
            for item in candidate:
                if isinstance(item, dict):
                    for key in ("primPath", "prim_path", "prim", "path"):
                        value = item.get(key)
                        if isinstance(value, str) and value.strip() != "":
                            return value.strip()
                if isinstance(item, str) and item.startswith("/"):
                    return item.strip()
        if isinstance(candidate, str) and candidate.startswith("/"):
            return candidate.strip()
        return ""

    @staticmethod
    def _coerce_semantic_id(candidate: Any) -> int | None:
        if isinstance(candidate, bool):
            return None
        if isinstance(candidate, (int, np.integer)):
            return int(candidate)
        if isinstance(candidate, str):
            text = candidate.strip()
            if text.isdigit():
                return int(text)
            return None
        if isinstance(candidate, dict):
            for key in ("semanticId", "semantic_id", "semantic", "id"):
                value = candidate.get(key)
                semantic_id = _IsaacYoloSegRenderer._coerce_semantic_id(value)
                if semantic_id is not None:
                    return semantic_id
            return None
        if isinstance(candidate, (list, tuple)):
            for item in candidate:
                semantic_id = _IsaacYoloSegRenderer._coerce_semantic_id(item)
                if semantic_id is not None:
                    return semantic_id
        return None

    def _extract_semantic_id(self, info: dict[str, Any], instance_id: int) -> int | None:
        mapping = info.get("idToSemantics") or info.get("id_to_semantics") or {}
        candidate = None
        if isinstance(mapping, dict):
            candidate = mapping.get(instance_id)
            if candidate is None:
                candidate = mapping.get(str(instance_id))
        return self._coerce_semantic_id(candidate)

    def _extract_class_name(
        self,
        *,
        instance_id: int,
        instance_info: dict[str, Any],
        semantic_id: int | None,
        semantic_info: dict[str, Any],
    ) -> str | None:
        if semantic_id is not None:
            class_name = self._extract_class_label(semantic_info, semantic_id)
            if isinstance(class_name, str) and class_name.strip() != "":
                return class_name.strip()
            class_name = self._extract_class_label(instance_info, semantic_id)
            if isinstance(class_name, str) and class_name.strip() != "":
                return class_name.strip()
        return self._extract_class_label(instance_info, instance_id)

    def _extract_segments(
        self,
        instance_map: np.ndarray,
        instance_info: dict[str, Any],
        semantic_map: np.ndarray | None,
        semantic_info: dict[str, Any],
    ) -> tuple[RenderedSegmentationInstance, ...]:
        segments: list[RenderedSegmentationInstance] = []
        for raw_instance_id in np.unique(instance_map):
            instance_id = int(raw_instance_id)
            if instance_id <= 0:
                continue
            mask = instance_map == raw_instance_id
            semantic_id = self._extract_semantic_id(instance_info, instance_id)
            if semantic_id is None and semantic_map is not None:
                semantic_values = semantic_map[mask]
                semantic_values = semantic_values[semantic_values > 0]
                if semantic_values.size > 0:
                    unique_semantic_ids, counts = np.unique(semantic_values, return_counts=True)
                    semantic_id = int(unique_semantic_ids[int(np.argmax(counts))])
            class_name = self._extract_class_name(
                instance_id=instance_id,
                instance_info=instance_info,
                semantic_id=semantic_id,
                semantic_info=semantic_info,
            )
            if class_name not in YOLO_CLASS_TO_ID:
                continue
            mask_area_px = int(np.count_nonzero(mask))
            if mask_area_px <= 0:
                continue
            polygon_and_bbox = _mask_to_largest_polygon(mask)
            if polygon_and_bbox is None:
                continue
            polygon_xy, bbox_xyxy = polygon_and_bbox
            segments.append(
                RenderedSegmentationInstance(
                    class_name=class_name,
                    prim_path=self._extract_prim_path(instance_info, instance_id),
                    instance_id=instance_id,
                    bbox_xyxy=bbox_xyxy,
                    polygon_xy=polygon_xy,
                    mask_area_px=mask_area_px,
                )
            )
        return tuple(segments)

    def render_sample(
        self,
        target_object: SceneObjectRecord,
        *,
        camera_position_xyz: tuple[float, float, float],
        look_at_xyz: tuple[float, float, float],
    ) -> RenderedSegmentationSample | None:
        if not self._initialized or self._camera is None:
            raise RuntimeError("Isaac renderer was not initialized correctly.")
        orientation = detection._look_at_quaternion(
            camera_position_xyz=camera_position_xyz,
            look_at_xyz=look_at_xyz,
        )
        set_world_pose = getattr(self._camera, "set_world_pose", None)
        if not callable(set_world_pose):
            raise RuntimeError("Isaac Camera.set_world_pose is unavailable.")
        set_world_pose(
            position=np.asarray(camera_position_xyz, dtype=np.float32),
            orientation=np.asarray(orientation, dtype=np.float32),
        )
        self._wait_for_render(frame_count=10)
        rgba = getattr(self._camera, "get_rgba", lambda: None)()
        instance_map, instance_info, semantic_map, semantic_info = self._capture_segmentation_payloads()
        if rgba is None or instance_map is None:
            return None
        rgb_image = _tone_map_rgb_image(rgba)
        return RenderedSegmentationSample(
            rgb_image=rgb_image,
            segments=self._extract_segments(instance_map, instance_info, semantic_map, semantic_info),
            camera_position_xyz=tuple(float(value) for value in camera_position_xyz),
            look_at_xyz=tuple(float(value) for value in look_at_xyz),
        )


def build_yolo_seg_dataset(
    dataset_dir: str | Path,
    *,
    scene_usda_path: str | Path,
    rooms_json_path: str | Path,
    occupancy_image_path: str | Path,
    config_path: str | Path,
    train_count: int = DEFAULT_TRAIN_COUNT,
    val_count: int = DEFAULT_VAL_COUNT,
    test_count: int = DEFAULT_TEST_COUNT,
    image_width: int = DEFAULT_IMAGE_WIDTH,
    image_height: int = DEFAULT_IMAGE_HEIGHT,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    root = Path(dataset_dir).resolve()
    if int(image_width) <= 0 or int(image_height) <= 0:
        raise ValueError(f"image size must be positive, got width={image_width} height={image_height}")
    split_targets = {
        "train": int(train_count),
        "val": int(val_count),
        "test": int(test_count),
    }
    for split, count in split_targets.items():
        if count < 0:
            raise ValueError(f"{split} count must be non-negative, got {count}")
    inventory = detection._collect_scene_inventory(scene_usda_path, rooms_json_path=rooms_json_path)
    scene_objects = list(inventory.retained_objects)
    if not scene_objects:
        raise RuntimeError(f"No retained scene objects were parsed from {scene_usda_path}")
    meta = MapMeta.from_config_file(config_path)
    occupancy_grid = load_occupancy_grid(occupancy_image_path, meta=meta)
    rng = random.Random(int(seed))
    metadata_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in split_targets}

    for split in split_targets:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (root / "metadata").mkdir(parents=True, exist_ok=True)

    renderer = _IsaacYoloSegRenderer(
        scene_usda_path=scene_usda_path,
        scene_objects=scene_objects,
        image_width=image_width,
        image_height=image_height,
    )
    try:
        for split, target_count in split_targets.items():
            schedule = detection._build_capture_schedule(scene_objects, sample_count=max(target_count, 1), rng=rng)
            produced_count = 0
            schedule_index = 0
            max_attempts = max(target_count * 12, 64)
            attempts = 0
            while produced_count < target_count and attempts < max_attempts:
                target_object = schedule[schedule_index % len(schedule)]
                schedule_index += 1
                attempts += 1
                camera_spec = detection._sample_camera_spec(
                    target_object=target_object,
                    meta=meta,
                    occupancy_grid=occupancy_grid,
                    rng=rng,
                )
                if camera_spec is None:
                    continue
                rendered = renderer.render_sample(
                    target_object,
                    camera_position_xyz=camera_spec.position_xyz,
                    look_at_xyz=camera_spec.look_at_xyz,
                )
                if rendered is None:
                    continue
                kept_segments = _filter_and_normalize_segments(
                    rendered,
                    image_width=int(image_width),
                    image_height=int(image_height),
                )
                if not kept_segments:
                    continue
                target_segment_count = sum(
                    _segment_matches_target_prim(str(segment["prim_path"]), target_object.prim_path)
                    for segment in kept_segments
                )
                stem = f"{split}_{produced_count:05d}"
                image_path = root / "images" / split / f"{stem}.png"
                label_path = root / "labels" / split / f"{stem}.txt"
                detection._write_image(image_path, rendered.rgb_image)
                _write_label_file(label_path, kept_segments)
                metadata_by_split[split].append(
                    {
                        "split": split,
                        "image_path": detection._relative_path(image_path, root),
                        "label_path": detection._relative_path(label_path, root),
                        "source_scene_usda": Path(scene_usda_path).resolve().as_posix(),
                        "target_class": target_object.class_name,
                        "target_prim_path": target_object.prim_path,
                        "target_room_scope": target_object.room_scope,
                        "target_visible": bool(target_segment_count > 0),
                        "target_segment_count": int(target_segment_count),
                        "camera_position_xyz": list(camera_spec.position_xyz),
                        "look_at_xyz": list(camera_spec.look_at_xyz),
                        "target_distance_m": float(camera_spec.target_distance_m),
                        "pitch_deg": float(camera_spec.pitch_deg),
                        "yaw_jitter_deg": float(camera_spec.yaw_jitter_deg),
                        "segments": kept_segments,
                    }
                )
                produced_count += 1
            if produced_count != target_count:
                raise RuntimeError(
                    f"Dataset split quota shortfall for {split}: expected={target_count} produced={produced_count} "
                    f"after {attempts} attempts"
                )
    finally:
        if not bool(getattr(renderer, "_defer_close_until_process_exit", False)):
            close = getattr(renderer, "close", None)
            if callable(close):
                close()

    for split, records in metadata_by_split.items():
        detection._write_jsonl(root / "metadata" / f"{split}.jsonl", records)

    detection._write_classes_file(root)
    detection._write_data_yaml(root)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_dir": root.as_posix(),
        "source_scene_usda": Path(scene_usda_path).resolve().as_posix(),
        "source_rooms_json": Path(rooms_json_path).resolve().as_posix(),
        "source_occupancy_image": Path(occupancy_image_path).resolve().as_posix(),
        "source_config": Path(config_path).resolve().as_posix(),
        "split_counts": {split: len(records) for split, records in metadata_by_split.items()},
        "class_names": list(YOLO_CLASS_NAMES),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "seed": int(seed),
        "raw_class_frequencies": inventory.raw_class_frequencies,
        "ignored_raw_class_frequencies": inventory.ignored_raw_class_frequencies,
        "retained_class_frequencies": inventory.retained_class_frequencies,
    }
    detection._write_json(root / "manifest.json", manifest)
    _write_readme(root, manifest)
    return manifest


def validate_yolo_seg_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    root = Path(dataset_dir).resolve()
    summary: dict[str, Any] = {
        "dataset_dir": root.as_posix(),
        "manifest_errors": [],
        "data_yaml_errors": [],
        "classes_errors": [],
        "splits": {},
    }
    manifest_path = root / "manifest.json"
    manifest: dict[str, Any] = {}
    if not manifest_path.exists():
        summary["manifest_errors"].append(f"Missing manifest.json: {manifest_path.as_posix()}")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
                summary["manifest_errors"].append(
                    f"Unexpected schema_version: {manifest.get('schema_version')!r}"
                )
        except Exception as exc:  # noqa: BLE001
            summary["manifest_errors"].append(f"Failed to parse manifest.json: {type(exc).__name__}: {exc}")
    classes_path = root / "classes.txt"
    class_names: list[str] = []
    if not classes_path.exists():
        summary["classes_errors"].append(f"Missing classes.txt: {classes_path.as_posix()}")
    else:
        class_names = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip() != ""]
        if class_names != list(YOLO_CLASS_NAMES):
            summary["classes_errors"].append(
                f"classes.txt contents mismatch: expected={list(YOLO_CLASS_NAMES)!r} actual={class_names!r}"
            )
    data_yaml_path = root / "data.yaml"
    if not data_yaml_path.exists():
        summary["data_yaml_errors"].append(f"Missing data.yaml: {data_yaml_path.as_posix()}")
    else:
        try:
            data_yaml = detection._parse_data_yaml(data_yaml_path)
            expected_names = {index: name for index, name in enumerate(YOLO_CLASS_NAMES)}
            actual_data_root = detection._normalize_dataset_root_text(data_yaml.get("path", ""), base_root=root)
            expected_data_root = detection._normalize_dataset_root_text(root.as_posix())
            if actual_data_root != expected_data_root:
                summary["data_yaml_errors"].append(f"data.yaml path mismatch: {data_yaml.get('path')!r}")
            for key, expected in (("train", "images/train"), ("val", "images/val"), ("test", "images/test")):
                if data_yaml.get(key) != expected:
                    summary["data_yaml_errors"].append(f"data.yaml {key} mismatch: {data_yaml.get(key)!r}")
            if data_yaml.get("nc") != len(YOLO_CLASS_NAMES):
                summary["data_yaml_errors"].append(f"data.yaml nc mismatch: {data_yaml.get('nc')!r}")
            if data_yaml.get("names") != expected_names:
                summary["data_yaml_errors"].append(f"data.yaml names mismatch: {data_yaml.get('names')!r}")
        except Exception as exc:  # noqa: BLE001
            summary["data_yaml_errors"].append(f"Failed to parse data.yaml: {type(exc).__name__}: {exc}")

    class_count = len(YOLO_CLASS_NAMES)
    manifest_split_counts = dict(manifest.get("split_counts", {})) if isinstance(manifest, dict) else {}
    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        metadata_path = root / "metadata" / f"{split}.jsonl"
        image_files = sorted(image_dir.glob("*.png")) if image_dir.exists() else []
        label_files = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []
        image_stems = {path.stem for path in image_files}
        label_stems = {path.stem for path in label_files}
        missing_paths: list[str] = []
        orphan_images = sorted(image_stems - label_stems)
        orphan_labels = sorted(label_stems - image_stems)
        invalid_class_ids: list[str] = []
        invalid_polygon_values: list[str] = []
        malformed_label_files: list[str] = []
        invalid_metadata: list[str] = []
        metadata_records = load_jsonl_records(metadata_path) if metadata_path.exists() else []
        if not metadata_path.exists():
            missing_paths.append(metadata_path.as_posix())
        for record_index, record in enumerate(metadata_records):
            image_path = root / str(record.get("image_path", ""))
            label_path = root / str(record.get("label_path", ""))
            if not image_path.exists():
                missing_paths.append(image_path.as_posix())
            if not label_path.exists():
                missing_paths.append(label_path.as_posix())
            if str(record.get("target_class", "")) not in YOLO_CLASS_TO_ID:
                invalid_metadata.append(f"{split}:{record_index}:target_class={record.get('target_class')!r}")
        for label_path in label_files:
            lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip() != ""]
            if not lines:
                malformed_label_files.append(f"{label_path.as_posix()}:empty")
                continue
            for line_index, line in enumerate(lines, start=1):
                parts = line.split()
                if len(parts) < 7 or len(parts) % 2 == 0:
                    malformed_label_files.append(f"{label_path.as_posix()}:{line_index}:fields={len(parts)}")
                    continue
                try:
                    class_id = int(parts[0])
                except Exception:  # noqa: BLE001
                    invalid_class_ids.append(f"{label_path.as_posix()}:{line_index}:class_id={parts[0]!r}")
                    continue
                if not (0 <= class_id < class_count):
                    invalid_class_ids.append(f"{label_path.as_posix()}:{line_index}:class_id={class_id}")
                try:
                    coord_values = [float(value) for value in parts[1:]]
                except Exception:  # noqa: BLE001
                    invalid_polygon_values.append(f"{label_path.as_posix()}:{line_index}:non_float")
                    continue
                if any(not (0.0 <= value <= 1.0) for value in coord_values):
                    invalid_polygon_values.append(f"{label_path.as_posix()}:{line_index}:range={tuple(coord_values)!r}")
                    continue
                points_xy = tuple((coord_values[index], coord_values[index + 1]) for index in range(0, len(coord_values), 2))
                if len(points_xy) < 3:
                    invalid_polygon_values.append(f"{label_path.as_posix()}:{line_index}:points={len(points_xy)}")
                    continue
                if _polygon_area_xy(tuple((int(x * 1_000_000), int(y * 1_000_000)) for x, y in points_xy)) <= 0.0:
                    invalid_polygon_values.append(f"{label_path.as_posix()}:{line_index}:degenerate")
        expected_records = manifest_split_counts.get(split)
        count_mismatch = None
        if expected_records is not None and int(expected_records) != len(image_files):
            count_mismatch = f"manifest={expected_records} actual_images={len(image_files)}"
        summary["splits"][split] = {
            "images": len(image_files),
            "labels": len(label_files),
            "metadata_records": len(metadata_records),
            "missing_paths": missing_paths,
            "orphan_images": orphan_images,
            "orphan_labels": orphan_labels,
            "invalid_class_ids": invalid_class_ids,
            "invalid_polygon_values": invalid_polygon_values,
            "malformed_label_files": malformed_label_files,
            "invalid_metadata": invalid_metadata,
            "count_mismatch": count_mismatch,
        }
    return summary


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and validate YOLO-seg datasets from USDA scenes.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build a YOLO-seg dataset from a USDA scene.")
    build_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    build_parser.add_argument("--scene-usda", type=Path, default=DEFAULT_SCENE_USDA_PATH)
    build_parser.add_argument("--rooms-json", type=Path, default=DEFAULT_ROOMS_JSON_PATH)
    build_parser.add_argument("--occupancy-image", type=Path, default=DEFAULT_OCCUPANCY_IMAGE_PATH)
    build_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    build_parser.add_argument("--train-count", type=int, default=DEFAULT_TRAIN_COUNT)
    build_parser.add_argument("--val-count", type=int, default=DEFAULT_VAL_COUNT)
    build_parser.add_argument("--test-count", type=int, default=DEFAULT_TEST_COUNT)
    build_parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH)
    build_parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    build_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    validate_parser = subparsers.add_parser("validate", help="Validate an existing YOLO-seg dataset.")
    validate_parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_cli()
    args = parser.parse_args(argv)
    if args.command == "build":
        manifest = build_yolo_seg_dataset(
            args.output_dir,
            scene_usda_path=args.scene_usda,
            rooms_json_path=args.rooms_json,
            occupancy_image_path=args.occupancy_image,
            config_path=args.config,
            train_count=args.train_count,
            val_count=args.val_count,
            test_count=args.test_count,
            image_width=args.image_width,
            image_height=args.image_height,
            seed=args.seed,
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0
    if args.command == "validate":
        summary = validate_yolo_seg_dataset(args.dataset_dir)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        has_errors = bool(summary["manifest_errors"] or summary["data_yaml_errors"] or summary["classes_errors"])
        for split_summary in summary["splits"].values():
            if any(
                split_summary[key]
                for key in (
                    "missing_paths",
                    "orphan_images",
                    "orphan_labels",
                    "invalid_class_ids",
                    "invalid_polygon_values",
                    "malformed_label_files",
                    "invalid_metadata",
                )
            ) or split_summary["count_mismatch"] is not None:
                has_errors = True
                break
        return 1 if has_errors else 0
    parser.error(f"Unsupported command: {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
