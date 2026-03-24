from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from runtime.global_route import MapMeta, load_occupancy_grid

YOLO_CLASS_NAMES: tuple[str, ...] = (
    "bed",
    "book",
    "cabinet",
    "chair",
    "curtain",
    "door",
    "fridge",
    "lamp",
    "mirror",
    "pillow",
    "range_hood",
    "shelf",
    "sink",
    "sofa",
    "table",
    "toilet",
    "tv",
    "vase",
    "washing_machine",
    "window",
)
YOLO_CLASS_TO_ID: dict[str, int] = {name: index for index, name in enumerate(YOLO_CLASS_NAMES)}
RAW_CLASS_TO_YOLO_CLASS: dict[str, str] = {
    "bed": "bed",
    "basin": "sink",
    "cabinet": "cabinet",
    "ceiling_light": "lamp",
    "chair": "chair",
    "chandelier": "lamp",
    "closestool": "toilet",
    "curtain": "curtain",
    "desk": "table",
    "dining_table": "table",
    "door": "door",
    "fridge": "fridge",
    "mirror": "mirror",
    "pillow": "pillow",
    "book": "book",
    "range_hood": "range_hood",
    "shelf": "shelf",
    "sofa": "sofa",
    "storage": "cabinet",
    "table": "table",
    "table_lamp": "lamp",
    "television": "tv",
    "throw_pillow": "pillow",
    "vase": "vase",
    "washing_machine": "washing_machine",
    "window": "window",
}
DEFAULT_TRAIN_COUNT = 240
DEFAULT_VAL_COUNT = 40
DEFAULT_TEST_COUNT = 40
DEFAULT_IMAGE_WIDTH = 640
DEFAULT_IMAGE_HEIGHT = 640
DEFAULT_SEED = 7
CAMERA_RADIUS_MIN_M = 1.2
CAMERA_RADIUS_MAX_M = 3.0
CAMERA_HEIGHT_M = 1.45
CAMERA_PITCH_DEG = -10.0
CAMERA_PITCH_JITTER_DEG = 7.0
CAMERA_YAW_JITTER_DEG = 25.0
CAMERA_SAMPLE_ATTEMPTS = 8
MIN_BBOX_WIDTH_PX = 12
MIN_BBOX_HEIGHT_PX = 12
MIN_BBOX_AREA_PX = 256
MAX_OCCLUSION_RATIO = 0.70
MANIFEST_SCHEMA_VERSION = "usda_yolo_v1"
_HEADER_RE = re.compile(r'^(def Scope|def Xform|over)\s+"([^"]+)"')
_REFERENCE_RE = re.compile(r"@\.\/Meshes\/(.+?)\.usd@")
_RAW_CLASS_RE = re.compile(r"^(.+)_([0-9]{4})$")
_VEC3_RE = re.compile(r"double3\s+xformOp:translate\s*=\s*\(([^)]+)\)")
_SCALE_RE = re.compile(r"float3\s+xformOp:scale\s*=\s*\(([^)]+)\)")
_QUAT_RE = re.compile(r"quatf\s+xformOp:orient\s*=\s*\(([^)]+)\)")


@dataclass(frozen=True)
class SceneObjectRecord:
    room_scope: str
    room_type: str
    room_polygon: tuple[tuple[float, float], ...]
    prim_path: str
    object_name: str
    mesh_path: str
    raw_class_name: str
    class_name: str
    translation_xyz: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    scale_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class CameraSpec:
    position_xyz: tuple[float, float, float]
    look_at_xyz: tuple[float, float, float]
    target_distance_m: float
    pitch_deg: float
    yaw_jitter_deg: float


@dataclass(frozen=True)
class RenderedBoundingBox:
    class_name: str
    prim_path: str
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    occlusion_ratio: float | None = None


@dataclass(frozen=True)
class RenderedSample:
    rgb_image: np.ndarray
    bounding_boxes: tuple[RenderedBoundingBox, ...]
    camera_position_xyz: tuple[float, float, float]
    look_at_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class _SceneInventory:
    retained_objects: tuple[SceneObjectRecord, ...]
    raw_class_frequencies: dict[str, int]
    ignored_raw_class_frequencies: dict[str, int]
    retained_class_frequencies: dict[str, int]


@dataclass(frozen=True)
class _ParsedBlock:
    kind: str
    name: str
    path: tuple[tuple[str, str], ...]
    header_text: str
    body_text: str
    body_lines: tuple[str, ...]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


DEFAULT_OUTPUT_DIR = _repo_root() / "artifacts" / "datasets" / "kujiale_0003_yolo"
DEFAULT_SCENE_USDA_PATH = _repo_root() / "datasets" / "InteriorAgent" / "kujiale_0003" / "kujiale_0003.usda"
DEFAULT_ROOMS_JSON_PATH = _repo_root() / "datasets" / "InteriorAgent" / "kujiale_0003" / "rooms.json"
DEFAULT_OCCUPANCY_IMAGE_PATH = _repo_root() / "datasets" / "InteriorAgent" / "kujiale_0003" / "occupancy map.png"
DEFAULT_CONFIG_PATH = _repo_root() / "datasets" / "InteriorAgent" / "kujiale_0003" / "config.txt"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def _debug_trace(event: str, **payload: Any) -> None:
    trace_path = os.environ.get("USDA_YOLO_TRACE_PATH", "").strip()
    if trace_path == "":
        return
    path = Path(trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"event": event, **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text == "":
                continue
            records.append(json.loads(text))
    return records


def _normalize_room_label(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]", "", str(value).strip().lower())
    replacements = {
        "livingroom": "livingroom",
        "living": "livingroom",
        "study": "studyroom",
        "studyroom": "studyroom",
        "bedroom": "bedroom",
        "bathroom": "bathroom",
        "kitchen": "kitchen",
        "storage": "storage",
        "balcony": "balcony",
        "unknown": "unknown",
        "other": "other",
        "wall": "wall",
        "floor": "floor",
        "ceiling": "ceiling",
    }
    return replacements.get(normalized, normalized)


def _map_raw_class_name(raw_class_name: str) -> str | None:
    return RAW_CLASS_TO_YOLO_CLASS.get(str(raw_class_name).strip())


def _parse_vector(raw: str, *, expected: int) -> tuple[float, ...]:
    parts = [segment.strip() for segment in raw.split(",") if segment.strip() != ""]
    if len(parts) != expected:
        raise ValueError(f"Expected {expected} values, got {len(parts)} from {raw!r}")
    return tuple(float(part) for part in parts)


def _iter_usda_blocks(lines: list[str], path: tuple[tuple[str, str], ...] = ()) -> Iterable[_ParsedBlock]:
    index = 0
    total = len(lines)
    while index < total:
        stripped = lines[index].strip()
        match = _HEADER_RE.match(stripped)
        if match is None:
            index += 1
            continue
        kind_token, name = match.groups()
        kind = "Scope" if kind_token == "def Scope" else "Xform" if kind_token == "def Xform" else "Over"
        header_lines = [lines[index]]
        brace_balance = lines[index].count("{") - lines[index].count("}")
        index += 1
        while brace_balance <= 0 and index < total:
            header_lines.append(lines[index])
            brace_balance += lines[index].count("{") - lines[index].count("}")
            index += 1
        body_lines: list[str] = []
        while index < total:
            line = lines[index]
            next_balance = brace_balance + line.count("{") - line.count("}")
            if next_balance <= 0:
                brace_balance = next_balance
                index += 1
                break
            body_lines.append(line)
            brace_balance = next_balance
            index += 1
        block = _ParsedBlock(
            kind=kind,
            name=name,
            path=path,
            header_text="\n".join(header_lines),
            body_text="\n".join(body_lines),
            body_lines=tuple(body_lines),
        )
        yield block
        yield from _iter_usda_blocks(list(body_lines), path=path + ((kind, name),))


def _room_scope_name_from_path(path: tuple[tuple[str, str], ...]) -> str | None:
    names = [name for _, name in path]
    if len(names) >= 3 and names[:2] == ["Root", "Meshes"]:
        return names[2]
    return None


def _is_top_level_room_scope(block: _ParsedBlock) -> bool:
    names = [name for _, name in block.path]
    return block.kind == "Scope" and names == ["Root", "Meshes"]


def _load_rooms_by_type(rooms_json_path: str | Path | None) -> dict[str, list[tuple[tuple[float, float], ...]]]:
    if rooms_json_path is None:
        return {}
    path = Path(rooms_json_path)
    if not path.exists():
        raise FileNotFoundError(f"rooms.json was not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"rooms.json must contain a list: {path}")
    grouped: dict[str, list[tuple[tuple[float, float], ...]]] = defaultdict(list)
    for item in payload:
        if not isinstance(item, dict):
            continue
        room_type = _normalize_room_label(str(item.get("room_type", "")).split("_", maxsplit=1)[0])
        polygon_payload = item.get("polygon", [])
        polygon: list[tuple[float, float]] = []
        if isinstance(polygon_payload, list):
            for point in polygon_payload:
                if not isinstance(point, list | tuple) or len(point) != 2:
                    continue
                polygon.append((float(point[0]), float(point[1])))
        if len(polygon) >= 3:
            grouped[room_type].append(tuple(polygon))
    return grouped


def _collect_scene_inventory(scene_usda_path: str | Path, *, rooms_json_path: str | Path | None = None) -> _SceneInventory:
    scene_path = Path(scene_usda_path)
    lines = scene_path.read_text(encoding="utf-8").splitlines()
    grouped_room_polygons = _load_rooms_by_type(rooms_json_path)
    room_assignment_counts: Counter[str] = Counter()
    room_scope_to_polygon: dict[str, tuple[tuple[float, float], ...]] = {}
    raw_class_frequencies: Counter[str] = Counter()
    ignored_raw_class_frequencies: Counter[str] = Counter()
    retained_class_frequencies: Counter[str] = Counter()
    retained_objects: list[SceneObjectRecord] = []

    for block in _iter_usda_blocks(lines):
        if _is_top_level_room_scope(block):
            room_scope = block.name
            room_type = _normalize_room_label(room_scope.split("_", maxsplit=1)[0])
            polygon_index = room_assignment_counts[room_type]
            polygons = grouped_room_polygons.get(room_type, [])
            if polygon_index < len(polygons):
                room_scope_to_polygon[room_scope] = polygons[polygon_index]
            room_assignment_counts[room_type] += 1
            continue

        if block.kind != "Xform":
            continue
        room_scope = _room_scope_name_from_path(block.path)
        if room_scope is None:
            continue
        reference_match = _REFERENCE_RE.search(block.header_text)
        if reference_match is None:
            continue
        mesh_name = reference_match.group(1)
        raw_class_match = _RAW_CLASS_RE.match(Path(mesh_name).name)
        if raw_class_match is None:
            continue
        raw_class_name = raw_class_match.group(1)
        raw_class_frequencies[raw_class_name] += 1
        class_name = _map_raw_class_name(raw_class_name)
        if class_name is None:
            ignored_raw_class_frequencies[raw_class_name] += 1
            continue

        translate_match = _VEC3_RE.search(block.body_text)
        scale_match = _SCALE_RE.search(block.body_text)
        quat_match = _QUAT_RE.search(block.body_text)
        translation = _parse_vector(translate_match.group(1), expected=3) if translate_match else (0.0, 0.0, 0.0)
        scale = _parse_vector(scale_match.group(1), expected=3) if scale_match else (1.0, 1.0, 1.0)
        orientation = _parse_vector(quat_match.group(1), expected=4) if quat_match else (1.0, 0.0, 0.0, 0.0)
        prim_path = "/" + "/".join([name for _, name in block.path] + [block.name])
        room_type = _normalize_room_label(room_scope.split("_", maxsplit=1)[0])
        room_polygon = room_scope_to_polygon.get(room_scope, ())
        retained_class_frequencies[class_name] += 1
        retained_objects.append(
            SceneObjectRecord(
                room_scope=room_scope,
                room_type=room_type,
                room_polygon=tuple(room_polygon),
                prim_path=prim_path,
                object_name=block.name,
                mesh_path=f"Meshes/{mesh_name}.usd",
                raw_class_name=raw_class_name,
                class_name=class_name,
                translation_xyz=(float(translation[0]), float(translation[1]), float(translation[2])),
                orientation_wxyz=(
                    float(orientation[0]),
                    float(orientation[1]),
                    float(orientation[2]),
                    float(orientation[3]),
                ),
                scale_xyz=(float(scale[0]), float(scale[1]), float(scale[2])),
            )
        )
    retained_objects.sort(key=lambda item: item.prim_path)
    return _SceneInventory(
        retained_objects=tuple(retained_objects),
        raw_class_frequencies=dict(sorted(raw_class_frequencies.items())),
        ignored_raw_class_frequencies=dict(sorted(ignored_raw_class_frequencies.items())),
        retained_class_frequencies=dict(sorted(retained_class_frequencies.items())),
    )


def load_scene_objects(scene_usda_path: str | Path, *, rooms_json_path: str | Path | None = None) -> list[SceneObjectRecord]:
    inventory = _collect_scene_inventory(scene_usda_path, rooms_json_path=rooms_json_path)
    return list(inventory.retained_objects)


def _point_in_polygon(point_xy: tuple[float, float], polygon: tuple[tuple[float, float], ...]) -> bool:
    x_value, y_value = float(point_xy[0]), float(point_xy[1])
    inside = False
    total = len(polygon)
    if total < 3:
        return False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        intersects = ((current_y > y_value) != (previous_y > y_value)) and (
            x_value < (previous_x - current_x) * (y_value - current_y) / max(previous_y - current_y, 1.0e-9) + current_x
        )
        if intersects:
            inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def _sample_camera_spec(
    *,
    target_object: SceneObjectRecord,
    meta: MapMeta,
    occupancy_grid: np.ndarray,
    rng: random.Random,
) -> CameraSpec | None:
    target_x, target_y, target_z = target_object.translation_xyz
    room_polygon = tuple(target_object.room_polygon)
    for _ in range(CAMERA_SAMPLE_ATTEMPTS):
        radius_m = rng.uniform(CAMERA_RADIUS_MIN_M, CAMERA_RADIUS_MAX_M)
        angle_rad = rng.uniform(0.0, math.tau)
        camera_x = float(target_x) + radius_m * math.cos(angle_rad)
        camera_y = float(target_y) + radius_m * math.sin(angle_rad)
        if room_polygon and not _point_in_polygon((camera_x, camera_y), room_polygon):
            continue
        try:
            row, col = meta.world_to_grid(camera_x, camera_y, clamp=False)
        except ValueError:
            continue
        if int(occupancy_grid[row, col]) != 0:
            continue
        yaw_jitter_deg = rng.uniform(-CAMERA_YAW_JITTER_DEG, CAMERA_YAW_JITTER_DEG)
        pitch_deg = CAMERA_PITCH_DEG + rng.uniform(-CAMERA_PITCH_JITTER_DEG, CAMERA_PITCH_JITTER_DEG)
        base_yaw = math.atan2(float(target_y) - camera_y, float(target_x) - camera_x)
        final_yaw = base_yaw + math.radians(yaw_jitter_deg)
        horizontal_distance = math.hypot(float(target_x) - camera_x, float(target_y) - camera_y)
        look_at_x = camera_x + math.cos(final_yaw) * horizontal_distance
        look_at_y = camera_y + math.sin(final_yaw) * horizontal_distance
        look_at_z = CAMERA_HEIGHT_M + math.tan(math.radians(pitch_deg)) * horizontal_distance
        look_at_z = float(np.clip(max(look_at_z, target_z), 0.25, 1.80))
        return CameraSpec(
            position_xyz=(float(camera_x), float(camera_y), float(CAMERA_HEIGHT_M)),
            look_at_xyz=(float(look_at_x), float(look_at_y), float(look_at_z)),
            target_distance_m=float(horizontal_distance),
            pitch_deg=float(pitch_deg),
            yaw_jitter_deg=float(yaw_jitter_deg),
        )
    return None


def _build_capture_schedule(
    scene_objects: list[SceneObjectRecord],
    *,
    sample_count: int,
    rng: random.Random,
) -> list[SceneObjectRecord]:
    grouped: dict[str, list[SceneObjectRecord]] = defaultdict(list)
    for item in scene_objects:
        grouped[item.class_name].append(item)
    available_classes = [class_name for class_name in YOLO_CLASS_NAMES if grouped.get(class_name)]
    if not available_classes:
        raise RuntimeError("No retained scene objects are available for dataset generation.")
    required_samples = max(int(sample_count), len(available_classes))
    schedule: list[SceneObjectRecord] = []
    while len(schedule) < required_samples:
        for class_name in available_classes:
            schedule.append(rng.choice(grouped[class_name]))
            if len(schedule) >= required_samples:
                break
    return schedule


def _normalize_rgb_image(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[-1] >= 3:
        rgb = array[..., :3]
    else:
        raise ValueError(f"RGB image must have at least 3 channels, got shape={array.shape}")
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.float32)
        max_value = float(np.nanmax(rgb)) if rgb.size > 0 else 0.0
        if max_value <= 1.5:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    return rgb


def _clamp_bbox_xyxy(
    bbox_xyxy: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    x_min, y_min, x_max, y_max = bbox_xyxy
    x0 = max(0, min(int(width) - 1, int(x_min)))
    y0 = max(0, min(int(height) - 1, int(y_min)))
    x1 = max(0, min(int(width) - 1, int(x_max)))
    y1 = max(0, min(int(height) - 1, int(y_max)))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _bbox_to_yolo(
    bbox_xyxy: tuple[int, int, int, int],
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x_min, y_min, x_max, y_max = bbox_xyxy
    width = float(x_max - x_min)
    height = float(y_max - y_min)
    center_x = float(x_min) + width * 0.5
    center_y = float(y_min) + height * 0.5
    return (
        center_x / float(image_width),
        center_y / float(image_height),
        width / float(image_width),
        height / float(image_height),
    )


def _filter_and_normalize_bboxes(
    rendered_sample: RenderedSample,
    *,
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for item in rendered_sample.bounding_boxes:
        if item.class_name not in YOLO_CLASS_TO_ID:
            continue
        if item.occlusion_ratio is not None and float(item.occlusion_ratio) > MAX_OCCLUSION_RATIO:
            continue
        clamped = _clamp_bbox_xyxy(
            (item.x_min, item.y_min, item.x_max, item.y_max),
            width=image_width,
            height=image_height,
        )
        if clamped is None:
            continue
        x_min, y_min, x_max, y_max = clamped
        width_px = int(x_max - x_min)
        height_px = int(y_max - y_min)
        area_px = int(width_px * height_px)
        if width_px < MIN_BBOX_WIDTH_PX or height_px < MIN_BBOX_HEIGHT_PX or area_px < MIN_BBOX_AREA_PX:
            continue
        center_x, center_y, width_norm, height_norm = _bbox_to_yolo(
            clamped,
            image_width=image_width,
            image_height=image_height,
        )
        kept.append(
            {
                "class_id": int(YOLO_CLASS_TO_ID[item.class_name]),
                "class_name": item.class_name,
                "prim_path": item.prim_path,
                "bbox_xyxy": [int(x_min), int(y_min), int(x_max), int(y_max)],
                "bbox_yolo": [
                    float(center_x),
                    float(center_y),
                    float(width_norm),
                    float(height_norm),
                ],
                "occlusion_ratio": None if item.occlusion_ratio is None else float(item.occlusion_ratio),
            }
        )
    return kept


def _label_lines_from_boxes(boxes: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for box in boxes:
        center_x, center_y, width_norm, height_norm = box["bbox_yolo"]
        lines.append(
            f"{int(box['class_id'])} "
            f"{float(center_x):.6f} "
            f"{float(center_y):.6f} "
            f"{float(width_norm):.6f} "
            f"{float(height_norm):.6f}"
        )
    return lines


def _write_label_file(path: Path, boxes: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_label_lines_from_boxes(boxes)) + "\n", encoding="utf-8")


def _write_image(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image_rgb, dtype=np.uint8), mode="RGB").save(path)


def _relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _write_classes_file(root: Path) -> None:
    (root / "classes.txt").write_text("\n".join(YOLO_CLASS_NAMES) + "\n", encoding="utf-8")


def _data_yaml_text(root: Path) -> str:
    lines = [
        f"path: {root.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(YOLO_CLASS_NAMES)}",
        "names:",
    ]
    for index, name in enumerate(YOLO_CLASS_NAMES):
        lines.append(f"  {index}: {name}")
    return "\n".join(lines) + "\n"


def _write_data_yaml(root: Path) -> None:
    (root / "data.yaml").write_text(_data_yaml_text(root), encoding="utf-8")


def _parse_data_yaml(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    names: dict[int, str] = {}
    in_names = False
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if raw_line.strip() == "" or raw_line.lstrip().startswith("#"):
            continue
        if in_names and raw_line.startswith("  "):
            stripped = raw_line.strip()
            if ":" not in stripped:
                raise ValueError(f"Malformed names entry at line {line_number}: {raw_line!r}")
            key_text, value_text = stripped.split(":", maxsplit=1)
            names[int(key_text.strip())] = value_text.strip().strip("'\"")
            continue
        in_names = False
        if ":" not in raw_line:
            raise ValueError(f"Malformed YAML line {line_number}: {raw_line!r}")
        key, value = raw_line.split(":", maxsplit=1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key == "names":
            in_names = True
            continue
        payload[key] = value
    payload["names"] = names
    if "nc" in payload:
        payload["nc"] = int(payload["nc"])
    return payload


def _normalize_dataset_root_text(path_text: str | Path, *, base_root: str | Path | None = None) -> str:
    text = str(path_text).strip().strip("'\"")
    if text == "":
        return ""
    text = text.replace("\\", "/")
    drive_match = re.match(r"^(?P<drive>[A-Za-z]):(?P<rest>/.*)?$", text)
    if drive_match:
        drive = drive_match.group("drive").lower()
        rest = drive_match.group("rest") or ""
        return f"/mnt/{drive}{rest}".rstrip("/")
    if text.startswith("/"):
        return text.rstrip("/")
    if base_root is None:
        return Path(text).resolve().as_posix().rstrip("/")
    return (Path(base_root).resolve() / text).resolve().as_posix().rstrip("/")


def _write_readme(root: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# USDA YOLO Dataset",
        "",
        "Synthetic object-detection dataset rendered from `datasets/InteriorAgent/kujiale_0003/kujiale_0003.usda`.",
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
            "- Labels use normalized YOLO `class_id cx cy w h` format.",
        ]
    )
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _quat_from_rotation_matrix(matrix: np.ndarray) -> tuple[float, float, float, float]:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s_value = math.sqrt(trace + 1.0) * 2.0
        w_value = 0.25 * s_value
        x_value = (m[2, 1] - m[1, 2]) / s_value
        y_value = (m[0, 2] - m[2, 0]) / s_value
        z_value = (m[1, 0] - m[0, 1]) / s_value
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s_value = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w_value = (m[2, 1] - m[1, 2]) / s_value
        x_value = 0.25 * s_value
        y_value = (m[0, 1] + m[1, 0]) / s_value
        z_value = (m[0, 2] + m[2, 0]) / s_value
    elif m[1, 1] > m[2, 2]:
        s_value = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w_value = (m[0, 2] - m[2, 0]) / s_value
        x_value = (m[0, 1] + m[1, 0]) / s_value
        y_value = 0.25 * s_value
        z_value = (m[1, 2] + m[2, 1]) / s_value
    else:
        s_value = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w_value = (m[1, 0] - m[0, 1]) / s_value
        x_value = (m[0, 2] + m[2, 0]) / s_value
        y_value = (m[1, 2] + m[2, 1]) / s_value
        z_value = 0.25 * s_value
    quat = np.asarray([w_value, x_value, y_value, z_value], dtype=np.float64)
    quat /= max(float(np.linalg.norm(quat)), 1.0e-9)
    return float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])


def _look_at_quaternion(
    *,
    camera_position_xyz: tuple[float, float, float],
    look_at_xyz: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    origin = np.asarray(camera_position_xyz, dtype=np.float64)
    target = np.asarray(look_at_xyz, dtype=np.float64)
    forward = target - origin
    norm = float(np.linalg.norm(forward))
    if norm <= 1.0e-9:
        return 1.0, 0.0, 0.0, 0.0
    forward /= norm
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(forward, world_up))) > 0.995:
        world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    z_axis = world_up - np.dot(world_up, forward) * forward
    z_axis /= max(float(np.linalg.norm(z_axis)), 1.0e-9)
    y_axis = np.cross(z_axis, forward)
    y_axis /= max(float(np.linalg.norm(y_axis)), 1.0e-9)
    rotation = np.stack([forward, y_axis, z_axis], axis=1)
    return _quat_from_rotation_matrix(rotation)


class _IsaacYoloRenderer:
    _defer_close_until_process_exit = True

    def __init__(
        self,
        *,
        scene_usda_path: str | Path,
        scene_objects: list[SceneObjectRecord],
        image_width: int,
        image_height: int,
    ) -> None:
        self._scene_usda_path = Path(scene_usda_path).resolve()
        self._scene_objects = list(scene_objects)
        self._image_width = int(image_width)
        self._image_height = int(image_height)
        self._sim_app = None
        self._camera = None
        self._stage = None
        self._syntheticdata = None
        self._timeline = None
        self._usd_bbox_cache = None
        self._initialized = False
        self._simulation_app_class = self._resolve_simulation_app_class()
        self._camera_class = None
        self._semantics_add = None
        self._setup()

    @staticmethod
    def _resolve_simulation_app_class():
        errors: list[str] = []
        try:
            from isaacsim import SimulationApp

            return SimulationApp
        except Exception as exc:  # noqa: BLE001
            errors.append(f"isaacsim.SimulationApp: {type(exc).__name__}: {exc}")
        try:
            from omni.isaac.kit import SimulationApp

            return SimulationApp
        except Exception as exc:  # noqa: BLE001
            errors.append(f"omni.isaac.kit.SimulationApp: {type(exc).__name__}: {exc}")
        raise RuntimeError(
            "Isaac Sim Python modules are unavailable. Run this command with Isaac Sim standalone python. "
            + " | ".join(errors)
        )

    @staticmethod
    def _enable_camera_extension() -> None:
        try:
            import omni.kit.app
        except Exception:  # noqa: BLE001
            return
        try:
            app = omni.kit.app.get_app()
            if app is None:
                return
            ext_mgr = app.get_extension_manager()
            if ext_mgr is None:
                return
            ext_mgr.set_extension_enabled_immediate("isaacsim.sensors.camera", True)
            ext_mgr.set_extension_enabled_immediate("isaacsim.core.utils", True)
        except Exception:  # noqa: BLE001
            return

    @staticmethod
    def _resolve_camera_class():
        errors: list[str] = []
        candidates = (
            ("isaacsim.sensors.camera", "Camera"),
            ("isaacsim.sensors.camera.camera", "Camera"),
            ("omni.isaac.sensor", "Camera"),
            ("omni.isaac.sensor.camera", "Camera"),
            ("omni.isaac.sensor.scripts.camera", "Camera"),
        )
        for module_name, class_name in candidates:
            try:
                module = __import__(module_name, fromlist=[class_name])
                camera_cls = getattr(module, class_name, None)
                if camera_cls is not None:
                    return camera_cls
                errors.append(f"{module_name}.{class_name}: attribute missing")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{module_name}.{class_name}: {type(exc).__name__}: {exc}")
        raise RuntimeError(
            "Isaac camera class is unavailable. Run this command with Isaac Sim standalone python. "
            + " | ".join(errors)
        )

    @staticmethod
    def _resolve_semantics_helper():
        errors: list[str] = []
        for module_name, attr_name in (
            ("isaacsim.core.utils.semantics", "add_labels"),
            ("isaacsim.core.utils.semantics", "add_update_semantics"),
            ("omni.isaac.core.utils.semantics", "add_update_semantics"),
        ):
            try:
                module = __import__(module_name, fromlist=[attr_name])
                helper = getattr(module, attr_name, None)
                if helper is not None:
                    return helper
                errors.append(f"{module_name}.{attr_name}: attribute missing")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{module_name}.{attr_name}: {type(exc).__name__}: {exc}")
        raise RuntimeError(
            "Isaac semantics helper is unavailable. Run this command with Isaac Sim standalone python. "
            + " | ".join(errors)
        )

    def _setup(self) -> None:
        _debug_trace("isaac_setup_start", scene_usda_path=self._scene_usda_path.as_posix())
        self._sim_app = self._simulation_app_class({"headless": True, "renderer": "RayTracedLighting"})
        _debug_trace("isaac_setup_after_sim_app")
        self._enable_camera_extension()
        _debug_trace("isaac_setup_after_enable_camera_extension")
        self._camera_class = self._resolve_camera_class()
        _debug_trace("isaac_setup_after_resolve_camera_class", camera_class=repr(self._camera_class))
        self._semantics_add = self._resolve_semantics_helper()
        _debug_trace("isaac_setup_after_resolve_semantics_helper", semantics_helper=repr(self._semantics_add))
        try:
            import omni.timeline
            import omni.usd
        except Exception as exc:  # noqa: BLE001
            self.close()
            raise RuntimeError(
                "Isaac Sim USD context is unavailable. Run this command with Isaac Sim standalone python."
            ) from exc
        self._timeline = omni.timeline.get_timeline_interface()
        self._syntheticdata = getattr(__import__("omni.syntheticdata", fromlist=["sensors"]), "sensors", None)
        _debug_trace("isaac_setup_after_imports")
        context = omni.usd.get_context()
        if context is None:
            self.close()
            raise RuntimeError("Isaac USD context is unavailable after SimulationApp initialization.")
        _debug_trace("isaac_setup_after_get_context")
        success = bool(context.open_stage(str(self._scene_usda_path)))
        _debug_trace("isaac_setup_after_open_stage", success=success)
        for _ in range(12):
            self._sim_app.update()
        _debug_trace("isaac_setup_after_stage_updates")
        if not success:
            self.close()
            raise RuntimeError(f"Failed to open USDA stage: {self._scene_usda_path}")
        self._stage = context.get_stage()
        if self._stage is None:
            self.close()
            raise RuntimeError(f"USD stage is not available after opening: {self._scene_usda_path}")
        try:
            from pxr import Usd, UsdGeom

            self._usd_bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            )
        except Exception:  # noqa: BLE001
            self._usd_bbox_cache = None
        _debug_trace("isaac_setup_after_get_stage")
        self._apply_semantics()
        _debug_trace("isaac_setup_after_apply_semantics")
        self._create_camera()
        _debug_trace("isaac_setup_after_create_camera")
        self._initialized = True
        _debug_trace("isaac_setup_complete")

    def _apply_semantics(self) -> None:
        assert self._stage is not None
        assert self._semantics_add is not None
        for item in self._scene_objects:
            prim = self._stage.GetPrimAtPath(item.prim_path)
            if not prim.IsValid():
                continue
            try:
                self._semantics_add(prim, labels=[item.class_name], instance_name="class", overwrite=True)
            except TypeError:
                self._semantics_add(prim, [item.class_name], "class")

    def _create_camera(self) -> None:
        assert self._stage is not None
        assert self._camera_class is not None
        camera_prim_path = "/World/YoloDatasetCamera"
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
        _debug_trace("isaac_create_camera_after_initialize")
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
        _debug_trace("isaac_create_camera_after_render_product", render_product_path=str(render_product_path))
        add_bbox = getattr(self._camera, "add_bounding_box_2d_tight_to_frame", None)
        if callable(add_bbox):
            add_bbox(init_params={"semanticTypes": ["class"]})
        resume = getattr(self._camera, "resume", None)
        if callable(resume):
            resume()
        if self._timeline is not None:
            self._timeline.play()
        for _ in range(6):
            self._sim_app.update()

    def _wait_for_render(self, frame_count: int = 10) -> None:
        if self._camera is None:
            return
        resume = getattr(self._camera, "resume", None)
        if callable(resume):
            resume()
        if self._timeline is not None:
            self._timeline.play()
        max_updates = max(8, int(frame_count) * 4)
        for _ in range(max_updates):
            self._sim_app.update()
            rgba = getattr(self._camera, "get_rgba", lambda: None)()
            array = self._coerce_array(rgba)
            if array is not None and array.ndim == 3 and array.shape[-1] >= 3 and array.size > 0:
                return

    @staticmethod
    def _coerce_array(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value
        detach = getattr(value, "detach", None)
        if callable(detach):
            try:
                value = detach()
            except Exception:  # noqa: BLE001
                pass
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            try:
                value = cpu()
            except Exception:  # noqa: BLE001
                pass
        numpy_fn = getattr(value, "numpy", None)
        if callable(numpy_fn):
            try:
                return np.asarray(numpy_fn())
            except Exception:  # noqa: BLE001
                pass
        try:
            return np.asarray(value)
        except Exception:  # noqa: BLE001
            return None

    def _capture_bbox_payload(self) -> tuple[np.ndarray | None, dict[str, Any]]:
        get_data = getattr(self._camera, "get_data", None)
        if callable(get_data):
            try:
                bbox_data, bbox_info = get_data("bounding_box_2d_tight")
                return self._coerce_array(bbox_data), dict(bbox_info or {})
            except Exception:  # noqa: BLE001
                pass
        current_frame_fn = getattr(self._camera, "get_current_frame", None)
        if callable(current_frame_fn):
            try:
                frame = current_frame_fn()
                if isinstance(frame, dict):
                    for key in ("bounding_box_2d_tight", "bbox_2d_tight"):
                        if key not in frame:
                            continue
                        payload = frame[key]
                        if isinstance(payload, dict):
                            return self._coerce_array(payload.get("data")), dict(payload.get("info", {}))
                        return self._coerce_array(payload), dict(frame.get(f"{key}_info", {}))
            except Exception:  # noqa: BLE001
                pass
        return None, {}

    @staticmethod
    def _extract_class_label(info: dict[str, Any], semantic_id: int) -> str | None:
        labels = info.get("idToLabels") or info.get("id_to_labels") or {}
        candidate = None
        if isinstance(labels, dict):
            candidate = labels.get(semantic_id)
            if candidate is None:
                candidate = labels.get(str(semantic_id))
        if candidate is None:
            return None
        if isinstance(candidate, dict):
            for key in ("class", "prim"):
                value = candidate.get(key)
                if isinstance(value, str) and value.strip() != "":
                    label = value.split(",")[-1].strip()
                    return label or None
            return None
        if isinstance(candidate, list):
            for item in candidate:
                if isinstance(item, dict) and isinstance(item.get("class"), str) and item["class"].strip() != "":
                    label = item["class"].split(",")[-1].strip()
                    return label or None
                if isinstance(item, str) and item.strip() != "":
                    label = item.split(",")[-1].strip()
                    return label or None
            return None
        if isinstance(candidate, str):
            return candidate.split(",")[-1].strip() or None
        return None

    def _project_world_bbox(self, target_object: SceneObjectRecord) -> tuple[int, int, int, int] | None:
        if self._stage is None or self._camera is None or self._usd_bbox_cache is None:
            return None
        try:
            prim = self._stage.GetPrimAtPath(target_object.prim_path)
            if not prim.IsValid():
                _debug_trace("isaac_project_bbox_invalid_prim", target_prim_path=target_object.prim_path)
                return None
            image_coords_from_world = getattr(self._camera, "get_image_coords_from_world_points", None)
            if not callable(image_coords_from_world):
                _debug_trace("isaac_project_bbox_missing_projection", target_prim_path=target_object.prim_path)
                return None
            world_bound = self._usd_bbox_cache.ComputeWorldBound(prim)
            aligned_box = world_bound.ComputeAlignedBox()
            min_point = aligned_box.GetMin()
            max_point = aligned_box.GetMax()
            corners = np.asarray(
                [
                    [min_point[0], min_point[1], min_point[2]],
                    [min_point[0], min_point[1], max_point[2]],
                    [min_point[0], max_point[1], min_point[2]],
                    [min_point[0], max_point[1], max_point[2]],
                    [max_point[0], min_point[1], min_point[2]],
                    [max_point[0], min_point[1], max_point[2]],
                    [max_point[0], max_point[1], min_point[2]],
                    [max_point[0], max_point[1], max_point[2]],
                ],
                dtype=np.float32,
            )
            image_points = np.asarray(image_coords_from_world(corners), dtype=np.float32)
            if image_points.ndim != 2 or image_points.shape[0] == 0 or image_points.shape[1] < 2:
                _debug_trace(
                    "isaac_project_bbox_invalid_points",
                    target_prim_path=target_object.prim_path,
                    image_points_shape=list(image_points.shape),
                )
                return None
            if not np.isfinite(image_points[:, :2]).all():
                _debug_trace(
                    "isaac_project_bbox_nonfinite_points",
                    target_prim_path=target_object.prim_path,
                    min_point=[float(value) for value in min_point],
                    max_point=[float(value) for value in max_point],
                )
                return None
            x_values = image_points[:, 0]
            y_values = image_points[:, 1]
            return (
                int(math.floor(float(np.min(x_values)))),
                int(math.floor(float(np.min(y_values)))),
                int(math.ceil(float(np.max(x_values)))),
                int(math.ceil(float(np.max(y_values)))),
            )
        except Exception:  # noqa: BLE001
            _debug_trace("isaac_project_bbox_exception", target_prim_path=target_object.prim_path)
            return None

    def render_sample(
        self,
        target_object: SceneObjectRecord,
        *,
        camera_position_xyz: tuple[float, float, float],
        look_at_xyz: tuple[float, float, float],
    ) -> RenderedSample | None:
        if not self._initialized or self._camera is None:
            raise RuntimeError("Isaac renderer was not initialized correctly.")
        orientation = _look_at_quaternion(camera_position_xyz=camera_position_xyz, look_at_xyz=look_at_xyz)
        set_world_pose = getattr(self._camera, "set_world_pose", None)
        if not callable(set_world_pose):
            raise RuntimeError("Isaac Camera.set_world_pose is unavailable.")
        set_world_pose(position=np.asarray(camera_position_xyz, dtype=np.float32), orientation=np.asarray(orientation, dtype=np.float32))
        self._wait_for_render(frame_count=10)
        rgba = getattr(self._camera, "get_rgba", lambda: None)()
        bbox_data, bbox_info = self._capture_bbox_payload()
        if rgba is None or bbox_data is None:
            projected_bbox = None if rgba is None else self._project_world_bbox(target_object)
            if rgba is not None and projected_bbox is not None:
                rgb_image = _normalize_rgb_image(rgba)
                _debug_trace(
                    "isaac_render_sample_projected_bbox_fallback",
                    target_prim_path=target_object.prim_path,
                    bbox_xyxy=list(projected_bbox),
                    rgb_shape=list(rgb_image.shape),
                )
                return RenderedSample(
                    rgb_image=rgb_image,
                    bounding_boxes=(
                        RenderedBoundingBox(
                            class_name=target_object.class_name,
                            prim_path=target_object.prim_path,
                            x_min=int(projected_bbox[0]),
                            y_min=int(projected_bbox[1]),
                            x_max=int(projected_bbox[2]),
                            y_max=int(projected_bbox[3]),
                            occlusion_ratio=None,
                        ),
                    ),
                    camera_position_xyz=tuple(float(value) for value in camera_position_xyz),
                    look_at_xyz=tuple(float(value) for value in look_at_xyz),
                )
            _debug_trace(
                "isaac_render_sample_empty",
                target_prim_path=target_object.prim_path,
                rgba_is_none=rgba is None,
                bbox_is_none=bbox_data is None,
                bbox_info_keys=sorted(str(key) for key in bbox_info.keys()),
            )
            return None
        rgb_image = _normalize_rgb_image(rgba)
        _debug_trace(
            "isaac_render_sample_ready",
            target_prim_path=target_object.prim_path,
            rgb_shape=list(rgb_image.shape),
            bbox_rows=int(len(bbox_data)),
        )
        boxes: list[RenderedBoundingBox] = []
        for row in bbox_data:
            semantic_id = int(row["semanticId"]) if "semanticId" in row.dtype.names else int(row[0])
            class_name = self._extract_class_label(bbox_info, semantic_id)
            if class_name is None:
                continue
            occlusion_ratio = None
            if "occlusionRatio" in row.dtype.names:
                occlusion_ratio = float(row["occlusionRatio"])
            boxes.append(
                RenderedBoundingBox(
                    class_name=class_name,
                    prim_path=target_object.prim_path if class_name == target_object.class_name else "",
                    x_min=int(row["x_min"]),
                    y_min=int(row["y_min"]),
                    x_max=int(row["x_max"]),
                    y_max=int(row["y_max"]),
                    occlusion_ratio=occlusion_ratio,
                )
            )
        return RenderedSample(
            rgb_image=rgb_image,
            bounding_boxes=tuple(boxes),
            camera_position_xyz=tuple(float(value) for value in camera_position_xyz),
            look_at_xyz=tuple(float(value) for value in look_at_xyz),
        )

    def close(self) -> None:
        camera = self._camera
        self._camera = None
        if camera is not None:
            try:
                destroy = getattr(camera, "destroy", None)
                if callable(destroy):
                    destroy()
            except Exception:  # noqa: BLE001
                pass
        app = self._sim_app
        self._sim_app = None
        if app is not None:
            try:
                app.close()
            except Exception:  # noqa: BLE001
                pass


def build_yolo_dataset(
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
    inventory = _collect_scene_inventory(scene_usda_path, rooms_json_path=rooms_json_path)
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

    renderer = _IsaacYoloRenderer(
        scene_usda_path=scene_usda_path,
        scene_objects=scene_objects,
        image_width=image_width,
        image_height=image_height,
    )
    try:
        for split, target_count in split_targets.items():
            schedule = _build_capture_schedule(scene_objects, sample_count=max(target_count, 1), rng=rng)
            produced_count = 0
            schedule_index = 0
            max_attempts = max(target_count * 12, 64)
            attempts = 0
            while produced_count < target_count and attempts < max_attempts:
                target_object = schedule[schedule_index % len(schedule)]
                schedule_index += 1
                attempts += 1
                camera_spec = _sample_camera_spec(
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
                kept_boxes = _filter_and_normalize_bboxes(
                    rendered,
                    image_width=int(image_width),
                    image_height=int(image_height),
                )
                if not kept_boxes:
                    continue
                stem = f"{split}_{produced_count:05d}"
                image_path = root / "images" / split / f"{stem}.png"
                label_path = root / "labels" / split / f"{stem}.txt"
                _write_image(image_path, rendered.rgb_image)
                _write_label_file(label_path, kept_boxes)
                metadata_by_split[split].append(
                    {
                        "split": split,
                        "image_path": _relative_path(image_path, root),
                        "label_path": _relative_path(label_path, root),
                        "source_scene_usda": Path(scene_usda_path).resolve().as_posix(),
                        "target_class": target_object.class_name,
                        "target_prim_path": target_object.prim_path,
                        "target_room_scope": target_object.room_scope,
                        "camera_position_xyz": list(camera_spec.position_xyz),
                        "look_at_xyz": list(camera_spec.look_at_xyz),
                        "target_distance_m": float(camera_spec.target_distance_m),
                        "pitch_deg": float(camera_spec.pitch_deg),
                        "yaw_jitter_deg": float(camera_spec.yaw_jitter_deg),
                        "boxes": kept_boxes,
                    }
                )
                produced_count += 1
            if produced_count != target_count:
                raise RuntimeError(
                    f"Dataset split quota shortfall for {split}: expected={target_count} produced={produced_count} "
                    f"after {attempts} attempts"
                )
    finally:
        # Isaac standalone terminates subsequent Python execution on SimulationApp.close().
        # Defer cleanup to interpreter shutdown so the dataset files can still be written.
        if not bool(getattr(renderer, "_defer_close_until_process_exit", False)):
            close = getattr(renderer, "close", None)
            if callable(close):
                close()

    for split, records in metadata_by_split.items():
        _write_jsonl(root / "metadata" / f"{split}.jsonl", records)

    _write_classes_file(root)
    _write_data_yaml(root)
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
    _write_json(root / "manifest.json", manifest)
    _write_readme(root, manifest)
    return manifest


def validate_yolo_dataset(dataset_dir: str | Path) -> dict[str, Any]:
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
            data_yaml = _parse_data_yaml(data_yaml_path)
            expected_names = {index: name for index, name in enumerate(YOLO_CLASS_NAMES)}
            actual_data_root = _normalize_dataset_root_text(data_yaml.get("path", ""), base_root=root)
            expected_data_root = _normalize_dataset_root_text(root.as_posix())
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
        invalid_bbox_values: list[str] = []
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
                if len(parts) != 5:
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
                    center_x, center_y, width_norm, height_norm = [float(value) for value in parts[1:]]
                except Exception:  # noqa: BLE001
                    invalid_bbox_values.append(f"{label_path.as_posix()}:{line_index}:non_float")
                    continue
                bbox_values = (center_x, center_y, width_norm, height_norm)
                if any(not (0.0 <= value <= 1.0) for value in bbox_values):
                    invalid_bbox_values.append(f"{label_path.as_posix()}:{line_index}:range={bbox_values!r}")
                if width_norm <= 0.0 or height_norm <= 0.0:
                    invalid_bbox_values.append(f"{label_path.as_posix()}:{line_index}:non_positive={bbox_values!r}")
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
            "invalid_bbox_values": invalid_bbox_values,
            "malformed_label_files": malformed_label_files,
            "invalid_metadata": invalid_metadata,
            "count_mismatch": count_mismatch,
        }
    return summary


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and validate YOLO datasets from USDA scenes.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build a YOLO dataset from a USDA scene.")
    build_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    build_parser.add_argument("--scene-usda", type=Path, default=DEFAULT_SCENE_USDA_PATH)
    build_parser.add_argument("--rooms-json", type=Path, default=DEFAULT_ROOMS_JSON_PATH)
    build_parser.add_argument("--occupancy-image", type=Path, default=DEFAULT_OCCUPANCY_IMAGE_PATH)
    build_parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    build_parser.add_argument("--train-count", type=int, default=DEFAULT_TRAIN_COUNT)
    build_parser.add_argument("--val-count", type=int, default=DEFAULT_VAL_COUNT)
    build_parser.add_argument("--test-count", type=int, default=DEFAULT_TEST_COUNT)
    build_parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH)
    build_parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    build_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    validate_parser = subparsers.add_parser("validate", help="Validate a generated YOLO dataset.")
    validate_parser.add_argument("--dataset-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_cli()
    args = parser.parse_args(argv)
    if args.command == "build":
        manifest = build_yolo_dataset(
            args.output_dir,
            scene_usda_path=args.scene_usda,
            rooms_json_path=args.rooms_json,
            occupancy_image_path=args.occupancy_image,
            config_path=args.config_path,
            train_count=args.train_count,
            val_count=args.val_count,
            test_count=args.test_count,
            image_width=args.image_width,
            image_height=args.image_height,
            seed=args.seed,
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default))
        return 0
    if args.command == "validate":
        summary = validate_yolo_dataset(args.dataset_dir)
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default))
        return 0
    parser.error(f"Unsupported command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
