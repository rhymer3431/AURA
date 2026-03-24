from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference.training import usda_yolo_dataset as dataset_mod


SCENE_USDA_PATH = ROOT / "datasets" / "InteriorAgent" / "kujiale_0003" / "kujiale_0003.usda"
ROOMS_JSON_PATH = ROOT / "datasets" / "InteriorAgent" / "kujiale_0003" / "rooms.json"
OCCUPANCY_IMAGE_PATH = ROOT / "datasets" / "InteriorAgent" / "kujiale_0003" / "occupancy map.png"
CONFIG_PATH = ROOT / "datasets" / "InteriorAgent" / "kujiale_0003" / "config.txt"


class _FakeIsaacYoloRenderer:
    def __init__(
        self,
        *,
        scene_usda_path,
        scene_objects,
        image_width,
        image_height,
    ) -> None:
        self.scene_usda_path = Path(scene_usda_path)
        self.scene_objects = list(scene_objects)
        self.image_width = int(image_width)
        self.image_height = int(image_height)

    def render_sample(
        self,
        target_object: dataset_mod.SceneObjectRecord,
        *,
        camera_position_xyz,
        look_at_xyz,
    ) -> dataset_mod.RenderedSample | None:
        image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        image[..., 1] = 96
        image[..., 2] = 192
        boxes = (
            dataset_mod.RenderedBoundingBox(
                class_name=target_object.class_name,
                prim_path=target_object.prim_path,
                x_min=16,
                y_min=20,
                x_max=min(self.image_width - 10, 52),
                y_max=min(self.image_height - 10, 56),
                occlusion_ratio=0.10,
            ),
        )
        return dataset_mod.RenderedSample(
            rgb_image=image,
            bounding_boxes=boxes,
            camera_position_xyz=tuple(float(value) for value in camera_position_xyz),
            look_at_xyz=tuple(float(value) for value in look_at_xyz),
        )

    def close(self) -> None:
        return None


def test_load_scene_objects_filters_and_maps_real_kujiale_scene() -> None:
    objects = dataset_mod.load_scene_objects(SCENE_USDA_PATH, rooms_json_path=ROOMS_JSON_PATH)

    assert objects
    class_names = {item.class_name for item in objects}
    raw_class_names = {item.raw_class_name for item in objects}
    assert "chair" in class_names
    assert "table" in class_names
    assert "washing_machine" in class_names
    assert "wall" not in raw_class_names
    assert "ceiling" not in raw_class_names
    assert all(item.class_name in dataset_mod.YOLO_CLASS_TO_ID for item in objects)
    assert any(item.room_scope.startswith("bedroom_") and len(item.room_polygon) >= 3 for item in objects)


def test_collect_scene_inventory_tracks_retained_and_ignored_counts() -> None:
    inventory = dataset_mod._collect_scene_inventory(SCENE_USDA_PATH, rooms_json_path=ROOMS_JSON_PATH)

    assert inventory.retained_class_frequencies["lamp"] == 14
    assert inventory.retained_class_frequencies["table"] == 12
    assert inventory.raw_class_frequencies["wall"] == 30
    assert inventory.raw_class_frequencies["door_handle"] == 15
    assert inventory.ignored_raw_class_frequencies["wall"] == 30
    assert inventory.ignored_raw_class_frequencies["ornament"] == 21


def test_bbox_helpers_normalize_for_yolo() -> None:
    clamped = dataset_mod._clamp_bbox_xyxy((10, 12, 42, 44), width=64, height=64)
    assert clamped == (10, 12, 42, 44)

    center_x, center_y, width_norm, height_norm = dataset_mod._bbox_to_yolo(
        clamped,
        image_width=64,
        image_height=64,
    )
    assert center_x == 26 / 64
    assert center_y == 28 / 64
    assert width_norm == 32 / 64
    assert height_norm == 32 / 64

    label_line = dataset_mod._label_lines_from_boxes(
        [
            {
                "class_id": dataset_mod.YOLO_CLASS_TO_ID["chair"],
                "bbox_yolo": [center_x, center_y, width_norm, height_norm],
            }
        ]
    )[0]
    assert label_line.startswith(f"{dataset_mod.YOLO_CLASS_TO_ID['chair']} ")


def test_normalize_dataset_root_text_handles_windows_drive_paths() -> None:
    normalized = dataset_mod._normalize_dataset_root_text(r"C:\Users\mango\project\isaac-aura\artifacts\dataset")
    assert normalized == "/mnt/c/Users/mango/project/isaac-aura/artifacts/dataset"


def test_build_yolo_dataset_with_fake_renderer_and_validate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(dataset_mod, "_IsaacYoloRenderer", _FakeIsaacYoloRenderer)
    dataset_dir = tmp_path / "kujiale_0003_yolo"

    manifest = dataset_mod.build_yolo_dataset(
        dataset_dir,
        scene_usda_path=SCENE_USDA_PATH,
        rooms_json_path=ROOMS_JSON_PATH,
        occupancy_image_path=OCCUPANCY_IMAGE_PATH,
        config_path=CONFIG_PATH,
        train_count=2,
        val_count=1,
        test_count=1,
        image_width=96,
        image_height=96,
        seed=5,
    )

    assert manifest["split_counts"] == {"train": 2, "val": 1, "test": 1}
    assert (dataset_dir / "data.yaml").exists()
    assert (dataset_dir / "classes.txt").exists()
    assert (dataset_dir / "metadata" / "train.jsonl").exists()
    train_metadata = dataset_mod.load_jsonl_records(dataset_dir / "metadata" / "train.jsonl")
    train_image = dataset_dir / train_metadata[0]["image_path"]
    assert train_image.suffix == dataset_mod.IMAGE_FILE_SUFFIX
    with Image.open(train_image) as image:
        assert image.format == "JPEG"

    validation = dataset_mod.validate_yolo_dataset(dataset_dir)
    assert validation["manifest_errors"] == []
    assert validation["data_yaml_errors"] == []
    assert validation["classes_errors"] == []
    assert validation["splits"]["train"]["missing_paths"] == []
    assert validation["splits"]["train"]["invalid_class_ids"] == []
    assert validation["splits"]["train"]["invalid_bbox_values"] == []
    assert validation["splits"]["train"]["malformed_label_files"] == []


def test_validate_yolo_dataset_reports_corruptions(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(dataset_mod, "_IsaacYoloRenderer", _FakeIsaacYoloRenderer)
    dataset_dir = tmp_path / "kujiale_0003_yolo"
    dataset_mod.build_yolo_dataset(
        dataset_dir,
        scene_usda_path=SCENE_USDA_PATH,
        rooms_json_path=ROOMS_JSON_PATH,
        occupancy_image_path=OCCUPANCY_IMAGE_PATH,
        config_path=CONFIG_PATH,
        train_count=1,
        val_count=1,
        test_count=1,
        image_width=96,
        image_height=96,
        seed=9,
    )

    train_metadata = dataset_mod.load_jsonl_records(dataset_dir / "metadata" / "train.jsonl")
    train_image = dataset_dir / train_metadata[0]["image_path"]
    train_label = dataset_dir / train_metadata[0]["label_path"]
    train_image.unlink()
    train_label.write_text("999 1.500000 0.500000 0.250000 0.250000\n", encoding="utf-8")
    (dataset_dir / "data.yaml").write_text("not: valid: yaml:\n", encoding="utf-8")

    validation = dataset_mod.validate_yolo_dataset(dataset_dir)

    assert validation["data_yaml_errors"]
    assert validation["splits"]["train"]["missing_paths"]
    assert validation["splits"]["train"]["invalid_class_ids"]
    assert validation["splits"]["train"]["invalid_bbox_values"]
