from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference.training import usda_yolo_seg_dataset as dataset_mod


SCENE_USDA_PATH = ROOT / "datasets" / "InteriorAgent" / "kujiale_0003" / "kujiale_0003.usda"
ROOMS_JSON_PATH = ROOT / "datasets" / "InteriorAgent" / "kujiale_0003" / "rooms.json"
OCCUPANCY_IMAGE_PATH = ROOT / "datasets" / "InteriorAgent" / "kujiale_0003" / "occupancy map.png"
CONFIG_PATH = ROOT / "datasets" / "InteriorAgent" / "kujiale_0003" / "config.txt"


class _FakeIsaacYoloSegRenderer:
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
    ) -> dataset_mod.RenderedSegmentationSample | None:
        image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        image[..., 0] = 32
        image[..., 1] = 160
        image[..., 2] = 96
        segments = (
            dataset_mod.RenderedSegmentationInstance(
                class_name=target_object.class_name,
                prim_path=target_object.prim_path,
                instance_id=7,
                bbox_xyxy=(12, 16, min(self.image_width - 8, 52), min(self.image_height - 6, 56)),
                polygon_xy=((12, 16), (52, 16), (52, 56), (12, 56)),
                mask_area_px=1600,
            ),
        )
        return dataset_mod.RenderedSegmentationSample(
            rgb_image=image,
            segments=segments,
            camera_position_xyz=tuple(float(value) for value in camera_position_xyz),
            look_at_xyz=tuple(float(value) for value in look_at_xyz),
        )

    def close(self) -> None:
        return None


def test_load_scene_objects_reuses_real_kujiale_parser() -> None:
    objects = dataset_mod.load_scene_objects(SCENE_USDA_PATH, rooms_json_path=ROOMS_JSON_PATH)

    assert objects
    class_names = {item.class_name for item in objects}
    raw_class_names = {item.raw_class_name for item in objects}
    assert "chair" in class_names
    assert "table" in class_names
    assert "tv" in class_names
    assert "wall" not in raw_class_names
    assert "ceiling" not in raw_class_names


def test_polygon_helpers_build_yolo_seg_line() -> None:
    polygon_xy = ((10, 8), (42, 8), (42, 36), (10, 36))
    polygon_yolo = dataset_mod._polygon_to_yolo(polygon_xy, image_width=64, image_height=64)

    assert polygon_yolo[:4] == [10 / 64, 8 / 64, 42 / 64, 8 / 64]
    label_line = dataset_mod._label_lines_from_segments(
        [
            {
                "class_id": dataset_mod.YOLO_CLASS_TO_ID["chair"],
                "polygon_yolo": polygon_yolo,
            }
        ]
    )[0]
    assert label_line.startswith(f"{dataset_mod.YOLO_CLASS_TO_ID['chair']} ")
    assert len(label_line.split()) == 1 + len(polygon_xy) * 2


def test_segment_matches_target_prim_accepts_same_or_child_path() -> None:
    assert dataset_mod._segment_matches_target_prim("/Root/Meshes/other/door_0001", "/Root/Meshes/other/door_0001")
    assert dataset_mod._segment_matches_target_prim(
        "/Root/Meshes/other/door_0001/Meshes/door_0001",
        "/Root/Meshes/other/door_0001",
    )
    assert not dataset_mod._segment_matches_target_prim(
        "/Root/Meshes/other/door_0002/Meshes/door_0002",
        "/Root/Meshes/other/door_0001",
    )


def test_tone_map_rgb_image_applies_srgb_curve_and_clips() -> None:
    rgb = np.asarray(
        [
            [[10, 20, 30], [200, 220, 240]],
        ],
        dtype=np.uint8,
    )

    tone_mapped = dataset_mod._tone_map_rgb_image(rgb, exposure_gain=1.5)

    assert tone_mapped.dtype == np.uint8
    assert tone_mapped[0, 0].tolist() == [69, 96, 117]
    assert tone_mapped[0, 1].tolist() == [255, 255, 255]


def test_mask_to_largest_polygon_extracts_largest_component() -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[4:10, 4:10] = 1
    mask[20:44, 18:48] = 1

    polygon_and_bbox = dataset_mod._mask_to_largest_polygon(mask)

    assert polygon_and_bbox is not None
    polygon_xy, bbox_xyxy = polygon_and_bbox
    assert len(polygon_xy) >= 3
    assert bbox_xyxy == (18, 20, 48, 44)


def test_extract_segments_uses_instance_to_semantic_mapping() -> None:
    renderer = dataset_mod._IsaacYoloSegRenderer.__new__(dataset_mod._IsaacYoloSegRenderer)
    instance_map = np.zeros((64, 64), dtype=np.uint32)
    semantic_map = np.zeros((64, 64), dtype=np.uint32)
    instance_map[12:42, 10:34] = 374
    instance_map[18:36, 40:58] = 417
    semantic_map[12:42, 10:34] = 6
    semantic_map[18:36, 40:58] = 5
    instance_info = {
        "idToLabels": {
            374: "/Root/Meshes/balcony_767847/table_0000/Meshes/table_0000",
            417: "/Root/Meshes/balcony_767847/washing_machine_0000/Meshes/washing_machine_0000",
        },
        "idToSemantics": {
            374: 6,
            417: 5,
        },
    }
    semantic_info = {
        "idToLabels": {
            5: {"class": "washing_machine"},
            6: {"class": "table"},
        }
    }

    segments = renderer._extract_segments(instance_map, instance_info, semantic_map, semantic_info)

    assert len(segments) == 2
    by_instance = {segment.instance_id: segment for segment in segments}
    assert by_instance[374].class_name == "table"
    assert by_instance[374].prim_path.endswith("/Meshes/table_0000")
    assert by_instance[417].class_name == "washing_machine"
    assert by_instance[417].prim_path.endswith("/Meshes/washing_machine_0000")


def test_extract_segments_falls_back_to_semantic_map_majority_vote() -> None:
    renderer = dataset_mod._IsaacYoloSegRenderer.__new__(dataset_mod._IsaacYoloSegRenderer)
    instance_map = np.zeros((64, 64), dtype=np.uint32)
    semantic_map = np.zeros((64, 64), dtype=np.uint32)
    instance_map[8:40, 8:40] = 460
    semantic_map[8:40, 8:36] = 6
    semantic_map[8:40, 36:40] = 2
    instance_info = {
        "idToLabels": {
            460: "/Root/Meshes/balcony_767847/table_0001/Meshes/table_0001",
        }
    }
    semantic_info = {
        "idToLabels": {
            2: {"class": "window"},
            6: {"class": "table"},
        }
    }

    segments = renderer._extract_segments(instance_map, instance_info, semantic_map, semantic_info)

    assert len(segments) == 1
    assert segments[0].class_name == "table"
    assert segments[0].prim_path.endswith("/Meshes/table_0001")


def test_build_yolo_seg_dataset_with_fake_renderer_and_validate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(dataset_mod, "_IsaacYoloSegRenderer", _FakeIsaacYoloSegRenderer)
    dataset_dir = tmp_path / "kujiale_0003_yolo_seg"

    manifest = dataset_mod.build_yolo_seg_dataset(
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
    assert train_image.suffix == dataset_mod.detection.IMAGE_FILE_SUFFIX
    with Image.open(train_image) as image:
        assert image.format == "JPEG"
    assert train_metadata[0]["target_visible"] is True
    assert train_metadata[0]["target_segment_count"] == 1

    validation = dataset_mod.validate_yolo_seg_dataset(dataset_dir)
    assert validation["manifest_errors"] == []
    assert validation["data_yaml_errors"] == []
    assert validation["classes_errors"] == []
    assert validation["splits"]["train"]["missing_paths"] == []
    assert validation["splits"]["train"]["invalid_class_ids"] == []
    assert validation["splits"]["train"]["invalid_polygon_values"] == []
    assert validation["splits"]["train"]["malformed_label_files"] == []


def test_validate_yolo_seg_dataset_reports_corruptions(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(dataset_mod, "_IsaacYoloSegRenderer", _FakeIsaacYoloSegRenderer)
    dataset_dir = tmp_path / "kujiale_0003_yolo_seg"
    dataset_mod.build_yolo_seg_dataset(
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
    train_label.write_text("999 1.500000 0.500000 0.250000 0.250000 0.750000 0.500000\n", encoding="utf-8")
    (dataset_dir / "data.yaml").write_text("not: valid: yaml:\n", encoding="utf-8")

    validation = dataset_mod.validate_yolo_seg_dataset(dataset_dir)

    assert validation["data_yaml_errors"]
    assert validation["splits"]["train"]["missing_paths"]
    assert validation["splits"]["train"]["invalid_class_ids"]
    assert validation["splits"]["train"]["invalid_polygon_values"]
