from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference.training import usda_room_scene_builder as builder_mod
from inference.training import usda_yolo_dataset as dataset_mod


SCENE_USDA_PATH = ROOT / "datasets" / "InteriorAgent" / "kujiale_0003" / "kujiale_0003.usda"
ROOMS_JSON_PATH = ROOT / "datasets" / "InteriorAgent" / "kujiale_0003" / "rooms.json"
OCCUPANCY_IMAGE_PATH = ROOT / "datasets" / "InteriorAgent" / "kujiale_0003" / "occupancy map.png"
CONFIG_PATH = ROOT / "datasets" / "InteriorAgent" / "kujiale_0003" / "config.txt"
EXPECTED_ROOM_SCOPES = [
    "bedroom_767840",
    "bedroom_3557416",
    "bathroom_767844",
    "bathroom_3479572",
    "bedroom_767842",
    "studyroom_767841",
    "livingroom_767839",
    "kitchen_914234340",
    "balcony_767847",
]


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


def _copy_source_scene(tmp_path: Path) -> tuple[Path, Path]:
    source_dir = tmp_path / "kujiale_0003"
    source_dir.mkdir(parents=True, exist_ok=True)
    copied_usda = source_dir / SCENE_USDA_PATH.name
    copied_rooms = source_dir / ROOMS_JSON_PATH.name
    copied_usda.write_text(SCENE_USDA_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    copied_rooms.write_text(ROOMS_JSON_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return copied_usda, copied_rooms


def _normalized_room_type(value: str) -> str:
    return dataset_mod._normalize_room_label(str(value).split("_", maxsplit=1)[0])


def test_discover_room_scopes_filters_expected_real_kujiale_scene() -> None:
    room_scopes = builder_mod.discover_room_scopes(SCENE_USDA_PATH)

    assert room_scopes == EXPECTED_ROOM_SCOPES


def test_build_room_scene_writes_expected_scopes_and_relative_refs(tmp_path: Path) -> None:
    copied_usda, copied_rooms = _copy_source_scene(tmp_path)

    summary = builder_mod.build_room_scenes(
        source_usda_path=copied_usda,
        source_rooms_json_path=copied_rooms,
        room_scopes=["livingroom_767839"],
    )

    output_dir = copied_usda.parent / "room_scenes"
    livingroom_scene = output_dir / "livingroom_767839.usda"
    text = livingroom_scene.read_text(encoding="utf-8")

    assert summary["output_dir"] == output_dir.resolve().as_posix()
    assert summary["selected_room_scopes"] == ["livingroom_767839"]
    assert 'def Scope "livingroom_767839"' in text
    assert 'def Scope "floor"' in text
    assert 'def Scope "wall"' in text
    assert 'def Scope "ceiling"' in text
    assert 'def "Rendering"' in text
    assert 'def Scope "other"' not in text
    assert "@../Meshes/" in text
    assert "@./Meshes/" not in text


def test_build_room_scene_preserves_bedroom_specific_polygon(tmp_path: Path) -> None:
    copied_usda, copied_rooms = _copy_source_scene(tmp_path)
    builder_mod.build_room_scenes(
        source_usda_path=copied_usda,
        source_rooms_json_path=copied_rooms,
        room_scopes=["bedroom_767842"],
    )

    generated_rooms = copied_usda.parent / "room_scenes" / "bedroom_767842.rooms.json"
    generated_payload = json.loads(generated_rooms.read_text(encoding="utf-8"))
    source_payload = json.loads(ROOMS_JSON_PATH.read_text(encoding="utf-8"))
    bedroom_payloads = [item for item in source_payload if _normalized_room_type(item["room_type"]) == "bedroom"]

    assert len(generated_payload) == 1
    assert generated_payload[0] == bedroom_payloads[2]
    assert generated_payload[0] != bedroom_payloads[0]


def test_generated_room_scene_load_scene_objects_uses_rewritten_mesh_refs(tmp_path: Path) -> None:
    copied_usda, copied_rooms = _copy_source_scene(tmp_path)
    builder_mod.build_room_scenes(
        source_usda_path=copied_usda,
        source_rooms_json_path=copied_rooms,
        room_scopes=["livingroom_767839"],
    )

    generated_scene = copied_usda.parent / "room_scenes" / "livingroom_767839.usda"
    generated_rooms = copied_usda.parent / "room_scenes" / "livingroom_767839.rooms.json"
    objects = dataset_mod.load_scene_objects(generated_scene, rooms_json_path=generated_rooms)

    assert objects
    assert {item.room_scope for item in objects} == {"livingroom_767839"}
    assert any(item.mesh_path.startswith("Meshes/") for item in objects)
    assert all(item.class_name in dataset_mod.YOLO_CLASS_TO_ID for item in objects)


def test_build_yolo_dataset_works_with_generated_room_scene(tmp_path: Path, monkeypatch) -> None:
    copied_usda, copied_rooms = _copy_source_scene(tmp_path)
    builder_mod.build_room_scenes(
        source_usda_path=copied_usda,
        source_rooms_json_path=copied_rooms,
        room_scopes=["livingroom_767839"],
    )

    generated_scene = copied_usda.parent / "room_scenes" / "livingroom_767839.usda"
    generated_rooms = copied_usda.parent / "room_scenes" / "livingroom_767839.rooms.json"
    dataset_dir = tmp_path / "kujiale_room_yolo"
    monkeypatch.setattr(dataset_mod, "_IsaacYoloRenderer", _FakeIsaacYoloRenderer)

    manifest = dataset_mod.build_yolo_dataset(
        dataset_dir,
        scene_usda_path=generated_scene,
        rooms_json_path=generated_rooms,
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
    assert manifest["source_scene_usda"] == generated_scene.resolve().as_posix()

    validation = dataset_mod.validate_yolo_dataset(dataset_dir)
    assert validation["manifest_errors"] == []
    assert validation["data_yaml_errors"] == []
    assert validation["classes_errors"] == []
    assert validation["splits"]["train"]["missing_paths"] == []
    assert validation["splits"]["train"]["invalid_class_ids"] == []
    assert validation["splits"]["train"]["invalid_bbox_values"] == []
