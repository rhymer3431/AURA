from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from runtime.global_route import MapMeta


@dataclass(frozen=True)
class OccupancySceneAsset:
    canonical_scene_preset: str
    label: str
    image_relpath: tuple[str, ...]
    config_relpath: tuple[str, ...]

    def image_path(self, repo_root: Path) -> Path:
        return repo_root.joinpath(*self.image_relpath).resolve()

    def config_path(self, repo_root: Path) -> Path:
        return repo_root.joinpath(*self.config_relpath).resolve()


_KUJIALE3 = OccupancySceneAsset(
    canonical_scene_preset="interior agent kujiale 3",
    label="Interior Agent Kujiale 3",
    image_relpath=("datasets", "InteriorAgent", "kujiale_0003", "occupancy map.png"),
    config_relpath=("datasets", "InteriorAgent", "kujiale_0003", "config.txt"),
)

_SCENE_ALIASES: dict[str, OccupancySceneAsset] = {
    "interior agent kujiale 3": _KUJIALE3,
    "interioragent kujiale 3": _KUJIALE3,
    "interioragent_kujiale3": _KUJIALE3,
    "interioragent-kujiale3": _KUJIALE3,
    "kujiale_0003": _KUJIALE3,
}


def normalize_scene_preset(value: str | None) -> str:
    return str(value or "").strip().lower()


def resolve_occupancy_scene_asset(scene_preset: str | None) -> OccupancySceneAsset | None:
    return _SCENE_ALIASES.get(normalize_scene_preset(scene_preset))


def build_occupancy_payload(*, repo_root: Path, api_base_url: str, scene_preset: str | None) -> dict[str, Any]:
    requested = str(scene_preset or "").strip()
    asset = resolve_occupancy_scene_asset(scene_preset)
    if asset is None:
        return {
            "available": False,
            "scenePreset": requested,
            "reason": "No occupancy map is registered for the selected scene preset.",
        }
    image_path = asset.image_path(repo_root)
    config_path = asset.config_path(repo_root)
    if not image_path.exists() or not config_path.exists():
        return {
            "available": False,
            "scenePreset": requested or asset.canonical_scene_preset,
            "canonicalScenePreset": asset.canonical_scene_preset,
            "reason": "Occupancy map assets are missing on disk.",
        }
    meta = MapMeta.from_config_file(config_path)
    query = urlencode({"scenePreset": asset.canonical_scene_preset})
    return {
        "available": True,
        "scenePreset": requested or asset.canonical_scene_preset,
        "canonicalScenePreset": asset.canonical_scene_preset,
        "label": asset.label,
        "imagePath": f"/api/occupancy/image?{query}",
        "imageUrl": f"{api_base_url}/api/occupancy/image?{query}",
        "imageWidth": int(meta.width),
        "imageHeight": int(meta.height),
        "xMin": float(meta.x_min),
        "xMax": float(meta.x_max),
        "yMin": float(meta.y_min),
        "yMax": float(meta.y_max),
        "resolutionMpp": float(meta.res),
    }


__all__ = [
    "OccupancySceneAsset",
    "build_occupancy_payload",
    "normalize_scene_preset",
    "resolve_occupancy_scene_asset",
]
