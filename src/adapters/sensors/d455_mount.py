from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


DEFAULT_D455_ASSET_RELATIVE_PATH = "/Isaac/Sensors/Intel/RealSense/rsd455.usd"
DEFAULT_D455_MOUNT_PRIM_PATH = "/World/realsense_d455"


@dataclass(frozen=True)
class D455AssetResolution:
    assets_root: str
    asset_path: str
    exists: bool | None
    source: str
    message: str = ""


@dataclass
class D455MountReport:
    asset_path: str
    prim_path: str
    prim_exists: bool
    asset_exists: bool | None = None
    reference_added: bool = False
    child_prim_paths: list[str] = field(default_factory=list)
    depth_sensor_paths: list[str] = field(default_factory=list)
    render_product_paths: list[str] = field(default_factory=list)
    camera_prim_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "asset_path": self.asset_path,
            "prim_path": self.prim_path,
            "prim_exists": bool(self.prim_exists),
            "asset_exists": self.asset_exists,
            "reference_added": bool(self.reference_added),
            "child_prim_paths": list(self.child_prim_paths),
            "depth_sensor_paths": list(self.depth_sensor_paths),
            "render_product_paths": list(self.render_product_paths),
            "camera_prim_paths": list(self.camera_prim_paths),
            "metadata": dict(self.metadata),
        }


def resolve_assets_root(
    *,
    explicit_root: str = "",
    getter: Callable[[], str | None] | None = None,
) -> D455AssetResolution:
    normalized_root = str(explicit_root).strip()
    source = "explicit"
    message = ""
    if normalized_root == "":
        source = "isaac_api"
        try:
            resolved = getter() if getter is not None else _default_assets_root_getter()
        except Exception as exc:  # noqa: BLE001
            resolved = None
            message = f"asset root lookup failed: {type(exc).__name__}: {exc}"
        normalized_root = "/Isaac" if resolved in {None, ""} else str(resolved)
        if message == "" and resolved in {None, ""}:
            source = "fallback"
            message = "Isaac asset root API unavailable; using /Isaac fallback."
    asset_path = _join_asset_root(normalized_root, DEFAULT_D455_ASSET_RELATIVE_PATH)
    exists = _stat_asset(asset_path)
    return D455AssetResolution(
        assets_root=normalized_root,
        asset_path=asset_path,
        exists=exists,
        source=source,
        message=message,
    )


def resolve_d455_asset_path(
    *,
    asset_path: str = "",
    assets_root: str = "",
    getter: Callable[[], str | None] | None = None,
) -> D455AssetResolution:
    normalized_asset = str(asset_path).strip()
    if normalized_asset != "":
        return D455AssetResolution(
            assets_root=str(assets_root).strip(),
            asset_path=normalized_asset,
            exists=_stat_asset(normalized_asset),
            source="explicit_asset_path",
        )
    return resolve_assets_root(explicit_root=assets_root, getter=getter)


def ensure_d455_mount(
    stage,
    *,
    asset_path: str,
    prim_path: str = DEFAULT_D455_MOUNT_PRIM_PATH,
) -> D455MountReport:
    normalized_prim = str(prim_path).strip() or DEFAULT_D455_MOUNT_PRIM_PATH
    prim = _define_prim(stage, normalized_prim, "Xform")
    reference_added = _add_reference(prim, asset_path)
    child_paths = _child_prim_paths(stage, normalized_prim)
    depth_sensor_paths = [
        path for path in child_paths if any(token in path.lower() for token in ("depth", "realsense", "sensor"))
    ]
    camera_paths = [path for path in child_paths if path.rsplit("/", maxsplit=1)[-1].lower().startswith("camera")]
    return D455MountReport(
        asset_path=str(asset_path),
        prim_path=normalized_prim,
        prim_exists=_is_valid_prim(prim),
        asset_exists=_stat_asset(asset_path),
        reference_added=reference_added,
        child_prim_paths=child_paths,
        depth_sensor_paths=depth_sensor_paths,
        camera_prim_paths=camera_paths,
    )


def dump_prim_tree(stage, *, root_path: str = "/World") -> str:
    entries: list[str] = []
    try:
        iterator = stage.Traverse()
    except Exception as exc:  # noqa: BLE001
        return f"<prim-tree unavailable: {type(exc).__name__}: {exc}>"
    prefix = str(root_path).rstrip("/")
    for prim in iterator:
        if not _is_valid_prim(prim):
            continue
        path = str(prim.GetPath())
        if prefix != "" and not path.startswith(prefix):
            continue
        entries.append(path)
    return "\n".join(entries)


def _default_assets_root_getter() -> str | None:
    from isaacsim.storage.native import get_assets_root_path

    return get_assets_root_path()


def _join_asset_root(root: str, suffix: str) -> str:
    normalized_root = str(root).rstrip("/")
    normalized_suffix = str(suffix)
    if normalized_root == "":
        return normalized_suffix
    if normalized_suffix.startswith(normalized_root):
        return normalized_suffix
    if normalized_suffix.startswith("/"):
        return f"{normalized_root}{normalized_suffix}"
    return f"{normalized_root}/{normalized_suffix}"


def _stat_asset(path: str) -> bool | None:
    normalized = str(path).strip()
    if normalized == "":
        return False
    if "://" not in normalized and not normalized.startswith("/Isaac/"):
        from pathlib import Path

        return Path(normalized).exists()
    try:
        import omni.client
    except Exception:  # noqa: BLE001
        return None
    try:
        result, _ = omni.client.stat(normalized)
    except Exception:  # noqa: BLE001
        return False
    return bool(result == omni.client.Result.OK)


def _define_prim(stage, prim_path: str, type_name: str):  # noqa: ANN001
    define_prim = getattr(stage, "DefinePrim", None)
    if callable(define_prim):
        return define_prim(prim_path, type_name)
    raise RuntimeError("Stage does not support DefinePrim().")


def _add_reference(prim, asset_path: str) -> bool:  # noqa: ANN001
    try:
        refs = prim.GetReferences()
    except Exception:  # noqa: BLE001
        return False
    add_reference = getattr(refs, "AddReference", None)
    if callable(add_reference):
        add_reference(str(asset_path))
        return True
    return False


def _child_prim_paths(stage, prim_path: str) -> list[str]:
    prefix = str(prim_path).rstrip("/")
    results: list[str] = []
    try:
        iterator = stage.Traverse()
    except Exception:  # noqa: BLE001
        return results
    for prim in iterator:
        if not _is_valid_prim(prim):
            continue
        path = str(prim.GetPath())
        if path == prefix or not path.startswith(f"{prefix}/"):
            continue
        results.append(path)
    return results


def _is_valid_prim(prim) -> bool:  # noqa: ANN001
    try:
        return bool(prim is not None and prim.IsValid())
    except Exception:  # noqa: BLE001
        return False
