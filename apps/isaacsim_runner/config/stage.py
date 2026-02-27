from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence


@dataclass(frozen=True)
class StageReferenceSpec:
    usd_path: str
    prim_path: str
    kind: str = "reference"


@dataclass
class StageLayoutConfig:
    world_prim_path: str = "/World"
    environment_prim_path: str = "/World/Environment"
    robots_prim_path: str = "/World/Robots"
    physics_scene_prim_path: str = "/World/PhysicsScene"
    key_light_prim_path: str = "/World/Environment/KeyLight"
    key_light_intensity: float = 500.0
    key_light_angle: float = 0.53
    enable_flat_grid: bool = True
    flat_grid_prim_path: Optional[str] = None
    environment_refs: list[StageReferenceSpec] = field(default_factory=list)
    object_refs: list[StageReferenceSpec] = field(default_factory=list)


def _sanitize_stage_token(token: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", str(token)).strip("_")
    return sanitized or "Ref"


def normalize_prim_path(path: str, fallback: str) -> str:
    value = str(path or "").strip()
    if not value:
        value = str(fallback or "").strip() or "/World"
    if not value.startswith("/"):
        value = f"/{value}"
    return value


def _split_reference_spec(spec: str) -> tuple[str, str]:
    # Format: "USD_PATH@/Prim/Path". If suffix after '@' does not look like a prim path,
    # keep the whole string as USD path (e.g. URIs containing user@host).
    if "@" not in spec:
        return spec, ""
    usd_candidate, prim_candidate = spec.rsplit("@", 1)
    if prim_candidate.strip().startswith("/"):
        return usd_candidate, prim_candidate
    return spec, ""


def parse_stage_reference_specs(
    specs: Optional[Sequence[str]],
    default_parent_prim: str,
    default_prefix: str,
    kind: str,
) -> list[StageReferenceSpec]:
    parent_prim = normalize_prim_path(default_parent_prim, "/World")
    out: list[StageReferenceSpec] = []
    for idx, raw in enumerate(specs or [], start=1):
        spec = str(raw or "").strip()
        if not spec:
            continue
        usd_part, prim_part = _split_reference_spec(spec)
        usd_path = str(usd_part).strip()
        if not usd_path:
            logging.warning("Skipping empty %s reference spec: %r", kind, raw)
            continue

        if str(prim_part).strip():
            prim_path = normalize_prim_path(str(prim_part).strip(), f"{parent_prim}/{default_prefix}_{idx:02d}")
        else:
            stem = _sanitize_stage_token(Path(usd_path).stem or f"{default_prefix}_{idx:02d}")
            prim_path = f"{parent_prim.rstrip('/')}/{default_prefix}_{idx:02d}_{stem}"

        out.append(StageReferenceSpec(usd_path=usd_path, prim_path=prim_path, kind=kind))
    return out


def build_stage_layout_config(args: argparse.Namespace) -> StageLayoutConfig:
    environment_prim = str(getattr(args, "stage_environment_prim", "/World/Environment"))
    if not environment_prim.startswith("/"):
        environment_prim = f"/{environment_prim}"

    object_root_default = f"{environment_prim.rstrip('/')}/Objects"
    object_root_prim = str(getattr(args, "stage_object_root_prim", object_root_default))
    if not object_root_prim.startswith("/"):
        object_root_prim = f"/{object_root_prim}"

    environment_refs = parse_stage_reference_specs(
        specs=getattr(args, "environment_ref", []) or [],
        default_parent_prim=environment_prim,
        default_prefix="EnvRef",
        kind="environment",
    )
    object_refs = parse_stage_reference_specs(
        specs=getattr(args, "object_ref", []) or [],
        default_parent_prim=object_root_prim,
        default_prefix="ObjRef",
        kind="object",
    )

    flat_grid_prim = getattr(args, "flat_grid_prim", None)
    flat_grid_prim = str(flat_grid_prim).strip() if flat_grid_prim is not None else ""
    if not flat_grid_prim:
        flat_grid_prim = None

    return StageLayoutConfig(
        world_prim_path=str(getattr(args, "stage_world_prim", "/World")),
        environment_prim_path=environment_prim,
        robots_prim_path=str(getattr(args, "stage_robots_prim", "/World/Robots")),
        physics_scene_prim_path=str(getattr(args, "stage_physics_scene_prim", "/World/PhysicsScene")),
        key_light_prim_path=str(getattr(args, "stage_key_light_prim", f"{environment_prim.rstrip('/')}/KeyLight")),
        key_light_intensity=float(getattr(args, "stage_key_light_intensity", 500.0)),
        key_light_angle=float(getattr(args, "stage_key_light_angle", 0.53)),
        enable_flat_grid=not bool(getattr(args, "disable_flat_grid", False)),
        flat_grid_prim_path=flat_grid_prim,
        environment_refs=environment_refs,
        object_refs=object_refs,
    )
