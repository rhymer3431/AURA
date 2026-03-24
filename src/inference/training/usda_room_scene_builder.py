from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import usda_yolo_dataset as detection

ROOM_SCOPE_TYPES: tuple[str, ...] = (
    "livingroom",
    "kitchen",
    "studyroom",
    "bedroom",
    "bathroom",
    "balcony",
)
COMMON_SCOPE_NAMES: tuple[str, ...] = ("floor", "wall", "ceiling")
_MESH_ASSET_RE = re.compile(r"@[^@]*?Meshes/(?P<suffix>[^@]+?\.usd)@")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def _normalize_relative_path(path: str) -> str:
    normalized = Path(path).as_posix()
    if normalized.startswith(".") or normalized.startswith("/"):
        return normalized
    return f"./{normalized}"


def _mesh_dir_relpath(*, source_usda_path: Path, output_dir: Path) -> str:
    source_mesh_dir = source_usda_path.resolve().parent / "Meshes"
    relative_path = os.path.relpath(source_mesh_dir, start=output_dir.resolve())
    return _normalize_relative_path(relative_path)


def _room_type_from_scope_name(scope_name: str) -> str:
    return detection._normalize_room_label(str(scope_name).split("_", maxsplit=1)[0])


def _is_room_scope_name(scope_name: str) -> bool:
    return _room_type_from_scope_name(scope_name) in ROOM_SCOPE_TYPES


def _is_top_level_mesh_scope(block: detection._ParsedBlock) -> bool:
    names = [name for _, name in block.path]
    return block.kind == "Scope" and names == ["Root", "Meshes"]


def _is_rendering_block(block: detection._ParsedBlock) -> bool:
    names = [name for _, name in block.path]
    return block.kind == "Def" and names == ["Root"] and block.name == "Rendering"


def _leading_indent(line: str) -> str:
    stripped = line.lstrip()
    return line[: len(line) - len(stripped)]


def _rewrite_mesh_asset_references(text: str, *, mesh_dir_relpath: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        suffix = match.group("suffix").replace("\\", "/")
        return f"@{mesh_dir_relpath}/{suffix}@"

    return _MESH_ASSET_RE.sub(_replace, text)


def _render_block_text(block: detection._ParsedBlock, *, mesh_dir_relpath: str) -> str:
    header_lines = block.header_text.splitlines()
    if not header_lines:
        raise ValueError(f"USDA block {block.name!r} is missing header text.")
    closing_indent = _leading_indent(header_lines[0])
    lines = header_lines + list(block.body_lines) + [f"{closing_indent}}}"]
    rewritten_lines = [_rewrite_mesh_asset_references(line, mesh_dir_relpath=mesh_dir_relpath) for line in lines]
    return "\n".join(rewritten_lines)


def _room_payloads_by_type(rooms_json_path: str | Path) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(Path(rooms_json_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"rooms.json must contain a list: {rooms_json_path}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in payload:
        if not isinstance(item, dict):
            continue
        room_type = detection._normalize_room_label(str(item.get("room_type", "")).split("_", maxsplit=1)[0])
        grouped[room_type].append(item)
    return grouped


def _load_source_scene_blocks(
    source_usda_path: str | Path,
) -> tuple[dict[str, detection._ParsedBlock], list[str], detection._ParsedBlock]:
    lines = Path(source_usda_path).read_text(encoding="utf-8").splitlines()
    top_level_mesh_blocks: dict[str, detection._ParsedBlock] = {}
    room_scopes_in_source_order: list[str] = []
    rendering_block: detection._ParsedBlock | None = None

    for block in detection._iter_usda_blocks(lines):
        if _is_top_level_mesh_scope(block):
            top_level_mesh_blocks[block.name] = block
            if _is_room_scope_name(block.name):
                room_scopes_in_source_order.append(block.name)
            continue
        if _is_rendering_block(block):
            rendering_block = block

    if rendering_block is None:
        raise ValueError(f"Could not find /Root/Rendering in {source_usda_path}")

    return top_level_mesh_blocks, room_scopes_in_source_order, rendering_block


def discover_room_scopes(source_usda_path: str | Path) -> list[str]:
    _, room_scopes_in_source_order, _ = _load_source_scene_blocks(source_usda_path)
    return list(room_scopes_in_source_order)


def _room_payload_by_scope(
    room_scopes_in_source_order: list[str],
    *,
    rooms_json_path: str | Path,
) -> dict[str, dict[str, Any]]:
    grouped_payloads = _room_payloads_by_type(rooms_json_path)
    assignment_counts: Counter[str] = Counter()
    payload_by_scope: dict[str, dict[str, Any]] = {}

    for room_scope in room_scopes_in_source_order:
        room_type = _room_type_from_scope_name(room_scope)
        payload_index = assignment_counts[room_type]
        payloads = grouped_payloads.get(room_type, [])
        if payload_index >= len(payloads):
            raise ValueError(
                f"rooms.json does not contain enough polygons for {room_scope!r} "
                f"(room_type={room_type!r}, index={payload_index})."
            )
        payload_by_scope[room_scope] = payloads[payload_index]
        assignment_counts[room_type] += 1

    return payload_by_scope


def _room_scene_text(
    *,
    room_block: detection._ParsedBlock,
    common_scope_blocks: list[detection._ParsedBlock],
    rendering_block: detection._ParsedBlock,
    mesh_dir_relpath: str,
) -> str:
    sections = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "Root"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        'def Xform "Root"',
        "{",
        '    def Scope "Meshes"',
        "    {",
        _render_block_text(room_block, mesh_dir_relpath=mesh_dir_relpath),
        "",
    ]
    for scope_block in common_scope_blocks:
        sections.append(_render_block_text(scope_block, mesh_dir_relpath=mesh_dir_relpath))
        sections.append("")
    sections.extend(
        [
            "    }",
            "",
            _render_block_text(rendering_block, mesh_dir_relpath=mesh_dir_relpath),
            "}",
            "",
        ]
    )
    return "\n".join(sections)


def build_room_scenes(
    *,
    source_usda_path: str | Path,
    source_rooms_json_path: str | Path,
    output_dir: str | Path | None = None,
    room_scopes: list[str] | tuple[str, ...] | None = None,
    include_other: bool = False,
) -> dict[str, Any]:
    source_usda = Path(source_usda_path).resolve()
    source_rooms_json = Path(source_rooms_json_path).resolve()
    if output_dir is None:
        output_dir_path = source_usda.parent / "room_scenes"
    else:
        output_dir_path = Path(output_dir).resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    top_level_mesh_blocks, room_scopes_in_source_order, rendering_block = _load_source_scene_blocks(source_usda)
    available_room_scopes = list(room_scopes_in_source_order)
    requested_room_scopes = list(room_scopes) if room_scopes else list(available_room_scopes)
    unknown_room_scopes = [scope for scope in requested_room_scopes if scope not in available_room_scopes]
    if unknown_room_scopes:
        raise ValueError(
            f"Unknown room scope(s): {unknown_room_scopes!r}. "
            f"Available room scopes: {available_room_scopes!r}"
        )

    selected_room_scopes = [scope for scope in available_room_scopes if scope in set(requested_room_scopes)]
    room_payload_by_scope = _room_payload_by_scope(room_scopes_in_source_order, rooms_json_path=source_rooms_json)
    mesh_dir_relpath = _mesh_dir_relpath(source_usda_path=source_usda, output_dir=output_dir_path)

    common_scope_blocks = [top_level_mesh_blocks[name] for name in COMMON_SCOPE_NAMES if name in top_level_mesh_blocks]
    if include_other and "other" in top_level_mesh_blocks:
        common_scope_blocks.append(top_level_mesh_blocks["other"])

    generated: list[dict[str, Any]] = []
    for room_scope in selected_room_scopes:
        room_block = top_level_mesh_blocks[room_scope]
        room_scene_path = output_dir_path / f"{room_scope}.usda"
        room_rooms_json_path = output_dir_path / f"{room_scope}.rooms.json"
        room_scene_path.write_text(
            _room_scene_text(
                room_block=room_block,
                common_scope_blocks=common_scope_blocks,
                rendering_block=rendering_block,
                mesh_dir_relpath=mesh_dir_relpath,
            ),
            encoding="utf-8",
        )
        room_rooms_json_path.write_text(
            json.dumps([room_payload_by_scope[room_scope]], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        generated.append(
            {
                "room_scope": room_scope,
                "scene_usda_path": room_scene_path.resolve().as_posix(),
                "rooms_json_path": room_rooms_json_path.resolve().as_posix(),
                "included_top_level_scopes": [room_scope, *[block.name for block in common_scope_blocks]],
            }
        )

    return {
        "source_usda_path": source_usda.as_posix(),
        "source_rooms_json_path": source_rooms_json.as_posix(),
        "output_dir": output_dir_path.as_posix(),
        "available_room_scopes": available_room_scopes,
        "selected_room_scopes": selected_room_scopes,
        "include_other": bool(include_other),
        "generated": generated,
    }


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build room-scoped USDA scenes from a source USDA scene.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build room-scoped USDA scenes.")
    build_parser.add_argument("--source-usda", type=Path, required=True)
    build_parser.add_argument("--source-rooms-json", type=Path, required=True)
    build_parser.add_argument("--output-dir", type=Path)
    build_parser.add_argument("--room-scope", action="append", default=None)
    build_parser.add_argument("--include-other", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_cli()
    args = parser.parse_args(argv)
    if args.command == "build":
        summary = build_room_scenes(
            source_usda_path=args.source_usda,
            source_rooms_json_path=args.source_rooms_json,
            output_dir=args.output_dir,
            room_scopes=args.room_scope,
            include_other=args.include_other,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default))
        return 0
    parser.error(f"Unsupported command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
