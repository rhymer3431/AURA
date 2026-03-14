from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from common.cv2_compat import cv2
from memory.models import (
    KeyframeRecord,
    MemoryContextBundle,
    RetrievedMemoryLine,
    ScratchpadState,
    memory_context_from_dict,
    memory_context_to_dict,
)

_SYSTEM_MESSAGE = "You must return only a single navigation decision string."
_TASK_FAMILIES = (
    "direct_pixel",
    "memory_pixel",
    "memory_turn_left",
    "memory_turn_right",
    "stop",
    "wait",
)
_TARGETS: tuple[dict[str, str], ...] = (
    {"class_name": "box", "ko_name": "상자", "shape": "box"},
    {"class_name": "chair", "ko_name": "의자", "shape": "chair"},
    {"class_name": "bottle", "ko_name": "병", "shape": "bottle"},
    {"class_name": "door", "ko_name": "문", "shape": "door"},
)
_COLORS: tuple[dict[str, tuple[int, int, int]], ...] = (
    {"name": "blue", "bgr": (215, 122, 48)},
    {"name": "green", "bgr": (76, 166, 93)},
    {"name": "orange", "bgr": (52, 153, 242)},
    {"name": "red", "bgr": (88, 82, 226)},
)
_ROOMS: tuple[str, ...] = ("hallway", "kitchen", "office", "storage")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
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


def _image_item(path: Path, *, load_images: bool) -> dict[str, Any]:
    if load_images:
        return {"type": "image", "image": Image.open(path).convert("RGB")}
    return {"type": "image", "path": path.as_posix()}


def _scratchpad_lines(bundle: MemoryContextBundle | None) -> list[str]:
    if bundle is None or bundle.scratchpad is None:
        return []
    scratchpad = bundle.scratchpad
    lines: list[str] = []
    if scratchpad.goal_summary.strip() != "":
        lines.append(f"Goal: {scratchpad.goal_summary.strip()}")
    if scratchpad.checked_locations:
        lines.append("Checked: " + ", ".join(scratchpad.checked_locations[-3:]))
    if scratchpad.recent_hint.strip() != "":
        lines.append(f"Hint: {scratchpad.recent_hint.strip()}")
    if scratchpad.next_priority.strip() != "":
        lines.append(f"Next: {scratchpad.next_priority.strip()}")
    return lines[:4]


def _memory_lines(bundle: MemoryContextBundle | None) -> list[str]:
    if bundle is None:
        return []
    return [str(line.text).strip() for line in bundle.text_lines if str(line.text).strip() != ""][:5]


def _build_prompt_text(
    *,
    instruction: str,
    width: int,
    height: int,
    events: dict[str, Any],
    memory_context: MemoryContextBundle | None,
) -> str:
    prompt_lines = [
        "You are an autonomous navigation assistant.",
        f"Instruction: {instruction}",
        f"Events: {events}",
        f"Current image size: width={width}, height={height}.",
        "The current image is already the waypoint-selection view for navigation.",
    ]
    scratchpad_lines = _scratchpad_lines(memory_context)
    if scratchpad_lines:
        prompt_lines.append("Scratchpad:")
        prompt_lines.extend(f"- {line}" for line in scratchpad_lines)
    memory_lines = _memory_lines(memory_context)
    if memory_lines:
        prompt_lines.append("Relevant memory:")
        prompt_lines.extend(f"- {line}" for line in memory_lines)
    else:
        prompt_lines.append("Relevant memory:")
        prompt_lines.append("- None")
    prompt_lines.extend(
        [
            "Respond with exactly one of these formats:",
            "- '<y>, <x>' for a waypoint in the current image",
            "- 'STOP' if the task is complete now",
            "- '←' to request a left turn before selecting a waypoint",
            "- '→' to request a right turn before selecting a waypoint",
            "- '↓' only if you need another frame before deciding",
            "Do not output JSON.",
        ]
    )
    return "\n".join(prompt_lines)


def build_messages_from_record(
    record: dict[str, Any],
    dataset_root: str | Path,
    *,
    load_images: bool = False,
    include_assistant: bool = True,
) -> list[dict[str, Any]]:
    root = Path(dataset_root).resolve()
    current_image_path = root / str(record["current_image"])
    current_image = cv2.imread(str(current_image_path), cv2.IMREAD_COLOR)
    if current_image is None:
        raise FileNotFoundError(f"Current image not found: {current_image_path}")
    memory_context = memory_context_from_dict(record.get("memory_context"))
    prompt = _build_prompt_text(
        instruction=str(record["instruction"]),
        width=int(current_image.shape[1]),
        height=int(current_image.shape[0]),
        events=dict(record.get("events", {})),
        memory_context=memory_context,
    )

    user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    history_images = list(record.get("history_images", []))
    if history_images:
        user_content.append({"type": "text", "text": "Recent history observation:"})
        for history in history_images[-1:]:
            frame_id = int(history.get("frame_id", -1))
            path = root / str(history["path"])
            user_content.append({"type": "text", "text": f"Recent history (frame {frame_id}):"})
            user_content.append(_image_item(path, load_images=load_images))

    if memory_context is not None:
        keyframes = [record for record in memory_context.keyframes if str(record.image_path).strip() != ""][:2]
        if keyframes:
            user_content.append({"type": "text", "text": "Retrieved memory keyframes:"})
            for index, keyframe in enumerate(keyframes, start=1):
                user_content.append(
                    {"type": "text", "text": f"Memory keyframe {index}: {keyframe.summary or f'Retrieved keyframe {index}'}"}
                )
                user_content.append(_image_item(root / str(keyframe.image_path), load_images=load_images))
        crop_path = str(memory_context.crop_path).strip()
        if crop_path != "":
            user_content.append({"type": "text", "text": "Retrieved memory crop:"})
            user_content.append(_image_item(root / crop_path, load_images=load_images))

    user_content.append({"type": "text", "text": f"Current observation (frame {int(record.get('frame_id', -1))}):"})
    user_content.append(_image_item(current_image_path, load_images=load_images))

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": _SYSTEM_MESSAGE}]},
        {"role": "user", "content": user_content},
    ]
    if include_assistant:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(record["decision_text"])}],
            }
        )
    return messages


def _relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _save_image(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")
    return path.as_posix()


def _background_canvas(image_size: int, *, room_id: str, seed: int) -> np.ndarray:
    image = np.full((image_size, image_size, 3), 232, dtype=np.uint8)
    wall_color = {
        "hallway": (212, 223, 235),
        "kitchen": (215, 230, 220),
        "office": (225, 220, 236),
        "storage": (216, 216, 216),
    }.get(room_id, (226, 226, 226))
    floor_color = {
        "hallway": (128, 139, 152),
        "kitchen": (118, 141, 128),
        "office": (132, 122, 145),
        "storage": (126, 126, 126),
    }.get(room_id, (128, 128, 128))
    image[:, :] = wall_color
    floor_top = int(image_size * 0.56)
    image[floor_top:, :] = floor_color

    vanishing_y = int(image_size * 0.24)
    left_bottom = int(image_size * 0.08)
    right_bottom = int(image_size * 0.92)
    left_mid = int(image_size * 0.33)
    right_mid = int(image_size * 0.67)
    floor_poly = np.asarray(
        [
            (left_bottom, image_size - 1),
            (left_mid, vanishing_y),
            (right_mid, vanishing_y),
            (right_bottom, image_size - 1),
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(image, floor_poly, (92, 98, 108))
    cv2.line(image, (left_bottom, image_size - 1), (left_mid, vanishing_y), (245, 245, 245), 2)
    cv2.line(image, (right_bottom, image_size - 1), (right_mid, vanishing_y), (245, 245, 245), 2)
    cv2.line(image, (image_size // 2, image_size - 1), (image_size // 2, vanishing_y), (220, 220, 220), 1)

    rng = np.random.default_rng(seed)
    for index in range(3):
        y0 = floor_top + index * int(image_size * 0.11)
        jitter = int(rng.integers(-4, 5))
        cv2.line(image, (left_bottom + 12, y0 + jitter), (right_bottom - 12, y0 + jitter), (110, 116, 124), 1)

    cv2.putText(
        image,
        room_id.upper(),
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (36, 36, 36),
        2,
        cv2.LINE_AA,
    )
    return image


def _draw_navigation_target(
    image: np.ndarray,
    *,
    center: tuple[int, int],
    target_class: str,
    color_bgr: tuple[int, int, int],
    shape: str,
    label: str,
) -> None:
    x, y = center
    if shape == "box":
        cv2.rectangle(image, (x - 28, y - 24), (x + 28, y + 24), color_bgr, -1)
        cv2.rectangle(image, (x - 28, y - 24), (x + 28, y + 24), (30, 30, 30), 2)
    elif shape == "chair":
        cv2.rectangle(image, (x - 18, y - 18), (x + 18, y + 18), color_bgr, -1)
        cv2.rectangle(image, (x - 22, y - 8), (x + 22, y + 8), color_bgr, -1)
        for offset in (-18, 18):
            cv2.line(image, (x + offset, y + 18), (x + offset, y + 34), (40, 40, 40), 3)
    elif shape == "bottle":
        cv2.rectangle(image, (x - 10, y - 30), (x + 10, y + 18), color_bgr, -1)
        cv2.circle(image, (x, y - 32), 8, color_bgr, -1)
        cv2.rectangle(image, (x - 5, y - 42), (x + 5, y - 30), color_bgr, -1)
    else:
        cv2.rectangle(image, (x - 26, y - 40), (x + 26, y + 40), color_bgr, 3)
        cv2.rectangle(image, (x - 20, y - 34), (x + 20, y + 34), (215, 215, 215), -1)
    cv2.putText(image, label.upper(), (x - 34, y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (28, 28, 28), 2, cv2.LINE_AA)
    cv2.putText(
        image,
        target_class[:5].upper(),
        (x - 34, y + 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (250, 250, 250),
        1,
        cv2.LINE_AA,
    )


def _draw_waypoint_hint(image: np.ndarray, center: tuple[int, int], *, label: str = "GOAL") -> None:
    x, y = center
    cv2.circle(image, (x, y), 18, (34, 220, 255), 2)
    cv2.line(image, (x - 24, y), (x + 24, y), (34, 220, 255), 1)
    cv2.line(image, (x, y - 24), (x, y + 24), (34, 220, 255), 1)
    cv2.putText(image, label, (x - 22, y - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (34, 220, 255), 1, cv2.LINE_AA)


def _draw_arrow_panel(image: np.ndarray, direction: str) -> None:
    height, width = image.shape[:2]
    center = (width // 2, int(height * 0.70))
    if direction == "left":
        points = np.asarray(
            [(center[0] + 30, center[1] - 30), (center[0] - 24, center[1]), (center[0] + 30, center[1] + 30)],
            dtype=np.int32,
        )
    else:
        points = np.asarray(
            [(center[0] - 30, center[1] - 30), (center[0] + 24, center[1]), (center[0] - 30, center[1] + 30)],
            dtype=np.int32,
        )
    cv2.fillConvexPoly(image, points, (40, 215, 250))
    cv2.rectangle(image, (center[0] - 14, center[1] - 10), (center[0] + 14, center[1] + 10), (40, 215, 250), -1)


def _build_keyframe_image(
    image_size: int,
    *,
    room_id: str,
    target_class: str,
    color_bgr: tuple[int, int, int],
    shape: str,
    color_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    image = np.full((image_size, image_size, 3), 235, dtype=np.uint8)
    image[:, :] = {
        "hallway": (220, 229, 239),
        "kitchen": (225, 238, 227),
        "office": (233, 228, 239),
        "storage": (225, 225, 225),
    }.get(room_id, (228, 228, 228))
    center = (image_size // 2, int(image_size * 0.54))
    _draw_navigation_target(
        image,
        center=center,
        target_class=target_class,
        color_bgr=color_bgr,
        shape=shape,
        label=color_name,
    )
    cv2.putText(
        image,
        f"MEMORY {room_id.upper()}",
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (45, 45, 45),
        2,
        cv2.LINE_AA,
    )
    crop = image[max(center[1] - 72, 0) : min(center[1] + 72, image_size), max(center[0] - 72, 0) : min(center[0] + 72, image_size)].copy()
    return image, crop


def _make_instruction(task_family: str, *, ko_name: str, color_name: str) -> str:
    if task_family == "direct_pixel":
        return f"현재 화면에 보이는 {color_name} {ko_name} 쪽으로 가줘"
    if task_family == "memory_pixel":
        return f"아까 봤던 {color_name} {ko_name}가 있던 곳으로 이동해"
    if task_family == "memory_turn_left":
        return f"이전에 본 {color_name} {ko_name}를 다시 찾으러 가자"
    if task_family == "memory_turn_right":
        return f"아까 지나쳤던 {color_name} {ko_name} 쪽으로 다시 가줘"
    if task_family == "stop":
        return f"찾던 {color_name} {ko_name} 앞에 도착했으면 멈춰"
    return f"아까 본 {color_name} {ko_name}를 다시 찾고 싶어"


def _make_memory_bundle(
    *,
    instruction: str,
    task_family: str,
    room_id: str,
    target_class: str,
    color_name: str,
    keyframe_rel_path: str,
    crop_rel_path: str,
    frame_id: int,
    direction_hint: str,
) -> dict[str, Any]:
    scratchpad = ScratchpadState(
        instruction=instruction,
        planner_mode="interactive",
        task_state="active",
        task_id=f"system2-{frame_id}",
        command_id=frame_id,
        goal_summary=f"Find {color_name} {target_class}.",
        checked_locations=["entry", room_id] if task_family == "wait" else ["entry"],
        recent_hint=f"Observed {color_name} {target_class} in {room_id}.",
        next_priority=f"Use the remembered evidence to ground the next System 2 decision toward the {direction_hint}.",
        updated_at=float(frame_id),
    )
    text_lines = [
        RetrievedMemoryLine(
            text=f"{color_name} {target_class} seen in {room_id} on the {direction_hint}.",
            score=4.5,
            source_type="object_memory",
            entity_id=f"{target_class}_{frame_id:04d}",
            keyframe_id=f"kf_{frame_id:04d}",
        )
    ]
    if task_family == "wait":
        text_lines.append(
            RetrievedMemoryLine(
                text=f"Another {target_class} was also reported near the junction. Confidence is low.",
                score=2.1,
                source_type="semantic_rule",
                entity_id=f"rule_{frame_id:04d}",
            )
        )
        scratchpad = ScratchpadState(
            instruction=scratchpad.instruction,
            planner_mode=scratchpad.planner_mode,
            task_state=scratchpad.task_state,
            task_id=scratchpad.task_id,
            command_id=scratchpad.command_id,
            goal_summary=scratchpad.goal_summary,
            checked_locations=scratchpad.checked_locations,
            recent_hint="Two memory candidates are competing.",
            next_priority="Ask for another frame before choosing a waypoint.",
            updated_at=scratchpad.updated_at,
        )
    keyframe = KeyframeRecord(
        keyframe_id=f"kf_{frame_id:04d}",
        image_path=keyframe_rel_path,
        crop_paths=[crop_rel_path],
        summary=f"{color_name} {target_class} visible in {room_id}.",
        timestamp=float(frame_id),
        source_frame_id=int(frame_id) - 1,
        robot_pose=(0.0, 0.0, 0.0),
        robot_yaw_rad=0.0,
        room_id=room_id,
        focus_labels=[target_class],
        focus_object_ids=[f"{target_class}_{frame_id:04d}"],
    )
    bundle = MemoryContextBundle(
        instruction=instruction,
        scratchpad=scratchpad,
        text_lines=text_lines,
        keyframes=[keyframe],
        crop_path=crop_rel_path,
        latent_backend_hint="llama.cpp_s2_only",
    )
    return memory_context_to_dict(bundle) or {}


def _sample_record(
    *,
    root: Path,
    split: str,
    sample_index: int,
    family: str,
    rng: np.random.Generator,
    image_size: int,
) -> dict[str, Any]:
    target = _TARGETS[sample_index % len(_TARGETS)]
    color = _COLORS[sample_index % len(_COLORS)]
    room_id = _ROOMS[sample_index % len(_ROOMS)]
    frame_id = sample_index + {"train": 1000, "val": 2000, "test": 3000}[split]
    instruction = _make_instruction(family, ko_name=target["ko_name"], color_name=color["name"])
    sample_id = f"{split}_{sample_index:04d}_{family}"
    split_image_dir = root / "images" / split

    current = _background_canvas(image_size, room_id=room_id, seed=int(rng.integers(0, 1_000_000)))
    history = _background_canvas(image_size, room_id=room_id, seed=int(rng.integers(0, 1_000_000)))
    keyframe_image, crop_image = _build_keyframe_image(
        image_size=max(192, image_size),
        room_id=room_id,
        target_class=target["class_name"],
        color_bgr=color["bgr"],
        shape=target["shape"],
        color_name=color["name"],
    )

    waypoint = (
        int(rng.integers(int(image_size * 0.28), int(image_size * 0.72))),
        int(rng.integers(int(image_size * 0.50), int(image_size * 0.82))),
    )
    object_center = (
        int(np.clip(waypoint[0] + int(rng.integers(-26, 27)), 48, image_size - 48)),
        int(np.clip(waypoint[1] - int(rng.integers(4, 28)), 52, image_size - 60)),
    )
    direction_hint = "left" if object_center[0] < image_size // 2 else "right"
    decision_text = f"{waypoint[1]}, {waypoint[0]}"
    decision_mode = "pixel_goal"
    memory_required = family != "direct_pixel"

    if family == "direct_pixel":
        _draw_navigation_target(
            current,
            center=object_center,
            target_class=target["class_name"],
            color_bgr=color["bgr"],
            shape=target["shape"],
            label=color["name"],
        )
        _draw_waypoint_hint(current, waypoint)
        _draw_navigation_target(
            history,
            center=(max(object_center[0] - 18, 42), max(object_center[1] - 8, 42)),
            target_class=target["class_name"],
            color_bgr=color["bgr"],
            shape=target["shape"],
            label=color["name"],
        )
        memory_context = None
    elif family == "memory_pixel":
        doorway_x = waypoint[0]
        cv2.rectangle(current, (doorway_x - 28, int(image_size * 0.27)), (doorway_x + 28, int(image_size * 0.63)), (242, 240, 226), 2)
        cv2.rectangle(current, (doorway_x - 24, int(image_size * 0.31)), (doorway_x + 24, int(image_size * 0.63)), (130, 120, 112), -1)
        _draw_waypoint_hint(current, waypoint)
        _draw_navigation_target(
            history,
            center=object_center,
            target_class=target["class_name"],
            color_bgr=color["bgr"],
            shape=target["shape"],
            label=color["name"],
        )
        memory_context = "pending"
    elif family == "memory_turn_left":
        _draw_arrow_panel(current, "left")
        decision_text = "←"
        decision_mode = "yaw_left"
        direction_hint = "left"
        memory_context = "pending"
    elif family == "memory_turn_right":
        _draw_arrow_panel(current, "right")
        decision_text = "→"
        decision_mode = "yaw_right"
        direction_hint = "right"
        memory_context = "pending"
    elif family == "stop":
        object_center = (image_size // 2, int(image_size * 0.62))
        _draw_navigation_target(
            current,
            center=object_center,
            target_class=target["class_name"],
            color_bgr=color["bgr"],
            shape=target["shape"],
            label=color["name"],
        )
        cv2.putText(current, "ARRIVED", (image_size // 2 - 48, int(image_size * 0.22)), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (44, 199, 96), 2, cv2.LINE_AA)
        decision_text = "STOP"
        decision_mode = "stop"
        memory_context = "pending"
    else:
        cv2.rectangle(current, (int(image_size * 0.23), int(image_size * 0.42)), (int(image_size * 0.78), int(image_size * 0.90)), (136, 136, 136), -1)
        cv2.putText(current, "OCCLUDED", (int(image_size * 0.30), int(image_size * 0.66)), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (248, 248, 248), 2, cv2.LINE_AA)
        decision_text = "↓"
        decision_mode = "wait"
        memory_context = "pending"

    current_rel = _relative_path(split_image_dir / f"{sample_id}_current.jpg", root)
    history_rel = _relative_path(split_image_dir / f"{sample_id}_history_1.jpg", root)
    keyframe_rel = _relative_path(split_image_dir / f"{sample_id}_memory_1.jpg", root)
    crop_rel = _relative_path(split_image_dir / f"{sample_id}_crop.jpg", root)
    _save_image(root / current_rel, current)
    _save_image(root / history_rel, history)
    _save_image(root / keyframe_rel, keyframe_image)
    _save_image(root / crop_rel, crop_image)

    memory_bundle = None
    if memory_context is not None:
        memory_bundle = _make_memory_bundle(
            instruction=instruction,
            task_family=family,
            room_id=room_id,
            target_class=target["class_name"],
            color_name=color["name"],
            keyframe_rel_path=keyframe_rel,
            crop_rel_path=crop_rel,
            frame_id=frame_id,
            direction_hint=direction_hint,
        )

    return {
        "schema_version": "system2_memory_lora_v1",
        "sample_id": sample_id,
        "split": split,
        "frame_id": frame_id,
        "instruction": instruction,
        "task_family": family,
        "decision_text": decision_text,
        "decision_mode": decision_mode,
        "memory_required": memory_required,
        "current_image": current_rel,
        "history_images": [{"frame_id": frame_id - 1, "path": history_rel}],
        "events": {
            "task_state": "active",
            "force_s2": family.startswith("memory_"),
            "stuck": family in {"memory_turn_left", "memory_turn_right"},
            "collision_risk": False,
        },
        "memory_context": memory_bundle,
        "metadata": {
            "scene_id": f"synthetic_{room_id}",
            "room_id": room_id,
            "target_class": target["class_name"],
            "target_color": color["name"],
            "target_side": direction_hint,
        },
    }


def _dataset_card(
    *,
    split_counts: dict[str, int],
    task_counts: dict[str, int],
    image_size: int,
    seed: int,
) -> str:
    lines = [
        "# System2 Memory LoRA Seed Dataset",
        "",
        "This synthetic seed set mirrors the `System2Session` contract used by `isaac-aura`.",
        "",
        "## Contents",
        "",
        f"- image size: `{image_size}x{image_size}`",
        f"- seed: `{seed}`",
        f"- train/val/test: `{split_counts['train']}/{split_counts['val']}/{split_counts['test']}`",
        "",
        "## Task Families",
        "",
    ]
    for family in _TASK_FAMILIES:
        lines.append(f"- `{family}`: {task_counts.get(family, 0)}")
    lines.extend(
        [
            "",
            "## Schema",
            "",
            "- `instruction`: Korean natural-language navigation command",
            "- `decision_text`: expected System 2 output string",
            "- `current_image` / `history_images`: image paths relative to the dataset root",
            "- `memory_context`: serialized `MemoryContextBundle` compatible with `MemoryService.build_memory_context()`",
            "",
            "Use this seed set for smoke tests and prompt-format stabilization, then replace or extend it with Isaac/real-robot captures using the same schema.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_seed_dataset(
    output_dir: str | Path,
    *,
    train_count: int = 24,
    val_count: int = 8,
    test_count: int = 8,
    image_size: int = 320,
    seed: int = 7,
) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    split_records: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    split_counts = {"train": int(train_count), "val": int(val_count), "test": int(test_count)}
    task_counts: Counter[str] = Counter()

    for split, count in split_counts.items():
        for sample_index in range(count):
            family = _TASK_FAMILIES[(sample_index + (0 if split == "train" else 2 if split == "val" else 4)) % len(_TASK_FAMILIES)]
            record = _sample_record(
                root=root,
                split=split,
                sample_index=sample_index,
                family=family,
                rng=rng,
                image_size=int(image_size),
            )
            split_records[split].append(record)
            task_counts[family] += 1

    for split, records in split_records.items():
        _write_jsonl(root / f"{split}.jsonl", records)

    manifest = {
        "schema_version": "system2_memory_lora_v1",
        "seed": int(seed),
        "image_size": int(image_size),
        "split_counts": split_counts,
        "task_counts": dict(task_counts),
    }
    _write_json(root / "manifest.json", manifest)
    (root / "README.md").write_text(
        _dataset_card(
            split_counts=split_counts,
            task_counts=dict(task_counts),
            image_size=int(image_size),
            seed=int(seed),
        ),
        encoding="utf-8",
    )
    return manifest


def validate_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    root = Path(dataset_dir).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary: dict[str, Any] = {"manifest": manifest, "splits": {}}
    for split in ("train", "val", "test"):
        records = load_jsonl_records(root / f"{split}.jsonl")
        missing_paths: list[str] = []
        for record in records:
            current_path = root / str(record["current_image"])
            if not current_path.exists():
                missing_paths.append(current_path.as_posix())
            for history in record.get("history_images", []):
                history_path = root / str(history["path"])
                if not history_path.exists():
                    missing_paths.append(history_path.as_posix())
            memory_context = memory_context_from_dict(record.get("memory_context"))
            if memory_context is not None:
                for keyframe in memory_context.keyframes:
                    keyframe_path = root / str(keyframe.image_path)
                    if not keyframe_path.exists():
                        missing_paths.append(keyframe_path.as_posix())
                crop_path = root / str(memory_context.crop_path)
                if str(memory_context.crop_path).strip() != "" and not crop_path.exists():
                    missing_paths.append(crop_path.as_posix())
        summary["splits"][split] = {
            "records": len(records),
            "missing_paths": missing_paths,
        }
    return summary


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic dataset utilities for System2 memory-aware LoRA training.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-seed", help="Build a reproducible synthetic seed dataset.")
    build_parser.add_argument("--output-dir", type=Path, required=True)
    build_parser.add_argument("--train-count", type=int, default=24)
    build_parser.add_argument("--val-count", type=int, default=8)
    build_parser.add_argument("--test-count", type=int, default=8)
    build_parser.add_argument("--image-size", type=int, default=320)
    build_parser.add_argument("--seed", type=int, default=7)

    validate_parser = subparsers.add_parser("validate", help="Validate a generated dataset.")
    validate_parser.add_argument("--dataset-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_cli()
    args = parser.parse_args(argv)
    if args.command == "build-seed":
        manifest = build_seed_dataset(
            output_dir=args.output_dir,
            train_count=args.train_count,
            val_count=args.val_count,
            test_count=args.test_count,
            image_size=args.image_size,
            seed=args.seed,
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0
    if args.command == "validate":
        summary = validate_dataset(args.dataset_dir)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    parser.error(f"Unsupported command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
