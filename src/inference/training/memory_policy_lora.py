from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from services.memory_policy_types import MemoryPolicyLabel


_SYSTEM_MESSAGE = "You are a text-only memory policy controller. Return exactly one label."
_LABELS = tuple(label.value for label in MemoryPolicyLabel)
_TASK_FAMILIES = (
    "route_direct_vision",
    "route_memory_vision",
    "turn_left",
    "turn_right",
    "stop",
    "wait",
)
_TARGETS = (
    ("apple", "사과"),
    ("box", "상자"),
    ("chair", "의자"),
    ("bottle", "병"),
    ("door", "문"),
)
_ROOMS = ("hallway", "kitchen", "office", "storage")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text == "":
                continue
            records.append(json.loads(text))
    return records


def build_prompt_from_record(record: dict[str, Any]) -> str:
    memory_context = dict(record.get("memory_context", {}))
    scratchpad = dict(memory_context.get("scratchpad", {}))
    retrieved_lines = [
        str(item)
        for item in memory_context.get("retrieved_lines", [])
        if str(item).strip() != ""
    ]
    semantic_rule_hints = [
        str(item)
        for item in memory_context.get("semantic_rule_hints", [])
        if str(item).strip() != ""
    ]
    current_pose = dict(record.get("current_pose", {}))
    lines = [
        "You are a text-only memory policy controller for remembered-object navigation.",
        f"Instruction: {str(record.get('instruction', '')).strip()}",
        f"Target class: {str(record.get('target_class', '')).strip() or 'unknown'}",
        f"Task state: {str(record.get('task_state', '')).strip() or 'active'}",
        "Current pose: "
        + (
            "unknown"
            if not current_pose
            else "x={x:.3f}, y={y:.3f}, z={z:.3f}".format(
                x=float(current_pose.get("x", 0.0)),
                y=float(current_pose.get("y", 0.0)),
                z=float(current_pose.get("z", 0.0)),
            )
        ),
        f"Visible target now: {'yes' if bool(record.get('visible_target_now', False)) else 'no'}",
        f"Candidate count: {int(record.get('candidate_count', 0))}",
        f"Top score: {float(record.get('top_score', 0.0)):.4f}",
        f"Score gap: {float(record.get('score_gap', 0.0)):.4f}",
        f"Retrieval confidence: {float(record.get('retrieval_confidence', 0.0)):.4f}",
        f"Ambiguity: {'yes' if bool(record.get('ambiguity', False)) else 'no'}",
        "Scratchpad:",
    ]
    scratchpad_lines = []
    if str(scratchpad.get("goal_summary", "")).strip() != "":
        scratchpad_lines.append(f"Goal: {str(scratchpad.get('goal_summary', '')).strip()}")
    checked_locations = [str(item) for item in scratchpad.get("checked_locations", []) if str(item).strip() != ""]
    if checked_locations:
        scratchpad_lines.append("Checked: " + ", ".join(checked_locations[-3:]))
    if str(scratchpad.get("recent_hint", "")).strip() != "":
        scratchpad_lines.append(f"Hint: {str(scratchpad.get('recent_hint', '')).strip()}")
    if str(scratchpad.get("next_priority", "")).strip() != "":
        scratchpad_lines.append(f"Next: {str(scratchpad.get('next_priority', '')).strip()}")
    if scratchpad_lines:
        lines.extend(f"- {line}" for line in scratchpad_lines[:4])
    else:
        lines.append("- None")
    lines.append("Retrieved memory lines:")
    if retrieved_lines:
        lines.extend(f"- {line}" for line in retrieved_lines[:5])
    else:
        lines.append("- None")
    lines.append("Semantic rule hints:")
    if semantic_rule_hints:
        lines.extend(f"- {line}" for line in semantic_rule_hints[:4])
    else:
        lines.append("- None")
    lines.append("Allowed labels:")
    lines.extend(f"- {label}" for label in _LABELS)
    lines.append("Return exactly one label.")
    return "\n".join(lines)


def build_messages_from_record(record: dict[str, Any], *, include_assistant: bool = True) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_MESSAGE},
        {"role": "user", "content": build_prompt_from_record(record)},
    ]
    if include_assistant:
        messages.append({"role": "assistant", "content": str(record["label"])})
    return messages


def validate_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    root = Path(dataset_dir).resolve()
    summary: dict[str, Any] = {"dataset_dir": root.as_posix(), "splits": {}}
    for split in ("train", "val", "test"):
        path = root / f"{split}.jsonl"
        records = load_jsonl_records(path)
        invalid_labels = [record.get("label", "") for record in records if str(record.get("label", "")) not in _LABELS]
        prompt_errors: list[str] = []
        for index, record in enumerate(records):
            try:
                prompt_text = build_prompt_from_record(record)
                if not prompt_text.endswith("Return exactly one label."):
                    prompt_errors.append(f"{split}:{index}:missing_suffix")
            except Exception as exc:  # noqa: BLE001
                prompt_errors.append(f"{split}:{index}:{type(exc).__name__}:{exc}")
        summary["splits"][split] = {
            "records": len(records),
            "invalid_labels": invalid_labels,
            "prompt_errors": prompt_errors,
        }
    return summary


def build_seed_dataset(
    dataset_dir: str | Path,
    *,
    train_count: int = 24,
    val_count: int = 8,
    test_count: int = 8,
) -> dict[str, Any]:
    root = Path(dataset_dir).resolve()
    split_specs = {"train": train_count, "val": val_count, "test": test_count}
    manifest_records: dict[str, int] = {}
    for split, count in split_specs.items():
        records = [_make_record(split=split, index=index) for index in range(count)]
        _write_jsonl(root / f"{split}.jsonl", records)
        manifest_records[split] = len(records)
    manifest = {
        "schema_version": "memory_policy_lora_v1",
        "split_counts": manifest_records,
        "label_frequencies": _label_frequencies(root),
    }
    _write_json(root / "manifest.json", manifest)
    _write_readme(root, manifest)
    return manifest


def _write_readme(root: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Memory Policy LoRA Seed Dataset",
        "",
        "This synthetic seed set targets the text-only remembered-object memory policy controller.",
        "",
        "## Label Frequencies",
        "",
    ]
    for label, count in manifest["label_frequencies"].items():
        lines.append(f"- `{label}`: {count}")
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _label_frequencies(root: Path) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for split in ("train", "val", "test"):
        for record in load_jsonl_records(root / f"{split}.jsonl"):
            counter[str(record.get("label", ""))] += 1
    return dict(sorted(counter.items()))


def _make_record(*, split: str, index: int) -> dict[str, Any]:
    family = _TASK_FAMILIES[index % len(_TASK_FAMILIES)]
    target_class, ko_name = _TARGETS[index % len(_TARGETS)]
    room_id = _ROOMS[index % len(_ROOMS)]
    label = {
        "route_direct_vision": MemoryPolicyLabel.ROUTE_DIRECT_VISION.value,
        "route_memory_vision": MemoryPolicyLabel.ROUTE_MEMORY_VISION.value,
        "turn_left": MemoryPolicyLabel.TURN_LEFT.value,
        "turn_right": MemoryPolicyLabel.TURN_RIGHT.value,
        "stop": MemoryPolicyLabel.STOP.value,
        "wait": MemoryPolicyLabel.WAIT.value,
    }[family]
    visible_target_now = family in {"route_direct_vision", "stop"}
    ambiguity = family == "wait"
    score_gap = 0.05 if ambiguity else 0.36
    retrieval_confidence = {
        "route_direct_vision": 0.22,
        "route_memory_vision": 0.74,
        "turn_left": 0.82,
        "turn_right": 0.81,
        "stop": 0.89,
        "wait": 0.18,
    }[family]
    candidate_count = 2 if family in {"route_memory_vision", "turn_left", "turn_right", "wait"} else 1
    retrieved_lines = {
        "route_direct_vision": [f"{target_class} is visible in front of the robot now."],
        "route_memory_vision": [f"{target_class} seen in {room_id} near the far shelf."],
        "turn_left": [f"{target_class} seen in {room_id} on the left side."],
        "turn_right": [f"{target_class} seen in {room_id} on the right side."],
        "stop": [f"{target_class} is already at the current stopping distance."],
        "wait": [f"Two {target_class} candidates are competing near the junction."],
    }[family]
    semantic_rule_hints = {
        "route_direct_vision": [f"find:{target_class}:{room_id} | preferred_room={room_id}"],
        "route_memory_vision": [f"find:{target_class}:{room_id} | preferred_room={room_id}"],
        "turn_left": [f"find:{target_class}:{room_id} | preferred_room={room_id}, preferred_side=left"],
        "turn_right": [f"find:{target_class}:{room_id} | preferred_room={room_id}, preferred_side=right"],
        "stop": [f"find:{target_class}:{room_id} | preferred_room={room_id}"],
        "wait": [f"find:{target_class}:{room_id} | preferred_room={room_id}, ambiguity=high"],
    }[family]
    recent_hint = {
        "route_direct_vision": f"Observed {target_class} in the current scene.",
        "route_memory_vision": f"Last seen in {room_id}.",
        "turn_left": f"Recent hint says the {target_class} is on the left.",
        "turn_right": f"Recent hint says the {target_class} is on the right.",
        "stop": f"Goal is already within stopping distance.",
        "wait": "Two memory candidates are competing.",
    }[family]
    next_priority = {
        "route_direct_vision": "Prefer the live observation over stale memory if possible.",
        "route_memory_vision": "Use memory recall to route toward the remembered room.",
        "turn_left": "Use the remembered evidence to ground the next turn toward the left.",
        "turn_right": "Use the remembered evidence to ground the next turn toward the right.",
        "stop": "If the object is already reached, stop.",
        "wait": "Ask for another frame before choosing a route.",
    }[family]
    instruction = {
        "route_direct_vision": f"지금 보이는 {ko_name}로 바로 가",
        "route_memory_vision": f"아까 봤던 {ko_name}를 다시 찾아가",
        "turn_left": f"아까 봤던 {ko_name}를 다시 찾아가",
        "turn_right": f"아까 지나쳤던 {ko_name} 쪽으로 다시 가줘",
        "stop": f"찾던 {ko_name} 앞에 도착했으면 멈춰",
        "wait": f"아까 본 {ko_name}를 다시 찾고 싶어",
    }[family]
    return {
        "schema_version": "memory_policy_lora_v1",
        "sample_id": f"{split}_{index:04d}",
        "instruction": instruction,
        "target_class": target_class,
        "task_state": "active",
        "current_pose": {
            "x": round(0.5 + index * 0.1, 3),
            "y": round((index % 3) * 0.2, 3),
            "z": 0.0,
        },
        "visible_target_now": visible_target_now,
        "memory_context": {
            "scratchpad": {
                "goal_summary": f"Find {target_class}.",
                "checked_locations": [room_id],
                "recent_hint": recent_hint,
                "next_priority": next_priority,
            },
            "retrieved_lines": retrieved_lines,
            "semantic_rule_hints": semantic_rule_hints,
        },
        "candidate_count": candidate_count,
        "top_score": retrieval_confidence,
        "score_gap": score_gap,
        "retrieval_confidence": retrieval_confidence,
        "ambiguity": ambiguity,
        "label": label,
        "meta": {
            "family": family,
            "language": "ko",
            "room_id": room_id,
            "hard_negative": family in {"route_direct_vision", "wait"},
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and validate text-only memory-policy LoRA datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-seed")
    build_parser.add_argument("--output-dir", type=Path, required=True)
    build_parser.add_argument("--train-count", type=int, default=24)
    build_parser.add_argument("--val-count", type=int, default=8)
    build_parser.add_argument("--test-count", type=int, default=8)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--dataset-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "build-seed":
        manifest = build_seed_dataset(
            args.output_dir,
            train_count=int(args.train_count),
            val_count=int(args.val_count),
            test_count=int(args.test_count),
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0
    if args.command == "validate":
        validation = validate_dataset(args.dataset_dir)
        print(json.dumps(validation, indent=2, ensure_ascii=False))
        return 0
    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
