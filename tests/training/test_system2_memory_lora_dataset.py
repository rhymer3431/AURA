from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference.training.system2_memory_lora import build_messages_from_record, build_seed_dataset, load_jsonl_records, validate_dataset


def test_build_seed_dataset_and_render_messages(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "system2_memory_lora_seed"
    manifest = build_seed_dataset(dataset_dir, train_count=6, val_count=2, test_count=2, image_size=224, seed=11)

    assert manifest["split_counts"] == {"train": 6, "val": 2, "test": 2}

    validation = validate_dataset(dataset_dir)
    assert validation["splits"]["train"]["records"] == 6
    assert validation["splits"]["train"]["missing_paths"] == []

    train_records = load_jsonl_records(dataset_dir / "train.jsonl")
    record = train_records[0]
    messages = build_messages_from_record(record, dataset_dir, load_images=False, include_assistant=True)

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert any(item.get("type") == "image" for item in messages[1]["content"])
    assert messages[2]["content"][0]["text"] == record["decision_text"]
