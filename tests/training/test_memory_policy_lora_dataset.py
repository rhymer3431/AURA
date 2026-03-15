from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference.training.memory_policy_lora import (
    build_messages_from_record,
    build_prompt_from_record,
    build_seed_dataset,
    load_jsonl_records,
    validate_dataset,
)


def test_build_seed_dataset_and_render_messages(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "memory_policy_lora_seed"
    manifest = build_seed_dataset(dataset_dir, train_count=6, val_count=2, test_count=2)

    assert manifest["split_counts"] == {"train": 6, "val": 2, "test": 2}

    validation = validate_dataset(dataset_dir)
    assert validation["splits"]["train"]["records"] == 6
    assert validation["splits"]["train"]["invalid_labels"] == []
    assert validation["splits"]["train"]["prompt_errors"] == []

    train_records = load_jsonl_records(dataset_dir / "train.jsonl")
    record = train_records[0]
    prompt_text = build_prompt_from_record(record)
    messages = build_messages_from_record(record, include_assistant=True)

    assert prompt_text.endswith("Return exactly one label.")
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == record["label"]
