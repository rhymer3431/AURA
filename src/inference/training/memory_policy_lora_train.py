from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from inference.training.memory_policy_lora import build_messages_from_record, load_jsonl_records, validate_dataset


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Training config must be a mapping: {path}")
    return payload


def _resolved_config(config_path: Path | None, overrides: argparse.Namespace) -> dict[str, Any]:
    config = _load_yaml(config_path) if config_path is not None else {}
    lora_config = dict(config.get("lora", {}))

    def set_if(name: str, key: str) -> None:
        value = getattr(overrides, name, None)
        if value not in (None, ""):
            config[key] = value

    set_if("dataset_dir", "dataset_dir")
    set_if("model_name_or_path", "model_name_or_path")
    set_if("output_dir", "output_dir")
    if overrides.max_length is not None:
        config["max_length"] = int(overrides.max_length)
    if overrides.per_device_train_batch_size is not None:
        config["per_device_train_batch_size"] = int(overrides.per_device_train_batch_size)
    if overrides.gradient_accumulation_steps is not None:
        config["gradient_accumulation_steps"] = int(overrides.gradient_accumulation_steps)
    if overrides.num_train_epochs is not None:
        config["num_train_epochs"] = float(overrides.num_train_epochs)
    if overrides.learning_rate is not None:
        config["learning_rate"] = float(overrides.learning_rate)
    if overrides.lora_r is not None:
        lora_config["r"] = int(overrides.lora_r)
    if overrides.lora_alpha is not None:
        lora_config["alpha"] = int(overrides.lora_alpha)
    if overrides.lora_dropout is not None:
        lora_config["dropout"] = float(overrides.lora_dropout)
    config["lora"] = lora_config
    return config


def _lazy_import_training_stack() -> dict[str, Any]:
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Training dependencies are unavailable for memory_policy_lora.") from exc

    return {
        "torch": torch,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
    }


class _JsonlDataset:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._records[index]


def _render_messages(messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> str:
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role", "")).strip().upper() or "USER"
        content = str(message.get("content", ""))
        lines.append(f"{role}: {content}")
    if add_generation_prompt:
        lines.append("ASSISTANT:")
    return "\n".join(lines)


class _TextCollator:
    def __init__(self, *, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_messages = [build_messages_from_record(record, include_assistant=False) for record in features]
        full_messages = [build_messages_from_record(record, include_assistant=True) for record in features]
        prompt_texts = [_render_messages(messages, add_generation_prompt=True) for messages in prompt_messages]
        full_texts = [_render_messages(messages, add_generation_prompt=False) for messages in full_messages]
        model_inputs = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_inputs = self.tokenizer(
            prompt_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = model_inputs["input_ids"].clone()
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1).tolist()
        for row_index, prompt_length in enumerate(prompt_lengths):
            labels[row_index, : int(prompt_length)] = -100
        labels[model_inputs["attention_mask"] == 0] = -100
        model_inputs["labels"] = labels
        return model_inputs


def _existing_suffixes(model, candidates: list[str]) -> list[str]:
    module_names = [name for name, _ in model.named_modules()]
    return [candidate for candidate in candidates if any(name.endswith(candidate) for name in module_names)]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoRA trainer for text-only memory policy control.")
    parser.add_argument("--config", type=Path, default=Path("experiments/memory_policy_lora/memory_policy_lora.yaml"))
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--model-name-or-path", type=str)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--per-device-train-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--num-train-epochs", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--lora-r", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-dropout", type=float)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = _resolved_config(args.config if args.config and args.config.exists() else None, args)
    dataset_dir = Path(config.get("dataset_dir", "")).resolve()
    output_dir = Path(config.get("output_dir", "artifacts/checkpoints/memory_policy_lora")).resolve()
    model_name_or_path = str(config.get("model_name_or_path", "")).strip()
    train_file = str(config.get("train_file", "train.jsonl"))
    eval_file = str(config.get("eval_file", "val.jsonl"))

    if dataset_dir == Path("").resolve():
        raise RuntimeError("dataset_dir must be provided in the config or CLI.")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    validation = validate_dataset(dataset_dir)
    resolved = {
        "dataset_dir": dataset_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "model_name_or_path": model_name_or_path,
        "validation": validation,
        "config": config,
    }
    if args.validate_only:
        print(json.dumps(resolved, indent=2, ensure_ascii=False, default=str))
        return 0

    if model_name_or_path == "":
        raise RuntimeError("model_name_or_path must be provided for training.")

    deps = _lazy_import_training_stack()
    torch = deps["torch"]
    tokenizer_cls = deps["AutoTokenizer"]
    model_cls = deps["AutoModelForCausalLM"]
    trainer_cls = deps["Trainer"]
    training_arguments_cls = deps["TrainingArguments"]
    lora_config_cls = deps["LoraConfig"]
    get_peft_model = deps["get_peft_model"]

    tokenizer = tokenizer_cls.from_pretrained(model_name_or_path, trust_remote_code=bool(config.get("trust_remote_code", True)))
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = None
    if bool(config.get("bf16", True)):
        torch_dtype = torch.bfloat16
    elif bool(config.get("fp16", False)):
        torch_dtype = torch.float16

    model = model_cls.from_pretrained(
        model_name_or_path,
        trust_remote_code=bool(config.get("trust_remote_code", True)),
        torch_dtype=torch_dtype,
    )
    if hasattr(model, "gradient_checkpointing_enable") and bool(config.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    lora_section = dict(config.get("lora", {}))
    target_modules = _existing_suffixes(model, list(lora_section.get("target_modules", [])))
    if not target_modules:
        raise RuntimeError("No configured LoRA target modules were found in the loaded model.")
    lora_cfg = lora_config_cls(
        r=int(lora_section.get("r", 16)),
        lora_alpha=int(lora_section.get("alpha", 32)),
        lora_dropout=float(lora_section.get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        modules_to_save=list(lora_section.get("modules_to_save", ["lm_head"])),
    )
    model = get_peft_model(model, lora_cfg)

    train_records = load_jsonl_records(dataset_dir / train_file)
    eval_records = load_jsonl_records(dataset_dir / eval_file)
    collator = _TextCollator(tokenizer=tokenizer, max_length=int(config.get("max_length", 2048)))
    training_args = training_arguments_cls(
        output_dir=output_dir.as_posix(),
        per_device_train_batch_size=int(config.get("per_device_train_batch_size", 4)),
        per_device_eval_batch_size=int(config.get("per_device_eval_batch_size", 4)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 8)),
        num_train_epochs=float(config.get("num_train_epochs", 2)),
        learning_rate=float(config.get("learning_rate", 2.0e-4)),
        warmup_ratio=float(config.get("warmup_ratio", 0.03)),
        logging_steps=int(config.get("logging_steps", 5)),
        save_steps=int(config.get("save_steps", 50)),
        eval_steps=int(config.get("eval_steps", 50)),
        save_total_limit=int(config.get("save_total_limit", 2)),
        lr_scheduler_type=str(config.get("lr_scheduler_type", "cosine")),
        weight_decay=float(config.get("weight_decay", 0.0)),
        max_grad_norm=float(config.get("max_grad_norm", 1.0)),
        bf16=bool(config.get("bf16", True)),
        fp16=bool(config.get("fp16", False)),
        evaluation_strategy="steps" if eval_records else "no",
        report_to=[],
        remove_unused_columns=False,
        seed=int(config.get("seed", 7)),
    )
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=_JsonlDataset(train_records),
        eval_dataset=_JsonlDataset(eval_records) if eval_records else None,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(output_dir.as_posix())
    tokenizer.save_pretrained(output_dir.as_posix())
    print(json.dumps(resolved, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
