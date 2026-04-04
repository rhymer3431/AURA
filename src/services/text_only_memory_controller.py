from __future__ import annotations

from dataclasses import replace

from aura_config.memory_policy_config import MemoryPolicyConfig

from .memory_policy_types import MemoryPolicyContext, MemoryPolicyDecision, MemoryPolicyLabel


_STOP_TOKENS = ("stop", "멈춰", "도착했으면")
_LEFT_TOKENS = ("left", "왼쪽")
_RIGHT_TOKENS = ("right", "오른쪽")


class TextOnlyMemoryController:
    def __init__(self, config: MemoryPolicyConfig | None = None) -> None:
        self.config = config or MemoryPolicyConfig()
        self._tokenizer = None
        self._model = None
        self._torch = None

    def predict(self, prompt_text: str, context: MemoryPolicyContext) -> MemoryPolicyDecision:
        backend = str(self.config.backend).strip().lower()
        if backend == "hf_generate":
            return self._predict_hf_generate(prompt_text, context)
        return self._predict_heuristic(context)

    def _predict_hf_generate(self, prompt_text: str, context: MemoryPolicyContext) -> MemoryPolicyDecision:
        try:
            label = self._generate_label(prompt_text)
        except Exception as exc:  # noqa: BLE001
            fallback = self._predict_heuristic(context)
            return replace(
                fallback,
                source="hf_generate",
                fallback_used=True,
                feature_snapshot={
                    **fallback.feature_snapshot,
                    "error": f"{type(exc).__name__}: {exc}",
                    "error_kind": "backend_failure",
                },
            )

        if label not in {item.value for item in MemoryPolicyLabel}:
            fallback = self._predict_heuristic(context)
            return replace(
                fallback,
                source="hf_generate",
                fallback_used=True,
                feature_snapshot={
                    **fallback.feature_snapshot,
                    "raw_output": label,
                    "error_kind": "parse_failure",
                },
            )

        return MemoryPolicyDecision(
            label=MemoryPolicyLabel(label),
            confidence=max(float(context.retrieval_confidence), 0.55),
            source="hf_generate",
            fallback_used=False,
            shadow_only=False,
            feature_snapshot=context.feature_snapshot(),
        )

    def _generate_label(self, prompt_text: str) -> str:
        if str(self.config.model_name_or_path).strip() == "":
            raise RuntimeError("hf_generate backend requires model_name_or_path.")
        if self._tokenizer is None or self._model is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError("transformers backend is unavailable") from exc
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path)
            self._model = AutoModelForCausalLM.from_pretrained(self.config.model_name_or_path)
            device = str(self.config.device).strip()
            if device != "":
                self._model.to(device)
            self._torch = torch
        tokenizer = self._tokenizer
        model = self._model
        torch = self._torch
        encoded = tokenizer(prompt_text, return_tensors="pt")
        device = str(self.config.device).strip()
        if device != "":
            encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=max(int(self.config.max_new_tokens), 1),
                do_sample=float(self.config.temperature) > 0.0,
                temperature=max(float(self.config.temperature), 1.0e-5),
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output[0][encoded["input_ids"].shape[1] :]
        return str(tokenizer.decode(generated, skip_special_tokens=True)).strip()

    def _predict_heuristic(self, context: MemoryPolicyContext) -> MemoryPolicyDecision:
        combined = self._combined_text(context)
        left_hits = sum(combined.count(token) for token in _LEFT_TOKENS)
        right_hits = sum(combined.count(token) for token in _RIGHT_TOKENS)
        instruction_lower = str(context.instruction).lower()
        if context.visible_target_now and any(token in instruction_lower for token in _STOP_TOKENS):
            label = MemoryPolicyLabel.STOP
            confidence = 0.85
        elif context.visible_target_now and (context.candidate_count == 0 or context.retrieval_confidence < 0.45):
            label = MemoryPolicyLabel.ROUTE_DIRECT_VISION
            confidence = 0.72
        elif context.ambiguity or context.candidate_count == 0 or context.retrieval_confidence < 0.15:
            label = MemoryPolicyLabel.WAIT
            confidence = 0.65 if context.ambiguity else 0.55
        elif left_hits > right_hits and context.retrieval_confidence >= 0.25:
            label = MemoryPolicyLabel.TURN_LEFT
            confidence = min(0.55 + 0.1 * left_hits + float(context.retrieval_confidence) * 0.2, 0.92)
        elif right_hits > left_hits and context.retrieval_confidence >= 0.25:
            label = MemoryPolicyLabel.TURN_RIGHT
            confidence = min(0.55 + 0.1 * right_hits + float(context.retrieval_confidence) * 0.2, 0.92)
        elif context.recall_result is not None and context.candidate_count > 0:
            label = MemoryPolicyLabel.ROUTE_MEMORY_VISION
            confidence = max(0.45, min(float(context.retrieval_confidence) + 0.25, 0.9))
        else:
            label = MemoryPolicyLabel.WAIT
            confidence = 0.5
        return MemoryPolicyDecision(
            label=label,
            confidence=float(confidence),
            source="heuristic",
            fallback_used=False,
            shadow_only=False,
            feature_snapshot={
                **context.feature_snapshot(),
                "left_hits": int(left_hits),
                "right_hits": int(right_hits),
            },
        )

    @staticmethod
    def _combined_text(context: MemoryPolicyContext) -> str:
        chunks = [str(context.instruction).lower()]
        if context.memory_context is not None and context.memory_context.scratchpad is not None:
            scratchpad = context.memory_context.scratchpad
            chunks.extend(
                [
                    str(scratchpad.goal_summary).lower(),
                    str(scratchpad.recent_hint).lower(),
                    str(scratchpad.next_priority).lower(),
                ]
            )
        if context.memory_context is not None:
            chunks.extend(str(line.text).lower() for line in context.memory_context.text_lines)
        chunks.extend(str(item).lower() for item in context.semantic_rule_hints)
        return " ".join(chunk for chunk in chunks if chunk.strip() != "")
