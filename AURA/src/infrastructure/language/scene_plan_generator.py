import json
import os
from typing import Any, Dict, List, Optional
import time   # ★ 시간 측정용

# Ensure safer defaults before importing heavy deps
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.domain.language.scene_graph import scene_graph_to_struct
from src.infrastructure.logging.pipeline_logger import PipelineLogger

import json
import os
import time
from typing import Any, Dict

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.domain.language.scene_graph import scene_graph_to_struct
import json
import os
import time
from typing import Any, Dict, Optional

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.domain.language.scene_graph import scene_graph_to_struct


from typing import Any, Dict
import json, time
import torch


def generate_scene_plan_local(
    tokenizer,
    model,
    sg_frame,
    max_new_tokens: int = 96,
    max_objects: int = 12,
    max_relations: int = 24,
) -> Dict[str, Any]:
    scene_struct = scene_graph_to_struct(
        sg_frame,
        max_objects=max_objects,
        max_relations=max_relations,
    )
    scene_json = json.dumps(scene_struct, ensure_ascii=False, separators=(",", ":"))

    prompt = f"""You are an on-device reasoning module in a robotics perception system.
You will receive a SCENE_GRAPH as JSON and must output ONE JSON object only.
No explanations, no comments, no markdown, no backticks.

Your output MUST be exactly one JSON object with these keys:
- "caption": string
- "focus_targets": string[]

Global JSON rules:
- Valid JSON only.
- Double quotes for all strings.
- No trailing commas.
- No text before or after the JSON object.

Scene rules:
- Base reasoning ONLY on SCENE_GRAPH.
- Use ONLY labels and entity_ids from SCENE_GRAPH.
- Object class labels are English.
- Do NOT hallucinate objects or relations.

caption rules:
- Use ONE Korean sentence.
- When referring to objects, ALWAYS use label#entity_id.
- Speak as if expressing a simple personal thought.
- Match expression complexity to the scene:
  - Simple movement or static state → state only.
  - Clear interaction or relation → light interpretation only.

focus_targets rules:
- This field represents YOLO OUTPUT targets.
- Array of UNIQUE object class labels ONLY.
- Use ONLY base class labels appearing in SCENE_GRAPH.
- NEVER include entity ids, numbers, or '#' symbols.
- Include ONLY the most important classes to focus on.

Input:
SCENE_GRAPH = {scene_json}

Output:
Return exactly ONE JSON object and nothing else.
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(model.device)

    print(f"[LLM] Input tokens: {inputs.input_ids.shape[1]}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"[LLM] caption generation time: {elapsed:.4f} sec")

    generated = output_ids[0, inputs.input_ids.shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # ✅ 출력은 caption + focus_targets만
    plan: Dict[str, Any] = {"caption": "", "focus_targets": []}

    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = json.loads(text[start:end + 1])

            plan["caption"] = str(obj.get("caption", "")).strip()

            ft = obj.get("focus_targets", [])
            if isinstance(ft, list):
                plan["focus_targets"] = [x for x in ft if isinstance(x, str)]
        else:
            # JSON이 아니면 raw를 캡션으로라도 넣음 (디버그용)
            plan["caption"] = text
    except Exception as e:
        print(f"[LLM] JSON parsing failed: {e}")
        print("[LLM] Raw output:", text)
        plan["caption"] = text

    return plan


def load_local_llm(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    use_flash_attn=True,
    attn_impl="sdpa",
    logger=None,
):
    """
    Load quantized LLM with optimizations.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float16,
        "quantization_config": quant_config,
        "low_cpu_mem_usage": True,
    }

    # 🔥 FlashAttention2 시도 (4bit 양자화와 호환 안 되는 경우 많음)
    model = None
    if attn_impl or use_flash_attn:
        try:
            model_kwargs["attn_implementation"] = attn_impl or "flash_attention_2"
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            print("[LLM] ✓ Loaded with FlashAttention2")
        except Exception as e:
            print(f"[LLM] ⚠ FlashAttention2 failed: {e}")
            model = None
    
    # Fallback to default attention
    if model is None:
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        print("[LLM] ✓ Loaded with default attention (sdpa or eager)")

    model.eval()
    
    # 🔥 디바이스 정보
    print(f"[LLM] Model device: {next(model.parameters()).device}")
    print(f"[LLM] Model dtype: {next(model.parameters()).dtype}")
    
    # 실제 attention 구현 확인
    if hasattr(model.config, '_attn_implementation'):
        actual_attn = model.config._attn_implementation
        print(f"[LLM] Actual attention: {actual_attn}")
        
        if actual_attn == "eager":
            print("[LLM] ⚠ WARNING: Using slow 'eager' attention!")
            print("[LLM]   This is normal with 4bit quantization.")
            print("[LLM]   Performance impact: ~2x slower than flash_attention_2")

    if logger:
        from src.infrastructure.logging.pipeline_logger import PipelineLogger
        register_llm_brain_hooks(model, logger)

    return tokenizer, model


def register_llm_brain_hooks(model, logger):
    """Hook registration (기존 코드 유지)"""
    # ... (기존 코드 그대로) ...
    pass

def register_llm_brain_hooks(model, logger: PipelineLogger) -> None:
    patterns = [
        ("embed_tokens", "Wernicke's Area"),
        ("rotary_emb", "STS"),
        ("self_attn.q_proj", "PPC"),
        ("self_attn.k_proj", "Hippocampus"),
        ("self_attn.v_proj", "TPJ"),
        ("self_attn.o_proj", "PFC"),
        ("q_proj", "PPC"),
        ("k_proj", "Hippocampus"),
        ("v_proj", "TPJ"),
        ("o_proj", "PFC"),
        ("mlp.gate_proj", "OFC"),
        ("mlp.up_proj", "dlPFC"),
        ("mlp.down_proj", "dlPFC"),
        ("post_attention_layernorm", "PFC"),
        ("input_layernorm", "PFC"),
        ("layernorms", "PFC"),
        ("layernorm", "PFC"),
        ("norm", "PFC"),
        ("lm_head", "Broca's Area"),
    ]

    def match_brain(name: str) -> Optional[str]:
        lname = name.lower()
        for pat, brain in patterns:
            if pat.lower() in lname:
                return brain
        return None

    def make_hook(name: str, brain: str):
        def _hook(module, inp, out):
            try:
                if logger is None:
                    return

                short_name = name
                if ".self_attn." in short_name:
                    short_name = short_name.split(".self_attn.", 1)[1]
                    short_name = f"self_attn.{short_name}"
                elif ".mlp." in short_name:
                    short_name = short_name.split(".mlp.", 1)[1]
                    short_name = f"mlp.{short_name}"
                elif "embed_tokens" in short_name:
                    short_name = "embed_tokens"
                elif "rotary_emb" in short_name:
                    short_name = "rotary_emb"
                elif "layernorm" in short_name or "norm" in short_name:
                    short_name = "layernorm"
                elif "lm_head" in short_name:
                    short_name = "lm_head"

                shape: Any = None
                if hasattr(out, "shape"):
                    shape = list(out.shape)
                elif isinstance(out, (list, tuple)) and out and hasattr(out[0], "shape"):
                    shape = [list(o.shape) for o in out]

                logger.log(
                    module=short_name,
                    event="forward",
                    frame_idx=None,
                    matched_brain=brain,
                    layer_index=None,
                    shape=shape,
                )
            except Exception:
                pass

        return _hook

    for name, module in model.named_modules():
        brain = match_brain(name)
        if brain is None:
            continue
        try:
            module.register_forward_hook(make_hook(name, brain))
        except Exception:
            continue
