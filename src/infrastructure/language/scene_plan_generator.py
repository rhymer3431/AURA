import json
import os
from typing import Any, Dict, List, Optional

# Ensure safer defaults before importing heavy deps
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("FLASH_ATTENTION_DISABLE", "1")

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


from domain.language.scene_graph import scene_graph_to_struct
from infrastructure.logging.pipeline_logger import PipelineLogger


def generate_scene_plan_local(
    tokenizer,
    model,
    sg_frame,
    max_new_tokens: int = 96,
    max_objects: int = 12,
    max_relations: int = 24,
) -> Dict[str, Any]:
    """
    Build a caption + focus_targets JSON plan from a scene graph.
    """

    scene_struct = scene_graph_to_struct(
        sg_frame,
        max_objects=max_objects,
        max_relations=max_relations,
    )
    scene_json = json.dumps(scene_struct, ensure_ascii=False, separators=(",", ":"))

    prompt = (
        "You are an on-device reasoning module inside a robotics perception system.\n"
        "You will receive a scene graph as JSON and must produce a scene plan.\n\n"
        "Your OUTPUT MUST BE A SINGLE JSON OBJECT ONLY (no explanation, no comments, no backticks).\n\n"
        "Output JSON schema:\n"
        "- caption: one short natural-language sentence (<= 20 words) describing the current frame.\n"
        "- focus_targets: list of object labels (strings) that the robot must pay attention to.\n"
        "- yolo_world_prompt: list of short noun labels (strings) that YOLO-World should detect next.\n\n"
        "Rules:\n"
        "- Use ONLY labels that actually appear in the scene graph JSON.\n"
        "- Do not mention indices, IDs, track_ids, or bounding boxes in the caption.\n"
        "- The caption must be a single sentence.\n"
        "- Base your reasoning ONLY on the given scene graph JSON.\n\n"
        "SCENE_JSON:\n"
        f"{scene_json}\n\n"
        "Respond with EXACTLY one JSON object with this shape and nothing else:\n"
        '{'
        '"caption": "<short sentence>", '
        '"focus_targets": ["label1", "label2"], '
        '"yolo_world_prompt": ["label1", "label2"]'
        "}"
    )


    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    generated = output_ids[0, input_ids["input_ids"].shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    plan = {"caption": text, "focus_targets": []}
    try:
        start = text.find("{")
        end = text.rfind("}")
        json_str = text[start : end + 1] if start != -1 and end != -1 and end > start else text
        obj = json.loads(json_str)
        caption = obj.get("caption", "").strip()
        ft = obj.get("focus_targets", [])
        if isinstance(caption, str):
            plan["caption"] = caption
        if isinstance(ft, list):
            plan["focus_targets"] = [str(x) for x in ft if isinstance(x, str)]
    except Exception:
        pass

    return plan


import torch
from typing import Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

def load_local_llm(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    device: str = "cuda",
    attn_impl: Optional[str] = None,
    logger: Optional[PipelineLogger] = None,
):
    """
    Load Qwen2.5-3B using bitsandbytes 8-bit quantization.
    - GPU 환경이면 device_map='auto'로 로딩
    - 로딩 실패 시 CPU fallback
    - LLM Brain Activity Hook 자동 등록
    """

    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto" if device == "cuda" else device,
        "quantization_config": quant_config,
    }
    if attn_impl is not None:
        model_kwargs["attn_implementation"] = attn_impl

    # Try GPU+8bit first
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    except Exception as e:
        # CPU fallback: less efficient but avoids CUDA/quant errors
        fallback_kwargs = {
            "device_map": "cpu",
            "torch_dtype": torch.float32,
        }
        if attn_impl is not None:
            fallback_kwargs["attn_implementation"] = "eager"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **fallback_kwargs,
        )
        if logger:
            try:
                logger.log(
                    module="LLM",
                    event="load_fallback_cpu",
                    frame_idx=None,
                    message=str(e),
                )
            except Exception:
                pass

    model.eval()

    # Register hook for brain activity mapping
    if logger is not None:
        try:
            register_llm_brain_hooks(model, logger)
        except Exception:
            if logger:
                logger.log(
                    module="LLM",
                    event="hook_registration_failed",
                    message="Could not attach brain hooks",
                )

    if logger:
        logger.log(
            module="LLM",
            event="load_success",
            message=f"Loaded local LLM ({model_name}) in 8bit on {device}",
        )

    return tokenizer, model

def register_llm_brain_hooks(model, logger: PipelineLogger) -> None:
    """
    Attach lightweight forward hooks to key submodules for brain-area logging.
    Uses substring pattern matching on module names; only registers when a brain area is determined.
    """
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
            # YOLO 훅과 포맷을 맞추기 위해 shape / frame_idx / layer_index까지 넣어줌
            try:
                if logger is None:
                    return

                short_name = name
                # strip leading prefixes like "model.layers.X."
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
                    frame_idx=None,          # YOLO 훅과 동일한 키
                    matched_brain=brain,     # ★ 여기서 반드시 들어감
                    layer_index=None,        # LLM은 별도 인덱스가 없으니 None
                    shape=shape,
                )
            except Exception:
                # 훅에서 에러가 나도 추론이 죽지 않도록
                pass

        return _hook

    # 실제 훅 등록
    for name, module in model.named_modules():
        brain = match_brain(name)
        if brain is None:
            continue
        try:
            module.register_forward_hook(make_hook(name, brain))
        except Exception:
            continue
