import json
import csv
from pathlib import Path
from typing import Dict, List, Optional


# Allowed brain areas provided by user
ALLOWED_BRAINS = [
    "Angular Gyrus",
    "Broca's Area",
    "DMN",
    "Dorsal Stream",
    "EBA",
    "FFA",
    "Hippocampus",
    "IT",
    "LOC",
    "MT/V5",
    "OFC",
    "PFC",
    "PPA",
    "PPC",
    "SMA/PMC",
    "STS",
    "TPJ",
    "V1",
    "V2",
    "V3",
    "V4",
    "Wernicke's Area",
    "dlPFC",
    "mPFC",
]


def load_events(jsonl_path: Path) -> List[Dict]:
    events: List[Dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    return events


def filter_and_match(events: List[Dict], mapping: Dict[str, str]) -> List[Dict]:
    matched: List[Dict] = []
    for ev in events:
        module = ev.get("module")
        ts = ev.get("ts")
        if module not in mapping:
            continue
        brain = ev.get("matched_brain") or mapping[module]
        if brain not in ALLOWED_BRAINS:
            continue
        matched.append(
            {
                "ts": float(ts),
                "module": module,
                "matched_brain": brain,
            }
        )
    matched.sort(key=lambda x: x["ts"])
    # compute active_time as diff to next event
    for i in range(len(matched)):
        if i + 1 < len(matched):
            matched[i]["active_time"] = matched[i + 1]["ts"] - matched[i]["ts"]
        else:
            matched[i]["active_time"] = 0.0
    return matched


def to_csv(rows: List[Dict], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ts", "module", "matched_brain", "active_time"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main(
    input_jsonl: Path = Path("logs/pipeline_log.jsonl"),
    output_csv: Path = Path("logs/brain_activity_log.csv"),
):
    module_to_brain: Dict[str, str] = {
    # ====================================
    # YOLO-World (Perception)
    # ====================================
    "yw_0": "V1",
    "yw_1": "V1",
    "yw_2": "V2",
    "yw_3": "V3",
    "yw_4": "V3",
    "yw_5": "V4",
    "yw_6": "V4",
    "yw_7": "IT",
    "yw_8": "IT",
    "yw_9": "IT",
    "yw_10": "Dorsal Stream",
    "yw_11": "TPJ",
    "yw_12": "PPC",
    "yw_13": "Dorsal Stream",
    "yw_14": "TPJ",
    "yw_15": "PPC",
    "yw_16": "V4",
    "yw_17": "TPJ",
    "yw_18": "PPC",
    "yw_19": "Dorsal Stream",
    "yw_20": "TPJ",
    "yw_21": "dlPFC",
    "yw_22": "PFC",

    "embed_tokens": "Wernicke's Area",

    # 시간적 연속 문맥 통합
    "rotary_emb": "STS",

    # Attention 게이트 (선택과 집중)
    "self_attn.q_proj": "PPC (Spatial Attention)",
    "self_attn.k_proj": "Hippocampus (Memory Indexing)",
    "self_attn.v_proj": "TPJ (Context Binding)",
    "self_attn.o_proj": "PFC (Attention Control)",

    # 고차원 reasoning
    "mlp.gate_proj": "OFC (Value Judgement)",
    "mlp.up_proj": "dlPFC (Abstract Reasoning)",
    "mlp.down_proj": "Working Memory",

    # 안정화, 충돌 억제
    "input_layernorm": "ACC",
    "post_attention_layernorm": "ACC",

    # 언어 행동 출력
    "lm_head": "Broca's Area",
}



    events = load_events(input_jsonl)
    rows = filter_and_match(events, module_to_brain)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    to_csv(rows, output_csv)


if __name__ == "__main__":
    main()
