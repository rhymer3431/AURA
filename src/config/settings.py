from pathlib import Path
import os

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YOLO_WEIGHT = Path(
    os.getenv(
        "YOLO_WORLD_WEIGHT",
        PROJECT_ROOT / "model_weights" / "yolo_world" / "yolov8s-worldv2.pt",
    )
)
DEVICE = torch.device(os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
DTYPE = torch.float32
