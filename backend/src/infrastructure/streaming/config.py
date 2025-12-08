import os
from pathlib import Path
import torch


class StreamServerConfig:
    """Runtime tunables sourced from env vars with sensible defaults."""

    def __init__(self) -> None:
        self.root_dir = Path(__file__).resolve().parents[3]
        self.video_path = Path(
            os.getenv("VIDEO_PATH", self.root_dir / "input" / "video.mp4")
        ).resolve()
        self.target_fps = float(os.getenv("TARGET_FPS", "30"))
        self.frame_max_width = int(os.getenv("FRAME_MAX_WIDTH", "960"))
        self.jpeg_quality = int(os.getenv("JPEG_QUALITY", "90"))
        self.llm_model_name = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
        self.llm_device = os.getenv(
            "LLM_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
        )
        # Default to eager attention to avoid flash-attn CUDA extension issues.
        self.llm_attn_impl = os.getenv("LLM_ATTN_IMPL", "eager")
        self.logging_enabled = (
            os.getenv("LOGGING_ENABLED", "1").lower() not in {"0", "false", "no"}
        )
