import os
from pathlib import Path
import torch


class StreamServerConfig:
    """Runtime tunables sourced from env vars with sensible defaults."""

    def __init__(self) -> None:
        self.root_dir = Path(__file__).resolve().parents[3]
        self.input_source = os.getenv("INPUT_SOURCE", "ros2").strip().lower()
        if self.input_source not in {"video", "ros2"}:
            raise ValueError(
                f"INPUT_SOURCE must be one of ['video', 'ros2'], got: {self.input_source}"
            )
        self.video_path = Path(
            os.getenv("VIDEO_PATH", self.root_dir / "input" / "video.mp4")
        ).resolve()
        self.yolo_weight = Path(
            os.getenv(
                "YOLO_WEIGHT",
                self.root_dir / "models" / "yoloe-26s-seg.pt",
            )
        ).resolve()
        self.ros_image_topic = os.getenv("ROS_IMAGE_TOPIC", "/camera/rgb")
        self.ros_slam_pose_topic = os.getenv("ROS_SLAM_POSE_TOPIC", "/orbslam/pose")
        self.ros_semantic_projected_map_topic = os.getenv(
            "ROS_SEMANTIC_PROJECTED_MAP_TOPIC", "/semantic_map/projected_map"
        )
        self.ros_semantic_octomap_cloud_topic = os.getenv(
            "ROS_SEMANTIC_OCTOMAP_CLOUD_TOPIC", "/semantic_map/octomap_cloud"
        )
        self.ros_queue_size = int(os.getenv("ROS_QUEUE_SIZE", "1"))
        self.target_fps = float(os.getenv("TARGET_FPS", "30"))
        self.frame_max_width = int(os.getenv("FRAME_MAX_WIDTH", "960"))
        self.jpeg_quality = int(os.getenv("JPEG_QUALITY", "90"))
        self.llm_enabled = (
            os.getenv("ENABLE_LLM", "1").strip().lower()
            not in {"0", "false", "no"}
        )
        self.llm_model_name = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
        self.llm_device = os.getenv(
            "LLM_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
        )
        # Default to eager attention to avoid flash-attn CUDA extension issues.
        self.llm_attn_impl = os.getenv("LLM_ATTN_IMPL", "eager")
        self.logging_enabled = (
            os.getenv("LOGGING_ENABLED", "1").lower() not in {"0", "false", "no"}
        )
