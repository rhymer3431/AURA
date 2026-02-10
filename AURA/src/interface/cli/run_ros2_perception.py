import argparse

import torch

from src.infrastructure.logging.pipeline_logger import PipelineLogger
from src.infrastructure.perception.perception_service_adapter import PerceptionServiceAdapter
from src.infrastructure.llm.local_scene_plan_worker import LocalScenePlanWorker
from src.infrastructure.streaming.config import StreamServerConfig
from src.interface.ros2.perception_ros2_node import Ros2PerceptionNode

try:
    import rclpy
except Exception as exc:  # pragma: no cover - import guard for non-ROS envs
    raise RuntimeError(
        "ROS2 Python packages are required. Source your ROS2 environment first."
    ) from exc


def _parse_args() -> argparse.Namespace:
    cfg = StreamServerConfig()
    parser = argparse.ArgumentParser(description="ROS2 perception pipeline subscriber")
    parser.add_argument("--image-topic", default="/image_raw")
    parser.add_argument("--metadata-topic", default="/aura/perception/metadata")
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument("--max-entities", type=int, default=16)
    parser.add_argument(
        "--yolo-weight",
        default=str(cfg.root_dir / "models" / "yoloe-26s-seg.pt"),
        help="Path to YOLO-World weights",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for perception",
    )
    parser.add_argument("--enable-llm", action="store_true")
    parser.add_argument("--llm-model-name", default=cfg.llm_model_name)
    parser.add_argument("--llm-device", default=cfg.llm_device)
    parser.add_argument("--llm-attn-impl", default=cfg.llm_attn_impl)
    parser.add_argument("--disable-logging", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rclpy.init()

    logger = PipelineLogger(enabled=not args.disable_logging)
    perception = PerceptionServiceAdapter(
        yolo_weight=args.yolo_weight,
        ltm_feat_dim=256,
        device=args.device,
        logger=logger,
    )

    scene_planner = None
    if args.enable_llm:
        scene_planner = LocalScenePlanWorker(
            model_name=args.llm_model_name,
            device=args.llm_device,
            attn_impl=args.llm_attn_impl,
            logger=logger,
        )

    node = Ros2PerceptionNode(
        perception=perception,
        scene_planner=scene_planner,
        image_topic=args.image_topic,
        metadata_topic=args.metadata_topic,
        target_fps=args.target_fps,
        max_entities=args.max_entities,
    )

    try:
        rclpy.spin(node)
    finally:
        if scene_planner is not None:
            scene_planner.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
