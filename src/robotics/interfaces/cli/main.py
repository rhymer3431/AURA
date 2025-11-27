# src/robotics/interfaces/cli/main.py

from pathlib import Path
import cv2

from robotics.infrastructure.config.yaml_loader import load_config
from robotics.infrastructure.detection.yolo_world import YoloWorldDetector
from robotics.infrastructure.tracking.bytetrack import ByteTrackAdapter
from robotics.domain.scene_graph.reasoning import SGCLReasoner
from robotics.domain.action.policy import PolicyEngine
from robotics.application.use_cases.process_frame import ProcessFrameUseCase
from robotics.application.use_cases.run_stream import VideoStreamRunner
from robotics.interfaces.viz.opencv_overlay import OpenCVOverlayVisualizer


def main():
    cfg = load_config("configs/server_dev.yaml")

    weight_path = Path("model_weights/yolo_world") / cfg["model"]["yolo_world_weight"]

    detector = YoloWorldDetector(
        weight_path=str(weight_path),
        conf_threshold=cfg["inference"]["conf_thresh"],
        iou_threshold=cfg["inference"]["iou_thresh"],
        classes=cfg["model"]["yolo_world_class_filter"],
    )

    tracker = ByteTrackAdapter(
        track_thresh=cfg["tracking"]["track_thresh"],
        track_buffer=cfg["tracking"]["track_buffer"],
    )

    sgcl = SGCLReasoner()
    policy = PolicyEngine()

    use_case = ProcessFrameUseCase(
        detector=detector,
        tracker=tracker,
        sgcl=sgcl,
        policy=policy,
    )

    visualizer = OpenCVOverlayVisualizer()

    runner = VideoStreamRunner(
        use_case,
        cfg["video_input"],
        visualizer=visualizer
    )
    runner.run()


if __name__ == "__main__":
    main()
