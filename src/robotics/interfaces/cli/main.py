# src/robotics/interfaces/cli/main.py

from pathlib import Path
import cv2
from robotics.infrastructure.config.yaml_loader import load_config
from robotics.infrastructure.detection.yolo_world import YoloWorldAdapter


from robotics.application.use_cases.process_frame import ProcessFrameUseCase
from robotics.application.use_cases.run_stream import VideoStreamRunner
from robotics.interfaces.viz.opencv_overlay import OpenCVOverlayVisualizer

def main():
    cfg = load_config("configs/server_dev.yaml")

    # 1. Detector
    weight_path = Path("model_weights/yolo_world") / cfg["model"]["yolo_world_weight"]
    detector = YoloWorldAdapter(
        weight_path=str(weight_path),
        conf_threshold=cfg["inference"]["conf_thresh"],
        iou_threshold=cfg["inference"]["iou_thresh"],
        classes=cfg["model"]["yolo_world_class_filter"],
    )



    # 3. UseCase (Detection + Tracking만)
    use_case = ProcessFrameUseCase(
        detector=detector,
    )

    # 4. 시각화
    visualizer = OpenCVOverlayVisualizer()
    # 5. 비디오 스트림 실행
    runner = VideoStreamRunner(
        use_case,
        cfg["video_input"],
        visualizer=visualizer
    )
    runner.run()


if __name__ == "__main__":
    main()
