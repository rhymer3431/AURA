# src/robotics/interfaces/cli/main.py

from pathlib import Path

from robotics.infrastructure.config.yaml_loader import load_config
from robotics.infrastructure.detection.yolo_world import YoloWorldAdapter
from robotics.domain.scene_graph.relations import SimpleRelationInfer
from robotics.domain.scene_graph.builder import SceneGraphBuilder
from robotics.domain.scene_graph.reasoning import SceneGraphReasoner
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

    # 2. Scene Graph Reasoner (currently heuristic-based)
    relation_predictor = SimpleRelationInfer()
    sg_builder = SceneGraphBuilder(relation_predictor)
    sg_reasoner = SceneGraphReasoner(sg_builder)

    # 3. UseCase (Detection + Tracking + Scene Graph)
    use_case = ProcessFrameUseCase(
        detector=detector,
        sg_reasoner=sg_reasoner,
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
