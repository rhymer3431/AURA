import threading
from pathlib import Path

from application.run_realtime import VideoStreamRunner
from application.scene_understanding import SceneUnderstandingUseCase
from config.yaml_loader import load_config
from domain.detect.service.detection_service import DetectionService
from domain.pipeline.service.realtime_pipeline import RealtimePipeline
from domain.reason.service.reasoning_service import ReasoningService
from domain.sgg.service.relation_predictor import SimpleRelationInfer
from domain.sgg.service.scene_graph_builder import SceneGraphBuilder
from domain.sgg.service.reasoning import SceneGraphReasoner
from infrastructure.model.yolo_world_adapter import YoloWorldAdapter
from infrastructure.utils.dearpygui_client import DearPyGuiClient, UiBridge
from infrastructure.utils.dearpygui_monitor import run_dpg
from infrastructure.utils.visualizer import OpenCVOverlayVisualizer


def main():
    config_dir = Path(__file__).resolve().parents[1] / "config"
    cfg = load_config(config_dir / "pipeline_realtime.yaml")
    project_root = Path(__file__).resolve().parents[2]

    weight_path = Path(cfg["model"]["yolo_world_weight"])
    if not weight_path.is_absolute():
        weight_path = project_root / weight_path

    detector_adapter = YoloWorldAdapter(
        weight_path=str(weight_path),
        device=cfg.get("device", "auto"),
        conf_threshold=cfg["inference"]["conf_thresh"],
        iou_threshold=cfg["inference"]["iou_thresh"],
        classes=cfg["model"].get("yolo_world_class_filter"),
    )
    detection_service = DetectionService(detector_adapter)

    relation_predictor = SimpleRelationInfer()
    sg_builder = SceneGraphBuilder(relation_predictor)
    sg_reasoner = SceneGraphReasoner(sg_builder)
    policy = ReasoningService()

    pipeline = RealtimePipeline(
        detector=detection_service,
        sg_reasoner=sg_reasoner,
        policy=policy,
    )
    use_case = SceneUnderstandingUseCase(pipeline)

    ui_bridge = UiBridge()
    vis_client = DearPyGuiClient(ui_bridge)
    visualizer = OpenCVOverlayVisualizer()

    runner = VideoStreamRunner(
        use_case,
        cfg["video_input"],
        visualizer=visualizer,
        vis_client=vis_client,
        target_fps=cfg.get("target_fps", 0),
        send_every=cfg.get("send_every", 1),
        jpeg_quality=cfg.get("jpeg_quality", 70),
        graph_send_interval=cfg.get("graph_send_interval", 5),
    )

    stop_event = threading.Event()
    pipeline_thread = threading.Thread(target=runner.run, daemon=True)
    pipeline_thread.start()

    try:
        run_dpg(ui_bridge, stop_event, target_fps=60)
    finally:
        stop_event.set()
        pipeline_thread.join(timeout=2)


if __name__ == "__main__":
    main()
