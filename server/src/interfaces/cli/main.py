# src/robotics/interfaces/cli/main.py
import threading
from pathlib import Path

from server.src.infrastructure.config.yaml_loader import load_config
from server.src.infrastructure.detection.yolo_world import YoloWorldAdapter
from server.src.infrastructure.sgg.dearpygui_client import DearPyGuiClient, UiBridge
from server.src.domain.scene_graph.relations import SimpleRelationInfer
from server.src.domain.scene_graph.builder import SceneGraphBuilder
from server.src.domain.scene_graph.reasoning import SceneGraphReasoner
from server.src.application.use_cases.process_frame import ProcessFrameUseCase
from server.src.application.use_cases.run_stream import VideoStreamRunner
from server.src.interfaces.viz.opencv_overlay import OpenCVOverlayVisualizer
from server.src.interfaces.ui.dearpygui_monitor import run_dpg


def main():
    cfg = load_config("configs/server_dev.yaml")
    Path(__file__).resolve().parents[4]  # project_root (reserved if needed)

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

    # 4. Local UI bridge + client (no HTTP/WebRTC)
    ui_bridge = UiBridge()
    vis_client = DearPyGuiClient(ui_bridge)
    visualizer = OpenCVOverlayVisualizer()

    # 5. Video stream runner (runs on worker thread)
    runner = VideoStreamRunner(
        use_case,
        cfg["video_input"],
        visualizer=visualizer,
        vis_client=vis_client,
        target_fps=0,          # 0 => run at max achievable FPS
        send_every=1,          # send every frame to UI
        jpeg_quality=70,
        graph_send_interval=5, # send graph every 5 frames
    )

    stop_event = threading.Event()
    pipeline_thread = threading.Thread(target=runner.run, daemon=True)
    pipeline_thread.start()

    # 6. Run DearPyGui UI on main thread
    try:
        run_dpg(ui_bridge, stop_event, target_fps=60)
    finally:
        stop_event.set()
        pipeline_thread.join(timeout=2)


if __name__ == "__main__":
    main()
