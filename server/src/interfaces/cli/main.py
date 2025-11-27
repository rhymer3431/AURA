# src/robotics/interfaces/cli/main.py

import subprocess
from pathlib import Path

from server.src.infrastructure.config.yaml_loader import load_config
from server.src.infrastructure.detection.yolo_world import YoloWorldAdapter
from server.src.infrastructure.sgg.vis_client import VisClient
from server.src.domain.scene_graph.relations import SimpleRelationInfer
from server.src.domain.scene_graph.builder import SceneGraphBuilder
from server.src.domain.scene_graph.reasoning import SceneGraphReasoner
from server.src.application.use_cases.process_frame import ProcessFrameUseCase
from server.src.application.use_cases.run_stream import VideoStreamRunner
from server.src.interfaces.viz.opencv_overlay import OpenCVOverlayVisualizer


def main():
    cfg = load_config("configs/server_dev.yaml")
    project_root = Path(__file__).resolve().parents[4]

    # 0. Spin up visualization backend/front (FastAPI + React dev server) using WebRTC server
    vis_server_proc = start_vis_server(project_root)
    webui_proc = start_webui(project_root)

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

    # 4. Visualization client (push frames/graphs to FastAPI)
    vis_client = VisClient(base_url="http://localhost:7000")
    visualizer = OpenCVOverlayVisualizer()

    # 5. 비디오 스트림 실행
    runner = VideoStreamRunner(
        use_case,
        cfg["video_input"],
        visualizer=visualizer,
        vis_client=vis_client,
        target_fps=0,       # 0 -> no pacing, run at max achievable FPS
        send_every=1,       # send every frame
        jpeg_quality=70,    # balance quality and bandwidth
    )
    try:
        runner.run()
    finally:
        for proc in (vis_server_proc, webui_proc):
            if proc and proc.poll() is None:
                proc.terminate()


def start_vis_server(project_root: Path):
    try:
        return subprocess.Popen(
            ["uvicorn", "server.webrtc_server:app", "--host", "0.0.0.0", "--port", "7000"],
            cwd=project_root,
        )
    except FileNotFoundError:
        print("Warning: uvicorn not found; visualization backend not started.")
        return None


def start_webui(project_root: Path):
    webui_dir = project_root / "webui"
    if not webui_dir.exists():
        print("Warning: webui directory not found; skipping frontend start.")
        return None
    try:
        return subprocess.Popen(
            ["npm", "start"],
            cwd=webui_dir,
        )
    except FileNotFoundError:
        print("Warning: npm not found; frontend not started.")
        return None


if __name__ == "__main__":
    main()
