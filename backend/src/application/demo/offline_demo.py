"""Offline video demo that reuses the adapters and LLM worker."""
import cv2
import torch
from typing import List
from pathlib import Path

from src.infrastructure.perception.perception_service_adapter import PerceptionServiceAdapter
from src.infrastructure.llm.local_scene_plan_worker import LocalScenePlanWorker
from src.infrastructure.logging.pipeline_logger import PipelineLogger
from src.domain.utils.box_iou_xyxy import box_iou_xyxy


def run_video_demo(
    video_path: str,
    skip_grin: bool = False,
    enable_logging: bool = True,
):
    logger = PipelineLogger(enabled=enable_logging)
    logger.log(module="Pipeline", event="start", frame_idx=None, video_path=video_path)

    cap = cv2.VideoCapture(video_path)
    system = PerceptionServiceAdapter(
        yolo_weight="yolov8s-worldv2.pt",
        ltm_feat_dim=256,
        device="cuda" if torch.cuda.is_available() else "cpu",
        logger=logger,
    )

    planner = LocalScenePlanWorker(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        device="cuda" if torch.cuda.is_available() else "cpu",
        attn_impl="eager",
        logger=logger,
    )

    frame_idx = 0
    last_entities = {}
    last_caption: str = ""
    last_focus_targets: List[str] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        run_grin = (frame_idx % 10 == 0) and (not skip_grin)
        sg_frame, _, _diff = system.process_frame(
            frame_bgr=frame,
            frame_idx=frame_idx,
            run_grin=run_grin,
            grin_horizon=16,
        )

        current_entities = {n.entity_id: n.box for n in sg_frame.nodes}
        change_detected = _detect_change(current_entities, last_entities)

        if change_detected and len(sg_frame.nodes) > 0:
            simple_sg = system.build_simple_scene_graph_frame(sg_frame)
            planner.submit(frame_idx, simple_sg)
            logger.log(
                module="LLMReasoner",
                event="llm_schedule",
                frame_idx=frame_idx,
                num_entities=len(sg_frame.nodes),
            )

        for idx, plan in planner.poll_results():
            if "error" in plan:
                print(f"[LLM ERROR F{idx}]: {plan['error']}")
                logger.log(
                    module="LLMReasoner",
                    event="error",
                    frame_idx=idx,
                    level="ERROR",
                    message=plan["error"],
                )
            else:
                caption = plan.get("caption", "")
                focus_targets = plan.get("focus_targets", [])
                if caption:
                    last_caption = caption
                if isinstance(focus_targets, list):
                    last_focus_targets = focus_targets
                    system.detector.update_focus_classes(last_focus_targets)

                print(f"[LLM F{idx}] Caption: {last_caption}")
                print(f"[LLM F{idx}] Focus: {last_focus_targets}")
                logger.log(
                    module="LLMReasoner",
                    event="reasoning_done",
                    frame_idx=idx,
                    caption=last_caption,
                    focus_targets=last_focus_targets,
                )

        last_entities = current_entities

        vis = frame.copy()
        for node in sg_frame.nodes:
            x1, y1, x2, y2 = map(int, node.box)
            label = f"{node.cls}#{node.entity_id}"
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis,
                label,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        if last_caption:
            cv2.putText(
                vis,
                last_caption[:60],
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
        if last_focus_targets:
            txt = "Focus: " + ", ".join(last_focus_targets)
            cv2.putText(
                vis,
                txt[:60],
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

        cv2.imshow("Perception w/ Async LLM", vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    planner.shutdown()
    cap.release()
    cv2.destroyAllWindows()

    try:
        logs_dir = Path(__file__).resolve().parents[3] / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        logger.to_csv(str(logs_dir / "pipeline_log.csv"))
        logger.to_jsonl(str(logs_dir / "pipeline_log.jsonl"))
    except Exception as e:
        print(f"[Logger] failed to save logs: {e}")


def _detect_change(
    current_entities: dict, last_entities: dict, iou_thresh: float = 0.5
) -> bool:
    if len(current_entities) != len(last_entities):
        return True
    if set(current_entities.keys()) != set(last_entities.keys()):
        return True
    for eid, box in current_entities.items():
        if box_iou_xyxy(box, last_entities[eid]) < iou_thresh:
            return True
    return False


if __name__ == "__main__":
    run_video_demo(video_path="input/video.mp4")
