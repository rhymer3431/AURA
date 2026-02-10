"""Offline video demo that reuses the adapters and LLM worker
(20 FPS base + REAL display-time preserved video).
"""

import cv2
import torch
import time
from typing import List
from pathlib import Path

from src.infrastructure.perception.perception_service_adapter import PerceptionServiceAdapter
from src.infrastructure.llm.local_scene_plan_worker import LocalScenePlanWorker
from src.infrastructure.logging.pipeline_logger import PipelineLogger
from src.utils.box_iou_xyxy import box_iou_xyxy
from src.domain.perception.scene_graph_relation import RelationType

relation_list = list(RelationType)

# ============================================================
# FPS & Video Save Config
# ============================================================
TARGET_FPS = 20
FRAME_TIME = 1.0 / TARGET_FPS

OUTPUT_VIDEO_PATH = "output/perception_demo_real_delay_sync.mp4"
VIDEO_CODEC = "mp4v"

# Purity-UI Mint (#2DBE60) — 항상 이 색만 사용
COLOR_DEFAULT = (96, 190, 45)   # BGR


# ============================================================
# Drawing Utilities (React-style)
# ============================================================
def draw_bbox_with_outline(img, x1, y1, x2, y2, color, thickness=2):
    halo = img.copy()
    cv2.rectangle(
        halo,
        (x1 - 1, y1 - 1),
        (x2 + 1, y2 + 1),
        color,
        1,
        cv2.LINE_AA,
    )
    cv2.addWeighted(halo, 0.25, img, 0.75, 0, img)

    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_label_badge(img, x1, y1, label, confidence, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.38
    thickness = 1

    conf_txt = ""
    if isinstance(confidence, (float, int)):
        conf_txt = f" {confidence * 100:.1f}%"

    text = label + conf_txt
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

    pad_x, pad_y = 6, 4
    box_w = tw + pad_x * 2
    box_h = th + pad_y * 2

    bx1 = x1
    by1 = max(0, y1 - box_h - 2)
    bx2 = bx1 + box_w
    by2 = by1 + box_h

    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (bx1, by1),
        (bx2, by2),
        color,
        -1,
        cv2.LINE_AA,
    )
    cv2.addWeighted(overlay, 0.9, img, 0.1, 0, img)

    cv2.putText(
        img,
        text,
        (bx1 + pad_x, by2 - pad_y),
        font,
        font_scale,
        (11, 18, 33),  # #0b1221 (BGR)
        thickness,
        cv2.LINE_AA,
    )


# ============================================================
# Main Loop
# ============================================================
def run_video_demo(video_path: str, skip_grin: bool = False, enable_logging: bool = True):
    logger = PipelineLogger(enabled=enable_logging)
    logger.log(module="Pipeline", event="start", frame_idx=None, video_path=video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(OUTPUT_VIDEO_PATH).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    video_writer = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH,
        fourcc,
        TARGET_FPS,
        (width, height),
    )

    system = PerceptionServiceAdapter(
        yolo_weight="models/yoloe-26s-seg.pt",
        ltm_feat_dim=256,
        device="cuda" if torch.cuda.is_available() else "cpu",
        logger=logger,
    )

    planner = LocalScenePlanWorker(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        device="cuda" if torch.cuda.is_available() else "cpu",
        attn_impl="sdpa",
        logger=logger,
    )

    frame_idx = 0
    last_entities = {}
    last_caption = ""
    last_focus_targets: List[str] = []

    paused = False
    last_vis = None

    # 🔑 “실제 화면 표시 시간” 기준점
    last_present_t = time.perf_counter()

    while True:
        loop_start_t = time.perf_counter()

        # --------------------------------------------------
        # Frame Processing
        # --------------------------------------------------
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            run_grin = (frame_idx % 10 == 0) and (not skip_grin)
            sg_frame, _, _ = system.process_frame(
                frame_bgr=frame,
                frame_idx=frame_idx,
                run_grin=run_grin,
                grin_horizon=16,
            )
            print(sg_frame)

            current_entities = {n.entity_id: n.box for n in sg_frame.nodes}
            change_detected = _detect_change(current_entities, last_entities)

            if change_detected and sg_frame.nodes:
                tensor_sg = system.build_scene_graph_tensor_frame(sg_frame)
                planner.submit(frame_idx, tensor_sg)

            for _, plan in planner.poll_results():
                if "error" not in plan:
                    if plan.get("caption"):
                        last_caption = plan["caption"]
                    if isinstance(plan.get("focus_targets"), list):
                        last_focus_targets = plan["focus_targets"]

            last_entities = current_entities

            # ---------------- Visualization ----------------
            vis = frame.copy()

            for node in sg_frame.nodes:
                x1, y1, x2, y2 = map(int, node.box)

                draw_bbox_with_outline(vis, x1, y1, x2, y2, COLOR_DEFAULT)
                draw_label_badge(
                    vis,
                    x1,
                    y1,
                    f"{node.cls}#{node.entity_id}",
                    float(node.score),
                    COLOR_DEFAULT,
                )

            last_vis = vis.copy()

        # --------------------------------------------------
        # Display + REAL timing video write
        # --------------------------------------------------
        now = time.perf_counter()
        delta_t = now - last_present_t
        last_present_t = now

        if last_vis is not None:
            cv2.imshow("Perception w/ Async LLM", last_vis)

            repeat = max(1, int(round(delta_t / FRAME_TIME)))
            for _ in range(repeat):
                video_writer.write(last_vis)

        # --------------------------------------------------
        # FPS floor (minimum pacing only)
        # --------------------------------------------------
        loop_elapsed = time.perf_counter() - loop_start_t
        remaining = FRAME_TIME - loop_elapsed
        if remaining > 0:
            time.sleep(remaining)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord(" "):
            paused = not paused
            print("[PAUSE]" if paused else "[RESUME]")

    planner.shutdown()
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


# ============================================================
# Change Detection
# ============================================================
def _detect_change(current_entities: dict, last_entities: dict, iou_thresh: float = 0.5) -> bool:
    if len(current_entities) != len(last_entities):
        return True
    if set(current_entities.keys()) != set(last_entities.keys()):
        return True
    for eid, box in current_entities.items():
        if box_iou_xyxy(box, last_entities[eid]) < iou_thresh:
            return True
    return False


# ============================================================
# Entrypoint
# ============================================================
if __name__ == "__main__":
    run_video_demo(video_path="input/video.mp4")
