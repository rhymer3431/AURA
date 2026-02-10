"""Application-level perception loop (framework-free)."""
import asyncio
import cv2
from typing import Dict, Any, Optional, List

from domain.streaming.ports import (
    PerceptionPort,
    ScenePlanPort,
    VideoSinkPort,
    MetadataSinkPort,
)
from infrastructure.streaming.visualization.draw_bbox import FrameVisualizer
from domain.utils.box_iou_xyxy import box_iou_xyxy


async def run_perception_stream(
    perception: PerceptionPort,
    scene_planner: ScenePlanPort,
    video_sink: VideoSinkPort,
    metadata_sink: MetadataSinkPort,
    video_path: str,
    target_fps: float,
    frame_max_width: int,
    stop_event: Optional[asyncio.Event] = None,
    on_shutdown: Optional[callable] = None,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    delay = 1.0 / target_fps if target_fps > 0 else 0

    frame_idx = 0
    last_entities: Dict[int, List[float]] = {}
    last_caption: str = ""
    last_focus_targets: List[str] = []

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break

            ok, frame_bgr = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_idx += 1
            run_grin = frame_idx % 10 == 0

            sg_frame, _ = perception.process_frame(
                frame_bgr, frame_idx, run_grin, max_entities=16
            )

            current_entities = {n.entity_id: n.box for n in sg_frame.nodes}
            change_detected = _detect_change(current_entities, last_entities)
            if change_detected and len(sg_frame.nodes) > 0:
                simple_sg = perception.build_simple_scene_graph_frame(sg_frame)
                scene_planner.submit(frame_idx, simple_sg)

            for idx, plan in scene_planner.poll_results():
                caption = plan.get("caption", "")
                focus_targets = plan.get("focus_targets", [])
                if caption:
                    last_caption = caption
                if isinstance(focus_targets, list):
                    last_focus_targets = [str(x) for x in focus_targets]
                    perception.update_focus_classes(last_focus_targets)

            last_entities = current_entities

            relation_name_map = getattr(perception, "RELATION_ID_TO_NAME", {})
            relations = _serialize_relations(sg_frame, relation_name_map)
            entity_records = perception.ltm_entities
            metadata = {
                "type": "metadata",
                "frameIdx": frame_idx,
                "caption": last_caption,
                "focusTargets": last_focus_targets,
                "entities": _serialize_frame_entities(sg_frame),
                "relations": relations,
                "entityRecords": entity_records,
            }
            
            await metadata_sink.send_metadata(metadata)

            await video_sink.send_frame(frame_idx, frame_bgr)

            await asyncio.sleep(delay)
    finally:
        cap.release()
        if on_shutdown is not None:
            await on_shutdown()


def _detect_change(
    current_entities: Dict[int, List[float]], last_entities: Dict[int, List[float]]
) -> bool:
    """Detect scene changes between frames using IoU on entity boxes."""
    if len(current_entities) != len(last_entities):
        return True
    if set(current_entities.keys()) != set(last_entities.keys()):
        return True
    for eid, box in current_entities.items():
        if box_iou_xyxy(box, last_entities[eid]) < 0.5:
            return True
    return False

 
def _serialize_frame_entities(sg_frame) -> List[Dict[str, Any]]:
    entities: List[Dict[str, Any]] = []
    for n in sg_frame.nodes:
        entities.append(
            {
                "entityId": int(n.entity_id),
                "trackId": n.track_id if n.track_id is not None else -1,
                "cls": n.cls,
                "box": [float(x) for x in n.box],
                "score": float(getattr(n, "score", 1.0)),
            }
        )
    return entities


def _serialize_relations(sg_frame, rel_id_to_name: Dict[int, str]) -> List[Dict[str, Any]]:
    relations: List[Dict[str, Any]] = []
    nodes = sg_frame.nodes
    for sub_idx, obj_idx, rel_id in getattr(sg_frame, "relations", []):
        if sub_idx >= len(nodes) or obj_idx >= len(nodes):
            continue
        subj = nodes[sub_idx]
        obj = nodes[obj_idx]
        relations.append(
            {
                "subjectEntityId": int(subj.entity_id),
                "objectEntityId": int(obj.entity_id),
                "relation": rel_id_to_name.get(rel_id, str(rel_id)),
                "relationId": int(rel_id),
            }
        )
    return relations
