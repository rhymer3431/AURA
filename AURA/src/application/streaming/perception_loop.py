"""Application-level perception loop (framework-free)."""
import asyncio
from typing import Dict, Optional, List

from src.domain.streaming.port import (
    FrameSourcePort,
    PerceptionPort,
    ScenePlanPort,
    VideoSinkPort,
    MetadataSinkPort,
)
from src.utils.streaming.serializer import (
    serialize_frame_entities,
    serialize_relations,
    serialize_sg_diff,
)
from src.utils.box_iou_xyxy import box_iou_xyxy


async def run_perception_stream(
    perception: PerceptionPort,
    scene_planner: ScenePlanPort,
    video_sink: VideoSinkPort,
    metadata_sink: MetadataSinkPort,
    frame_source: FrameSourcePort,
    target_fps: float,
    stop_event: Optional[asyncio.Event] = None,
    on_shutdown: Optional[callable] = None,
    use_source_timing: bool = False,
) -> None:
    delay = 1.0 / target_fps if target_fps > 0 else 0

    frame_idx = 0
    last_entities: Dict[int, List[float]] = {}
    last_caption: str = ""
    last_focus_targets: List[str] = []

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break

            frame_bgr = await frame_source.read_frame()
            if frame_bgr is None:
                if stop_event is not None and stop_event.is_set():
                    break
                await asyncio.sleep(0.01)
                continue

            frame_idx += 1
            run_grin = frame_idx % 10 == 0

            # -------------------------------------------------
            # Perception processing (diff calculation included)
            # -------------------------------------------------
            sg_frame, sg_diff, _ = perception.process_frame(
                frame_bgr, frame_idx, run_grin, max_entities=16
            )

            # Scene plan block remains the same
            current_entities = {n.entity_id: n.box for n in sg_frame.nodes}
            change_detected = _detect_change(current_entities, last_entities)

            if change_detected and len(sg_frame.nodes) > 0:
                tensor_sg = perception.build_scene_graph_tensor_frame(sg_frame)
                scene_planner.submit(frame_idx, tensor_sg)

            for idx, plan in scene_planner.poll_results():
                caption = plan.get("caption", "")
                focus_targets = plan.get("focus_targets", [])
                if caption:
                    last_caption = caption
                if isinstance(focus_targets, list):
                    last_focus_targets = [str(x) for x in focus_targets]
                    perception.update_focus_classes(last_focus_targets)

            last_entities = current_entities

            # -------------------------------------------------
            # Tracking metadata is sent every frame
            # -------------------------------------------------
            metadata = {
                "type": "metadata",
                "frameIdx": frame_idx,
                "caption": last_caption,
                "focusTargets": last_focus_targets,
                "entities": serialize_frame_entities(sg_frame),
                "entityRecords": perception.ltm_entities,
            }

            live_metadata_getter = getattr(frame_source, "get_live_metadata", None)
            if callable(live_metadata_getter):
                try:
                    live_metadata = live_metadata_getter()
                    if isinstance(live_metadata, dict):
                        metadata.update(live_metadata)
                except Exception:
                    pass

            # -------------------------------------------------
            # Scene graph (nodes/edges) is sent only when it changes
            #   - First frame (sg_diff is None) sends the initial snapshot
            #   - Afterwards only when there is an actual diff
            # -------------------------------------------------
            has_graph_diff = sg_diff is not None and any(len(v) > 0 for v in sg_diff.values())
            should_send_scene_graph = sg_diff is None or has_graph_diff

            if should_send_scene_graph:
                relation_name_map = getattr(perception, "RELATION_ID_TO_NAME", {})
                metadata["relations"] = serialize_relations(sg_frame, relation_name_map)
                if has_graph_diff:
                    metadata["sceneGraphDiff"] = serialize_sg_diff(sg_diff, relation_name_map)

            # -------------------------------------------------
            # Metadata is always sent
            # -------------------------------------------------
            await metadata_sink.send_metadata(metadata)

            # -------------------------------------------------
            # Video frame transmission
            # -------------------------------------------------
            await video_sink.send_frame(frame_idx, frame_bgr)

            if delay > 0 and not use_source_timing:
                await asyncio.sleep(delay)

    finally:
        await frame_source.close()
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
