from __future__ import annotations

import re
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np

from common.cv2_compat import cv2
from memory import (
    EpisodeRecord,
    EpisodicMemoryStore,
    KeyframeRecord,
    MemoryContextBundle,
    MemoryConsolidator,
    MemoryNavigationTarget,
    MemoryQueryEngine,
    RecallQuery,
    RetrievedMemoryLine,
    ScratchpadState,
    SQLiteMemoryPersistence,
    SemanticMemoryStore,
    SpatialMemoryStore,
    TemporalMemoryStore,
    WorkingMemory,
)
from perception.speaker_events import SpeakerEvent
from services.semantic_consolidation import SemanticConsolidationService


class MemoryService:
    def __init__(self, *, db_path: str | None = None, keyframe_dir: str | None = None) -> None:
        self.spatial_store = SpatialMemoryStore()
        self.temporal_store = TemporalMemoryStore()
        self.episodic_store = EpisodicMemoryStore()
        self.semantic_store = SemanticMemoryStore()
        self.working_memory = WorkingMemory()
        self.query_engine = MemoryQueryEngine(self.spatial_store, self.semantic_store, self.working_memory)
        self.consolidator = MemoryConsolidator(self.episodic_store, self.semantic_store)
        self.semantic_consolidation = SemanticConsolidationService(self.semantic_store)
        self.persistence = SQLiteMemoryPersistence(Path(db_path)) if db_path is not None else None
        if self.persistence is not None:
            self.persistence.initialize()
        base_memory_dir = Path(db_path).resolve().parent if db_path is not None else Path("state") / "memory"
        self.keyframe_dir = Path(keyframe_dir) if keyframe_dir is not None else (base_memory_dir / "keyframes")
        self.keyframe_dir.mkdir(parents=True, exist_ok=True)
        self.keyframes: dict[str, KeyframeRecord] = {}
        self._keyframe_order: list[str] = []
        self._keyframe_seq = 0
        self.scratchpad = ScratchpadState()
        self._active_episode_id = ""

    def observe_objects(self, observations) -> list[object]:  # noqa: ANN001
        results = []
        for observation in observations:
            result = self.spatial_store.associate_observation(observation)
            self.temporal_store.remember_observation(observation, object_id=result.object_node.object_id)
            self._record_episode_observation(result.place_node.place_id, result.object_node.object_id)
            results.append(result)
        return results

    def update_from_observation(self, observation) -> object:  # noqa: ANN001
        return self.observe_objects([observation])[0]

    def record_perception_frame(
        self,
        *,
        frame_id: int,
        rgb_image,
        observations,
        robot_pose_xyz: tuple[float, float, float] | None = None,
        robot_yaw_rad: float = 0.0,
        instruction: str = "",
    ) -> list[object]:  # noqa: ANN001
        observation_list = list(observations)
        results = self.observe_objects(observation_list)
        keyframe = self._maybe_store_keyframe(
            frame_id=int(frame_id),
            rgb_image=rgb_image,
            observation_list=observation_list,
            results=results,
            robot_pose_xyz=robot_pose_xyz,
            robot_yaw_rad=float(robot_yaw_rad),
        )
        self._update_object_memory_summaries(observation_list, results, keyframe=keyframe)
        self._update_scratchpad_from_frame(
            instruction=str(instruction),
            observation_list=observation_list,
            results=results,
        )
        return results

    def set_planner_task(
        self,
        *,
        instruction: str,
        planner_mode: str,
        task_state: str,
        task_id: str = "",
        command_id: int = -1,
    ) -> None:
        normalized_instruction = str(instruction).strip()
        normalized_mode = str(planner_mode).strip().lower()
        normalized_state = str(task_state).strip().lower() or "active"
        previous = self.scratchpad
        keep_locations = (
            previous.instruction == normalized_instruction
            and previous.planner_mode == normalized_mode
            and previous.task_state in {"pending", "active"}
        )
        self.scratchpad = ScratchpadState(
            instruction=normalized_instruction,
            planner_mode=normalized_mode,
            task_state=normalized_state,
            task_id=str(task_id),
            command_id=int(command_id),
            goal_summary=self._summarize_instruction(normalized_instruction),
            checked_locations=list(previous.checked_locations if keep_locations else []),
            recent_hint=str(previous.recent_hint if keep_locations else ""),
            next_priority=self._default_next_priority(normalized_instruction),
            updated_at=time.time(),
        )

    def clear_planner_task(self, *, task_state: str = "idle", reason: str = "") -> None:
        self.scratchpad = ScratchpadState(
            task_state=str(task_state).strip().lower() or "idle",
            recent_hint=str(reason).strip(),
            updated_at=time.time(),
        )

    def build_memory_context(
        self,
        *,
        instruction: str,
        current_pose: tuple[float, float, float] | None = None,
        max_text_lines: int = 5,
        max_keyframes: int = 2,
    ) -> MemoryContextBundle | None:
        normalized_instruction = str(instruction).strip()
        if normalized_instruction == "":
            return None
        semantic_terms, spatial_terms, temporal_terms = self._decompose_query(normalized_instruction)
        scored_objects: list[tuple[float, object, object | None, str]] = []
        now = time.time()
        for object_node in self.spatial_store.objects.values():
            place = self.spatial_store.place_for_object(object_node.object_id)
            summary = str(object_node.metadata.get("memory_summary", "")).strip()
            room_id = "" if place is None else str(place.room_id)
            semantic_score = self._semantic_score(
                semantic_terms=semantic_terms,
                class_name=str(object_node.class_name),
                summary=summary,
                aliases=[str(alias) for alias in object_node.aliases],
                room_id=room_id,
            )
            spatial_score = self._spatial_score(
                spatial_terms=spatial_terms,
                room_id=room_id,
                metadata=dict(object_node.metadata),
            )
            temporal_score = self._temporal_score(
                temporal_terms=temporal_terms,
                last_seen=float(object_node.last_seen),
                now=now,
            )
            recency_bonus = max(0.0, 1.0 / (1.0 + max(now - float(object_node.last_seen), 0.0)))
            confidence_bonus = max(float(object_node.confidence), 0.0)
            proximity_bonus = 0.0
            if current_pose is not None:
                current_xy = np.asarray(current_pose[:2], dtype=np.float32)
                object_xy = np.asarray(object_node.last_pose[:2], dtype=np.float32)
                proximity_bonus = max(0.0, 1.0 / (1.0 + float(np.linalg.norm(object_xy - current_xy))))
            score = semantic_score + spatial_score + temporal_score + recency_bonus + confidence_bonus + proximity_bonus
            if score <= 0.0:
                continue
            scored_objects.append((score, object_node, place, summary))

        scored_objects.sort(key=lambda item: (item[0], float(item[1].last_seen)), reverse=True)
        text_lines: list[RetrievedMemoryLine] = []
        seen_entities: set[str] = set()
        for score, object_node, place, summary in scored_objects:
            if object_node.object_id in seen_entities:
                continue
            seen_entities.add(object_node.object_id)
            text_lines.append(
                RetrievedMemoryLine(
                    text=summary or self._fallback_memory_summary(object_node, place),
                    score=float(score),
                    source_type="object_memory",
                    entity_id=str(object_node.object_id),
                    keyframe_id=str(object_node.metadata.get("keyframe_id", "")),
                )
            )
            if len(text_lines) >= max(int(max_text_lines), 0):
                break

        rules = self.semantic_store.matching_rules(
            intent=self._infer_intent(normalized_instruction),
            target_class=self._infer_target_class(normalized_instruction),
            room_id=self._infer_room_id(normalized_instruction),
        )
        for rule in rules:
            if len(text_lines) >= max(int(max_text_lines), 0):
                break
            text_lines.append(
                RetrievedMemoryLine(
                    text=str(rule.description),
                    score=max(float(rule.success_rate), 0.0) + float(rule.support_count),
                    source_type="semantic_rule",
                    entity_id=str(rule.rule_key),
                )
            )

        keyframes: list[KeyframeRecord] = []
        seen_keyframes: set[str] = set()
        for line in text_lines:
            if line.keyframe_id == "" or line.keyframe_id in seen_keyframes:
                continue
            record = self.keyframes.get(line.keyframe_id)
            if record is None:
                continue
            seen_keyframes.add(record.keyframe_id)
            keyframes.append(record)
            if len(keyframes) >= max(int(max_keyframes), 0):
                break

        if len(keyframes) < max(int(max_keyframes), 0):
            for keyframe_id in reversed(self._keyframe_order):
                if keyframe_id in seen_keyframes:
                    continue
                record = self.keyframes.get(keyframe_id)
                if record is None:
                    continue
                if self._keyframe_matches_query(record, spatial_terms=spatial_terms, semantic_terms=semantic_terms):
                    seen_keyframes.add(record.keyframe_id)
                    keyframes.append(record)
                if len(keyframes) >= max(int(max_keyframes), 0):
                    break

        crop_path = ""
        for line in text_lines:
            if line.entity_id == "":
                continue
            object_node = self.spatial_store.objects.get(line.entity_id)
            if object_node is None:
                continue
            maybe_crop = str(object_node.metadata.get("keyframe_crop_path", "")).strip()
            if maybe_crop != "":
                crop_path = maybe_crop
                break

        return MemoryContextBundle(
            instruction=normalized_instruction,
            scratchpad=self.scratchpad if self.scratchpad.instruction != "" else None,
            text_lines=text_lines[: max(int(max_text_lines), 0)],
            keyframes=keyframes[: max(int(max_keyframes), 0)],
            crop_path=crop_path,
            latent_backend_hint="llama.cpp_s2_only",
        )

    def resolve_navigation_target(
        self,
        *,
        instruction: str,
        current_pose: tuple[float, float, float] | None = None,
        target_class: str = "",
        room_id: str = "",
    ) -> MemoryNavigationTarget | None:
        normalized_instruction = str(instruction).strip()
        if normalized_instruction == "":
            return None
        recall = self.query_engine.recall_object(
            RecallQuery(
                query_text=normalized_instruction,
                target_class=str(target_class).strip(),
                intent="goto_remembered_object",
                room_id=str(room_id).strip(),
            ),
            current_pose=current_pose,
        )
        selected_object = recall.selected_object
        selected_place = recall.selected_place
        if selected_object is None and selected_place is None:
            return None
        memory_pose = (
            selected_object.last_pose
            if selected_object is not None
            else selected_place.pose  # type: ignore[union-attr]
        )
        goal_pose = (
            selected_place.pose
            if selected_place is not None
            else selected_object.last_pose  # type: ignore[union-attr]
        )
        goal_pose = (float(goal_pose[0]), float(goal_pose[1]), 0.0)
        place = selected_place if selected_place is not None else self.spatial_store.place_for_object(selected_object.object_id)  # type: ignore[union-attr]
        return MemoryNavigationTarget(
            instruction=normalized_instruction,
            object_id="" if selected_object is None else str(selected_object.object_id),
            place_id="" if place is None else str(place.place_id),
            memory_pose_xyz=(float(memory_pose[0]), float(memory_pose[1]), float(memory_pose[2])),
            goal_pose_xyz=(float(goal_pose[0]), float(goal_pose[1]), float(goal_pose[2])),
            room_id="" if place is None else str(place.room_id),
            summary=""
            if selected_object is None
            else str(selected_object.metadata.get("memory_summary", "")).strip(),
        )

    def record_speaker_event(self, event: SpeakerEvent) -> None:
        self.temporal_store.record_event(
            "speaker_event",
            timestamp=float(event.timestamp),
            track_id=event.speaker_id,
            payload={"yaw_rad": float(event.direction_yaw_rad), "confidence": float(event.confidence), **event.metadata},
        )

    def record_speaker_binding(self, *, person_id: str, track_id: str, timestamp: float, pose=None) -> None:  # noqa: ANN001
        self.temporal_store.record_event(
            "speaker_binding",
            timestamp=float(timestamp),
            track_id=str(track_id),
            person_id=str(person_id),
            pose=pose,
        )
        if self._active_episode_id == "":
            return
        record = self.episodic_store.get(self._active_episode_id)
        if record is not None:
            record.speaker_person_id = str(person_id)

    def record_follow_target(self, *, person_id: str, track_id: str) -> None:
        if self._active_episode_id == "":
            return
        record = self.episodic_store.get(self._active_episode_id)
        if record is not None:
            record.follow_target_id = str(person_id or track_id)

    def record_candidate_attempt(
        self,
        *,
        object_id: str = "",
        place_id: str = "",
        semantic_rule_keys: list[str] | None = None,
    ) -> None:
        if self._active_episode_id == "":
            return
        record = self.episodic_store.get(self._active_episode_id)
        if record is None:
            return
        if object_id != "" and object_id not in record.candidate_object_ids:
            record.candidate_object_ids.append(object_id)
        if place_id != "" and place_id not in record.candidate_place_ids:
            record.candidate_place_ids.append(place_id)
        for rule_key in semantic_rule_keys or []:
            if rule_key != "" and rule_key not in record.semantic_rules_applied:
                record.semantic_rules_applied.append(rule_key)

    def record_recovery_action(self, action: str) -> None:
        if self._active_episode_id == "":
            return
        record = self.episodic_store.get(self._active_episode_id)
        if record is None:
            return
        normalized = str(action).strip()
        if normalized != "":
            record.recovery_actions.append(normalized)

    def start_episode(self, *, command_text: str, intent: str, target_json: dict[str, object]) -> str:
        episode_id = f"episode_{uuid.uuid4().hex[:12]}"
        self._active_episode_id = episode_id
        record = EpisodeRecord(
            episode_id=episode_id,
            command_text=str(command_text),
            intent=str(intent),
            target_json=dict(target_json),
            started_at=time.time(),
        )
        self.episodic_store.put(record)
        return episode_id

    def finish_episode(
        self,
        *,
        success: bool,
        failure_reason: str = "",
        recovery_actions: list[str] | None = None,
        summary_text: str = "",
    ) -> None:
        if self._active_episode_id == "":
            return
        record = self.episodic_store.get(self._active_episode_id)
        if record is None:
            return
        record.success = bool(success)
        record.failure_reason = str(failure_reason)
        if recovery_actions:
            record.recovery_actions.extend(str(action) for action in recovery_actions if str(action).strip() != "")
        if summary_text != "":
            record.summary_text = str(summary_text)
        record.ended_at = time.time()
        self.semantic_consolidation.summarize_episode(record)
        self.consolidator.consolidate_episode(record.episode_id)
        self._active_episode_id = ""

    def recall_object(
        self,
        *,
        query_text: str,
        target_class: str,
        intent: str,
        room_id: str = "",
        current_pose: tuple[float, float, float] | None = None,
    ):
        result = self.query_engine.recall_object(
            RecallQuery(
                query_text=str(query_text),
                target_class=str(target_class),
                intent=str(intent),
                room_id=str(room_id),
            ),
            current_pose=current_pose,
        )
        self.record_candidate_attempt(
            object_id=result.selected_object.object_id if result.selected_object is not None else "",
            place_id=result.selected_place.place_id if result.selected_place is not None else "",
            semantic_rule_keys=[rule.rule_key for rule in result.semantic_rules],
        )
        return result

    def preview_object_recall(
        self,
        *,
        query_text: str,
        target_class: str,
        intent: str,
        room_id: str = "",
        current_pose: tuple[float, float, float] | None = None,
    ):
        return self.query_engine.recall_object(
            RecallQuery(
                query_text=str(query_text),
                target_class=str(target_class),
                intent=str(intent),
                room_id=str(room_id),
            ),
            current_pose=current_pose,
        )

    def record_memory_policy_event(
        self,
        *,
        label: str,
        source: str,
        fallback_used: bool,
        shadow_only: bool,
        feature_snapshot: dict[str, object] | None = None,
        prompt_text: str = "",
    ) -> None:
        if self._active_episode_id == "":
            return
        record = self.episodic_store.get(self._active_episode_id)
        if record is None:
            return
        record.policy_events.append(
            {
                "label": str(label),
                "source": str(source),
                "fallback_used": bool(fallback_used),
                "shadow_only": bool(shadow_only),
                "feature_snapshot": dict(feature_snapshot or {}),
                "prompt_preview": str(prompt_text).splitlines()[:6],
                "timestamp": time.time(),
            }
        )

    def reacquire_follow_target(self, track_id: str, *, now: float, max_age_sec: float = 6.0):
        return self.temporal_store.reacquire_track(track_id, now=now, max_age_sec=max_age_sec)

    def persist_snapshot(self) -> int | None:
        if self.persistence is None:
            return None
        payload = {
            "places": [asdict(place) for place in self.spatial_store.places.values()],
            "objects": [asdict(obj) for obj in self.spatial_store.objects.values()],
            "semantic_rules": [asdict(rule) for rule in self.semantic_store.list()],
            "scratchpad": asdict(self.scratchpad),
            "keyframes": [asdict(record) for record in self.keyframes.values()],
        }
        return self.persistence.save_snapshot("memory_service", payload)

    def _maybe_store_keyframe(
        self,
        *,
        frame_id: int,
        rgb_image,
        observation_list: list[object],
        results: list[object],
        robot_pose_xyz: tuple[float, float, float] | None,
        robot_yaw_rad: float,
    ) -> KeyframeRecord | None:
        image = np.asarray(rgb_image, dtype=np.uint8) if rgb_image is not None else None
        if image is None or image.ndim != 3 or image.shape[-1] != 3:
            return None
        if not self._should_store_keyframe(
            observation_list=observation_list,
            results=results,
            robot_pose_xyz=robot_pose_xyz,
            robot_yaw_rad=float(robot_yaw_rad),
        ):
            return None

        timestamp = max((float(getattr(item, "timestamp", 0.0)) for item in observation_list), default=time.time())
        room_id = next(
            (
                str(getattr(result.place_node, "room_id", "") or getattr(observation, "room_id", ""))
                for observation, result in zip(observation_list, results, strict=False)
                if str(getattr(result.place_node, "room_id", "") or getattr(observation, "room_id", "")).strip() != ""
            ),
            "",
        )
        focus_labels = [str(getattr(item, "class_name", "")).strip() for item in observation_list if str(getattr(item, "class_name", "")).strip() != ""]
        focus_labels = list(dict.fromkeys(focus_labels))
        focus_object_ids = [
            str(result.object_node.object_id)
            for result in results
            if str(result.object_node.object_id).strip() != ""
        ]
        focus_object_ids = list(dict.fromkeys(focus_object_ids))

        self._keyframe_seq += 1
        keyframe_id = f"kf_{self._keyframe_seq:04d}"
        image_path = self.keyframe_dir / f"{keyframe_id}.jpg"
        cv2.imwrite(str(image_path), image)
        crop_paths: list[str] = []
        for index, (observation, result) in enumerate(zip(observation_list, results, strict=False), start=1):
            bbox = getattr(observation, "metadata", {}).get("bbox_xyxy") if hasattr(observation, "metadata") else None
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x0, y0, x1, y1 = [int(value) for value in bbox]
            x0 = max(x0, 0)
            y0 = max(y0, 0)
            x1 = min(x1, image.shape[1] - 1)
            y1 = min(y1, image.shape[0] - 1)
            if x1 <= x0 or y1 <= y0:
                continue
            crop = image[y0 : y1 + 1, x0 : x1 + 1]
            if crop.size == 0:
                continue
            crop_name = re.sub(r"[^a-z0-9_]+", "_", str(getattr(observation, "class_name", "object")).lower()).strip("_") or "object"
            crop_path = self.keyframe_dir / f"{keyframe_id}_{index:02d}_{crop_name}.jpg"
            cv2.imwrite(str(crop_path), crop)
            crop_paths.append(str(crop_path))
            result.object_node.metadata["keyframe_crop_path"] = str(crop_path)

        record = KeyframeRecord(
            keyframe_id=keyframe_id,
            image_path=str(image_path),
            crop_paths=crop_paths,
            summary=self._summarize_keyframe(observation_list=observation_list, room_id=room_id),
            timestamp=float(timestamp),
            source_frame_id=int(frame_id),
            robot_pose=(0.0, 0.0, 0.0) if robot_pose_xyz is None else tuple(float(value) for value in robot_pose_xyz[:3]),
            robot_yaw_rad=float(robot_yaw_rad),
            room_id=room_id,
            focus_labels=focus_labels,
            focus_object_ids=focus_object_ids,
        )
        self.keyframes[record.keyframe_id] = record
        self._keyframe_order.append(record.keyframe_id)
        while len(self._keyframe_order) > 256:
            stale_id = self._keyframe_order.pop(0)
            self.keyframes.pop(stale_id, None)
        for result in results:
            result.object_node.metadata["keyframe_id"] = record.keyframe_id
        return record

    def _should_store_keyframe(
        self,
        *,
        observation_list: list[object],
        results: list[object],
        robot_pose_xyz: tuple[float, float, float] | None,
        robot_yaw_rad: float,
    ) -> bool:
        if not self._keyframe_order:
            return True
        latest = self.keyframes.get(self._keyframe_order[-1])
        if latest is None:
            return True
        room_id = next(
            (
                str(getattr(result.place_node, "room_id", "") or getattr(observation, "room_id", ""))
                for observation, result in zip(observation_list, results, strict=False)
                if str(getattr(result.place_node, "room_id", "") or getattr(observation, "room_id", "")).strip() != ""
            ),
            "",
        )
        room_changed = room_id != "" and room_id != latest.room_id
        pose_changed = False
        if robot_pose_xyz is not None:
            current_pose = np.asarray(robot_pose_xyz[:3], dtype=np.float32)
            last_pose = np.asarray(latest.robot_pose[:3], dtype=np.float32)
            pose_changed = float(np.linalg.norm(current_pose - last_pose)) >= 1.0
        yaw_changed = abs(float(robot_yaw_rad) - float(latest.robot_yaw_rad)) >= 0.75
        new_object = any(not bool(getattr(result, "matched_existing", True)) for result in results)
        target_match = any(self._matches_instruction(str(getattr(item, "class_name", ""))) for item in observation_list)
        last_age = max(time.time() - float(latest.timestamp), 0.0)
        return bool(room_changed or pose_changed or yaw_changed or new_object or target_match or last_age >= 10.0)

    def _update_object_memory_summaries(
        self,
        observation_list: list[object],
        results: list[object],
        *,
        keyframe: KeyframeRecord | None,
    ) -> None:
        for observation, result in zip(observation_list, results, strict=False):
            room_id = str(getattr(result.place_node, "room_id", "") or getattr(observation, "room_id", "")).strip()
            object_node = result.object_node
            object_node.metadata["room_id"] = room_id
            object_node.metadata["last_bbox_xyxy"] = list(getattr(observation, "metadata", {}).get("bbox_xyxy", []))
            object_node.metadata["last_observed_timestamp"] = float(getattr(observation, "timestamp", 0.0))
            object_node.metadata["bearing_deg"] = self._bearing_deg(getattr(observation, "metadata", {}))
            object_node.metadata["memory_summary"] = self._build_observation_summary(
                class_name=str(getattr(observation, "class_name", object_node.class_name)),
                room_id=room_id,
                metadata=dict(getattr(observation, "metadata", {})),
            )
            if keyframe is not None:
                object_node.metadata["keyframe_id"] = keyframe.keyframe_id

    def _update_scratchpad_from_frame(
        self,
        *,
        instruction: str,
        observation_list: list[object],
        results: list[object],
    ) -> None:
        if self.scratchpad.task_state not in {"pending", "active"}:
            return
        active_instruction = str(instruction).strip() or str(self.scratchpad.instruction).strip()
        if active_instruction == "":
            return
        checked_locations = list(self.scratchpad.checked_locations)
        for observation, result in zip(observation_list, results, strict=False):
            room_id = str(getattr(result.place_node, "room_id", "") or getattr(observation, "room_id", "")).strip()
            if room_id != "" and room_id not in checked_locations:
                checked_locations.append(room_id)
        checked_locations = checked_locations[-8:]
        matched_label = next(
            (
                str(getattr(observation, "class_name", "")).strip()
                for observation in observation_list
                if self._matches_instruction(str(getattr(observation, "class_name", "")))
            ),
            "",
        )
        recent_hint = self.scratchpad.recent_hint
        next_priority = self.scratchpad.next_priority
        if matched_label != "":
            recent_hint = f"Observed {matched_label} in the current scene."
            next_priority = "Use the remembered evidence to ground the next System 2 decision."
        elif checked_locations:
            recent_hint = f"Checked {checked_locations[-1]}."
            next_priority = self._default_next_priority(active_instruction)
        self.scratchpad = ScratchpadState(
            instruction=self.scratchpad.instruction,
            planner_mode=self.scratchpad.planner_mode,
            task_state=self.scratchpad.task_state,
            task_id=self.scratchpad.task_id,
            command_id=self.scratchpad.command_id,
            goal_summary=self.scratchpad.goal_summary,
            checked_locations=checked_locations,
            recent_hint=recent_hint,
            next_priority=next_priority,
            updated_at=time.time(),
        )

    def _summarize_keyframe(self, *, observation_list: list[object], room_id: str) -> str:
        labels = [str(getattr(item, "class_name", "")).strip() for item in observation_list if str(getattr(item, "class_name", "")).strip() != ""]
        labels = list(dict.fromkeys(labels))
        if labels:
            joined = ", ".join(labels[:3])
            if room_id != "":
                return f"{joined} visible in {room_id}."
            return f"{joined} visible in the current scene."
        if room_id != "":
            return f"Scene context captured in {room_id}."
        return "Scene context captured."

    def _build_observation_summary(self, *, class_name: str, room_id: str, metadata: dict[str, object]) -> str:
        room_text = room_id if room_id != "" else "the scene"
        bearing_text = self._bearing_text(metadata)
        depth_m = metadata.get("depth_m")
        summary = f"{class_name} seen in {room_text}"
        if bearing_text != "":
            summary += f" on the {bearing_text}"
        if isinstance(depth_m, (int, float)):
            summary += f" at about {float(depth_m):.1f}m"
        return summary + "."

    def _fallback_memory_summary(self, object_node, place) -> str:  # noqa: ANN001
        room_id = "" if place is None else str(place.room_id)
        return self._build_observation_summary(
            class_name=str(object_node.class_name),
            room_id=room_id,
            metadata=dict(object_node.metadata),
        )

    def _decompose_query(self, instruction: str) -> tuple[list[str], list[str], list[str]]:
        tokens = [token for token in re.findall(r"[A-Za-z0-9_+-]{2,}|[가-힣]{2,}", instruction.lower()) if len(token) >= 2]
        spatial_terms = [token for token in tokens if token in {"left", "right", "front", "back", "near", "table", "hallway", "kitchen", "sink", "corridor", "왼쪽", "오른쪽", "앞", "뒤", "근처", "테이블", "복도", "주방", "싱크대"}]
        temporal_terms = [token for token in tokens if token in {"recent", "recently", "before", "earlier", "previous", "last", "아까", "전에", "방금", "최근"}]
        semantic_terms = [
            token
            for token in tokens
            if token not in spatial_terms
            and token not in temporal_terms
            and token not in {"find", "look", "move", "go", "target", "찾아", "가", "봐", "보이는"}
        ]
        return semantic_terms, spatial_terms, temporal_terms

    def _semantic_score(
        self,
        *,
        semantic_terms: list[str],
        class_name: str,
        summary: str,
        aliases: list[str],
        room_id: str,
    ) -> float:
        if not semantic_terms:
            return 0.25
        haystack = " ".join([class_name.lower(), summary.lower(), room_id.lower(), *(alias.lower() for alias in aliases)])
        score = 0.0
        for term in semantic_terms:
            if term in haystack:
                score += 2.0
        return score

    def _spatial_score(self, *, spatial_terms: list[str], room_id: str, metadata: dict[str, object]) -> float:
        if not spatial_terms:
            return 0.0
        score = 0.0
        room_lower = room_id.lower()
        bearing_text = self._bearing_text(metadata)
        metadata_text = " ".join([room_lower, bearing_text.lower(), str(metadata.get("memory_summary", "")).lower()])
        for term in spatial_terms:
            if term in metadata_text:
                score += 1.0
        return score

    def _temporal_score(self, *, temporal_terms: list[str], last_seen: float, now: float) -> float:
        if not temporal_terms:
            return 0.0
        age_sec = max(float(now) - float(last_seen), 0.0)
        return max(0.0, 2.0 - min(age_sec / 15.0, 2.0))

    def _keyframe_matches_query(self, record: KeyframeRecord, *, spatial_terms: list[str], semantic_terms: list[str]) -> bool:
        if not semantic_terms and not spatial_terms:
            return True
        haystack = " ".join(
            [
                str(record.summary).lower(),
                str(record.room_id).lower(),
                *(label.lower() for label in record.focus_labels),
            ]
        )
        return any(term in haystack for term in [*semantic_terms, *spatial_terms])

    def _matches_instruction(self, label: str) -> bool:
        instruction = str(self.scratchpad.instruction).strip().lower()
        normalized_label = str(label).strip().lower()
        if instruction == "" or normalized_label == "":
            return False
        return normalized_label in instruction

    def _infer_intent(self, instruction: str) -> str:
        lowered = instruction.lower()
        if any(token in lowered for token in ("find", "search", "찾아", "찾기")):
            return "find"
        if any(token in lowered for token in ("inspect", "look", "봐", "확인")):
            return "inspect"
        return "find"

    def _infer_target_class(self, instruction: str) -> str:
        semantic_terms, _, _ = self._decompose_query(instruction)
        return semantic_terms[0] if semantic_terms else ""

    def _infer_room_id(self, instruction: str) -> str:
        _, spatial_terms, _ = self._decompose_query(instruction)
        for term in spatial_terms:
            if term in {"kitchen", "hallway", "corridor", "주방", "복도"}:
                return term
        return ""

    def _summarize_instruction(self, instruction: str) -> str:
        if instruction.strip() == "":
            return ""
        semantic_terms, spatial_terms, _ = self._decompose_query(instruction)
        target = semantic_terms[0] if semantic_terms else instruction.strip()
        if spatial_terms:
            return f"Find {target} with focus on {' '.join(spatial_terms[:2])}."
        return f"Find {target}."

    def _default_next_priority(self, instruction: str) -> str:
        target = self._infer_target_class(instruction)
        if target != "":
            return f"Continue searching for {target} with the latest memory cues."
        return "Continue searching using the latest memory cues."

    def _bearing_deg(self, metadata: dict[str, object]) -> float | None:
        value = metadata.get("bearing_yaw_rad")
        if not isinstance(value, (int, float)):
            return None
        return float(np.degrees(float(value)))

    def _bearing_text(self, metadata: dict[str, object]) -> str:
        bearing_deg = self._bearing_deg(metadata)
        if bearing_deg is None:
            return ""
        if bearing_deg <= -15.0:
            return "left"
        if bearing_deg >= 15.0:
            return "right"
        return "center"

    def _record_episode_observation(self, place_id: str, object_id: str) -> None:
        if self._active_episode_id == "":
            return
        record = self.episodic_store.get(self._active_episode_id)
        if record is None:
            return
        if place_id != "" and place_id not in record.visited_places:
            record.visited_places.append(place_id)
        if object_id != "" and object_id not in record.objects_seen:
            record.objects_seen.append(object_id)
