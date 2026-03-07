from __future__ import annotations

import time
from typing import Iterable

from .association import association_score, xy_distance
from .models import (
    AssociationResult,
    ObjectNode,
    ObjectSnapshot,
    ObsObject,
    PlaceNode,
    RelationEdge,
    pose3,
)


class SpatialMemoryStore:
    def __init__(
        self,
        *,
        place_merge_distance_m: float = 1.2,
        object_match_threshold: float = 0.45,
        static_conflict_distance_m: float = 1.5,
    ) -> None:
        self.place_merge_distance_m = float(place_merge_distance_m)
        self.object_match_threshold = float(object_match_threshold)
        self.static_conflict_distance_m = float(static_conflict_distance_m)
        self.places: dict[str, PlaceNode] = {}
        self.objects: dict[str, ObjectNode] = {}
        self.relations: list[RelationEdge] = []
        self._place_seq = 0
        self._object_seq = 0

    def _next_place_id(self) -> str:
        self._place_seq += 1
        return f"place_{self._place_seq:04d}"

    def _next_object_id(self, class_name: str) -> str:
        self._object_seq += 1
        return f"{class_name.lower()}_{self._object_seq:04d}"

    def ensure_place(
        self,
        pose: tuple[float, float, float] | list[float],
        *,
        room_id: str = "",
        place_id: str = "",
        timestamp: float | None = None,
    ) -> PlaceNode:
        now = time.time() if timestamp is None else float(timestamp)
        normalized_pose = pose3(pose)
        if place_id != "" and place_id in self.places:
            place = self.places[place_id]
            place.pose = normalized_pose
            place.room_id = room_id or place.room_id
            place.visit_count += 1
            place.last_seen = now
            return place

        best: PlaceNode | None = None
        best_distance = float("inf")
        for candidate in self.places.values():
            distance = xy_distance(normalized_pose, candidate.pose)
            if distance < self.place_merge_distance_m and distance < best_distance:
                best = candidate
                best_distance = distance
        if best is not None:
            best.pose = normalized_pose
            best.room_id = room_id or best.room_id
            best.visit_count += 1
            best.last_seen = now
            return best

        place = PlaceNode(
            place_id=self._next_place_id(),
            pose=normalized_pose,
            room_id=room_id,
            visit_count=1,
            first_seen=now,
            last_seen=now,
        )
        self.places[place.place_id] = place
        return place

    def retrieve_subgraph(
        self,
        *,
        place_id: str = "",
        pose: tuple[float, float, float] | list[float] | None = None,
        radius_m: float = 3.0,
    ) -> tuple[list[PlaceNode], list[ObjectNode]]:
        target_place_id = place_id
        if target_place_id == "" and pose is not None:
            center = pose3(pose)
            nearby_places = [place for place in self.places.values() if xy_distance(place.pose, center) <= float(radius_m)]
        else:
            nearby_places = [self.places[target_place_id]] if target_place_id in self.places else []
        place_ids = {place.place_id for place in nearby_places}
        nearby_objects = [obj for obj in self.objects.values() if obj.last_place_id in place_ids]
        return nearby_places, nearby_objects

    def recall_objects(self, *, class_name: str = "", query_text: str = "") -> list[ObjectNode]:
        normalized_class = class_name.strip().lower()
        normalized_query = query_text.strip().lower()
        results: list[ObjectNode] = []
        for candidate in self.objects.values():
            labels = [candidate.class_name.lower(), *(alias.lower() for alias in candidate.aliases)]
            if normalized_class != "" and candidate.class_name.lower() == normalized_class:
                results.append(candidate)
                continue
            if normalized_query != "" and any(token in normalized_query for token in labels):
                results.append(candidate)
        return sorted(results, key=lambda item: (item.last_seen, item.confidence), reverse=True)

    def associate_observation(self, observation: ObsObject) -> AssociationResult:
        now = float(observation.timestamp)
        anchor_place = self.ensure_place(
            observation.pose,
            room_id=observation.room_id,
            place_id=observation.place_id,
            timestamp=now,
        )
        _, local_objects = self.retrieve_subgraph(place_id=anchor_place.place_id, radius_m=self.place_merge_distance_m * 2.5)

        best_candidate: ObjectNode | None = None
        best_score = -1.0
        for candidate in self._candidate_objects(observation, local_objects):
            score, _ = association_score(observation, candidate, now=now)
            if score > best_score:
                best_candidate = candidate
                best_score = score

        if best_candidate is None or best_score < self.object_match_threshold:
            object_node = ObjectNode(
                object_id=self._next_object_id(observation.class_name),
                class_name=observation.class_name,
                track_id=observation.track_id,
                last_pose=pose3(observation.pose),
                last_place_id=anchor_place.place_id,
                first_seen=now,
                last_seen=now,
                confidence=float(observation.confidence),
                movable=bool(observation.movable),
                state=observation.state,
                embedding_id=observation.embedding_id,
                snapshots=[
                    ObjectSnapshot(
                        timestamp=now,
                        pose=pose3(observation.pose),
                        confidence=float(observation.confidence),
                        note="created",
                    )
                ],
                metadata=dict(observation.metadata),
            )
            self.objects[object_node.object_id] = object_node
            conflict_flag = False
            matched_existing = False
        else:
            object_node = best_candidate
            conflict_flag = (
                not bool(object_node.movable)
                and xy_distance(object_node.last_pose, pose3(observation.pose)) > self.static_conflict_distance_m
            )
            object_node.track_id = observation.track_id or object_node.track_id
            object_node.last_pose = pose3(observation.pose)
            object_node.last_place_id = anchor_place.place_id
            object_node.last_seen = now
            object_node.confidence = max(float(object_node.confidence), float(observation.confidence))
            object_node.state = observation.state or object_node.state
            object_node.embedding_id = observation.embedding_id or object_node.embedding_id
            object_node.conflict_flag = bool(conflict_flag)
            object_node.metadata.update(observation.metadata)
            object_node.snapshots.append(
                ObjectSnapshot(
                    timestamp=now,
                    pose=pose3(observation.pose),
                    confidence=float(observation.confidence),
                    note="updated",
                )
            )
            matched_existing = True

        self._upsert_relation(
            source_id=object_node.object_id,
            target_id=anchor_place.place_id,
            relation_type="in_room" if anchor_place.room_id != "" else "seen_from",
            timestamp=now,
            metadata={"room_id": anchor_place.room_id},
        )
        return AssociationResult(
            object_node=object_node,
            place_node=anchor_place,
            matched_existing=matched_existing,
            conflict_flag=bool(conflict_flag),
            score=max(best_score, 0.0),
        )

    def place_for_object(self, object_id: str) -> PlaceNode | None:
        obj = self.objects.get(object_id)
        if obj is None:
            return None
        return self.places.get(obj.last_place_id)

    def _candidate_objects(self, observation: ObsObject, local_objects: Iterable[ObjectNode]) -> list[ObjectNode]:
        candidates = []
        for candidate in self.objects.values():
            if observation.track_id != "" and candidate.track_id == observation.track_id:
                candidates.append(candidate)
        if candidates:
            return candidates
        for candidate in local_objects:
            if candidate.class_name == observation.class_name:
                candidates.append(candidate)
        return candidates

    def _upsert_relation(
        self,
        *,
        source_id: str,
        target_id: str,
        relation_type: str,
        timestamp: float,
        metadata: dict[str, str],
    ) -> None:
        for relation in self.relations:
            if (
                relation.source_id == source_id
                and relation.target_id == target_id
                and relation.relation_type == relation_type
            ):
                relation.last_updated = float(timestamp)
                relation.metadata.update(metadata)
                return
        self.relations.append(
            RelationEdge(
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                last_updated=float(timestamp),
                metadata=dict(metadata),
            )
        )
