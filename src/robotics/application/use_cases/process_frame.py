# src/robotics/application/use_cases/process_frame.py

from __future__ import annotations
from typing import Optional
from dataclasses import dataclass

from robotics.domain.ports.detection_port import DetectionPort
from robotics.domain.ports.tracking_port import TrackingPort
from robotics.domain.scene_graph.entities import SceneGraph, SceneNode, SceneEdge
from robotics.domain.scene_graph.reasoning import SGCLReasoner
from robotics.domain.action.policy import PolicyEngine
from robotics.application.dto import FrameContext


@dataclass
class ProcessFrameUseCase:
    detector: DetectionPort
    tracker: Optional[TrackingPort]
    sgcl: SGCLReasoner
    policy: PolicyEngine

    def execute(self, ctx: FrameContext) -> FrameContext:
        """
        1) Detection → 도메인 DetectionResult
        2) Tracking → track_id 보정
        3) SceneGraph 생성 (노드 우선)
        4) SGCL Logical Rule → Alerts
        5) Policy → Action 결정
        """

        # 1) YOLO-World Inference
        detections = self.detector.detect(ctx.frame_bgr)

        # 2) Tracking
        if self.tracker:
            detections = self.tracker.update_tracks(
                detections,
                ctx.frame_bgr.shape
            )

        # DTO 갱신
        ctx.detections = detections

        # 3) SceneGraph 생성(Edges는 추후 SGG 연동)
        nodes = [
            SceneNode(
                node_id=det.track_id if det.track_id is not None else idx,
                label=det.class_name,
                bbox=det.bbox,
                attributes={
                    "score": det.score,
                }
            )
            for idx, det in enumerate(detections)
        ]

        # Edges는 지금은 빈 상태로 유지
        edges: list[SceneEdge] = []

        graph = SceneGraph(nodes=nodes, edges=edges)
        ctx.scene_graph = graph

        # 4) SGCL(Static Geometric & Context Logic)
        alerts = self.sgcl.infer_risks(graph)
        ctx.alerts = alerts

        # 5) Policy 결정
        action = self.policy.decide(graph, alerts)
        ctx.action = action

        return ctx
