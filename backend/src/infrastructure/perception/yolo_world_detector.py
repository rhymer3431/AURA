# pipeline.py (perception_pipeline.py)
# -*- coding: utf-8 -*-

import torch
from typing import Dict, List, Optional

from infrastructure.perception.node_encoder import YoloWorldTrackNodeEncoder
from infrastructure.logging.pipeline_logger import PipelineLogger

class YoloWorldDetector:
    """
    Wrapper around YoloWorldTrackNodeEncoder used elsewhere in this project.
    """

    def __init__(
        self,
        weight: str = "yolov8s-worldv2.pt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        conf: float = 0.25,
        logger: Optional[PipelineLogger] = None,
    ):
        self.device = device
        self.conf = conf
        self.logger = logger
        self.encoder = YoloWorldTrackNodeEncoder(weight=weight, device=device, logger=logger)

        # names를 항상 "라벨 문자열 리스트"로 정규화
        names = getattr(self.encoder, "names", None)
        if names is None and hasattr(self.encoder, "model"):
            names = getattr(self.encoder.model, "names", [])
        if isinstance(names, dict):
            # dict: {id: "label"} 형태
            self.default_classes = [str(v) for v in names.values()]
        else:
            self.default_classes = [str(v) for v in names] if names is not None else []

        self.current_classes = list(self.default_classes)

    @torch.no_grad()
    def detect(self, frame_bgr) -> List[Dict]:
        results = self.encoder.model(frame_bgr, conf=self.conf, verbose=False)[0]
        if self.logger is not None:
            self.logger.log(
                module="YOLOWorld",
                event="forward_done",
                frame_idx=None,
                num_boxes=len(results.boxes) if results.boxes is not None else 0,
            )
        nodes = self.encoder.nodes_from_result(results)
        dets = []
        for i, n in enumerate(nodes):
            dets.append(
                {
                    "track_id": n["track_id"] if n["track_id"] is not None else i,
                    "cls": n["cls_name"],
                    "box": n["bbox"],
                    "score": n["score"],
                    "roi_feat": n["roi_feat"].detach().cpu(),
                    "face_feat": n.get("face_feat", None).detach().cpu() if n.get("face_feat", None) is not None else None,
                }
            )
        return dets

    def update_focus_classes(self, focus_targets: List[str]):
        """
        Update YOLO-World open-vocabulary classes based on LLM focus.
        - 입력 리스트에서 문자열만 필터링해서 사용.
        """
        if focus_targets:
            normalized: List[str] = []
            for x in focus_targets:
                # 문자열이 아니면 str로 캐스팅해서라도 사용 (e.g., int → "1")
                if isinstance(x, (str, int, float)):
                    s = str(x).strip()
                    if s:
                        normalized.append(s)

            if normalized:
                self.encoder.model.set_classes(normalized)
                self.current_classes = list(normalized)
                print(f"[YOLO-World] Focus classes updated: {self.current_classes}")
                return

        # focus_targets가 비었거나 유효한 문자열이 없으면 기본 클래스로 reset
        if self.default_classes:
            self.encoder.model.set_classes(self.default_classes)
        self.current_classes = list(self.default_classes)
        print(f"[YOLO-World] Focus classes reset to default ({len(self.default_classes)})")

