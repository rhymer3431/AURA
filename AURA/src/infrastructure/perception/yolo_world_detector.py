# pipeline.py (perception_pipeline.py)
# -*- coding: utf-8 -*-

import shutil
from pathlib import Path
from typing import Dict, List, Optional

import torch

from src.infrastructure.perception.node_encoder import YoloWorldTrackNodeEncoder
from src.infrastructure.logging.pipeline_logger import PipelineLogger


# NOTE:
# YOLOE-26 starts with placeholder class names ("0", "1", ...). Without
# explicitly setting text prompts, detections may stay empty in practice.
_FALLBACK_OPEN_VOCAB_CLASSES: List[str] = [
    "person",
    "bus",
    "car",
    "truck",
    "bicycle",
    "motorcycle",
    "chair",
    "couch",
    "bed",
    "dining table",
    "tv",
    "laptop",
    "cell phone",
    "book",
    "bottle",
    "cup",
    "bowl",
    "refrigerator",
    "microwave",
    "oven",
    "sink",
    "toilet",
    "clock",
    "potted plant",
    "backpack",
    "handbag",
    "suitcase",
    "mouse",
    "keyboard",
]


class YoloWorldDetector:
    """
    Wrapper around YoloWorldTrackNodeEncoder used elsewhere in this project.
    """

    def __init__(
        self,
        weight: str = "models/yoloe-26s-seg.pt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        conf: float = 0.25,
        logger: Optional[PipelineLogger] = None,
    ):
        self.device = device
        self.conf = conf
        self.logger = logger
        self._mobileclip_repair_attempted = False
        self._classes_applied = False
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

        # YOLOE-26 placeholder vocab("0","1",...)이면 fallback open-vocab 사용
        if self._is_placeholder_vocab(self.default_classes):
            self.default_classes = list(_FALLBACK_OPEN_VOCAB_CLASSES)

        self.current_classes = list(self.default_classes)
        self._classes_applied = self._apply_classes(self.current_classes, reason="startup-default")
        if self._classes_applied:
            print(f"[YOLO] Default classes applied ({len(self.current_classes)}).")
        elif self.current_classes:
            print("[YOLO] Failed to apply default classes; detections may be empty.")

    @torch.no_grad()
    def detect(self, frame_bgr) -> List[Dict]:
        # Retry once lazily if startup class-application failed.
        if not self._classes_applied and self.current_classes:
            self._classes_applied = self._apply_classes(self.current_classes, reason="lazy-detect-init")

        results = self.encoder.model(frame_bgr, conf=self.conf, verbose=False)[0]
        if self.logger is not None:
            self.logger.log(
                module="YOLO",
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
                }
            )
        return dets

    def update_focus_classes(self, focus_targets: List[str]):
        """
        Update model classes based on LLM focus when supported by the backend.
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
                self._classes_applied = self._apply_classes(normalized, reason="focus-update")
                self.current_classes = list(normalized)
                print(f"[YOLO] Focus classes updated: {self.current_classes}")
                return

        # focus_targets가 비었거나 유효한 문자열이 없으면 기본 클래스로 reset
        self._classes_applied = self._apply_classes(self.default_classes, reason="focus-reset")
        self.current_classes = list(self.default_classes)
        print(f"[YOLO] Focus classes reset to default ({len(self.default_classes)})")

    @staticmethod
    def _is_placeholder_vocab(classes: List[str]) -> bool:
        if not classes:
            return True
        numeric_count = sum(1 for c in classes if c.strip().isdigit())
        return numeric_count >= max(1, int(len(classes) * 0.8))

    def _apply_classes(self, classes: List[str], reason: str) -> bool:
        if not classes:
            return False
        if not hasattr(self.encoder.model, "set_classes"):
            return False

        try:
            self.encoder.model.set_classes(classes)
            latest_names = getattr(self.encoder.model, "names", None)
            if latest_names is not None:
                self.encoder.names = latest_names
            return True
        except Exception as exc:
            if self._repair_mobileclip_if_needed(exc):
                try:
                    self.encoder.model.set_classes(classes)
                    latest_names = getattr(self.encoder.model, "names", None)
                    if latest_names is not None:
                        self.encoder.names = latest_names
                    print("[YOLO] set_classes recovered after MobileCLIP repair.")
                    return True
                except Exception as retry_exc:
                    print(f"[YOLO] set_classes retry failed ({reason}): {retry_exc}")
            else:
                print(f"[YOLO] set_classes failed ({reason}): {exc}")
            return False

    def _repair_mobileclip_if_needed(self, exc: Exception) -> bool:
        if self._mobileclip_repair_attempted:
            return False

        msg = str(exc).lower()
        looks_like_mobileclip_failure = (
            "mobileclip" in msg
            or "pytorchstreamreader" in msg
            or "central directory" in msg
        )
        if not looks_like_mobileclip_failure:
            return False

        self._mobileclip_repair_attempted = True
        try:
            from ultralytics.utils import SETTINGS
            from ultralytics.utils.downloads import attempt_download_asset
        except Exception as import_exc:
            print(f"[YOLO] MobileCLIP repair unavailable (import error): {import_exc}")
            return False

        try:
            downloaded = Path(
                attempt_download_asset("mobileclip2_b.ts", release="latest")
            ).resolve()
            if not downloaded.exists():
                print("[YOLO] MobileCLIP repair failed: downloaded file not found.")
                return False

            aura_root = Path(__file__).resolve().parents[3]
            target = aura_root / "models" / "mobileclip2_b.ts"
            target.parent.mkdir(parents=True, exist_ok=True)

            if downloaded != target.resolve():
                shutil.copy2(downloaded, target)

            # Validate file eagerly so we fail fast on broken downloads.
            torch.jit.load(str(target), map_location="cpu")
            SETTINGS.update(weights_dir=str(target.parent.resolve()))
            print(f"[YOLO] MobileCLIP repaired: {target}")
            return True
        except Exception as repair_exc:
            print(f"[YOLO] MobileCLIP repair failed: {repair_exc}")
            return False

