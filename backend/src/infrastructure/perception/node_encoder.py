import torch
import torchvision.ops as ops
from typing import Optional, List, Tuple, Any
from ultralytics import YOLOWorld

from infrastructure.logging.pipeline_logger import PipelineLogger


class YoloWorldTrackNodeEncoder:
    """
    - YOLO-World(yolov8s-worldv2.pt) track() 결과에서 노드 로드
    - P3 hook으로 ROI feature 추출
    - 각 주요 YOLO-World 레이어에 forward hook을 걸어 모듈별 로그를 남김 (yw_#)
    """

    def __init__(self, weight: str = "yolov8s-worldv2.pt", device: Optional[str] = None, logger: Optional[PipelineLogger] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger

        # 1) YOLO-World 로드
        self.model = YOLOWorld(weight).to(self.device)
        self.model.eval()

        # 2) 텍스트 임베딩 캐시
        txt = self.model.model.txt_feats  # (1, num_classes, D_text)
        if txt.dim() == 3 and txt.shape[0] == 1:
            txt = txt.squeeze(0)
        self.text_emb = txt.detach().to(self.device)  # (num_classes, D_text)

        self.names = self.model.model.names  # cls_id -> name

        # 3) Feature hook (P3)
        self.features = {"P3": None}
        self.model.model.model[12].register_forward_hook(self._hook_p3)

        # 4) Layer-wise logging hooks (yw_0 ... yw_22)
        self._register_layer_hooks()

    # -----------------------
    # Hook utilities
    # -----------------------
    def _hook_p3(self, module, inp, out):
        # out: (B=1, C, Hf, Wf)
        self.features["P3"] = out

    def _register_layer_hooks(self):
        """
        Register forward hooks on key YOLO-World layers with semantic names (yw_#).
        """
        # brain areas restricted to provided list
        layer_map: List[Tuple[int, str, str]] = [
            (0, "yw_0", "V1"),
            (1, "yw_1", "V1"),
            (2, "yw_2", "V2"),
            (3, "yw_3", "V3"),
            (4, "yw_4", "V3"),
            (5, "yw_5", "V4"),
            (6, "yw_6", "V4"),
            (7, "yw_7", "IT"),
            (8, "yw_8", "IT"),
            (9, "yw_9", "IT"),
            (10, "yw_10", "Dorsal Stream"),
            (11, "yw_11", "TPJ"),
            (12, "yw_12", "PPC"),  # also used for P3 hook
            (13, "yw_13", "Dorsal Stream"),
            (14, "yw_14", "TPJ"),
            (15, "yw_15", "PPC"),
            (16, "yw_16", "V4"),
            (17, "yw_17", "TPJ"),
            (18, "yw_18", "PPC"),
            (19, "yw_19", "Dorsal Stream"),
            (20, "yw_20", "TPJ"),
            (21, "yw_21", "dlPFC"),
            (22, "yw_22", "PFC"),
        ]

        modules = self.model.model.model
        for idx, tag, brain in layer_map:
            if idx >= len(modules):
                continue

            def _make_hook(layer_idx: int, layer_tag: str, brain_area: str):
                def _hook(module, inp, out):
                    if self.logger is None:
                        return
                    try:
                        shape: Any = None
                        if hasattr(out, "shape"):
                            shape = list(out.shape)
                        elif isinstance(out, (list, tuple)) and out and hasattr(out[0], "shape"):
                            shape = [list(o.shape) for o in out]
                        self.logger.log(
                            module=layer_tag,
                            event="forward",
                            frame_idx=None,
                            matched_brain=brain_area,
                            layer_index=layer_idx,
                            shape=shape,
                        )
                    except Exception:
                        # logging must not break inference
                        pass

                return _hook

            modules[idx].register_forward_hook(_make_hook(idx, tag, brain))

    # -----------------------
    # Public API
    # -----------------------
    def nodes_from_result(self, result):
        """
        YOLO track() step 결과로부터 node 추출.
        return: list of dict per detection:
            track_id, cls_id, cls_name, score, bbox,
            roi_feat, face_feat, text_feat, node_feat
        """
        if result.boxes is None or len(result.boxes) == 0:
            return []

        P3 = self.features["P3"]
        if P3 is None:
            raise RuntimeError("P3 feature not captured. Check hook index (12).")

        _, C, Hf, Wf = P3.shape

        frame_bgr = result.orig_img  # BGR
        H, W = frame_bgr.shape[:2]

        nodes = []

        for b in result.boxes:
            track_id = int(b.id.item()) if b.id is not None else -1

            cls_id = int(b.cls.item())
            score = float(b.conf.item())
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cls_name = self.names[cls_id]

            # ROIAlign 좌표 변환
            sx = Wf / float(W)
            sy = Hf / float(H)
            x1f, y1f, x2f, y2f = x1 * sx, y1 * sy, x2 * sx, y2 * sy

            roi_box = torch.tensor([[0, x1f, y1f, x2f, y2f]], device=self.device, dtype=torch.float32)

            roi_feat = ops.roi_align(
                input=P3,
                boxes=roi_box,
                output_size=(7, 7),
                spatial_scale=1.0,
                sampling_ratio=-1,
                aligned=True,
            )  # (1, C, 7, 7)
            head_feat = None
            if cls_name == "person":
                head_y2f = (y1 + (y2 - y1) * 0.4) * sy  # 상단 40%
                head_box = torch.tensor([[0, x1f, y1f, x2f, head_y2f]], device=self.device, dtype=torch.float32)
                head_feat = ops.roi_align(
                    input=P3,
                    boxes=head_box,
                    output_size=(7, 7),
                    spatial_scale=1.0,
                    sampling_ratio=-1,
                    aligned=True,
                )

            roi_feat = roi_feat.mean(dim=(2, 3)).squeeze(0)  # (C,)
            if head_feat is not None:
                head_feat = head_feat.mean(dim=(2, 3)).squeeze(0)  # (C,)
            roi_feat = roi_feat.detach()
            if head_feat is not None:
                head_feat = head_feat.detach()

            text_feat = self.text_emb[cls_id]  # (D_text,)
            node_feat = torch.cat([roi_feat, text_feat], dim=0)  # (C + D_text,)

            nodes.append({
                "track_id": track_id,
                "cls_id": cls_id,
                "cls_name": cls_name,
                "score": score,
                "bbox": [x1, y1, x2, y2],
                "roi_feat": roi_feat,
                "face_feat": head_feat,
                "text_feat": text_feat,
                "node_feat": node_feat,
            })

        return nodes
