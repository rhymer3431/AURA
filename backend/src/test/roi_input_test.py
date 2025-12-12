import torch
import torch.nn as nn
import cv2

from src.domain.perception.entity_node import EntityNode
from src.infrastructure.perception.perception_service_adapter import PerceptionServiceAdapter


class ROIToTextEmbedding(nn.Module):
    """
    ROI feature를 YOLO-World 텍스트 임베딩 차원(512D 등)으로 변환하는 모듈.
    """
    def __init__(self, in_dim=256, out_dim=512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        return self.proj(x)


def build_focus_text_embedding(node: EntityNode, proj_model: ROIToTextEmbedding):
    """
    EntityNode에서 ROI feature를 가져와 YOLO-World 텍스트 임베딩으로 변환.
    """
    roi_feat = node.roi_feat.float().detach()

    if roi_feat.dim() == 1:
        roi_feat = roi_feat.unsqueeze(0)  # (1, 256)

    focus_emb = proj_model(roi_feat)
    focus_emb = torch.nn.functional.normalize(focus_emb, dim=-1)

    return focus_emb


def run_yolo_with_focus(yolo_model, image_tensor, base_text_embs, focus_emb):
    """
    YOLO-World에 focus 텍스트 임베딩을 추가하여 실행.
    """
    text_embs = base_text_embs + [focus_emb]

    preds = yolo_model(
        image_tensor,
        text_embs=text_embs,  # 핵심
        verbose=False,
    )
    return preds


def focus_detection_pipeline(node, yolo_model, image_tensor, proj_model, base_text_embs):
    """
    EntityNode 기반 YOLO-World 탐색 강화 실행.
    """
    focus_emb = build_focus_text_embedding(node, proj_model)

    preds = run_yolo_with_focus(
        yolo_model=yolo_model,
        image_tensor=image_tensor,
        base_text_embs=base_text_embs,
        focus_emb=focus_emb,
    )
    return preds


# ============================================
#        실제 파이프라인 실행 루프
# ============================================
def pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    proj_model = ROIToTextEmbedding(in_dim=256, out_dim=512).to(device)

    cap = cv2.VideoCapture("input/video.mp4")

    system = PerceptionServiceAdapter(
        yolo_weight="yolov8s-worldv2.pt",
        ltm_feat_dim=256,
        device=device,
    )

    frame_idx = 0

    # 기본 YOLO-World 텍스트 임베딩 (system에서 가져오기)
    base_text_embs = system.detector.text_embeddings  # YOLO-World 기본 prompt embedding

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # ======================
        # 1) 1차 YOLO-World 검출
        # ======================
        sg_frame, _, _diff = system.process_frame(
            frame_bgr=frame,
            frame_idx=frame_idx,
            run_grin=False,
            grin_horizon=16,
        )

        vis = frame.copy()

        # -------------------------------------------------------
        # 2) 재입력 테스트: 노드 하나 선택 (예: 첫 번째 사람)
        # -------------------------------------------------------
        target_node = None
        for node in sg_frame.nodes:
            if node.cls == "person":
                target_node = node
                break

        # 만약 타겟 노드가 있다면 → focus YOLO 실행
        if target_node is not None:
            # YOLO 입력 이미지 준비
            img = torch.from_numpy(frame[..., ::-1]).permute(2, 0, 1).float() / 255.0
            img = img.unsqueeze(0).to(device)

            # ===============================
            # 3) focus YOLO-World 재입력 수행
            # ===============================
            preds = focus_detection_pipeline(
                node=target_node,
                yolo_model=system.detector.yolo,  # YOLO-World 원본 모델
                image_tensor=img,
                proj_model=proj_model,
                base_text_embs=base_text_embs,
            )

            # ===============================
            # 4) focus YOLO 결과 시각화
            # ===============================
            focus_boxes = preds[0].boxes.xyxy.cpu().numpy()
            focus_scores = preds[0].boxes.conf.cpu().numpy()

            for i, box in enumerate(focus_boxes):
                x1, y1, x2, y2 = map(int, box)
                score = focus_scores[i]

                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    vis,
                    f"focus:{score:.2f}",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                )

        # ========================
        # 5) 원래 detection도 표시
        # ========================
        for node in sg_frame.nodes:
            x1, y1, x2, y2 = map(int, node.box)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"{node.cls}#{node.entity_id}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Perception + Focus YOLO", vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()
