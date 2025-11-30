# sgg_yoloworld_track_transformer.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision.ops as ops

from ultralytics import YOLOWorld
from relation_classes import REL_CLASSES, NUM_REL_CLASSES


class YoloWorldTrackNodeEncoder:
    """
    - YOLO-World(yolov8s-worldv2.pt)의 track()을 사용
    - model.model.model[12]에 hook 걸어서 P3 feature 추출
    - P3에서 ROIAlign → roi_feat
    - txt_feats에서 text_feat
    - 둘 concat → node_feat
    """

    def __init__(self, weight="yolov8s-worldv2.pt", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 1) YOLO-World 로드
        self.model = YOLOWorld(weight).to(self.device)
        self.model.eval()

        # 2) 텍스트 임베딩 (1, num_classes, dim) → (num_classes, dim)
        txt = self.model.model.txt_feats
        if txt.dim() == 3 and txt.shape[0] == 1:
            txt = txt.squeeze(0)
        self.text_emb = txt.detach().cpu()  # (num_classes, D_text)

        self.names = self.model.model.names  # cls_id → name

        # 3) Feature hook (P3)
        self.features = {"P3": None}
        # 사용자가 출력해 준 named_modules 기준으로 12번이 C3+Attn(P3 레벨)
        self.model.model.model[12].register_forward_hook(self._hook_p3)

    def _hook_p3(self, module, inp, out):
        # out: (B=1, C, Hf, Wf)
        self.features["P3"] = out

    def nodes_from_result(self, result):
        """
        YOLO track() 한 step의 Result 객체에서 노드 추출
        result: ultralytics.engine.results.Results
        return: list of dict (각 객체 = node)
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
            # track id (없으면 -1)
            track_id = int(b.id.item()) if b.id is not None else -1

            cls_id = int(b.cls.item())
            score = float(b.conf.item())
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cls_name = self.names[cls_id]

            # 1) ROIAlign을 위한 feature map 좌표 변환
            sx = Wf / float(W)
            sy = Hf / float(H)
            x1f, y1f, x2f, y2f = x1 * sx, y1 * sy, x2 * sx, y2 * sy

            roi_box = torch.tensor([[0, x1f, y1f, x2f, y2f]],
                                   device=self.device, dtype=torch.float32)

            roi_feat = ops.roi_align(
                input=P3,        # (1, C, Hf, Wf)
                boxes=roi_box,   # (1, 5)
                output_size=(7, 7),
                spatial_scale=1.0,
                sampling_ratio=-1,
                aligned=True,
            )  # (1, C, 7, 7)

            # (1, C)
            roi_feat = roi_feat.mean(dim=(2, 3)).detach().cpu()

            # 2) Text feature (num_classes, D_text)에서 cls_id 인덱싱
            text_feat = self.text_emb[cls_id]  # (D_text,)

            # 3) Node feature concat
            node_feat = torch.cat([roi_feat[0], text_feat], dim=0)  # (C+D_text,)

            nodes.append({
                "track_id": track_id,
                "cls_id": cls_id,
                "cls_name": cls_name,
                "score": score,
                "bbox": [x1, y1, x2, y2],  # 원본 이미지 좌표
                "roi_feat": roi_feat[0],   # (C,)
                "text_feat": text_feat,    # (D_text,)
                "node_feat": node_feat,    # (C + D_text,)
            })

        return nodes


class RelationTransformerHead(nn.Module):
    """
    Transformer 기반 REACT/SGTR 스타일 간단화 버전:
    - 입력: node_feats (N, D_node), boxes (N,4) [0~1 정규화된 xyxy]
    - 1) node_feats를 TransformerEncoder로 컨텍스트 인코딩
    - 2) 모든 (i, j), i!=j 쌍에 대해 [h_i, h_j, geo_ij] → relation logits
    """

    def __init__(
        self,
        node_dim: int,
        num_rel_classes: int,
        nhead: int = 4,
        num_layers: int = 2,
        geo_dim: int = 8,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.num_rel_classes = num_rel_classes
        self.geo_dim = geo_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=node_dim,
            nhead=nhead,
            dim_feedforward=node_dim * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mlp_rel = nn.Sequential(
            nn.Linear(node_dim * 2 + geo_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_rel_classes),
        )

    def forward(self, node_feats: torch.Tensor, boxes_norm: torch.Tensor):
        """
        node_feats: (N, D_node)
        boxes_norm: (N, 4) [x1,y1,x2,y2] / (W,H,W,H) 으로 0~1 정규화
        return:
            pair_idx: (M, 2) (subject_idx, object_idx)
            rel_logits: (M, num_rel_classes)
        """
        N = node_feats.size(0)
        if N < 2:
            # 관계를 만들 수 없음
            return torch.empty((0, 2), dtype=torch.long, device=node_feats.device), \
                   torch.empty((0, self.num_rel_classes), device=node_feats.device)

        # 1) Transformer context
        x = node_feats.unsqueeze(0)  # (1, N, D)
        x_ctx = self.encoder(x).squeeze(0)  # (N, D)

        # 2) 모든 (i,j), i!=j 쌍 생성
        # i: subject, j: object
        idx_i = []
        idx_j = []
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                idx_i.append(i)
                idx_j.append(j)
        idx_i = torch.tensor(idx_i, dtype=torch.long, device=node_feats.device)
        idx_j = torch.tensor(idx_j, dtype=torch.long, device=node_feats.device)

        # 3) feature gather
        hi = x_ctx[idx_i]  # (M, D)
        hj = x_ctx[idx_j]  # (M, D)

        bi = boxes_norm[idx_i]  # (M, 4)
        bj = boxes_norm[idx_j]  # (M, 4)

        # 4) 간단한 geometry feature (8차원)
        #    [x1_s, y1_s, x2_s, y2_s, x1_o, y1_o, x2_o, y2_o]
        geo_ij = torch.cat([bi, bj], dim=1)  # (M, 8)

        # 5) relation logits
        rel_in = torch.cat([hi, hj, geo_ij], dim=1)  # (M, 2D + 8)
        rel_logits = self.mlp_rel(rel_in)  # (M, num_rel_classes)

        pair_idx = torch.stack([idx_i, idx_j], dim=1)  # (M,2)

        return pair_idx, rel_logits


def run_sgg_video(
    video_path="input/video.mp4",
    weight="yolov8s-worldv2.pt",
    device=None,
    rel_conf_thresh=0.5,
):
    """
    YOLO-World track() + Transformer SGG
    - 프레임마다 node + relation 추론
    - 콘솔에 간단히 출력
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Node Encoder 초기화
    encoder = YoloWorldTrackNodeEncoder(weight=weight, device=device)

    # node_dim = roi_feat_dim + text_dim
    # P3 채널 수와 txt_feats dim 을 합산해서 추론해야 하지만,
    # 여기서는 첫 프레임에서 한 번 계산하는 방식으로 설정
    # (lazy init)
    relation_head = None

    # 2) YOLO-World track API 사용
    # persist=True → track id 유지, stream=True → generator 반환
    results_gen = encoder.model.track(
        source=video_path,
        stream=True,
        persist=True,
        conf=0.25,
        iou=0.45,
    )

    frame_idx = 0
    for result in results_gen:
        frame_idx += 1

        # 2-1) 해당 프레임에서 nodes 추출
        nodes = encoder.nodes_from_result(result)
        if len(nodes) == 0:
            print(f"[Frame {frame_idx}] No nodes.")
            continue

        # node_feats, boxes 준비
        node_feats = torch.stack([n["node_feat"] for n in nodes])  # (N, D_node)
        bboxes = torch.tensor([n["bbox"] for n in nodes], dtype=torch.float32)  # (N,4)

        frame_bgr = result.orig_img
        H, W = frame_bgr.shape[:2]
        boxes_norm = bboxes.clone()
        boxes_norm[:, [0, 2]] /= W
        boxes_norm[:, [1, 3]] /= H

        node_feats = node_feats.to(device)
        boxes_norm = boxes_norm.to(device)

        # 2-2) Relation head lazy init
        if relation_head is None:
            node_dim = node_feats.size(1)
            relation_head = RelationTransformerHead(
                node_dim=node_dim,
                num_rel_classes=NUM_REL_CLASSES,
                nhead=4,
                num_layers=2,
                geo_dim=8,
                hidden_dim=512,
            ).to(device)
            relation_head.eval()
            print(f"[Init] Relation head with node_dim={node_dim}, num_rel_classes={NUM_REL_CLASSES}")

        # 2-3) 관계 추론
        with torch.no_grad():
            pair_idx, rel_logits = relation_head(node_feats, boxes_norm)
            if rel_logits.numel() == 0:
                print(f"[Frame {frame_idx}] No relations.")
                continue
            rel_scores = rel_logits.softmax(dim=-1)  # (M, C)

        # 2-4) 간단히 top-1 relation만 출력 (no_relation 제외)
        M = pair_idx.size(0)
        print(f"\n[Frame {frame_idx}] Nodes={len(nodes)}, Candidate pairs={M}")

        for k in range(M):
            i, j = pair_idx[k].tolist()
            subj = nodes[i]
            obj = nodes[j]

            # top-1
            scores_k, cls_k = torch.max(rel_scores[k], dim=-1)
            rel_id = int(cls_k.item())
            score_rel = float(scores_k.item())
            rel_name = REL_CLASSES[rel_id]

            # no_relation 이거나 threshold보다 낮으면 skip
            if rel_name == "no_relation" or score_rel < rel_conf_thresh:
                continue

            print(
                f"  ({subj['track_id']}:{subj['cls_name']}) "
                f"-[{rel_name} {score_rel:.2f}]-> "
                f"({obj['track_id']}:{obj['cls_name']})"
            )


if __name__ == "__main__":
    run_sgg_video(
        video_path="input/video.mp4",
        weight="yolov8s-worldv2.pt",
        rel_conf_thresh=0.7,  # 출력 필터용 threshold (임시)
    )
