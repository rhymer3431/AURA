import torch
import torch.nn as nn
import torch.nn.functional as F


class PSGCriterion(nn.Module):
    """
    3-in-1 PSG Relation Loss
        1) Background Downsampling
        2) Class-weighted Focal Loss
        3) Hard Positive Mining
    """

    def __init__(
        self,
        num_classes,
        bg_sampling_ratio=0.1,      # background 10%만 사용
        focal_gamma=2,
        class_weights=None,         # [num_classes]
        hard_pos_ratio=1.0          # positive 중 hard example 비율 (1.0 = 전체)
    ):
        super().__init__()
        self.num_classes = num_classes
        self.bg_sampling_ratio = bg_sampling_ratio
        self.focal_gamma = focal_gamma
        self.hard_pos_ratio = hard_pos_ratio

        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights))
        else:
            self.class_weights = None

        self.ce = nn.CrossEntropyLoss(weight=self.class_weights, reduction="none")

    def forward(self, logits, labels):
        """
        logits: (M, num_classes)
        labels: (M,)
        """

        device = logits.device
        M = labels.size(0)

        # --------------------------------------------------------
        # 1) positive / background 분리
        # --------------------------------------------------------
        pos_mask = labels > 0
        bg_mask = labels == 0

        pos_idx = torch.nonzero(pos_mask, as_tuple=False).squeeze(1)
        bg_idx = torch.nonzero(bg_mask, as_tuple=False).squeeze(1)

        # background가 너무 많으면 downsampling
        if len(bg_idx) > 0:
            num_bg_keep = max(1, int(len(bg_idx) * self.bg_sampling_ratio))
            perm = torch.randperm(len(bg_idx), device=device)[:num_bg_keep]
            bg_idx = bg_idx[perm]

        # 최종 사용 인덱스
        keep_idx = torch.cat([pos_idx, bg_idx], dim=0)
        logits = logits[keep_idx]
        labels = labels[keep_idx]

        # --------------------------------------------------------
        # 2) Hard Positive Mining
        #    positive gradient가 큰 애부터 선별
        # --------------------------------------------------------
        if len(pos_idx) > 0 and self.hard_pos_ratio < 1.0:
            # positive 데이터 중 loss 큰 순서 정렬하여 상위만 유지
            pos_logits = logits[labels > 0]
            pos_labels = labels[labels > 0]

            pos_loss = self.ce(pos_logits, pos_labels).detach()
            num_keep = max(1, int(len(pos_loss) * self.hard_pos_ratio))
            _, hard_ids = torch.topk(pos_loss, num_keep)

            # hard positive만 남기기
            hard_pos_idx = pos_idx[hard_ids]

            # background 유지하면서 최종 idx 재구성
            keep_idx = torch.cat([hard_pos_idx, bg_idx], dim=0)
            logits = logits[torch.cat([hard_ids, torch.arange(len(bg_idx), device=device)])]
            labels = labels[keep_idx]

        # --------------------------------------------------------
        # 3) Class-weighted Focal Loss
        # --------------------------------------------------------
        ce = self.ce(logits, labels)         # (K,)
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.focal_gamma * ce

        return focal.mean()
