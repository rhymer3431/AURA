# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class GRINStubModel(nn.Module):
    def __init__(self, node_dim: int):
        super().__init__()
        self.fc = nn.Linear(node_dim, node_dim)

    def _ensure_feature_dim(self, feat: torch.Tensor) -> None:
        in_dim = int(feat.shape[-1])
        if self.fc.in_features == in_dim and self.fc.out_features == in_dim:
            return
        # GRIN stub accepts dynamic detector feature sizes (e.g. 256, 768).
        self.fc = nn.Linear(in_dim, in_dim, device=feat.device, dtype=feat.dtype)

    def forward(self, node_feats: torch.Tensor, boxes: torch.Tensor, t_idx: torch.Tensor):
        last_feat = node_feats[-1]
        self._ensure_feature_dim(last_feat)
        future_feat = self.fc(last_feat)
        return {
            "future_feat": future_feat,
            "future_box": boxes[-1],
        }
