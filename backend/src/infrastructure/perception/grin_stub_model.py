# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class GRINStubModel(nn.Module):
    def __init__(self, node_dim: int):
        super().__init__()
        self.fc = nn.Linear(node_dim, node_dim)

    def forward(self, node_feats: torch.Tensor, boxes: torch.Tensor, t_idx: torch.Tensor):
        last_feat = node_feats[-1]
        future_feat = self.fc(last_feat)
        return {
            "future_feat": future_feat,
            "future_box": boxes[-1],
        }