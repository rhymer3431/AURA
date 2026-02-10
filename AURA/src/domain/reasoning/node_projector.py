
import torch
import torch.nn as nn
class YoloNodeProjector(nn.Module):
    """
    ROI feature와 Text embedding을 SGG용 node embedding으로 투영하는 레이어.

    - 입력:
        roi_feats:  (N, roi_dim)
        text_feats: (N, text_dim)
    - 출력:
        node_feats: (N, out_dim)
    """

    def __init__(self, roi_dim: int, text_dim: int, out_dim: int = 512):
        super().__init__()
        self.roi_proj = nn.Linear(roi_dim, out_dim)
        self.txt_proj = nn.Linear(text_dim, out_dim)

    def forward(self, roi_feats: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
        """
        roi_feats:  (N, roi_dim)
        text_feats: (N, text_dim)
        """
        # 둘 다 같은 device에 있다고 가정 (GPU)
        roi_emb = self.roi_proj(roi_feats)
        txt_emb = self.txt_proj(text_feats)
        # 간단히 합으로 fusion (원하면 concat 후 MLP 등으로 확장 가능)
        node_feats = roi_emb + txt_emb
        return node_feats  # (N, out_dim)