import torch
import torch.nn.functional as F
from typing import Any
from pathlib import Path

from torchreid.reid.models import osnet_ain_x1_0

from src.domain.memory.feature_similarity_port import FeatureSimilarityPort


class OSNetSimilarityAdapter(FeatureSimilarityPort):
    """
    OSNet-AIN-x1_0 feature extractor (NO torch.hub)
    """

    def __init__(
        self,
        weight_path: str = "weights/osnet_ain_x1_0.pth",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device)

        weight = Path(weight_path)
        if not weight.exists():
            raise FileNotFoundError(f"Weight not found: {weight}")

        print(f"[OSNetAIN] Loading weights: {weight}")

        # only load architecture, no hub download
        self.model = osnet_ain_x1_0(pretrained=False)
        state_dict = torch.load(weight, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)

        self.model.to(self.device).eval()

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1,3,1,1)

    @torch.no_grad()
    def embed(self, roi_img: torch.Tensor) -> torch.Tensor:
        if roi_img.dim() == 3:
            roi_img = roi_img.unsqueeze(0)

        x = (roi_img.to(self.device) - self.mean) / self.std
        feat = self.model(x)
        return F.normalize(feat, dim=1).squeeze(0)

    # Port compatibility -----------------
    def get_embedding(self, roi_img: torch.Tensor) -> torch.Tensor:
        return self.embed(roi_img)

    def cosine_similarity(self, query: torch.Tensor, prototypes: Any) -> float:
        """
        Compute the best cosine similarity between query and a set of prototypes.
        Accepts list/tuple of tensors or a tensor shaped (N, C).
        """
        if isinstance(prototypes, (list, tuple)):
            prototypes = torch.stack(prototypes)
        if prototypes.dim() == 1:
            prototypes = prototypes.unsqueeze(0)

        if query.dim() == 1:
            query = query.unsqueeze(0)

        query = self.normalize(query)
        prototypes = self.normalize(prototypes.to(query.device))

        sims = torch.matmul(query, prototypes.t())
        return float(sims.max().item())

    def cosine_sim(self, a: torch.Tensor, b: torch.Tensor) -> float:
        # Backwards-compat alias
        return self.cosine_similarity(a, b)

    def normalize(self, feat: torch.Tensor) -> torch.Tensor:
        return F.normalize(feat, dim=-1)
