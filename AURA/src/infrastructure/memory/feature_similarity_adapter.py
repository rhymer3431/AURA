# src/infrastructure/memory/torch_similarity_adapter.py
import torch
import torch.nn.functional as F
from typing import Any

from src.domain.memory.feature_similarity_port import FeatureSimilarityPort


class FeatureSimilarityAdapter(FeatureSimilarityPort):

    def normalize(self, feat: torch.Tensor) -> torch.Tensor:
        return F.normalize(feat, dim=-1)

    def get_embedding(self, feat: torch.Tensor) -> torch.Tensor:
        # Assume incoming feature is already extracted; just normalize.
        return self.normalize(feat)

    def cosine_similarity(self, query: torch.Tensor, prototypes: Any) -> float:
        if isinstance(prototypes, (list, tuple)):
            prototypes = torch.stack(prototypes)
        if prototypes.dim() == 1:
            prototypes = prototypes.unsqueeze(0)

        if query.dim() == 1:
            query = query.unsqueeze(0)

        q = self.normalize(query)
        p = self.normalize(prototypes.to(q.device))
        return float((q @ p.t()).max().item())

    def cosine_sim(self, query: torch.Tensor, prototypes: torch.Tensor) -> float:
        # Backwards-compat alias
        return self.cosine_similarity(query, prototypes)
