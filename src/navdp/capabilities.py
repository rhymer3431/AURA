from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class NavDPCheckpointCapabilities:
    checkpoint_path: str
    supports_pixelgoal: bool
    supports_imagegoal: bool
    cond_tokens: int
    state_dict_keys: int


@lru_cache(maxsize=8)
def inspect_checkpoint_capabilities(checkpoint_path: str, *, memory_size: int = 8) -> NavDPCheckpointCapabilities:
    import torch

    resolved_path = str(Path(checkpoint_path).expanduser().resolve())
    state_dict = torch.load(resolved_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected checkpoint state_dict to be a dict, got {type(state_dict)!r}")

    cond_embedding = state_dict.get("cond_pos_embed.position_embedding.weight")
    cond_tokens = -1
    if hasattr(cond_embedding, "shape") and len(cond_embedding.shape) >= 1:
        cond_tokens = int(cond_embedding.shape[0])

    expected_cond_tokens = (int(memory_size) * 16) + 4
    supports_pixelgoal = any(str(key).startswith("pixel_encoder.") for key in state_dict) and cond_tokens == expected_cond_tokens
    supports_imagegoal = any(str(key).startswith("image_encoder.") for key in state_dict) and cond_tokens == expected_cond_tokens

    return NavDPCheckpointCapabilities(
        checkpoint_path=resolved_path,
        supports_pixelgoal=bool(supports_pixelgoal),
        supports_imagegoal=bool(supports_imagegoal),
        cond_tokens=cond_tokens,
        state_dict_keys=len(state_dict),
    )
