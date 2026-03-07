from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from .client import InProcessNavDPClient, InProcessNavDPClientConfig


def discover_default_checkpoint(repo_root: Path | None = None) -> str:
    root = repo_root or Path(__file__).resolve().parents[3]
    candidates = [
        root / "artifacts" / "models" / "navdp-cross-modal.ckpt",
        root / "artifacts" / "models" / "navdp-weights.ckpt",
        root / "navdp-weights.ckpt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def create_inprocess_navdp_client(
    *,
    intrinsic: np.ndarray | None = None,
    backend: str = "auto",
    checkpoint_path: str = "",
    device: str = "cpu",
    amp: bool = False,
    amp_dtype: str = "float16",
    tf32: bool = False,
    stop_threshold: float = -3.0,
) -> InProcessNavDPClient:
    resolved_checkpoint = checkpoint_path or discover_default_checkpoint()
    client = InProcessNavDPClient(
        InProcessNavDPClientConfig(
            backend=backend,
            checkpoint_path=resolved_checkpoint,
            device=device,
            amp=amp,
            amp_dtype=amp_dtype,
            tf32=tf32,
            stop_threshold=stop_threshold,
        )
    )
    if intrinsic is not None:
        try:
            client.navigator_reset(np.asarray(intrinsic, dtype=np.float32), batch_size=1)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"In-process NavDP client reset failed: {type(exc).__name__}: {exc}", stacklevel=2)
    return client
