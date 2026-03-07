from __future__ import annotations

import io
from dataclasses import asdict

import numpy as np

from .shm_ring import ShmSlotRef


def encode_ndarray(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    return buffer.getvalue()


def decode_ndarray(payload: bytes) -> np.ndarray:
    return np.load(io.BytesIO(payload), allow_pickle=False)


def ref_to_dict(ref: ShmSlotRef) -> dict[str, int | str]:
    return asdict(ref)


def ref_from_dict(payload: dict[str, object]) -> ShmSlotRef:
    return ShmSlotRef(
        name=str(payload["name"]),
        slot_index=int(payload["slot_index"]),
        payload_size=int(payload["payload_size"]),
        sequence=int(payload["sequence"]),
    )
