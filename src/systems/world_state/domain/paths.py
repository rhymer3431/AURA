"""Path and provider resolution helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def repo_dir() -> str:
    return str(Path(__file__).resolve().parents[4])


def resolve_default_policy_path(base_dir: str) -> str:
    candidates = [
        os.path.abspath(os.path.join(base_dir, "artifacts", "models", "policy_fp16_b1.engine")),
        os.path.abspath(os.path.join(base_dir, "policy_fp16_b1.engine")),
        os.path.abspath(os.path.join(base_dir, "artifacts", "models", "g1_policy_fp16.engine")),
        os.path.abspath(os.path.join(base_dir, "g1_policy_fp16.engine")),
        os.path.abspath(os.path.join(base_dir, "artifacts", "models", "policy.onnx")),
        os.path.abspath(os.path.join(base_dir, "policy.onnx")),
        os.path.abspath(os.path.join(base_dir, "tuned", "exported", "policy_tuned.onnx")),
        os.path.abspath(os.path.join(base_dir, "go_omniverse", "tuned", "exported", "policy_tuned.onnx")),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0]


def resolve_default_robot_usd_path(base_dir: str) -> str:
    candidates = [
        os.path.abspath(os.path.join(base_dir, "robots", "g1", "g1_d455.usd")),
        os.path.abspath(os.path.join(base_dir, "robots", "g1", "g1.usd")),
        os.path.abspath(os.path.join(base_dir, "go_omniverse", "robots", "g1", "g1_d455.usd")),
        os.path.abspath(os.path.join(base_dir, "go_omniverse", "robots", "g1", "g1.usd")),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0]


def resolve_environment_reference(scene_usd: str | None, env_url: str) -> Optional[str]:
    if scene_usd:
        return os.path.abspath(scene_usd)

    from isaacsim.storage.native import get_assets_root_path

    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        return None
    return assets_root_path + env_url


def select_onnx_providers(onnx_device: str) -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()

    if onnx_device == "cpu":
        return ["CPUExecutionProvider"]

    if onnx_device == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                f"--onnx_device cuda requested, but CUDAExecutionProvider is unavailable. "
                f"Available providers: {available}"
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]
