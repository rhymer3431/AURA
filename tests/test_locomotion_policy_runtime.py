from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from locomotion.controller import (
    _build_height_scan_grid,
    _extract_raycast_hit_position,
    create_policy_session,
    infer_policy_backend,
)
from locomotion.paths import resolve_default_policy_path


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "artifacts" / "models" / "policy.onnx"


def _build_tensorrt_engine(onnx_path: Path, engine_path: Path) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with onnx_path.open("rb") as onnx_file:
        parsed = parser.parse(onnx_file.read())
    if not parsed:
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError(f"TensorRT ONNX parse failed: {errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT engine build returned None")

    engine_path.write_bytes(serialized_engine)


def test_infer_policy_backend_detects_engine_suffix() -> None:
    assert infer_policy_backend("artifacts/models/g1_policy_fp32.engine") == "tensorrt"
    assert infer_policy_backend("artifacts/models/policy.onnx") == "onnxruntime"


def test_resolve_default_policy_path_prefers_built_engine(tmp_path: Path) -> None:
    models_dir = tmp_path / "artifacts" / "models"
    models_dir.mkdir(parents=True)
    engine_path = models_dir / "g1_policy_fp32.engine"
    onnx_path = models_dir / "policy.onnx"
    engine_path.write_bytes(b"engine")
    onnx_path.write_bytes(b"onnx")

    assert resolve_default_policy_path(str(tmp_path)) == str(engine_path.resolve())


def test_height_scan_grid_matches_official_layout() -> None:
    grid = _build_height_scan_grid()

    assert grid.shape == (187, 2)
    assert len(np.unique(grid[:, 0])) == 17
    assert len(np.unique(grid[:, 1])) == 11
    np.testing.assert_allclose(grid[0], np.asarray([-0.8, -0.5], dtype=np.float32))
    np.testing.assert_allclose(grid[-1], np.asarray([0.8, 0.5], dtype=np.float32))


def test_extract_raycast_hit_position_supports_supported_shapes() -> None:
    class _Hit:
        def __init__(self, position) -> None:
            self.position = position

    np.testing.assert_allclose(
        _extract_raycast_hit_position({"hit": True, "position": (1.0, 2.0, 3.0)}),
        np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        _extract_raycast_hit_position((True, _Hit((4.0, 5.0, 6.0)))),
        np.asarray([4.0, 5.0, 6.0], dtype=np.float32),
    )
    assert _extract_raycast_hit_position({"hit": False}) is None
    assert _extract_raycast_hit_position((False, None)) is None


def test_onnx_policy_session_runs_exported_policy() -> None:
    session = create_policy_session(str(POLICY_PATH), providers=["CPUExecutionProvider"], device_preference="cpu")
    try:
        outputs = session.run(np.zeros((310,), dtype=np.float32))
    finally:
        session.close()

    assert session.backend_name == "onnxruntime"
    assert session.input_shape == (1, 310)
    assert session.output_shape == (1, 37)
    assert outputs.shape == (1, 37)
    assert outputs.dtype == np.float32


def test_tensorrt_policy_session_rejects_cpu_preference() -> None:
    with pytest.raises(RuntimeError, match="requires CUDA"):
        create_policy_session("artifacts/models/g1_policy_fp32.engine", providers=[], device_preference="cpu")


@pytest.mark.skipif(importlib.util.find_spec("tensorrt") is None, reason="TensorRT is not installed")
@pytest.mark.skipif(importlib.util.find_spec("cuda") is None, reason="cuda-python is not installed")
def test_tensorrt_policy_session_matches_onnx(tmp_path: Path) -> None:
    engine_path = tmp_path / "policy.engine"
    _build_tensorrt_engine(POLICY_PATH, engine_path)

    onnx_session = create_policy_session(str(POLICY_PATH), providers=["CPUExecutionProvider"], device_preference="cpu")
    trt_session = create_policy_session(str(engine_path), providers=[], device_preference="cuda")
    sample = np.random.default_rng(0).standard_normal((310,), dtype=np.float32)

    try:
        onnx_outputs = onnx_session.run(sample)
        trt_outputs = trt_session.run(sample)
    finally:
        onnx_session.close()
        trt_session.close()

    assert trt_session.backend_name == "tensorrt"
    assert np.allclose(onnx_outputs, trt_outputs, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(importlib.util.find_spec("tensorrt") is None, reason="TensorRT is not installed")
@pytest.mark.skipif(importlib.util.find_spec("cuda") is None, reason="cuda-python is not installed")
def test_tensorrt_policy_session_fails_for_incompatible_engine(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    engine_path = tmp_path / "g1_policy_fp32.engine"
    neighbor_onnx_path = tmp_path / "policy.onnx"
    neighbor_onnx_path.write_bytes(POLICY_PATH.read_bytes())
    engine_path.write_bytes(b"invalid-engine")

    with pytest.raises(RuntimeError, match="Failed to deserialize TensorRT engine"):
        create_policy_session(str(engine_path), providers=[], device_preference="cuda")

    captured = capsys.readouterr()
    assert "TensorRT engine is incompatible or unreadable" in captured.out
    assert engine_path.read_bytes() == b"invalid-engine"
