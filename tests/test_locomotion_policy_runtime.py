from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from locomotion.controller import create_policy_session, infer_policy_backend


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
