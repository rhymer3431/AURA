from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from locomotion.controller import (
    G1PolicyController,
    _build_height_scan_grid,
    _build_policy_observation,
    _extract_raycast_hit_path,
    _extract_raycast_hit_position,
    _path_has_ground_token,
    create_policy_session,
    infer_policy_backend,
)
from locomotion.paths import resolve_default_policy_path
from locomotion.runtime import _validate_default_policy_device


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
    assert infer_policy_backend("artifacts/models/g1_policy_fp16.engine") == "tensorrt"
    assert infer_policy_backend("artifacts/models/policy.onnx") == "onnxruntime"


def test_resolve_default_policy_path_prefers_artifacts_fp16_engine(tmp_path: Path) -> None:
    models_dir = tmp_path / "artifacts" / "models"
    models_dir.mkdir(parents=True)
    engine_path = models_dir / "g1_policy_fp16.engine"
    (models_dir / "g1_policy_fp32.engine").write_bytes(b"fallback-engine")
    onnx_path = models_dir / "policy.onnx"
    engine_path.write_bytes(b"engine")
    onnx_path.write_bytes(b"onnx")

    assert resolve_default_policy_path(str(tmp_path)) == str(engine_path.resolve())


def test_resolve_default_policy_path_falls_back_to_artifacts_fp32_engine(tmp_path: Path) -> None:
    models_dir = tmp_path / "artifacts" / "models"
    models_dir.mkdir(parents=True)
    fallback_engine_path = models_dir / "g1_policy_fp32.engine"
    fallback_engine_path.write_bytes(b"fallback-engine")

    assert resolve_default_policy_path(str(tmp_path)) == str(fallback_engine_path.resolve())


def test_resolve_default_policy_path_falls_back_to_legacy_src_engine(tmp_path: Path) -> None:
    policy_dir = tmp_path / "src" / "locomotion" / "models"
    policy_dir.mkdir(parents=True)
    tuned_engine_path = policy_dir / "policy_fp16.engine"
    tuned_engine_path.write_bytes(b"tuned-engine")

    assert resolve_default_policy_path(str(tmp_path)) == str(tuned_engine_path.resolve())


def test_validate_default_policy_device_rejects_cpu_for_default_engine() -> None:
    args = SimpleNamespace(policy="", onnx_device="cpu")

    with pytest.raises(RuntimeError, match="requires CUDA/TensorRT"):
        _validate_default_policy_device(args, "/tmp/artifacts/models/g1_policy_fp16.engine")


def test_build_policy_observation_without_height_scan_matches_123_dim_layout() -> None:
    obs = _build_policy_observation(
        lin_vel_b=np.zeros(3, dtype=np.float32),
        ang_vel_b=np.zeros(3, dtype=np.float32),
        gravity_b=np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
        command=np.zeros(3, dtype=np.float32),
        joint_pos=np.zeros(37, dtype=np.float32),
        default_pos=np.zeros(37, dtype=np.float32),
        joint_vel=np.zeros(37, dtype=np.float32),
        default_vel=np.zeros(37, dtype=np.float32),
        previous_action=np.zeros(37, dtype=np.float32),
        height_scan=None,
    )

    assert obs.shape == (123,)


def test_build_policy_observation_with_height_scan_matches_310_dim_layout() -> None:
    obs = _build_policy_observation(
        lin_vel_b=np.zeros(3, dtype=np.float32),
        ang_vel_b=np.zeros(3, dtype=np.float32),
        gravity_b=np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
        command=np.zeros(3, dtype=np.float32),
        joint_pos=np.zeros(37, dtype=np.float32),
        default_pos=np.zeros(37, dtype=np.float32),
        joint_vel=np.zeros(37, dtype=np.float32),
        default_vel=np.zeros(37, dtype=np.float32),
        previous_action=np.zeros(37, dtype=np.float32),
        height_scan=np.zeros(187, dtype=np.float32),
    )

    assert obs.shape == (310,)


def test_validate_io_disables_height_scan_for_123_dim_policy() -> None:
    controller = object.__new__(G1PolicyController)
    controller.default_pos = np.zeros(37, dtype=np.float32)
    controller.policy_session = SimpleNamespace(input_shape=(1, 123), output_shape=(1, 37))
    controller._include_height_scan = True

    controller._validate_io()

    assert controller._include_height_scan is False


def test_validate_io_keeps_height_scan_for_310_dim_policy() -> None:
    controller = object.__new__(G1PolicyController)
    controller.default_pos = np.zeros(37, dtype=np.float32)
    controller.policy_session = SimpleNamespace(input_shape=(1, 310), output_shape=(1, 37))
    controller._include_height_scan = False

    controller._validate_io()

    assert controller._include_height_scan is True


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


def test_extract_raycast_hit_path_and_ground_token_detection() -> None:
    class _Hit:
        def __init__(self, collision: str) -> None:
            self.collision = collision

    assert _extract_raycast_hit_path({"collision": "/World/Environment/floor/collider"}) == "/World/Environment/floor/collider"
    assert _extract_raycast_hit_path((True, _Hit("/World/ground/mesh"))) == "/World/ground/mesh"
    assert _path_has_ground_token("/World/ground")
    assert _path_has_ground_token("/World/Environment/FloorMesh")
    assert not _path_has_ground_token("/World/Environment/Wall")


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
        create_policy_session("artifacts/models/g1_policy_fp16.engine", providers=[], device_preference="cpu")


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
    engine_path = tmp_path / "g1_policy_fp16.engine"
    neighbor_onnx_path = tmp_path / "policy.onnx"
    neighbor_onnx_path.write_bytes(POLICY_PATH.read_bytes())
    engine_path.write_bytes(b"invalid-engine")

    with pytest.raises(RuntimeError, match="Failed to deserialize TensorRT engine"):
        create_policy_session(str(engine_path), providers=[], device_preference="cuda")

    captured = capsys.readouterr()
    assert "TensorRT engine is incompatible or unreadable" in captured.out
    assert engine_path.read_bytes() == b"invalid-engine"
