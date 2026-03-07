from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import adapters.sensors.d455_mount as d455_mount


class _FakeReferences:
    def __init__(self, stage, prim_path: str) -> None:  # noqa: ANN001
        self.stage = stage
        self.prim_path = prim_path
        self.references: list[str] = []

    def AddReference(self, asset_path: str) -> None:
        self.references.append(str(asset_path))
        self.stage.DefinePrim(f"{self.prim_path}/Camera_OmniVision_OV9782_Color", "Camera")
        self.stage.DefinePrim(f"{self.prim_path}/Camera_OmniVision_OV9782_Depth", "Camera")
        self.stage.DefinePrim(f"{self.prim_path}/DepthSensor", "Xform")


class _FakePrim:
    def __init__(self, stage, path: str, type_name: str) -> None:  # noqa: ANN001
        self.stage = stage
        self.path = str(path)
        self.type_name = str(type_name)
        self.refs = _FakeReferences(stage, self.path)

    def IsValid(self) -> bool:
        return True

    def GetPath(self) -> str:
        return self.path

    def GetReferences(self) -> _FakeReferences:
        return self.refs


class _FakeStage:
    def __init__(self) -> None:
        self.prims: dict[str, _FakePrim] = {}

    def DefinePrim(self, path: str, type_name: str):
        prim = _FakePrim(self, path, type_name)
        self.prims[str(path)] = prim
        return prim

    def Traverse(self):
        return [self.prims[key] for key in sorted(self.prims)]


def test_d455_asset_path_resolution_and_mount(monkeypatch) -> None:
    monkeypatch.setattr(d455_mount, "_stat_asset", lambda path: True)
    resolution = d455_mount.resolve_d455_asset_path(asset_path="", assets_root="/Isaac", getter=lambda: "/Isaac")
    stage = _FakeStage()
    report = d455_mount.ensure_d455_mount(stage, asset_path=resolution.asset_path, prim_path="/World/realsense_d455")

    assert resolution.asset_path == "/Isaac/Sensors/Intel/RealSense/rsd455.usd"
    assert report.prim_exists is True
    assert report.reference_added is True
    assert "/World/realsense_d455/Camera_OmniVision_OV9782_Color" in report.camera_prim_paths
    assert "/World/realsense_d455/DepthSensor" in report.depth_sensor_paths
