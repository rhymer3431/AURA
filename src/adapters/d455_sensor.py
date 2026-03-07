from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


DEFAULT_G1_CAMERA_BASE_SUFFIX = "/torso_link/d435_link/Realsense/RSD455"
DEFAULT_G1_CAMERA_SUFFIX_COLOR = f"{DEFAULT_G1_CAMERA_BASE_SUFFIX}/Camera_OmniVision_OV9782_Color"
DEFAULT_G1_DEPTH_CAMERA_SUFFIX_CANDIDATES = (
    f"{DEFAULT_G1_CAMERA_BASE_SUFFIX}/Camera_OmniVision_OV9782_Depth",
    f"{DEFAULT_G1_CAMERA_BASE_SUFFIX}/Camera_Realsense_Depth",
    f"{DEFAULT_G1_CAMERA_BASE_SUFFIX}/Camera_Depth",
    f"{DEFAULT_G1_CAMERA_BASE_SUFFIX}/Camera_Pseudo_Depth",
)
DEFAULT_G1_PRIM_CANDIDATES = (
    "/World/envs/env_0/Robot",
    "/World/G1",
    "/World/Robot",
)


@dataclass
class D455SensorAdapterConfig:
    use_d455: bool = True
    image_width: int = 640
    image_height: int = 640
    depth_max_m: float = 5.0
    strict_d455: bool = False
    force_runtime_mount: bool = False


@dataclass
class D455CaptureMeta:
    rgb_source: str = "missing"
    depth_source: str = "missing"
    fallback_used: bool = False
    rgb_shape: tuple[int, ...] | None = None
    depth_shape: tuple[int, ...] | None = None
    depth_min_m: float | None = None
    depth_max_m: float | None = None
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "rgb_source": self.rgb_source,
            "depth_source": self.depth_source,
            "fallback_used": bool(self.fallback_used),
            "note": self.note,
        }
        if self.rgb_shape is not None:
            payload["rgb_shape"] = list(self.rgb_shape)
        if self.depth_shape is not None:
            payload["depth_shape"] = list(self.depth_shape)
        if self.depth_min_m is not None:
            payload["depth_min_m"] = float(self.depth_min_m)
        if self.depth_max_m is not None:
            payload["depth_max_m"] = float(self.depth_max_m)
        return payload


class D455SensorAdapter:
    def __init__(self, cfg: D455SensorAdapterConfig):
        self.cfg = cfg
        self.rgb_prim_path: str | None = None
        self.depth_prim_path: str | None = None
        self._rgb_camera = None
        self._depth_camera = None
        self._intrinsic = self._default_intrinsic(cfg.image_width, cfg.image_height)
        self._fallback_reason: str | None = None
        self._last_capture_meta = D455CaptureMeta(note="capture not attempted")
        self._runtime_camera_mode = False
        self._stage = None

    @property
    def intrinsic(self) -> np.ndarray:
        return self._intrinsic.copy()

    @property
    def last_capture_meta(self) -> dict[str, Any]:
        return self._last_capture_meta.as_dict()

    def _finalize_capture_meta(
        self,
        rgb: np.ndarray | None,
        depth: np.ndarray | None,
        *,
        rgb_source: str,
        depth_source: str,
        fallback_used: bool,
        note: str = "",
    ) -> dict[str, Any]:
        depth_min = None
        depth_max = None
        if depth is not None and depth.size > 0:
            depth_min = float(np.min(depth))
            depth_max = float(np.max(depth))
        self._last_capture_meta = D455CaptureMeta(
            rgb_source=rgb_source,
            depth_source=depth_source,
            fallback_used=fallback_used,
            rgb_shape=tuple(rgb.shape) if rgb is not None else None,
            depth_shape=tuple(depth.shape) if depth is not None else None,
            depth_min_m=depth_min,
            depth_max_m=depth_max,
            note=note,
        )
        return self._last_capture_meta.as_dict()

    def initialize(self, simulation_app, stage) -> tuple[bool, str]:
        self._stage = stage
        if not self.cfg.use_d455:
            self._fallback_reason = "D455 adapter disabled by --navdp_use_d455=false."
            self._rgb_camera = None
            self._depth_camera = None
            msg = "D455 adapter disabled; using env depth pseudo-RGB fallback."
            if self.cfg.strict_d455:
                return False, f"{msg} strict_d455=true requires live D455 RGB-D."
            return True, msg

        self._fallback_reason = None
        rgb_candidate, depth_candidate = self._resolve_camera_prims(stage)
        existing_rgb_candidate = rgb_candidate
        existing_depth_candidate = depth_candidate
        self._runtime_camera_mode = False
        runtime_created = False
        runtime_rgb, runtime_depth = self._resolve_runtime_camera_paths(stage)
        if runtime_rgb is not None and runtime_depth is not None:
            # Avoid creating duplicate OmniVision cameras when the USD already provides a valid D455 rig.
            if rgb_candidate is None or depth_candidate is None:
                runtime_created = self._ensure_runtime_camera_prims(stage, runtime_rgb, runtime_depth)
            runtime_rgb_valid = self._is_valid_prim_path(stage, runtime_rgb)
            runtime_depth_valid = self._is_valid_prim_path(stage, runtime_depth)
            if rgb_candidate is None and runtime_rgb_valid:
                rgb_candidate = runtime_rgb
            if depth_candidate is None and runtime_depth_valid:
                depth_candidate = runtime_depth
            self._runtime_camera_mode = bool(
                (rgb_candidate == runtime_rgb and runtime_rgb_valid)
                or (depth_candidate == runtime_depth and runtime_depth_valid)
            )
        if rgb_candidate is None:
            self._fallback_reason = "D455 RGB camera prim not found"
            msg = "D455 RGB camera prim unavailable; using env depth pseudo-RGB fallback."
            if self.cfg.strict_d455:
                return False, f"{msg} strict_d455=true requires D455 RGB+Depth prims."
            return True, msg
        if depth_candidate is None:
            self._fallback_reason = "D455 depth camera prim not found"
            msg = "D455 depth camera prim unavailable; using env depth fallback."
            if self.cfg.strict_d455:
                return False, f"{msg} strict_d455=true requires D455 RGB+Depth prims."
            # Keep RGB prim and continue with best-effort mode.

        self.rgb_prim_path = rgb_candidate
        self.depth_prim_path = depth_candidate

        try:
            camera_cls, camera_source = self._resolve_camera_class()
        except Exception as exc:  # noqa: BLE001
            self._fallback_reason = f"camera import failed: {type(exc).__name__}: {exc}"
            msg = (
                "Failed to import Isaac camera module; "
                "using env depth pseudo-RGB fallback. "
                f"reason={type(exc).__name__}: {exc}"
            )
            if self.cfg.strict_d455:
                return False, msg
            return True, msg

        try:
            self._rgb_camera = camera_cls(
                prim_path=self.rgb_prim_path,
                resolution=(int(self.cfg.image_width), int(self.cfg.image_height)),
                annotator_device="cpu",
            )
            if self.depth_prim_path is not None:
                self._depth_camera = camera_cls(
                    prim_path=self.depth_prim_path,
                    resolution=(int(self.cfg.image_width), int(self.cfg.image_height)),
                    annotator_device="cpu",
                )
            self._enable_depth_annotators(self._rgb_camera)
            self._enable_depth_annotators(self._depth_camera)

            for _ in range(4):
                simulation_app.update()
            self._rgb_camera.initialize()
            if self._depth_camera is not None:
                self._depth_camera.initialize()
            self._enable_depth_annotators(self._rgb_camera)
            self._enable_depth_annotators(self._depth_camera)
            for _ in range(4):
                simulation_app.update()
            self._intrinsic = self._read_intrinsic(self._rgb_camera, self.cfg.image_width, self.cfg.image_height)
            if self.cfg.strict_d455 and self._depth_camera is None:
                self._fallback_reason = "D455 depth camera prim missing after initialization."
                return (
                    False,
                    "D455 depth camera prim missing after initialization; strict_d455=true requires live D455 RGB-D.",
                )
            return (
                True,
                "D455 camera initialized: "
                f"rgb={self.rgb_prim_path} depth={self.depth_prim_path} "
                f"camera_class={camera_source} runtime_mount={self._runtime_camera_mode} "
                f"runtime_created={runtime_created} "
                f"existing_rgb={existing_rgb_candidate is not None} "
                f"existing_depth={existing_depth_candidate is not None}",
            )
        except Exception as exc:  # noqa: BLE001
            self._rgb_camera = None
            self._depth_camera = None
            self._fallback_reason = f"camera init failed: {type(exc).__name__}: {exc}"
            msg = (
                "D455 camera initialization failed; "
                "using env depth pseudo-RGB fallback. "
                f"reason={type(exc).__name__}: {exc}"
            )
            if self.cfg.strict_d455:
                return False, msg
            return True, msg

    def get_rgb_camera_pose_world(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        pos, quat = self._camera_pose_from_wrapper(self._rgb_camera)
        if pos is not None and quat is not None:
            return pos, quat
        if self._stage is not None and self.rgb_prim_path is not None:
            return self._camera_pose_from_stage(self._stage, self.rgb_prim_path)
        return None, None

    @staticmethod
    def _resolve_camera_class():
        def _try_candidates() -> tuple[type | None, str | None, list[str]]:
            candidates = (
                ("isaacsim.sensors.camera", "Camera"),
                ("isaacsim.sensors.camera.camera", "Camera"),
                ("omni.isaac.sensor", "Camera"),
                ("omni.isaac.sensor.camera", "Camera"),
                ("omni.isaac.sensor.scripts.camera", "Camera"),
            )
            errors: list[str] = []
            for module_name, class_name in candidates:
                try:
                    module = __import__(module_name, fromlist=[class_name])
                    camera_cls = getattr(module, class_name, None)
                    if camera_cls is not None:
                        return camera_cls, f"{module_name}.{class_name}", errors
                    errors.append(f"{module_name}.{class_name}: attribute missing")
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{module_name}.{class_name}: {type(exc).__name__}: {exc}")
            return None, None, errors

        camera_cls, camera_source, errors = _try_candidates()
        if camera_cls is not None and camera_source is not None:
            return camera_cls, camera_source

        D455SensorAdapter._enable_camera_extension()
        camera_cls, camera_source, errors = _try_candidates()
        if camera_cls is not None and camera_source is not None:
            return camera_cls, camera_source
        raise RuntimeError("No compatible Camera class found. " + " | ".join(errors))

    @staticmethod
    def _enable_camera_extension() -> None:
        try:
            import omni.kit.app
        except Exception:  # noqa: BLE001
            return
        try:
            app = omni.kit.app.get_app()
            if app is None:
                return
            ext_mgr = app.get_extension_manager()
            if ext_mgr is None:
                return
            ext_mgr.set_extension_enabled_immediate("isaacsim.sensors.camera", True)
        except Exception:  # noqa: BLE001
            return

    def capture_rgbd(self, env) -> tuple[np.ndarray | None, np.ndarray | None]:
        rgb, depth, _ = self.capture_rgbd_with_meta(env)
        return rgb, depth

    def capture_rgbd_with_meta(self, env) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
        rgb = self._capture_rgb()
        depth, depth_source = self._capture_depth_from_camera()
        rgb_source = "d455_rgb_camera" if rgb is not None else "missing"
        if depth is None:
            depth_source = "missing"
        fallback_used = False

        if self.cfg.strict_d455:
            if rgb is None or depth is None:
                meta = self._finalize_capture_meta(
                    rgb=rgb,
                    depth=depth,
                    rgb_source=rgb_source,
                    depth_source=depth_source,
                    fallback_used=True,
                    note="strict_d455=true requires D455 RGB+Depth every frame.",
                )
                return None, None, meta
        elif depth is None or rgb is None:
            depth_from_env = self._capture_depth_from_env(env)
            if depth is None and depth_from_env is not None:
                depth = depth_from_env
                depth_source = "env_depth_sensor"
                fallback_used = True
            if rgb is None and depth_from_env is not None:
                depth_for_rgb = self._sanitize_depth(depth_from_env, self.cfg.depth_max_m)
                rgb = self._depth_to_pseudo_rgb(depth_for_rgb, self.cfg.depth_max_m)
                depth = depth_for_rgb
                rgb_source = "pseudo_rgb_from_env_depth"
                depth_source = "env_depth_sensor"
                fallback_used = True

        if rgb is None:
            meta = self._finalize_capture_meta(
                rgb=rgb,
                depth=depth,
                rgb_source=rgb_source,
                depth_source=depth_source,
                fallback_used=True,
                note="RGB capture unavailable.",
            )
            return None, None, meta
        if depth is None:
            meta = self._finalize_capture_meta(
                rgb=rgb,
                depth=depth,
                rgb_source=rgb_source,
                depth_source=depth_source,
                fallback_used=True,
                note="Depth capture unavailable.",
            )
            return rgb, None, meta

        depth = self._sanitize_depth(depth, self.cfg.depth_max_m)
        if depth.shape[:2] != rgb.shape[:2]:
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            fallback_used = True
        meta = self._finalize_capture_meta(
            rgb=rgb,
            depth=depth,
            rgb_source=rgb_source,
            depth_source=depth_source,
            fallback_used=fallback_used,
        )
        return rgb, depth, meta

    def _resolve_camera_prims(self, stage) -> tuple[str | None, str | None]:
        rgb_candidates: list[str] = []
        depth_candidates: list[str] = []
        for root in DEFAULT_G1_PRIM_CANDIDATES:
            rgb_candidates.append(f"{root}{DEFAULT_G1_CAMERA_SUFFIX_COLOR}")
            for depth_suffix in DEFAULT_G1_DEPTH_CAMERA_SUFFIX_CANDIDATES:
                depth_candidates.append(f"{root}{depth_suffix}")

        rgb_prim = self._first_valid_prim(stage, rgb_candidates)
        depth_prim = self._first_valid_prim(stage, depth_candidates)
        if rgb_prim is None or depth_prim is None:
            discovered_rgb, discovered_depth = self._discover_camera_prims(stage)
            if rgb_prim is None:
                rgb_prim = discovered_rgb
            if depth_prim is None:
                depth_prim = discovered_depth
        return rgb_prim, depth_prim

    @staticmethod
    def _first_valid_prim(stage, candidates: list[str]) -> str | None:
        for candidate in candidates:
            prim = stage.GetPrimAtPath(candidate)
            if prim.IsValid():
                return candidate
        return None

    @staticmethod
    def _resolve_runtime_camera_paths(stage) -> tuple[str | None, str | None]:
        robot_root = None
        for root in DEFAULT_G1_PRIM_CANDIDATES:
            prim = stage.GetPrimAtPath(root)
            if prim.IsValid():
                robot_root = root
                break
        if robot_root is None:
            return None, None

        torso_path = f"{robot_root}/torso_link"
        attach_path = torso_path if stage.GetPrimAtPath(torso_path).IsValid() else robot_root
        runtime_base = f"{attach_path}/NavDP_D455_Runtime"
        rgb_path = f"{runtime_base}/Camera_OmniVision_OV9782_Color"
        depth_path = f"{runtime_base}/Camera_OmniVision_OV9782_Depth"
        return rgb_path, depth_path

    @staticmethod
    def _is_valid_prim_path(stage, path: str | None) -> bool:
        if path is None:
            return False
        try:
            prim = stage.GetPrimAtPath(path)
        except Exception:  # noqa: BLE001
            return False
        return bool(prim.IsValid())

    def _ensure_runtime_camera_prims(self, stage, rgb_path: str, depth_path: str) -> bool:
        try:
            runtime_base = rgb_path.rsplit("/", maxsplit=1)[0]
            base_prim = stage.DefinePrim(runtime_base, "Xform")
            rgb_prim = stage.DefinePrim(rgb_path, "Camera")
            depth_prim = stage.DefinePrim(depth_path, "Camera")
            return bool(base_prim.IsValid() and rgb_prim.IsValid() and depth_prim.IsValid())
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _discover_camera_prims(stage) -> tuple[str | None, str | None]:
        rgb_candidates: list[str] = []
        depth_candidates: list[str] = []
        depth_names = {
            "Camera_OmniVision_OV9782_Depth",
            "Camera_Realsense_Depth",
            "Camera_Depth",
            "Camera_Pseudo_Depth",
        }
        try:
            iterator = stage.Traverse()
        except Exception:  # noqa: BLE001
            return None, None

        for prim in iterator:
            if not prim.IsValid():
                continue
            path = str(prim.GetPath())
            name = path.rsplit("/", maxsplit=1)[-1]
            if name == "Camera_OmniVision_OV9782_Color":
                rgb_candidates.append(path)
            if name in depth_names:
                depth_candidates.append(path)

        def _pick(paths: list[str]) -> str | None:
            if len(paths) == 0:
                return None
            ranked = sorted(
                paths,
                key=lambda p: (
                    0 if "/Robot/" in p or "/robot/" in p else 1,
                    len(p),
                    p,
                ),
            )
            return ranked[0]

        return _pick(rgb_candidates), _pick(depth_candidates)

    def _capture_rgb(self) -> np.ndarray | None:
        if self._rgb_camera is None:
            return None
        try:
            rgba = self._rgb_camera.get_rgba()
            if rgba is None:
                return None
            arr = self._coerce_array(rgba)
            if arr is None:
                return None
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim != 3 or arr.shape[-1] < 3:
                return None
            rgb = arr[..., :3]
            if rgb.dtype != np.uint8:
                rgb = rgb.astype(np.float32)
                if np.nanmax(rgb) <= 1.5:
                    rgb = rgb * 255.0
                rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
            return rgb
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _coerce_array(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value
        detach = getattr(value, "detach", None)
        if callable(detach):
            try:
                value = detach()
            except Exception:  # noqa: BLE001
                pass
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            try:
                value = cpu()
            except Exception:  # noqa: BLE001
                pass
        numpy_fn = getattr(value, "numpy", None)
        if callable(numpy_fn):
            try:
                return np.asarray(numpy_fn())
            except Exception:  # noqa: BLE001
                pass
        try:
            return np.asarray(value)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _normalize_depth_candidate(depth: Any) -> np.ndarray | None:
        depth_arr = D455SensorAdapter._coerce_array(depth)
        if depth_arr is None:
            return None
        try:
            depth_arr = np.asarray(depth_arr, dtype=np.float32)
        except Exception:  # noqa: BLE001
            return None
        if depth_arr.ndim == 4 and depth_arr.shape[0] == 1:
            depth_arr = depth_arr[0]
        if depth_arr.ndim == 3 and depth_arr.shape[-1] == 1:
            depth_arr = depth_arr[..., 0]
        if depth_arr.ndim != 2:
            return None
        return depth_arr

    def _extract_depth_from_camera(self, camera) -> np.ndarray | None:
        if camera is None:
            return None

        for method_name in ("get_depth", "get_distance_to_image_plane", "get_depth_image"):
            method = getattr(camera, method_name, None)
            if method is None:
                continue
            try:
                candidate = self._normalize_depth_candidate(method())
                if candidate is not None:
                    return candidate
            except Exception:  # noqa: BLE001
                continue

        current_frame_fn = getattr(camera, "get_current_frame", None)
        if current_frame_fn is not None:
            try:
                frame = current_frame_fn()
                if isinstance(frame, dict):
                    for key in (
                        "distance_to_image_plane",
                        "distance_to_camera",
                        "depth",
                        "depthLinear",
                        "depth_linear",
                    ):
                        if key not in frame:
                            continue
                        candidate = self._normalize_depth_candidate(frame.get(key))
                        if candidate is not None:
                            return candidate
            except Exception:  # noqa: BLE001
                return None
        return None

    @staticmethod
    def _camera_pose_from_wrapper(camera) -> tuple[np.ndarray | None, np.ndarray | None]:
        if camera is None:
            return None, None
        get_world_pose = getattr(camera, "get_world_pose", None)
        if get_world_pose is None:
            return None, None
        try:
            pos_raw, quat_raw = get_world_pose()
            pos = np.asarray(pos_raw, dtype=np.float32).reshape(-1)
            quat = np.asarray(quat_raw, dtype=np.float32).reshape(-1)
            if pos.shape[0] < 3 or quat.shape[0] < 4:
                return None, None
            return pos[:3].copy(), quat[:4].copy()
        except Exception:  # noqa: BLE001
            return None, None

    @staticmethod
    def _camera_pose_from_stage(stage, prim_path: str) -> tuple[np.ndarray | None, np.ndarray | None]:
        try:
            from pxr import Usd, UsdGeom
        except Exception:  # noqa: BLE001
            return None, None
        try:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                return None, None
            xform = UsdGeom.Xformable(prim)
            world_tf = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            trans = world_tf.ExtractTranslation()
            quat_gf = world_tf.ExtractRotationQuat()
            imag = quat_gf.GetImaginary()
            pos = np.asarray([trans[0], trans[1], trans[2]], dtype=np.float32)
            quat = np.asarray([quat_gf.GetReal(), imag[0], imag[1], imag[2]], dtype=np.float32)
            return pos, quat
        except Exception:  # noqa: BLE001
            return None, None

    def _capture_depth_from_camera(self) -> tuple[np.ndarray | None, str]:
        depth = self._extract_depth_from_camera(self._depth_camera)
        if depth is not None:
            return depth, "d455_depth_camera"

        depth = self._extract_depth_from_camera(self._rgb_camera)
        if depth is not None:
            return depth, "d455_rgb_camera_depth"
        return None, "missing"

    @staticmethod
    def _enable_depth_annotators(camera) -> None:
        if camera is None:
            return
        for method_name in (
            "add_distance_to_image_plane_to_frame",
            "add_distance_to_camera_to_frame",
        ):
            method = getattr(camera, method_name, None)
            if method is None:
                continue
            try:
                method()
            except Exception:  # noqa: BLE001
                continue

    @staticmethod
    def _capture_depth_from_env(env) -> np.ndarray | None:
        try:
            sensor = env.unwrapped.scene.sensors["depth_sensor"]
            depth_tensor = sensor.data.output["distance_to_image_plane"][0]
            depth_np = depth_tensor.detach().cpu().numpy().astype(np.float32)
            if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
                depth_np = depth_np[:, :, 0]
            if depth_np.ndim == 2:
                return depth_np
        except Exception:  # noqa: BLE001
            return None
        return None

    @staticmethod
    def _sanitize_depth(depth_m: np.ndarray, depth_max_m: float) -> np.ndarray:
        depth = np.asarray(depth_m, dtype=np.float32)
        depth = np.nan_to_num(depth, nan=float(depth_max_m), posinf=float(depth_max_m), neginf=0.0)
        depth = np.clip(depth, 0.0, float(depth_max_m))
        return depth

    @staticmethod
    def _depth_to_pseudo_rgb(depth_m: np.ndarray, depth_max_m: float) -> np.ndarray:
        depth = np.asarray(depth_m, dtype=np.float32)
        depth = np.nan_to_num(depth, nan=float(depth_max_m), posinf=float(depth_max_m), neginf=0.0)
        denom = float(max(depth_max_m, 1.0e-6))
        norm = np.clip(depth / denom, 0.0, 1.0)
        gray = np.clip((1.0 - norm) * 255.0, 0.0, 255.0).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)

    @staticmethod
    def _default_intrinsic(width: int, height: int) -> np.ndarray:
        fx = float(max(width, height))
        fy = float(max(width, height))
        cx = float(width) * 0.5
        cy = float(height) * 0.5
        return np.asarray(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def _read_intrinsic(self, camera, width: int, height: int) -> np.ndarray:
        for method_name in ("get_intrinsics_matrix", "get_intrinsic_matrix"):
            method = getattr(camera, method_name, None)
            if method is None:
                continue
            try:
                matrix = np.asarray(method(), dtype=np.float32)
                if matrix.shape == (3, 3):
                    return matrix
            except Exception:  # noqa: BLE001
                continue
        return self._default_intrinsic(width, height)


__all__ = ["D455SensorAdapter", "D455SensorAdapterConfig"]
