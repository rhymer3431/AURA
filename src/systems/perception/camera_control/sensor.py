"""Camera sensor and pitch-control primitives shared by G1 runtime modes."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time

import numpy as np

from .math import (
    quaternion_from_axis_angle_wxyz,
    quaternion_multiply_wxyz,
    rotation_matrix_from_quaternion_wxyz,
)


@dataclass(slots=True)
class CameraFrame:
    """Single RGB-D capture plus camera calibration and pose."""

    rgb: np.ndarray
    depth: np.ndarray
    intrinsic: np.ndarray
    camera_pos_w: np.ndarray
    camera_rot_w: np.ndarray
    stamp_s: float


class G1NavCameraSensor:
    """Attach and read an RGB-D camera from a G1 body link."""

    def __init__(
        self,
        prim_path: str,
        resolution: tuple[int, int],
        translation: tuple[float, float, float],
        orientation_wxyz: tuple[float, float, float, float],
        clipping_range: tuple[float, float],
        initial_pitch_deg: float = 0.0,
        pitch_limits_deg: tuple[float, float] = (-45.0, 45.0),
        attach_streams: bool = True,
        annotator_device: str = "cpu",
    ):
        self.prim_path = str(prim_path)
        self.control_prim_path = str(prim_path)
        self.camera_prim_path: str | None = str(prim_path)
        self.resolution = (int(resolution[0]), int(resolution[1]))
        self.translation = tuple(float(v) for v in translation)
        self.orientation_wxyz = tuple(float(v) for v in orientation_wxyz)
        self.clipping_range = (float(clipping_range[0]), float(clipping_range[1]))
        self.attach_streams = bool(attach_streams)
        self.annotator_device = str(annotator_device)
        self._camera = None
        self._control_prim = None
        self._control_translation = np.asarray(self.translation, dtype=np.float32)
        self._base_orientation_wxyz = np.asarray(self.orientation_wxyz, dtype=np.float32)
        lower_limit = float(min(pitch_limits_deg[0], pitch_limits_deg[1]))
        upper_limit = float(max(pitch_limits_deg[0], pitch_limits_deg[1]))
        self._pitch_limits_deg = (lower_limit, upper_limit)
        self._pitch_lock = threading.Lock()
        self._target_pitch_deg = self._clamp_pitch_deg(initial_pitch_deg)
        self._applied_pitch_deg: float | None = None
        self._pitch_dirty = True

    def _clamp_pitch_deg(self, pitch_deg: float) -> float:
        lower_limit, upper_limit = self._pitch_limits_deg
        return float(min(max(float(pitch_deg), lower_limit), upper_limit))

    def _orientation_for_pitch(self, pitch_deg: float) -> np.ndarray:
        # Keep the external API intuitive: positive pitch looks upward, negative looks downward.
        pitch_quat = quaternion_from_axis_angle_wxyz(
            np.asarray((0.0, 1.0, 0.0), dtype=np.float64),
            np.deg2rad(-pitch_deg),
        )
        return quaternion_multiply_wxyz(np.asarray(self._base_orientation_wxyz, dtype=np.float64), pitch_quat).astype(
            np.float32
        )

    @staticmethod
    def _iter_descendants(prim):
        for child in prim.GetChildren():
            yield child
            yield from G1NavCameraSensor._iter_descendants(child)

    def _select_camera_descendant(self, root_prim) -> str | None:
        camera_paths = []
        for prim in self._iter_descendants(root_prim):
            if prim.GetTypeName() == "Camera":
                camera_paths.append(prim.GetPath().pathString)
        if not camera_paths:
            return None

        preferred_tokens = ("color", "rgb", "left", "camera")
        for token in preferred_tokens:
            for path in camera_paths:
                if token in path.lower():
                    return path
        return camera_paths[0]

    def _resolve_existing_paths(self):
        from isaacsim.core.utils.prims import get_prim_at_path

        prim = get_prim_at_path(self.prim_path)
        if not prim.IsValid():
            self.control_prim_path = self.prim_path
            self.camera_prim_path = self.prim_path
            return False

        self.control_prim_path = self.prim_path
        if prim.GetTypeName() == "Camera":
            self.camera_prim_path = self.prim_path
            return True

        self.camera_prim_path = self._select_camera_descendant(prim)
        return True

    def pitch_status(self) -> dict[str, object]:
        with self._pitch_lock:
            target_pitch_deg = self._target_pitch_deg
            applied_pitch_deg = self._applied_pitch_deg
            lower_limit, upper_limit = self._pitch_limits_deg
        return {
            "prim_path": self.prim_path,
            "control_prim_path": self.control_prim_path,
            "camera_prim_path": self.camera_prim_path,
            "ready": self._camera is not None,
            "target_pitch_deg": target_pitch_deg,
            "applied_pitch_deg": applied_pitch_deg,
            "min_pitch_deg": lower_limit,
            "max_pitch_deg": upper_limit,
        }

    def set_pitch_deg(self, pitch_deg: float) -> float:
        clamped_pitch_deg = self._clamp_pitch_deg(pitch_deg)
        with self._pitch_lock:
            self._target_pitch_deg = clamped_pitch_deg
            self._pitch_dirty = True
        return clamped_pitch_deg

    def add_pitch_deg(self, delta_pitch_deg: float) -> float:
        with self._pitch_lock:
            next_pitch_deg = self._clamp_pitch_deg(self._target_pitch_deg + float(delta_pitch_deg))
            self._target_pitch_deg = next_pitch_deg
            self._pitch_dirty = True
        return next_pitch_deg

    def apply_pending_pitch(self) -> bool:
        if self._camera is None:
            return False

        with self._pitch_lock:
            target_pitch_deg = self._target_pitch_deg
            needs_update = self._pitch_dirty or self._applied_pitch_deg is None
            if not needs_update and abs(float(self._applied_pitch_deg) - target_pitch_deg) <= 1e-6:
                return False
            self._pitch_dirty = False

        orientation = self._orientation_for_pitch(target_pitch_deg)
        try:
            # Avoid writing pose changes to articulation-linked rig roots while PhysX is running.
            # We only rotate the Camera prim itself, which keeps the pitch control local to perception.
            self._camera.set_local_pose(
                translation=np.asarray(self._control_translation, dtype=np.float32),
                orientation=orientation,
                camera_axes="world",
            )
        except Exception:
            with self._pitch_lock:
                self._pitch_dirty = True
            raise

        with self._pitch_lock:
            self._applied_pitch_deg = target_pitch_deg
        return True

    def _enable_camera_extension(self):
        import omni.kit.app

        extension_manager = omni.kit.app.get_app().get_extension_manager()
        extension_manager.set_extension_enabled_immediate("isaacsim.sensors.camera", True)

    def attach(self):
        if self._control_prim is not None:
            return

        self._enable_camera_extension()

        from isaacsim.core.utils.prims import get_prim_at_path
        from isaacsim.sensors.camera import Camera

        existing_root = self._resolve_existing_paths()
        existing_camera = False if self.camera_prim_path is None else bool(get_prim_at_path(self.camera_prim_path).IsValid())

        if existing_root:
            if existing_camera and self.camera_prim_path:
                self._camera = Camera(
                    prim_path=self.camera_prim_path,
                    name="navdp_camera",
                    resolution=self.resolution,
                    annotator_device=self.annotator_device,
                )
                if hasattr(self._camera, "set_clipping_range"):
                    self._camera.set_clipping_range(*self.clipping_range)
                self._camera.initialize(attach_rgb_annotator=False)
                if self.attach_streams:
                    self._camera.add_rgb_to_frame()
                    self._camera.add_distance_to_image_plane_to_frame()
                control_translation, control_orientation = self._camera.get_local_pose(camera_axes="world")
                self._control_translation = np.asarray(control_translation, dtype=np.float32)
                self._base_orientation_wxyz = np.asarray(control_orientation, dtype=np.float32)
                self.control_prim_path = self._camera.prim_path
                self.camera_prim_path = self._camera.prim_path
                print(f"[INFO] Using existing camera prim for pitch control: {self.control_prim_path}")
                self.apply_pending_pitch()
                return

            runtime_camera_path = f"{self.control_prim_path.rstrip('/')}/AuraRuntimeCamera"
            self._camera = Camera(
                prim_path=runtime_camera_path,
                name="navdp_camera",
                resolution=self.resolution,
                translation=np.asarray(self.translation, dtype=np.float32),
                orientation=np.asarray(self.orientation_wxyz, dtype=np.float32),
                annotator_device=self.annotator_device,
            )
            if hasattr(self._camera, "set_clipping_range"):
                self._camera.set_clipping_range(*self.clipping_range)
            self._camera.initialize(attach_rgb_annotator=False)
            if self.attach_streams:
                self._camera.add_rgb_to_frame()
                self._camera.add_distance_to_image_plane_to_frame()
            self.camera_prim_path = self._camera.prim_path
            self.control_prim_path = self._camera.prim_path
            self._control_translation = np.asarray(self.translation, dtype=np.float32)
            self._base_orientation_wxyz = np.asarray(self.orientation_wxyz, dtype=np.float32)
            print(f"[INFO] Using runtime child camera for pitch control: {self.control_prim_path}")
            self.apply_pending_pitch()
            return

        self._camera = Camera(
            prim_path=self.camera_prim_path or self.prim_path,
            name="navdp_camera",
            resolution=self.resolution,
            translation=None if existing_camera else np.asarray(self.translation, dtype=np.float32),
            orientation=None if existing_camera else np.asarray(self.orientation_wxyz, dtype=np.float32),
            annotator_device=self.annotator_device,
        )
        if hasattr(self._camera, "set_clipping_range"):
            self._camera.set_clipping_range(*self.clipping_range)
        # Isaac Camera creates its render product during initialize(). Attaching annotators before that
        # can bind them to a None render product path and produce repeated "not attached to None" warnings.
        self._camera.initialize(attach_rgb_annotator=False)
        if self.attach_streams:
            self._camera.add_rgb_to_frame()
            self._camera.add_distance_to_image_plane_to_frame()
        self.camera_prim_path = self._camera.prim_path
        self.control_prim_path = self._camera.prim_path
        if existing_camera:
            control_translation, control_orientation = self._camera.get_local_pose(camera_axes="world")
            self._control_translation = np.asarray(control_translation, dtype=np.float32)
            self._base_orientation_wxyz = np.asarray(control_orientation, dtype=np.float32)
            print(f"[INFO] Using existing camera prim directly for pitch control: {self.control_prim_path}")
        else:
            self._control_translation = np.asarray(self.translation, dtype=np.float32)
            self._base_orientation_wxyz = np.asarray(self.orientation_wxyz, dtype=np.float32)
            print(f"[INFO] Using runtime camera prim for pitch control: {self.control_prim_path}")
        self.apply_pending_pitch()

    def intrinsic_matrix(self) -> np.ndarray:
        if self._camera is None:
            raise RuntimeError("Camera must be attached before querying intrinsics.")
        intrinsic = self._camera.get_intrinsics_matrix()
        return np.asarray(intrinsic, dtype=np.float32)

    def capture_frame(self) -> CameraFrame | None:
        if self._camera is None:
            raise RuntimeError("Camera must be attached before capture.")
        if not self.attach_streams:
            raise RuntimeError("Camera streams are disabled for this sensor instance.")
        self.apply_pending_pitch()

        rgb = self._camera.get_rgb() if hasattr(self._camera, "get_rgb") else None
        if rgb is None and hasattr(self._camera, "get_rgba"):
            rgb = self._camera.get_rgba()
        depth = self._camera.get_depth() if hasattr(self._camera, "get_depth") else None
        if depth is None and hasattr(self._camera, "get_current_frame"):
            depth = self._camera.get_current_frame().get("distance_to_image_plane")
        if rgb is None or depth is None:
            return None

        rgb = np.asarray(rgb)
        depth = np.asarray(depth)
        if rgb.size == 0 or depth.size == 0:
            return None

        if rgb.ndim == 4:
            rgb = rgb[0]
        if rgb.shape[-1] == 4:
            rgb = rgb[:, :, :3]
        if np.issubdtype(rgb.dtype, np.floating):
            if rgb.max(initial=0.0) <= 1.0:
                rgb = rgb * 255.0
            rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
        else:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[:, :, 0]
        if depth.ndim == 3:
            depth = depth[0]

        position_w, orientation_wxyz = self._camera.get_world_pose(camera_axes="world")
        intrinsic = self.intrinsic_matrix()
        return CameraFrame(
            rgb=rgb,
            depth=np.asarray(depth, dtype=np.float32),
            intrinsic=intrinsic,
            camera_pos_w=np.asarray(position_w, dtype=np.float32),
            camera_rot_w=rotation_matrix_from_quaternion_wxyz(np.asarray(orientation_wxyz, dtype=np.float32)).astype(
                np.float32
            ),
            stamp_s=time.monotonic(),
        )

    def shutdown(self):
        if self._camera is not None:
            self._camera.destroy()
            self._camera = None
        self._control_prim = None
