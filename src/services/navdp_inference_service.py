from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
from PIL import Image

from common.cv2_compat import cv2
from inference.policy_agent import NavDP_Agent


def _load_rgb_bgr(image_file, *, batch_size: int) -> np.ndarray:
    image = Image.open(image_file.stream).convert("RGB")
    image_np = np.asarray(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return image_bgr.reshape((batch_size, -1, image_bgr.shape[1], 3))


def _load_depth(depth_file, *, batch_size: int) -> np.ndarray:
    depth = Image.open(depth_file.stream).convert("I")
    depth_np = np.asarray(depth)[:, :, np.newaxis]
    depth_np = depth_np.astype(np.float32) / 10000.0
    return depth_np.reshape((batch_size, -1, depth_np.shape[1], 1))


class NavDPInferenceService:
    def __init__(
        self,
        *,
        checkpoint: str,
        device: str,
        amp: bool,
        amp_dtype: str,
        tf32: bool,
    ) -> None:
        self.checkpoint = str(checkpoint)
        self.device = str(device)
        self.amp = bool(amp)
        self.amp_dtype = str(amp_dtype)
        self.tf32 = bool(tf32)
        self.navigator = None
        self.last_input_meta: dict[str, Any] = {}

    def _ensure_navigator(self, intrinsic: np.ndarray) -> None:
        if self.navigator is not None:
            return
        print(
            "[NavDP] init device={} amp={} amp_dtype={} tf32={}".format(
                self.device, self.amp, self.amp_dtype, self.tf32
            )
        )
        self.navigator = NavDP_Agent(
            intrinsic,
            image_size=224,
            memory_size=8,
            predict_size=24,
            temporal_depth=16,
            heads=8,
            token_dim=384,
            navi_model=self.checkpoint,
            device=self.device,
            use_amp=self.amp,
            amp_dtype=self.amp_dtype,
            enable_tf32=self.tf32,
        )

    def _build_input_meta(self, goal_data: dict[str, Any], image: np.ndarray, depth: np.ndarray, batch_size: int) -> dict[str, Any]:
        sensor_meta = goal_data.get("sensor_meta", {})
        client_meta = goal_data.get("client_meta", {})
        depth_min = float(np.min(depth)) if depth.size > 0 else 0.0
        depth_max = float(np.max(depth)) if depth.size > 0 else 0.0
        depth_nonzero_ratio = float(np.count_nonzero(depth > 1.0e-6)) / float(depth.size) if depth.size > 0 else 0.0
        return {
            "batch_size": int(batch_size),
            "image_shape": list(image.shape),
            "depth_shape": list(depth.shape),
            "depth_min_m": depth_min,
            "depth_max_m": depth_max,
            "depth_nonzero_ratio": depth_nonzero_ratio,
            "sensor_meta": sensor_meta if isinstance(sensor_meta, dict) else {},
            "client_meta": client_meta if isinstance(client_meta, dict) else {},
        }

    def navigator_reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        intrinsic = np.asarray(payload.get("intrinsic"), dtype=np.float32)
        threshold = np.asarray(payload.get("stop_threshold"))
        batch_size = np.asarray(payload.get("batch_size"))
        self._ensure_navigator(intrinsic)
        self.navigator.reset(batch_size, threshold)
        return {"algo": "navdp"}

    def navigator_reset_env(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.navigator is None:
            raise RuntimeError("navigator_reset must be called before navigator_reset_env")
        self.navigator.reset_env(int(payload.get("env_id")))
        return {"algo": "navdp"}

    def pointgoal_step(self, image_file, depth_file, goal_data: dict[str, Any]) -> dict[str, Any]:
        if self.navigator is None:
            raise RuntimeError("navigator_reset must be called before pointgoal_step")
        start_time = time.time()
        goal_x = np.asarray(goal_data["goal_x"])
        goal_y = np.asarray(goal_data["goal_y"])
        goal = np.stack((goal_x, goal_y, np.zeros_like(goal_x)), axis=1)
        batch_size = int(self.navigator.batch_size)

        phase1_time = time.time()
        image = _load_rgb_bgr(image_file, batch_size=batch_size)
        depth = _load_depth(depth_file, batch_size=batch_size)
        input_meta = self._build_input_meta(goal_data, image, depth, batch_size)
        self.last_input_meta = input_meta
        print(
            "[NAVDP_IO] pointgoal_step sensor={}/{} depth=[{:.4f},{:.4f}] nonzero={:.4f}".format(
                input_meta["sensor_meta"].get("rgb_source", "-"),
                input_meta["sensor_meta"].get("depth_source", "-"),
                input_meta["depth_min_m"],
                input_meta["depth_max_m"],
                input_meta["depth_nonzero_ratio"],
            )
        )

        phase2_time = time.time()
        execute_trajectory, all_trajectory, all_values = self.navigator.step_pointgoal(goal, image, depth)
        phase3_time = time.time()
        print(
            "phase1:%f, phase2:%f, phase3:%f, all:%f"
            % (phase1_time - start_time, phase2_time - phase1_time, phase3_time - phase2_time, time.time() - start_time)
        )
        return {
            "trajectory": execute_trajectory.tolist(),
            "all_trajectory": all_trajectory.tolist(),
            "all_values": all_values.tolist(),
            "input_meta": input_meta,
        }

    def pixelgoal_step(self, image_file, depth_file, goal_data: dict[str, Any]) -> dict[str, Any]:
        if self.navigator is None:
            raise RuntimeError("navigator_reset must be called before pixelgoal_step")
        start_time = time.time()
        goal_x = np.asarray(goal_data["goal_x"])
        goal_y = np.asarray(goal_data["goal_y"])
        goal = np.stack((goal_x, goal_y), axis=1)
        batch_size = int(self.navigator.batch_size)

        phase1_time = time.time()
        image = _load_rgb_bgr(image_file, batch_size=batch_size)
        depth = _load_depth(depth_file, batch_size=batch_size)

        phase2_time = time.time()
        execute_trajectory, all_trajectory, all_values = self.navigator.step_pixelgoal(goal, image, depth)
        phase3_time = time.time()
        print(
            "phase1:%f, phase2:%f, phase3:%f, all:%f"
            % (phase1_time - start_time, phase2_time - phase1_time, phase3_time - phase2_time, time.time() - start_time)
        )
        return {
            "trajectory": execute_trajectory.tolist(),
            "all_trajectory": all_trajectory.tolist(),
            "all_values": all_values.tolist(),
        }

    def imagegoal_step(self, image_file, depth_file, goal_file) -> dict[str, Any]:
        if self.navigator is None:
            raise RuntimeError("navigator_reset must be called before imagegoal_step")
        start_time = time.time()
        batch_size = int(self.navigator.batch_size)

        phase1_time = time.time()
        image = _load_rgb_bgr(image_file, batch_size=batch_size)
        goal = _load_rgb_bgr(goal_file, batch_size=batch_size)
        depth = _load_depth(depth_file, batch_size=batch_size)

        phase2_time = time.time()
        execute_trajectory, all_trajectory, all_values = self.navigator.step_imagegoal(goal, image, depth)
        phase3_time = time.time()
        print(
            "phase1:%f, phase2:%f, phase3:%f, all:%f"
            % (phase1_time - start_time, phase2_time - phase1_time, phase3_time - phase2_time, time.time() - start_time)
        )
        return {
            "trajectory": execute_trajectory.tolist(),
            "all_trajectory": all_trajectory.tolist(),
            "all_values": all_values.tolist(),
        }

    def nogoal_step(self, image_file, depth_file) -> dict[str, Any]:
        if self.navigator is None:
            raise RuntimeError("navigator_reset must be called before nogoal_step")
        start_time = time.time()
        batch_size = int(self.navigator.batch_size)

        phase1_time = time.time()
        image = _load_rgb_bgr(image_file, batch_size=batch_size)
        depth = _load_depth(depth_file, batch_size=batch_size)

        phase2_time = time.time()
        execute_trajectory, all_trajectory, all_values = self.navigator.step_nogoal(image, depth)
        phase3_time = time.time()
        print(
            "phase1:%f, phase2:%f, phase3:%f, all:%f"
            % (phase1_time - start_time, phase2_time - phase1_time, phase3_time - phase2_time, time.time() - start_time)
        )
        return {
            "trajectory": execute_trajectory.tolist(),
            "all_trajectory": all_trajectory.tolist(),
            "all_values": all_values.tolist(),
        }

    def point_image_mixgoal_step(self, image_file, depth_file, image_goal_file, goal_data: dict[str, Any]) -> dict[str, Any]:
        if self.navigator is None:
            raise RuntimeError("navigator_reset must be called before navdp_step_ip_mixgoal")
        start_time = time.time()
        batch_size = int(self.navigator.batch_size)

        point_goal_x = np.asarray(goal_data["goal_x"])
        point_goal_y = np.asarray(goal_data["goal_y"])
        point_goal = np.stack((point_goal_x, point_goal_y, np.zeros_like(point_goal_x)), axis=1)

        image_goal = _load_rgb_bgr(image_goal_file, batch_size=batch_size)

        phase1_time = time.time()
        image = _load_rgb_bgr(image_file, batch_size=batch_size)
        depth = _load_depth(depth_file, batch_size=batch_size)

        phase2_time = time.time()
        execute_trajectory, all_trajectory, all_values = self.navigator.step_point_image_goal(
            point_goal, image_goal, image, depth
        )
        phase3_time = time.time()
        print(
            "phase1:%f, phase2:%f, phase3:%f, all:%f"
            % (phase1_time - start_time, phase2_time - phase1_time, phase3_time - phase2_time, time.time() - start_time)
        )
        return {
            "trajectory": execute_trajectory.tolist(),
            "all_trajectory": all_trajectory.tolist(),
            "all_values": all_values.tolist(),
        }

    @staticmethod
    def parse_goal_data(raw_goal_data: str | None) -> dict[str, Any]:
        if raw_goal_data is None:
            raise ValueError("missing form field: goal_data")
        return json.loads(raw_goal_data)
