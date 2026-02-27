from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional, Tuple

from .contracts import Plan, Pose2D, SkillCall, SlamMode, pose_from_dict
from .exploration import ExplorationBehavior
from .look_at_controller import LookAtController, LookAtStatus
from .manipulation_groot_trt import GrootManipulator
from .memory import SceneMemory
from .nav2_client import Nav2Client
from .slam_monitor import SLAMMonitor


class TaskExecutor:
    def __init__(
        self,
        memory: SceneMemory,
        nav_client: Nav2Client,
        manipulator: GrootManipulator,
        exploration: ExplorationBehavior,
        slam_monitor: SLAMMonitor,
        perception: Optional[Any] = None,
        look_at_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.memory = memory
        self.nav_client = nav_client
        self.manipulator = manipulator
        self.exploration = exploration
        self.slam_monitor = slam_monitor
        self.perception = perception
        self.look_at_cfg = dict(look_at_cfg or {})

        self.pause_event = asyncio.Event()
        self.pause_event.set()
        self._exploration_task: Optional[asyncio.Task] = None
        self._current_target: Optional[str] = None
        self._look_at_controller: Optional[LookAtController] = None
        self._look_at_task: Optional[asyncio.Task] = None
        self._look_at_status = LookAtStatus()
        self._look_at_last_log_ts = 0.0

        self._init_look_at_controller()

    def _init_look_at_controller(self) -> None:
        if self.perception is None:
            return
        if not hasattr(self.manipulator, "get_camera_aim") or not hasattr(self.manipulator, "command_camera_aim"):
            logging.warning("look_at disabled: manipulator camera-aim API unavailable.")
            return
        if not hasattr(self.perception, "get_tracked_target"):
            logging.warning("look_at disabled: perception tracker API unavailable.")
            return
        self._look_at_controller = LookAtController(
            get_target=lambda label, max_age: self.perception.get_tracked_target(label, max_age),
            get_camera_aim=self.manipulator.get_camera_aim,
            command_camera_aim=self.manipulator.command_camera_aim,
        )

    def get_look_at_status(self) -> Dict[str, Any]:
        status = self._look_at_status
        return {
            "state": status.state,
            "object": status.object_label,
            "target_score": status.target_score,
            "detections_ok": status.detections_ok,
            "message": status.message,
            "updated_at": status.updated_at,
        }

    async def on_slam_mode_changed(self, mode: str, c_loc: float) -> None:
        if mode == SlamMode.EXPLORATION:
            logging.warning("Task executor paused (C_loc=%.2f).", c_loc)
            self.pause_event.clear()
            if self._exploration_task is None or self._exploration_task.done():
                self._exploration_task = asyncio.create_task(self._run_exploration())
            return

        logging.info("Task executor resumed (C_loc=%.2f).", c_loc)
        self.pause_event.set()

    async def execute_plan(self, plan: Plan) -> bool:
        if not plan.skills:
            logging.warning("Planner returned empty plan.")
            return False

        for idx, skill in enumerate(plan.skills):
            logging.info("Executing skill %s/%s: %s args=%s", idx + 1, len(plan.skills), skill.name, skill.args)
            ok = await self._execute_with_retry(skill)
            if not ok:
                logging.error("Plan failed at skill '%s'.", skill.name)
                return False
        logging.info("Plan completed successfully.")
        return True

    async def _execute_with_retry(self, skill: SkillCall) -> bool:
        max_retries = max(0, skill.retry_policy.max_retries)
        for attempt in range(max_retries + 1):
            await self.pause_event.wait()
            ok, reason = await self._execute_once(skill)
            if ok:
                return True

            logging.warning(
                "Skill '%s' failed on attempt %s/%s: reason=%s",
                skill.name,
                attempt + 1,
                max_retries + 1,
                reason,
            )
            if skill.name == "navigate":
                await self._on_navigation_failure(reason)
            if attempt < max_retries:
                await asyncio.sleep(skill.retry_policy.backoff_s * (attempt + 1))
        return False

    async def _execute_once(self, skill: SkillCall) -> Tuple[bool, str]:
        name = skill.name.strip().lower()
        if name == "locate":
            return await self._skill_locate(skill)
        if name == "navigate":
            return await self._skill_navigate(skill)
        if name == "pick":
            return await self._skill_pick(skill)
        if name == "return":
            return await self._skill_return(skill)
        if name == "inspect":
            return await self._skill_inspect(skill)
        if name == "fetch":
            return await self._skill_fetch(skill)
        if name == "look_at":
            return await self._skill_look_at(skill)
        return False, f"UNKNOWN_SKILL:{name}"

    async def _skill_locate(self, skill: SkillCall) -> Tuple[bool, str]:
        target = str(skill.args.get("object") or skill.args.get("target") or "apple").lower()
        timeout_s = float(skill.args.get("timeout_s", 8.0))
        self._current_target = target
        self.memory.set_task_focus([target])

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            await self.pause_event.wait()
            found = self.memory.get_object_pose(target)
            if found is not None:
                pose, confidence, last_seen = found
                logging.info(
                    "Locate success: target=%s pose=(%.2f,%.2f) conf=%.2f last_seen=%.2f",
                    target,
                    pose.x,
                    pose.y,
                    confidence,
                    last_seen,
                )
                return True, "OK"
            await asyncio.sleep(0.2)
        return False, "NOT_FOUND"

    async def _skill_navigate(self, skill: SkillCall) -> Tuple[bool, str]:
        target_pose = self._resolve_target_pose(skill)
        if target_pose is None:
            return False, "NO_TARGET_POSE"

        timeout_s = float(skill.args.get("timeout_s", 20.0))
        result = await self.nav_client.navigate_to_pose(
            target_pose, timeout_s=timeout_s, pause_event=self.pause_event
        )
        if result.success:
            return True, "OK"
        return False, result.reason

    def _resolve_target_pose(self, skill: SkillCall) -> Optional[Pose2D]:
        if "pose" in skill.args and isinstance(skill.args["pose"], dict):
            return pose_from_dict(skill.args["pose"])

        target = str(skill.args.get("target", "")).lower()
        target_obj = str(skill.args.get("object", "")).lower() or self._current_target
        if target == "start":
            return self.memory.get_start_pose()
        if target in {"object", ""} and target_obj:
            pose_info = self.memory.get_object_pose(target_obj)
            if pose_info is not None:
                return pose_info[0]
        return None

    async def _skill_pick(self, skill: SkillCall) -> Tuple[bool, str]:
        target = str(skill.args.get("object") or self._current_target or "object")
        instruction = str(skill.args.get("instruction", ""))
        ok = await self.manipulator.pick(target, instruction=instruction, pause_event=self.pause_event)
        return (True, "OK") if ok else (False, "MANIP_FAIL")

    async def _skill_return(self, skill: SkillCall) -> Tuple[bool, str]:
        pose = self.memory.get_start_pose()
        if pose is None:
            return False, "NO_START_POSE"
        result = await self.nav_client.navigate_to_pose(
            pose,
            timeout_s=float(skill.args.get("timeout_s", 20.0)),
            pause_event=self.pause_event,
        )
        return (True, "OK") if result.success else (False, result.reason)

    async def _skill_inspect(self, skill: SkillCall) -> Tuple[bool, str]:
        target = str(skill.args.get("object") or self._current_target or "object")
        instruction = str(skill.args.get("instruction", ""))
        ok = await self.manipulator.inspect(target, instruction=instruction, pause_event=self.pause_event)
        return (True, "OK") if ok else (False, "INSPECT_FAIL")

    async def _skill_fetch(self, skill: SkillCall) -> Tuple[bool, str]:
        target = str(skill.args.get("object") or self._current_target or "object")
        for child in [
            SkillCall(name="locate", args={"object": target}),
            SkillCall(name="navigate", args={"target": "object", "object": target}),
            SkillCall(name="pick", args={"object": target}),
            SkillCall(name="return", args={"target": "start"}),
        ]:
            ok, reason = await self._execute_once(child)
            if not ok:
                return False, f"FETCH_{reason}"
        return True, "OK"

    async def _skill_look_at(self, skill: SkillCall) -> Tuple[bool, str]:
        if self._look_at_controller is None:
            self._init_look_at_controller()
        if self._look_at_controller is None:
            return False, "LOOK_AT_UNAVAILABLE"

        target = str(skill.args.get("object", "")).strip().lower()
        if target in {"", "none", "null", "stop"}:
            await self._stop_look_at("stopped")
            return True, "STOPPED"

        overrides: Dict[str, Any] = dict(self.look_at_cfg)
        for key in (
            "max_rate_hz",
            "deadband_px",
            "timeout_sec",
            "grace_sec",
            "smoothing",
            "fallback_behavior",
            "kx",
            "ky",
            "max_rate_deg_s",
            "target_max_age_s",
        ):
            if key in skill.args and skill.args[key] is not None:
                overrides[key] = skill.args[key]

        status = self._look_at_controller.activate(target, overrides=overrides)
        self._look_at_status = status
        if self._look_at_task is None or self._look_at_task.done():
            self._look_at_task = asyncio.create_task(self._run_look_at_loop(), name="look_at_loop")
        logging.info(
            "look_at activated: object=%s max_rate_hz=%.1f deadband_px=%.1f timeout_sec=%.1f",
            target,
            self._look_at_controller.params.max_rate_hz,
            self._look_at_controller.params.deadband_px,
            self._look_at_controller.params.timeout_sec,
        )
        return True, "OK"

    async def _on_navigation_failure(self, reason: str) -> None:
        if reason == "POSE_UNCERTAIN":
            await self.slam_monitor.force_exploration("navigation_pose_uncertain")

    async def _run_exploration(self) -> None:
        try:
            recovered = await self.exploration.recover(self.nav_client, self.slam_monitor)
            if recovered:
                logging.info("Exploration recovered localization.")
            else:
                logging.warning("Exploration completed without relocalization.")
        finally:
            if self.slam_monitor.mode == SlamMode.LOCALIZATION:
                self.pause_event.set()
            self._exploration_task = None

    async def _run_look_at_loop(self) -> None:
        try:
            while True:
                if self._look_at_controller is None:
                    return
                status = self._look_at_controller.step()
                self._look_at_status = status

                now = time.time()
                if now - self._look_at_last_log_ts >= 1.0:
                    logging.info(
                        "look_at status: state=%s object=%s score=%.2f detections_ok=%s msg=%s",
                        status.state,
                        status.object_label,
                        status.target_score,
                        status.detections_ok,
                        status.message,
                    )
                    self._look_at_last_log_ts = now

                if status.object_label == "" and status.state in {"idle", "target_lost", "stopped"}:
                    return
                await asyncio.sleep(max(0.01, 1.0 / self._look_at_controller.params.max_rate_hz))
        except asyncio.CancelledError:
            raise
        finally:
            self._look_at_task = None

    async def _stop_look_at(self, reason: str) -> None:
        if self._look_at_controller is not None:
            self._look_at_status = self._look_at_controller.stop(reason=reason)
        if self._look_at_task is not None:
            task = self._look_at_task
            self._look_at_task = None
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def stop(self) -> None:
        await self._stop_look_at("stopped")
