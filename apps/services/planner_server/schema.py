from __future__ import annotations

import re
from typing import Dict, List, Tuple

ALLOWED_SKILLS = {"locate", "navigate", "pick", "return", "inspect", "fetch", "look_at"}


def _infer_object_name(command: str) -> str:
    text = command.lower()
    mapping = {
        "사과": "apple",
        "apple": "apple",
        "병": "bottle",
        "bottle": "bottle",
        "컵": "cup",
        "cup": "cup",
        "책": "book",
        "book": "book",
    }
    for key, value in mapping.items():
        if key in text:
            return value
    return "apple"


def _intent_flags(command: str) -> Tuple[bool, bool, bool, bool, bool, bool]:
    text = command.lower()
    wants_pick = any(token in text for token in ["pick", "grasp", "집", "잡", "들"])
    wants_return = any(token in text for token in ["return", "back", "다시", "와", "bring", "fetch", "가져"])
    wants_inspect = any(token in text for token in ["inspect", "check", "확인", "검사"])
    wants_nav_only = any(token in text for token in ["navigate", "go to", "이동"])
    wants_look = any(token in text for token in ["look at", "look_at", "track", "gaze"])
    wants_stop_look = any(token in text for token in ["stop look", "stop tracking", "look_at stop", "cancel look"])
    return wants_pick, wants_return, wants_inspect, wants_nav_only, wants_look, wants_stop_look


def _infer_look_target(command: str, fallback: str) -> str:
    text = command.strip().lower()
    m = re.search(r"object\s*=\s*['\"]?([a-z0-9_\-]+)", text)
    if m:
        return m.group(1)
    m = re.search(r"look[_\s]*at\s+([a-z0-9_\-]+)", text)
    if m:
        return m.group(1)
    m = re.search(r"track\s+([a-z0-9_\-]+)", text)
    if m:
        return m.group(1)
    return fallback


def generate_stub_plan(user_command: str, world_state: Dict) -> Dict:
    target = _infer_object_name(user_command)
    wants_pick, wants_return, wants_inspect, wants_nav_only, wants_look, wants_stop_look = _intent_flags(
        user_command
    )
    look_target = _infer_look_target(user_command, fallback=target)

    if wants_stop_look:
        return {
            "plan": [
                {
                    "skill": "look_at",
                    "args": {"object": ""},
                    "success_criteria": "look_at_stopped",
                    "retry_policy": {"max_retries": 0, "backoff_s": 0.0},
                }
            ],
            "notes": "look_at stop plan",
        }

    if wants_look and not any([wants_pick, wants_return, wants_inspect, wants_nav_only]):
        return {
            "plan": [
                {
                    "skill": "look_at",
                    "args": {"object": look_target, "timeout_sec": 3.0},
                    "success_criteria": "camera_tracks_target",
                    "retry_policy": {"max_retries": 0, "backoff_s": 0.0},
                }
            ],
            "notes": "look_at plan",
        }

    # Default behavior should remain task-complete even when command encoding/tokenization fails.
    if not any([wants_pick, wants_return, wants_inspect, wants_nav_only]):
        wants_pick = True
        wants_return = True
        wants_inspect = True

    if wants_nav_only and not wants_pick:
        return {
            "plan": [
                {
                    "skill": "navigate",
                    "args": {"target": "object", "object": target},
                    "success_criteria": "reach_target_pose",
                    "retry_policy": {"max_retries": 1, "backoff_s": 1.0},
                }
            ],
            "notes": "navigation-only plan",
        }

    if "fetch" in user_command.lower():
        return {
            "plan": [
                {
                    "skill": "fetch",
                    "args": {"object": target},
                    "success_criteria": "object_delivered_to_start_pose",
                    "retry_policy": {"max_retries": 1, "backoff_s": 1.0},
                },
                {
                    "skill": "inspect",
                    "args": {"object": target, "instruction": "Confirm fetched object state"},
                    "success_criteria": "inspection_done",
                    "retry_policy": {"max_retries": 0, "backoff_s": 0.0},
                },
            ],
            "notes": "fetch macro plan",
        }

    plan: List[Dict] = [
        {
            "skill": "locate",
            "args": {"object": target},
            "success_criteria": "target_object_is_localized",
            "retry_policy": {"max_retries": 1, "backoff_s": 1.0},
        }
    ]

    if wants_pick or wants_return:
        plan.append(
            {
                "skill": "navigate",
                "args": {"target": "object", "object": target},
                "success_criteria": "robot_near_target",
                "retry_policy": {"max_retries": 1, "backoff_s": 1.0},
            }
        )
        plan.append(
            {
                "skill": "pick",
                "args": {"object": target, "instruction": f"Pick up the {target}"},
                "success_criteria": "object_grasped",
                "retry_policy": {"max_retries": 1, "backoff_s": 1.0},
            }
        )

    if wants_return:
        plan.append(
            {
                "skill": "return",
                "args": {"target": "start"},
                "success_criteria": "robot_back_to_start_pose",
                "retry_policy": {"max_retries": 1, "backoff_s": 1.0},
            }
        )

    if wants_inspect or wants_return:
        plan.append(
            {
                "skill": "inspect",
                "args": {"object": target, "instruction": f"Inspect held {target}"},
                "success_criteria": "inspection_completed",
                "retry_policy": {"max_retries": 0, "backoff_s": 0.0},
            }
        )

    if len(plan) == 1:
        plan.extend(
            [
                {
                    "skill": "navigate",
                    "args": {"target": "object", "object": target},
                    "success_criteria": "robot_near_target",
                    "retry_policy": {"max_retries": 1, "backoff_s": 1.0},
                },
                {
                    "skill": "inspect",
                    "args": {"object": target},
                    "success_criteria": "inspection_completed",
                    "retry_policy": {"max_retries": 0, "backoff_s": 0.0},
                },
            ]
        )

    return {"plan": plan, "notes": "mock planner output (JSON only)"}


def validate_plan_payload(payload: Dict) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Planner output must be a JSON object.")
    if "plan" not in payload or not isinstance(payload["plan"], list):
        raise ValueError("Planner output must include 'plan' list.")
    if "notes" in payload and not isinstance(payload["notes"], str):
        raise ValueError("Planner output field 'notes' must be a string.")

    for i, step in enumerate(payload["plan"]):
        if not isinstance(step, dict):
            raise ValueError(f"Plan step {i} must be an object.")
        skill = step.get("skill")
        if not isinstance(skill, str) or skill.strip().lower() not in ALLOWED_SKILLS:
            raise ValueError(f"Plan step {i} has invalid skill: {skill!r}")

        args = step.get("args", {})
        if args is not None and not isinstance(args, dict):
            raise ValueError(f"Plan step {i} 'args' must be object.")

        if skill.strip().lower() == "look_at":
            object_label = args.get("object", "")
            if not isinstance(object_label, str):
                raise ValueError(f"Plan step {i} look_at args.object must be a string.")
            for key in ("max_rate_hz", "deadband_px", "timeout_sec", "grace_sec", "smoothing", "kx", "ky"):
                if key in args and not isinstance(args[key], (int, float)):
                    raise ValueError(f"Plan step {i} look_at args.{key} must be a number.")
            if "fallback_behavior" in args and not isinstance(args["fallback_behavior"], str):
                raise ValueError(f"Plan step {i} look_at args.fallback_behavior must be a string.")

        retry = step.get("retry_policy", {})
        if retry is not None:
            if not isinstance(retry, dict):
                raise ValueError(f"Plan step {i} 'retry_policy' must be object.")
            max_retries = retry.get("max_retries", 1)
            backoff_s = retry.get("backoff_s", 1.0)
            if not isinstance(max_retries, int) or max_retries < 0:
                raise ValueError(f"Plan step {i} has invalid max_retries.")
            if not isinstance(backoff_s, (int, float)) or backoff_s < 0:
                raise ValueError(f"Plan step {i} has invalid backoff_s.")
