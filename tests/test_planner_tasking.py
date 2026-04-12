from __future__ import annotations

from types import SimpleNamespace
import unittest

from systems.planner.api.runtime import AuraTaskingAdapter


class _FakeRuntimeController:
    def __init__(self, snapshot: dict[str, object]) -> None:
        self.snapshot = dict(snapshot)
        self.navigation_calls: list[dict[str, object]] = []
        self.report_messages: list[str] = []

    def start_navigation_instruction(self, instruction: str, language: str = "en") -> dict[str, object]:
        self.navigation_calls.append({"instruction": instruction, "language": language})
        return {
            "instruction": instruction,
            "language": language,
            "command_revision": len(self.navigation_calls),
            "session_id": "test-session",
            "session_reset_required": True,
        }

    def navigation_snapshot(self, *, origin_pose: dict[str, object] | None = None) -> dict[str, object]:
        del origin_pose
        return dict(self.snapshot)

    def check_binary_question(self, question: str) -> str:
        del question
        return "true"

    def set_last_report(self, message: str) -> None:
        self.report_messages.append(str(message))


class PlannerTaskingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = AuraTaskingAdapter(
            completion=None,
            model="test-model",
            timeout=1.0,
        )

    def test_check_state_command_expands_to_navigate_inspect_return_report(self) -> None:
        task_frame = self.adapter.plan_task_frame("check whether the tv is off and come back")
        self.assertEqual(task_frame["intent"], "check_state")
        self.assertEqual(task_frame["target"]["object"], "tv")
        self.assertEqual(task_frame["query"]["attribute"], "power_state")
        self.assertEqual(task_frame["query"]["expected_value"], "off")
        self.assertTrue(task_frame["constraints"]["return_after_check"])

        subgoals = self.adapter.initialize_subgoals(task_frame)
        self.assertEqual([item["type"] for item in subgoals], ["navigate", "inspect", "return", "report"])

    def test_navigate_and_return_command_expands_to_navigate_return_report(self) -> None:
        task_frame = self.adapter.plan_task_frame("go to the purple box cart and come back")
        self.assertEqual(task_frame["intent"], "navigate_to_object")
        self.assertEqual(task_frame["target"]["object"], "purple_box_cart")
        self.assertTrue(task_frame["constraints"]["return_after_check"])

        subgoals = self.adapter.initialize_subgoals(task_frame)
        self.assertEqual([item["type"] for item in subgoals], ["navigate", "return", "report"])

    def test_navigation_subgoal_starts_navigation_pipeline(self) -> None:
        task_frame = self.adapter.plan_task_frame("go to the purple box cart and come back")
        subgoals = self.adapter.initialize_subgoals(task_frame)
        controller = _FakeRuntimeController(
            {
                "planner_target_mode": "none",
                "has_goal": False,
                "goal_world_xy": None,
                "goal_pixel_xy": None,
                "system2_status": "move",
                "system2_decision_mode": "move",
                "system2_text": "keep moving",
                "action_override_mode": None,
                "locomotion_command": [0.2, 0.0, 0.0],
                "state_label": "tracking",
                "goal_reached": False,
                "return_pose_distance": None,
                "return_pose_reached": False,
            }
        )
        task_state = SimpleNamespace(task_frame=task_frame, subgoals=subgoals, origin_pose=None)

        event = self.adapter.step(subgoals, {"controller": controller, "task_state": task_state})

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event["type"], "navigate")
        self.assertEqual(event["status"], "running")
        self.assertEqual(subgoals[0]["status"], "running")
        self.assertEqual(
            controller.navigation_calls,
            [{"instruction": "Go to the purple box cart.", "language": "en"}],
        )

    def test_stop_result_advances_from_navigation_to_next_subgoal(self) -> None:
        task_frame = self.adapter.plan_task_frame("check whether the tv is off and come back")
        subgoals = self.adapter.initialize_subgoals(task_frame)
        controller = _FakeRuntimeController(
            {
                "planner_target_mode": "none",
                "has_goal": False,
                "goal_world_xy": None,
                "goal_pixel_xy": None,
                "system2_status": "STOP",
                "system2_decision_mode": "stop",
                "system2_text": "STOP",
                "action_override_mode": None,
                "locomotion_command": [0.0, 0.0, 0.0],
                "state_label": "done",
                "goal_reached": False,
                "return_pose_distance": None,
                "return_pose_reached": False,
            }
        )
        task_state = SimpleNamespace(task_frame=task_frame, subgoals=subgoals, origin_pose=None)

        event = self.adapter.step(subgoals, {"controller": controller, "task_state": task_state})
        current_subgoal = self.adapter.orchestrator.state_machine.current_subgoal(subgoals)

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event["type"], "navigate")
        self.assertEqual(event["status"], "succeeded")
        self.assertEqual(subgoals[0]["status"], "succeeded")
        self.assertIsNotNone(current_subgoal)
        assert current_subgoal is not None
        self.assertEqual(current_subgoal["type"], "inspect")
        self.assertEqual(
            controller.navigation_calls,
            [{"instruction": "Go to the TV.", "language": "en"}],
        )


if __name__ == "__main__":
    unittest.main()
