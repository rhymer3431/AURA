from __future__ import annotations

import unittest

from systems.control.api.runtime_args import build_arg_parser as build_control_arg_parser
from systems.control.api.runtime_controller import InternVlaNavDpController
from systems.planner.service import PlannerSystem, build_arg_parser as build_planner_arg_parser


class RuntimePlannerStatusTests(unittest.TestCase):
    def test_control_runtime_parser_supports_viewer_publish_toggle(self) -> None:
        parser = build_control_arg_parser()

        default_args = parser.parse_args([])
        disabled_args = parser.parse_args(["--no-viewer-publish"])
        enabled_args = parser.parse_args(["--viewer-publish"])

        self.assertTrue(default_args.viewer_publish)
        self.assertFalse(disabled_args.viewer_publish)
        self.assertTrue(enabled_args.viewer_publish)

    def test_planner_system_owns_task_and_subgoal_status(self) -> None:
        planner_args = build_planner_arg_parser().parse_args(["--planner-model-base-url", ""])

        class _NavStub:
            def __init__(self):
                self.cancel_calls = 0
                self.commands = []

            def cancel(self):
                self.cancel_calls += 1
                return {"ok": True}

            def command(self, instruction: str, language: str = "en", *, task_id: str | None = None):
                self.commands.append({"instruction": instruction, "language": language, "task_id": task_id})
                return {"ok": True}

        planner = PlannerSystem(planner_args)
        planner._navigation = _NavStub()  # type: ignore[attr-defined]
        response = planner.submit_task("check whether the tv is off and come back", "en", task_id="planner-fixed")

        self.assertEqual(response["task_id"], "planner-fixed")
        self.assertEqual(response["task_frame"]["intent"], "check_state")
        self.assertEqual(response["task_status"], "running")
        self.assertEqual(response["current_subgoal"]["type"], "navigate")
        self.assertEqual([item["type"] for item in response["subgoals"]], ["navigate", "inspect", "return", "report"])
        self.assertEqual(planner._navigation.cancel_calls, 0)  # type: ignore[attr-defined]
        self.assertEqual(planner._navigation.commands[0]["instruction"], "Go to the TV.")  # type: ignore[attr-defined]
        self.assertEqual(planner._navigation.commands[0]["language"], "en")  # type: ignore[attr-defined]
        self.assertEqual(planner._navigation.commands[0]["task_id"], "planner-fixed")  # type: ignore[attr-defined]
        self.assertEqual(len(planner._navigation.commands), 1)  # type: ignore[attr-defined]

    def test_planner_system_creates_navigation_task_for_go_to_purple_box(self) -> None:
        planner_args = build_planner_arg_parser().parse_args(["--planner-model-base-url", ""])

        class _NavStub:
            def __init__(self):
                self.cancel_calls = 0
                self.commands = []

            def cancel(self):
                self.cancel_calls += 1
                return {"ok": True}

            def command(self, instruction: str, language: str = "en", *, task_id: str | None = None):
                self.commands.append({"instruction": instruction, "language": language, "task_id": task_id})
                return {"ok": True}

        planner = PlannerSystem(planner_args)
        planner._navigation = _NavStub()  # type: ignore[attr-defined]
        response = planner.submit_task("go to purple box", "en", task_id="planner-purple-box")

        self.assertEqual(response["task_id"], "planner-purple-box")
        self.assertEqual(response["task_frame"]["intent"], "navigate_to_object")
        self.assertEqual(response["task_frame"]["target"]["object"], "purple_box_cart")
        self.assertEqual(response["task_status"], "running")
        self.assertEqual(response["current_subgoal"]["type"], "navigate")
        self.assertEqual([item["type"] for item in response["subgoals"]], ["navigate", "report"])
        self.assertEqual(planner._navigation.cancel_calls, 0)  # type: ignore[attr-defined]
        self.assertEqual(planner._navigation.commands[0]["instruction"], "Go to the purple box cart.")  # type: ignore[attr-defined]
        self.assertEqual(planner._navigation.commands[0]["language"], "en")  # type: ignore[attr-defined]
        self.assertEqual(planner._navigation.commands[0]["task_id"], "planner-purple-box")  # type: ignore[attr-defined]
        self.assertEqual(len(planner._navigation.commands), 1)  # type: ignore[attr-defined]

    def test_control_runtime_status_is_navigation_only(self) -> None:
        args = build_control_arg_parser().parse_args([])
        controller = InternVlaNavDpController(args)
        try:
            status = controller.runtime_status()
        finally:
            controller.shutdown()

        self.assertEqual(status["executionMode"], "NAV")
        self.assertIn("routeState", status)
        self.assertIn("locomotion_command", status)
        self.assertNotIn("task_status", status)


if __name__ == "__main__":
    unittest.main()
