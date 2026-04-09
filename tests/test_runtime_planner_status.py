from __future__ import annotations

import unittest

from g1_play.args import build_arg_parser
from g1_play.navdp_runtime import InternVlaNavDpController


class RuntimePlannerStatusTests(unittest.TestCase):
    def test_command_status_exposes_planner_fields(self) -> None:
        args = build_arg_parser().parse_args([])
        args.planner_base_url = ""
        controller = InternVlaNavDpController(args)
        try:
            response = controller.apply_runtime_command("check whether the tv is off and come back", "en")
            status = controller.command_api_status()
        finally:
            controller.shutdown()

        self.assertIn("task_id", response)
        self.assertEqual(response["task_frame"]["intent"], "check_state")
        self.assertEqual(status["task_status"], "running")
        self.assertEqual(status["task_frame"]["intent"], "check_state")
        self.assertEqual(status["current_subgoal"]["type"], "navigate")
        self.assertEqual([item["type"] for item in status["subgoals"]], ["navigate", "inspect", "return", "report"])


if __name__ == "__main__":
    unittest.main()
