from __future__ import annotations

import unittest

from g1_play.tasking.aura_adapter import AuraTaskingAdapter


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


if __name__ == "__main__":
    unittest.main()
