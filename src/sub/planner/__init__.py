from __future__ import annotations

from pathlib import Path
from typing import Final

from ..shared import load_module, root_path


NAME: Final[str] = "Planner Subsystem"
DESCRIPTION: Final[str] = "Task-frame planning, planner endpoint integration, subgoal expansion, and reporting."

MODULES: Final[dict[str, str]] = {
    "adapter": "g1_play.tasking.aura_adapter",
    "service": "g1_play.tasking.planner_service",
    "llm_client": "g1_play.tasking.llm_client",
    "normalizer": "g1_play.tasking.normalizer",
    "ontology": "g1_play.tasking.ontology",
    "orchestration": "g1_play.tasking.orchestration",
    "reporting": "g1_play.tasking.reporting",
    "schemas": "g1_play.tasking.schemas",
    "task_frames": "g1_play.tasking.task_frames",
    "validator": "g1_play.tasking.validator",
}

ENTRYPOINTS: Final[dict[str, Path]] = {
    "planner_launcher": root_path("scripts", "serve_planner_qwen3_nothink.ps1"),
}

PUBLIC_APIS: Final[tuple[str, ...]] = (
    "g1_play.tasking.aura_adapter.AuraTaskingAdapter",
    "g1_play.tasking.planner_service.PlannerService",
    "g1_play.tasking.orchestration.SubgoalOrchestrator",
    "g1_play.tasking.reporting.render_report_message",
)


def load(alias: str):
    return load_module(MODULES[alias])
