from __future__ import annotations

import argparse

from services.memory_service import MemoryService
from services.task_orchestrator import TaskOrchestrator


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Structured memory agent scaffold.")
    parser.add_argument("--memory-db-path", type=str, default="state/memory/memory.sqlite")
    parser.add_argument("--bootstrap-rule", type=str, default="find:apple:kitchen")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    memory_service = MemoryService(db_path=args.memory_db_path)
    memory_service.semantic_store.remember_rule(
        args.bootstrap_rule,
        "prioritize table-like surfaces",
        succeeded=True,
        metadata={"source": "bootstrap"},
    )
    orchestrator = TaskOrchestrator(memory_service)
    print(f"[MEMORY_AGENT] db={args.memory_db_path} state={orchestrator.snapshot()['state']}")
    return 0
