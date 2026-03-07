from __future__ import annotations

import argparse
import time

from runtime.supervisor import Supervisor, SupervisorConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct local stack scaffold using the in-process bus.")
    parser.add_argument("--command", type=str, default="아까 봤던 사과를 찾아가")
    parser.add_argument("--memory-db-path", type=str, default="state/memory/memory.sqlite")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    supervisor = Supervisor(config=SupervisorConfig(memory_db_path=args.memory_db_path))
    request = supervisor.submit_task(str(args.command))
    command = supervisor.step(now=time.time(), robot_pose=(0.0, 0.0, 0.0))
    print(f"[LOCAL_STACK] task_id={request.task_id} state={supervisor.snapshot()['state']} action={command.action_type if command else 'none'}")
    return 0
