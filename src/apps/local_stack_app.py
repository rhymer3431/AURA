from __future__ import annotations

import argparse
import time

from apps.runtime_common import build_frame_source, frame_sample_to_batch, infer_demo_scene
from runtime.supervisor import Supervisor, SupervisorConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct local stack scaffold using the in-process bus.")
    parser.add_argument("--command", type=str, default="아까 봤던 사과를 찾아가")
    parser.add_argument("--memory-db-path", type=str, default="state/memory/memory.sqlite")
    parser.add_argument("--scene", type=str, default="")
    parser.add_argument("--detector-model-path", type=str, default="")
    parser.add_argument("--detector-device", type=str, default="")
    parser.add_argument("--frame-source", type=str, choices=("auto", "live", "synthetic"), default="auto")
    parser.add_argument("--strict-live", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    supervisor = Supervisor(
        config=SupervisorConfig(
            memory_db_path=args.memory_db_path,
            detector_model_path=args.detector_model_path,
            detector_device=args.detector_device,
        )
    )
    scene = str(args.scene).strip() or infer_demo_scene(args.command)
    frame_source = build_frame_source(
        mode=args.frame_source,
        scene=scene,
        source_name="local_stack",
        room_id="kitchen" if scene == "apple" else "",
        strict_live=bool(args.strict_live),
    )
    source_report = frame_source.start()
    if args.frame_source == "live" and source_report.status != "ready":
        print(f"[LOCAL_STACK] live frame source unavailable: {source_report.notice}")
        frame_source.close()
        return 1
    sample = frame_source.read()
    if sample is not None:
        supervisor.process_frame(frame_sample_to_batch(sample), publish=False)
    request = supervisor.submit_task(str(args.command))
    command = supervisor.step(now=time.time(), robot_pose=(0.0, 0.0, 0.0), publish=False)
    snapshot = supervisor.snapshot()
    print(
        "[LOCAL_STACK] "
        f"task_id={request.task_id} state={snapshot['state']} "
        f"detector={snapshot['detector_backend']} action={command.action_type if command else 'none'}"
    )
    frame_source.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
