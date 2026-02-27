from __future__ import annotations

import json
from pathlib import Path

from isaacsim import SimulationApp

from apps.isaacsim_runner.isaac_runner import _resolve_robot_placement_prim, _find_robot_prim_path

USD_PATHS = [
    Path(r"C:/Users/mango/project/isaac-aura/g1/g1_d455.usd"),
    Path(r"C:/Users/mango/project/isaac-aura/apps/isaac_ros2_bridge_bundle/robot_model/model_data/g1/g1_29dof_with_hand/g1_29dof_with_hand.usd"),
]


def main() -> None:
    sim = SimulationApp({"headless": True})
    out = []
    try:
        import omni.usd  # type: ignore

        ctx = omni.usd.get_context()
        for p in USD_PATHS:
            ok = ctx.open_stage(str(p))
            if not ok:
                out.append({"usd": str(p), "error": "open_stage failed"})
                continue
            for _ in range(120):
                sim.update()
            stage = ctx.get_stage()
            root = _find_robot_prim_path(stage)
            placement = ""
            if root:
                prim = _resolve_robot_placement_prim(stage, root)
                if prim.IsValid():
                    placement = prim.GetPath().pathString
            out.append({"usd": str(p), "articulation_root": root, "placement_prim": placement})
    finally:
        sim.close()

    path = Path(r"C:/Users/mango/project/isaac-aura/tmp/check_runner_placement.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
