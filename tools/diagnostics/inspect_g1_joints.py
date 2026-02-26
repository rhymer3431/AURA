import json
from pathlib import Path

from isaacsim import SimulationApp


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output_path = root / "tmp" / "inspect_g1_joints_data.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "usd_path": "",
        "articulation_roots": [],
        "selected_root": "",
        "joint_count": 0,
        "joint_names": [],
        "joint_positions": [],
        "body29_joint_names": [],
        "body29_joint_positions": [],
        "error": None,
    }

    simulation_app = SimulationApp({"headless": True})
    try:
        import omni.usd  # type: ignore
        from isaacsim.core.api import SimulationContext  # type: ignore
        from omni.isaac.core.articulations import Articulation  # type: ignore
        from pxr import UsdPhysics  # type: ignore

        usd_path = (root / "g1" / "g1_d455.usd").resolve()
        result["usd_path"] = str(usd_path)
        context = omni.usd.get_context()
        context.open_stage(str(usd_path))
        for _ in range(120):
            simulation_app.update()

        stage = context.get_stage()
        roots = []
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                roots.append(prim.GetPath().pathString)
        result["articulation_roots"] = roots
        if not roots:
            result["error"] = "no articulation roots found"
            return

        selected = roots[0]
        for candidate in roots:
            if "g1" in candidate.lower():
                selected = candidate
                break
        result["selected_root"] = selected

        sim = SimulationContext(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0)
        sim.initialize_physics()
        sim.play()
        for _ in range(30):
            sim.step(render=False)

        robot = Articulation(selected)
        robot.initialize()

        result["joint_count"] = int(robot.num_dof)
        result["joint_names"] = list(robot.dof_names)
        joint_positions = [float(v) for v in robot.get_joint_positions().tolist()]
        result["joint_positions"] = joint_positions
        result["body29_joint_names"] = result["joint_names"][:29]
        result["body29_joint_positions"] = joint_positions[:29]

        sim.stop()
    except Exception as exc:  # pragma: no cover - runtime-dependent
        result["error"] = str(exc)
    finally:
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        simulation_app.close()


if __name__ == "__main__":
    main()
