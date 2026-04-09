from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEMS_ROOT = REPO_ROOT / "src" / "systems"
SUBSYSTEMS = {"navigation", "inference", "world_state", "planner", "control"}
LAYERS = {"api", "application", "domain", "infrastructure"}


def _module_parts(path: Path) -> tuple[str, ...]:
    rel = path.relative_to(REPO_ROOT / "src")
    return rel.with_suffix("").parts


def _iter_python_files() -> list[Path]:
    return [
        path
        for path in SYSTEMS_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    ]


def _import_targets(module: ast.AST) -> list[str]:
    targets: list[str] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            targets.append(node.module)
    return targets


def test_five_subsystems_exist_with_expected_bins() -> None:
    for subsystem in SUBSYSTEMS:
        assert (SYSTEMS_ROOT / subsystem).is_dir()
    assert (SYSTEMS_ROOT / "control" / "bin" / "run_sim_g1_internvla_navdp_windows.bat").is_file()
    assert (SYSTEMS_ROOT / "navigation" / "bin" / "run_navdp_server_windows.bat").is_file()
    assert (SYSTEMS_ROOT / "inference" / "bin" / "run_internvla_nav_server_windows.bat").is_file()


def test_legacy_runtime_packages_are_not_imported_from_systems_tree() -> None:
    for path in _iter_python_files():
        text = path.read_text(encoding="utf-8")
        assert "g1_play" not in text, path
        assert "from navdp " not in text, path
        assert "from navdp." not in text, path


def test_cross_subsystem_imports_only_use_api_or_shared_contracts() -> None:
    for path in _iter_python_files():
        parts = _module_parts(path)
        if len(parts) < 3 or parts[0] != "systems":
            continue
        current_subsystem = parts[1]
        current_layer = parts[2] if len(parts) > 2 else ""
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for target in _import_targets(tree):
            if not target.startswith("systems."):
                continue
            target_parts = tuple(target.split("."))
            if len(target_parts) < 3:
                continue
            target_subsystem = target_parts[1]
            target_layer = target_parts[2]
            if target_subsystem == "shared":
                assert target_parts[:3] == ("systems", "shared", "contracts"), (path, target)
                continue
            if target_subsystem != current_subsystem:
                assert target_layer == "api", (path, target)
            if current_layer == "domain" and target_subsystem == current_subsystem:
                assert target_layer == "domain", (path, target)
