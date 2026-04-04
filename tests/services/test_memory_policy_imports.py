from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_services_imports_ignore_shadowed_config_module(tmp_path: Path) -> None:
    shadow_dir = tmp_path / "shadow"
    shadow_dir.mkdir(parents=True, exist_ok=True)
    (shadow_dir / "config.py").write_text("raise RuntimeError('shadowed config import used')\n", encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(shadow_dir), str(src_dir), env.get("PYTHONPATH", "")]).strip(os.pathsep)

    command = [
        sys.executable,
        "-c",
        (
            "import services.task_orchestrator as mod; "
            "assert mod.MemoryPolicyConfig.__module__ == 'aura_config.memory_policy_config'"
        ),
    ]
    result = subprocess.run(
        command,
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
