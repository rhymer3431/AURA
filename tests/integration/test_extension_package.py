from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
EXT_ROOT = ROOT / "exts" / "isaac.aura.live_smoke"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(EXT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXT_ROOT))

from isaac.aura.live_smoke_ext import get_extension_contract


def test_extension_manifest_exists_and_declares_python_module() -> None:
    manifest_path = EXT_ROOT / "config" / "extension.toml"
    payload = tomllib.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["package"]["title"] == "Isaac Aura Live Smoke"
    assert payload["core"]["reloadable"] is True
    assert payload["python"]["module"][0]["name"] == "isaac.aura.live_smoke_ext"


def test_extension_contract_points_to_editor_smoke_entry() -> None:
    contract = get_extension_contract()

    assert contract["name"] == "isaac.aura.live_smoke"
    assert contract["entrypoint"] == "apps.editor_smoke_entry.run_extension_smoke"
