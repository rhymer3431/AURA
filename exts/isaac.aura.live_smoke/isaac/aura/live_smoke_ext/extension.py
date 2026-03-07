from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from typing import Any


try:
    import omni.ext
except Exception:  # noqa: BLE001
    omni = None
    _BaseExt = object
else:
    _BaseExt = omni.ext.IExt


EXTENSION_NAME = "isaac.aura.live_smoke"
ACTION_NAME = "run_smoke"
MENU_PATH = "Isaac Aura"
MENU_LABEL = "Run Live Smoke"
AUTO_RUN_ENV = "ISAAC_AURA_LIVE_SMOKE_AUTO_RUN"
AUTO_RUN_ARGS_ENV = "ISAAC_AURA_LIVE_SMOKE_ARGS"


def get_extension_contract() -> dict[str, str]:
    return {
        "name": EXTENSION_NAME,
        "action": ACTION_NAME,
        "menu_path": MENU_PATH,
        "menu_label": MENU_LABEL,
        "entrypoint": "apps.editor_smoke_entry.run_extension_smoke",
    }


class LiveSmokeExtension(_BaseExt):
    def __init__(self) -> None:
        super().__init__()
        self._added_sys_path = False
        self._menu_items = []
        self._action_registry = None
        self._ext_id = ""

    def on_startup(self, ext_id: str) -> None:
        self._ext_id = str(ext_id)
        self._ensure_repo_src_on_path()
        self._register_action()
        self._register_menu()
        if os.environ.get(AUTO_RUN_ENV, "").strip() == "1":
            self.run_smoke()

    def on_shutdown(self) -> None:
        self._unregister_menu()
        self._unregister_action()
        self._remove_repo_src_from_path()

    def run_smoke(self, argv: list[str] | None = None) -> int:
        from apps.editor_smoke_entry import run_extension_smoke

        resolved_argv = list(argv or self._auto_run_argv())
        return run_extension_smoke(argv=resolved_argv)

    def _auto_run_argv(self) -> list[str]:
        raw = os.environ.get(AUTO_RUN_ARGS_ENV, "").strip()
        if raw == "":
            return []
        return shlex.split(raw, posix=os.name != "nt")

    def _ensure_repo_src_on_path(self) -> None:
        repo_root = Path(__file__).resolve().parents[5]
        src_root = repo_root / "src"
        src_str = str(src_root)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)
            self._added_sys_path = True

    def _remove_repo_src_from_path(self) -> None:
        if not self._added_sys_path:
            return
        repo_root = Path(__file__).resolve().parents[5]
        src_root = str(repo_root / "src")
        try:
            sys.path.remove(src_root)
        except ValueError:
            pass
        self._added_sys_path = False

    def _register_action(self) -> None:
        try:
            import omni.kit.actions.core
        except Exception:  # noqa: BLE001
            return
        try:
            self._action_registry = omni.kit.actions.core.get_action_registry()
            self._action_registry.register_action(
                EXTENSION_NAME,
                ACTION_NAME,
                self.run_smoke,
                display_name=MENU_LABEL,
                description="Run Isaac Aura live smoke diagnostics inside the current Isaac Sim Full App session.",
            )
        except Exception:  # noqa: BLE001
            self._action_registry = None

    def _unregister_action(self) -> None:
        if self._action_registry is None:
            return
        try:
            self._action_registry.deregister_action(EXTENSION_NAME, ACTION_NAME)
        except Exception:  # noqa: BLE001
            pass
        self._action_registry = None

    def _register_menu(self) -> None:
        try:
            from omni.kit.menu.utils import MenuItemDescription, add_menu_items
        except Exception:  # noqa: BLE001
            return
        try:
            self._menu_items = [
                MenuItemDescription(
                    name=MENU_LABEL,
                    onclick_action=(EXTENSION_NAME, ACTION_NAME),
                )
            ]
            add_menu_items(self._menu_items, MENU_PATH)
        except Exception:  # noqa: BLE001
            self._menu_items = []

    def _unregister_menu(self) -> None:
        if not self._menu_items:
            return
        try:
            from omni.kit.menu.utils import remove_menu_items
        except Exception:  # noqa: BLE001
            self._menu_items = []
            return
        try:
            remove_menu_items(self._menu_items, MENU_PATH)
        except Exception:  # noqa: BLE001
            pass
        self._menu_items = []
