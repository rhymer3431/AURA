from __future__ import annotations

from importlib import import_module
from pathlib import Path
from types import ModuleType


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def root_path(*parts: str) -> Path:
    return repo_root().joinpath(*parts)


def load_module(module_name: str) -> ModuleType:
    return import_module(module_name)
