from pathlib import Path

import yaml


def load_config(path) -> dict:
    """Load a YAML config file into a Python dict."""
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML config: {config_path}\n{e}")

    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config format: expected dict, got {type(cfg)}")

    return cfg
