# src/robotics/infrastructure/config/yaml_loader.py

from pathlib import Path
import yaml


def load_config(path) -> dict:
    """
    YAML 설정파일을 로드하여 Python dict로 반환한다.

    Parameters
    ----------
    path : str | Path
        YAML 파일 경로

    Returns
    -------
    dict
        파싱된 설정값 딕셔너리

    Raises
    ------
    FileNotFoundError
        지정한 경로에 파일이 없을 경우
    ValueError
        YAML 파싱 중 오류 발생 시
    """

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
