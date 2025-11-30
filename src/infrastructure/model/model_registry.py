from pathlib import Path
from typing import Optional

from ultralytics import YOLOWorld

from config.settings import DEFAULT_YOLO_WEIGHT, DEVICE

_yolo_model = None
_yolo_model_path: Optional[str] = None


def get_yolo(weight_path: Optional[str | Path] = None):
    """
    Load a YOLO-World model lazily and reuse the instance for subsequent calls.
    """
    global _yolo_model, _yolo_model_path
    path = Path(weight_path or DEFAULT_YOLO_WEIGHT)
    if _yolo_model is None or _yolo_model_path != str(path):
        _yolo_model = YOLOWorld(str(path)).to(DEVICE).eval()
        _yolo_model_path = str(path)
    return _yolo_model
