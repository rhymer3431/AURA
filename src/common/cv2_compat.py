from __future__ import annotations

import io

import numpy as np
from PIL import Image

try:
    import cv2 as cv2  # type: ignore[no-redef]
except Exception:  # noqa: BLE001
    class _Cv2Compat:
        COLOR_RGB2BGR = 0
        COLOR_BGR2RGB = 1
        INTER_NEAREST = 0
        INTER_AREA = 1
        INTER_CUBIC = 2

        @staticmethod
        def cvtColor(image: np.ndarray, code: int) -> np.ndarray:
            if code not in (_Cv2Compat.COLOR_RGB2BGR, _Cv2Compat.COLOR_BGR2RGB):
                raise ValueError(f"Unsupported cvtColor code: {code}")
            return np.asarray(image)[..., ::-1].copy()

        @staticmethod
        def imencode(ext: str, image: np.ndarray) -> tuple[bool, np.ndarray]:
            suffix = str(ext).lower()
            fmt = "JPEG" if suffix in {".jpg", ".jpeg"} else "PNG"
            buffer = io.BytesIO()
            Image.fromarray(np.asarray(image)).save(buffer, format=fmt)
            return True, np.frombuffer(buffer.getvalue(), dtype=np.uint8)

        @staticmethod
        def resize(
            image: np.ndarray,
            dsize: tuple[int, int],
            fx: float | None = None,
            fy: float | None = None,
            interpolation: int = 0,
        ) -> np.ndarray:
            array = np.asarray(image)
            if dsize == (-1, -1):
                if fx is None or fy is None:
                    raise ValueError("fx/fy are required when dsize is (-1, -1)")
                width = max(int(round(array.shape[1] * float(fx))), 1)
                height = max(int(round(array.shape[0] * float(fy))), 1)
            else:
                width, height = int(dsize[0]), int(dsize[1])
            resample = Image.NEAREST
            if interpolation == _Cv2Compat.INTER_CUBIC:
                resample = Image.BICUBIC
            elif interpolation == _Cv2Compat.INTER_AREA:
                resample = Image.BILINEAR
            pil_image = Image.fromarray(array)
            return np.asarray(pil_image.resize((width, height), resample=resample))

    cv2 = _Cv2Compat()
