import cv2
import numpy as np
from typing import List, Optional, Tuple


class FrameVisualizer:
    def __init__(
        self,
        default_color: Tuple[int, int, int] = (0, 200, 255),
        focus_color: Tuple[int, int, int] = (0, 255, 140),
        text_color: Tuple[int, int, int] = (255, 255, 255),
        border_thickness: int = 2,
        alpha: float = 0.22,
        glow_thickness: int = 6,
    ):
        self.default_color = default_color
        self.focus_color = focus_color
        self.text_color = text_color
        self.border_thickness = border_thickness
        self.alpha = alpha
        self.glow_thickness = glow_thickness

    def draw(
        self,
        frame_bgr,
        nodes: List,
        focus_targets: Optional[List[str]] = None,
    ):
        h, w = frame_bgr.shape[:2]
        focus_targets = set(focus_targets or [])
        overlay = frame_bgr.copy()
        output = frame_bgr.copy()

        for n in nodes:
            x1, y1, x2, y2 = map(int, n.box)
            label = str(n.cls)
            tid = "" if n.track_id is None else f"#{n.track_id}"
            text = f"{label}{tid}"

            color = (
                self.focus_color
                if label in focus_targets
                else self.default_color
            )

            # Glow for focused objects
            if label in focus_targets:
                cv2.rectangle(output, (x1-2, y1-2), (x2+2, y2+2),
                              color, self.glow_thickness)

            # Fill soft overlay
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

            # Clean border
            cv2.rectangle(output, (x1, y1), (x2, y2),
                          color, self.border_thickness)

            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2
            )

            # Prevent label from going off-screen
            label_y = max(y1 - th - 10, 0)

            # Label rounded rectangle
            cv2.rectangle(
                overlay,
                (x1, label_y),
                (x1 + tw + 14, label_y + th + 8),
                color,
                -1,
            )
            cv2.putText(
                output,
                text,
                (x1 + 7, label_y + th),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                self.text_color,
                1,
                cv2.LINE_AA,
            )

        cv2.addWeighted(overlay, self.alpha,
                        output, 1 - self.alpha, 0, output)
        return output
