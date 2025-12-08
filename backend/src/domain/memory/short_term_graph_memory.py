from collections import deque
from typing import List, Optional

from domain.perception.scene_graph_frame import SceneGraphFrame
from infrastructure.logging.pipeline_logger import PipelineLogger


class ShortTermGraphMemory:
    def __init__(self, max_frames: int = 60, logger: Optional[PipelineLogger] = None):
        self.buffer: deque[SceneGraphFrame] = deque(maxlen=max_frames)
        self.logger = logger

    def _log(self, event: str, frame_idx: Optional[int] = None, **payload):
        if self.logger is not None:
            self.logger.log(
                module="ShortTermMemory",
                event=event,
                frame_idx=frame_idx,
                **payload,
            )

    def push(self, sg_frame: SceneGraphFrame):
        self.buffer.append(sg_frame)
        self._log(
            event="stm_push",
            frame_idx=sg_frame.frame_idx,
            size=len(self.buffer),
            matched_brain="Hippocampus",
        )

    def get_sequence_for_grin(
        self,
        end_frame: int,
        horizon: int = 16,
        stride: int = 1,
    ) -> List[SceneGraphFrame]:
        candidates = [
            f
            for f in self.buffer
            if end_frame - horizon + 1 <= f.frame_idx <= end_frame
        ]
        candidates = sorted(candidates, key=lambda f: f.frame_idx)
        seq = candidates[::stride]
        if seq:
            self._log(
                event="stm_get_sequence",
                frame_idx=seq[-1].frame_idx,
                horizon=horizon,
                stride=stride,
                length=len(seq),
                matched_brain="Hippocampus",
            )
        return seq
