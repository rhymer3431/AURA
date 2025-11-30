from pathlib import Path
import cv2


class FileWriter:
    def __init__(self, path: str, fps: int = 20, frame_size: tuple[int, int] | None = None):
        self.path = Path(path)
        self.fps = fps
        self.frame_size = frame_size
        self.writer = None

    def open(self, frame_size: tuple[int, int]):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, frame_size)

    def write(self, frame):
        if self.writer is None:
            if self.frame_size is None:
                raise RuntimeError("Call open() with frame size before writing.")
            self.open(self.frame_size)
        self.writer.write(frame)

    def close(self):
        if self.writer:
            self.writer.release()
