from typing import Optional


class LoggerPort:
    """Port interface for logging from the memory domain."""

    def log(
        self,
        module: str,
        event: str,
        frame_idx: Optional[int] = None,
        level: str = "INFO",
        **payload,
    ):
        raise NotImplementedError
