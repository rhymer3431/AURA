# src/infrastructure/memory/pipeline_logger_adapter.py
from src.domain.memory.logger_port import LoggerPort
from src.infrastructure.logging.pipeline_logger import PipelineLogger


class LoggerAdapter(LoggerPort):
    def __init__(self, logger: PipelineLogger | None):
        self.logger = logger

    def log(self, *args, **kwargs):
        if self.logger:
            self.logger.log(*args, **kwargs)
