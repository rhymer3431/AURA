from abc import ABC, abstractmethod

from domain.reason.entity.reasoning_unit import ReasoningUnit


class LlmPort(ABC):
    @abstractmethod
    def generate(self, unit: ReasoningUnit) -> ReasoningUnit:
        raise NotImplementedError
