from domain.reason.entity.reasoning_unit import ReasoningUnit
from domain.reason.repository.llm_port import LlmPort


class LlmRepository(LlmPort):
    def __init__(self, adapter: LlmPort):
        self.adapter = adapter

    def generate(self, unit: ReasoningUnit) -> ReasoningUnit:
        return self.adapter.generate(unit)
