from domain.reason.entity.reasoning_unit import ReasoningUnit
from domain.reason.repository.llm_port import LlmPort


class GrinAdapter(LlmPort):
    def generate(self, unit: ReasoningUnit) -> ReasoningUnit:
        raise NotImplementedError("GRIN adapter not implemented yet.")
