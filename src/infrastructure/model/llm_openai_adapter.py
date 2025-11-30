from domain.reason.entity.reasoning_unit import ReasoningUnit
from domain.reason.repository.llm_port import LlmPort


class OpenAILlmAdapter(LlmPort):
    def generate(self, unit: ReasoningUnit) -> ReasoningUnit:
        raise NotImplementedError("OpenAI LLM adapter not implemented yet.")
