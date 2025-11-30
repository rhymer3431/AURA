from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ReasoningUnit:
    prompt: str
    context: Dict[str, Any]
    output: str | None = None
