from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class MemoryItem:
    key: str
    value: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryState:
    short_term: List[MemoryItem] = field(default_factory=list)
    long_term: List[MemoryItem] = field(default_factory=list)
