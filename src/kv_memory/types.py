from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class ConversationTurn:
    role: str
    content: str
    timestamp: float
    importance: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryUnit:
    unit_id: str
    turn_start: int
    turn_end: int
    text: str
    timestamp: float
    importance: float
    embedding: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
