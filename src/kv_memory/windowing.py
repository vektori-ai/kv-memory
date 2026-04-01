from __future__ import annotations

from typing import Iterable

from .types import ConversationTurn, MemoryUnit


def build_turn_windows(
    turns: Iterable[ConversationTurn],
    *,
    window_size: int,
    stride: int,
) -> list[MemoryUnit]:
    turns_list = list(turns)
    if not turns_list:
        return []

    units: list[MemoryUnit] = []
    index = 0
    unit_num = 0

    while index < len(turns_list):
        end_index = min(index + window_size, len(turns_list))
        window = turns_list[index:end_index]
        if not window:
            break

        text = "\n".join(f"{turn.role}: {turn.content}" for turn in window)
        timestamp = max(turn.timestamp for turn in window)
        importance = sum(turn.importance for turn in window) / len(window)

        unit = MemoryUnit(
            unit_id=f"unit-{unit_num}",
            turn_start=index,
            turn_end=end_index - 1,
            text=text,
            timestamp=timestamp,
            importance=importance,
            metadata={"size": len(window)},
        )
        units.append(unit)

        unit_num += 1
        if end_index == len(turns_list):
            break
        index += stride

    return units
