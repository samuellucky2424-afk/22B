from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


def now_ns() -> int:
    return time.perf_counter_ns()


@dataclass(slots=True)
class FrameEnvelope:
    stream_id: str
    frame_id: int
    received_ns: int
    deadline_ns: int
    payload: Any
    width: int
    height: int
    fps_hint: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_late(self) -> bool:
        return now_ns() > self.deadline_ns

    @property
    def age_ms(self) -> float:
        return (now_ns() - self.received_ns) / 1_000_000.0


@dataclass(slots=True)
class BatchEnvelope:
    frames: list[FrameEnvelope]
    profile_name: str
    deadline_ns: int

    @property
    def size(self) -> int:
        return len(self.frames)


@dataclass(slots=True)
class GeneratedFrame:
    stream_id: str
    frame_id: int
    encoded_payload: bytes
    produced_ns: int
    metadata: dict[str, Any] = field(default_factory=dict)
