from __future__ import annotations

from typing import Protocol

from v2v_rt.types import FrameEnvelope


class FrameSource(Protocol):
    async def start(self) -> None:
        """Start accepting frames."""

    async def receive(self) -> FrameEnvelope:
        """Return the next frame for scheduling."""

    async def stop(self) -> None:
        """Stop accepting frames."""
