from __future__ import annotations

from typing import Protocol

from v2v_rt.types import GeneratedFrame


class FrameSink(Protocol):
    async def send_many(self, frames: list[GeneratedFrame]) -> None:
        """Publish generated frames to the caller-facing stream."""


class LoggingFrameSink:
    async def send_many(self, frames: list[GeneratedFrame]) -> None:
        for frame in frames:
            latency = frame.metadata.get("latency_ms", "?")
            print(
                f"produced stream={frame.stream_id} frame={frame.frame_id} "
                f"bytes={len(frame.encoded_payload)} latency_ms={latency}"
            )
