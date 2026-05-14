from __future__ import annotations

import asyncio
import heapq
from dataclasses import dataclass

from v2v_rt.config import ProfileConfig, SLOConfig
from v2v_rt.types import BatchEnvelope, FrameEnvelope, now_ns


@dataclass(slots=True)
class SchedulerStats:
    accepted: int = 0
    dropped_late: int = 0
    dropped_backpressure: int = 0
    emitted_batches: int = 0


class SLOBatchScheduler:
    """Earliest-deadline-first micro-batcher for live frames."""

    def __init__(self, slo: SLOConfig, initial_profile: ProfileConfig) -> None:
        self.slo = slo
        self.profile = initial_profile
        self.stats = SchedulerStats()
        self._cv = asyncio.Condition()
        self._heap: list[tuple[int, int, FrameEnvelope]] = []
        self._counter = 0

    async def submit(self, frame: FrameEnvelope) -> None:
        async with self._cv:
            if self.slo.drop_late_frames and frame.is_late:
                self.stats.dropped_late += 1
                return

            max_depth = max(self.profile.max_batch_size * 4, 4)
            if len(self._heap) >= max_depth:
                heapq.heappop(self._heap)
                self.stats.dropped_backpressure += 1

            heapq.heappush(self._heap, (frame.deadline_ns, self._counter, frame))
            self._counter += 1
            self.stats.accepted += 1
            self._cv.notify()

    async def next_batch(self) -> BatchEnvelope:
        async with self._cv:
            while not self._heap:
                await self._cv.wait()

            queue_budget_ns = self.slo.max_queue_ms * 1_000_000
            first_deadline = self._heap[0][0]
            wait_until_ns = min(first_deadline, now_ns() + queue_budget_ns)

            while len(self._heap) < self.profile.max_batch_size and now_ns() < wait_until_ns:
                timeout_s = max((wait_until_ns - now_ns()) / 1_000_000_000.0, 0.0)
                try:
                    await asyncio.wait_for(self._cv.wait(), timeout=timeout_s)
                except asyncio.TimeoutError:
                    break

            frames: list[FrameEnvelope] = []
            while self._heap and len(frames) < self.profile.max_batch_size:
                _, _, frame = heapq.heappop(self._heap)
                if self.slo.drop_late_frames and frame.is_late:
                    self.stats.dropped_late += 1
                    continue
                frames.append(frame)

            if not frames:
                return await self.next_batch()

            self.stats.emitted_batches += 1
            return BatchEnvelope(
                frames=frames,
                profile_name=self.profile.name,
                deadline_ns=min(frame.deadline_ns for frame in frames),
            )

    def set_profile(self, profile: ProfileConfig) -> None:
        self.profile = profile
