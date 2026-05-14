from __future__ import annotations

import pytest

from v2v_rt.config import ProfileConfig, SLOConfig
from v2v_rt.orchestration.scheduler import SLOBatchScheduler
from v2v_rt.types import FrameEnvelope, now_ns


@pytest.mark.asyncio
async def test_scheduler_batches_by_deadline() -> None:
    profile = ProfileConfig(
        name="test",
        width=320,
        height=180,
        fps=30,
        max_batch_size=2,
        denoise_steps=1,
    )
    slo = SLOConfig(frame_deadline_ms=50, max_queue_ms=1, max_batch_size=2)
    scheduler = SLOBatchScheduler(slo, profile)
    t0 = now_ns()

    await scheduler.submit(
        FrameEnvelope(
            stream_id="a",
            frame_id=0,
            received_ns=t0,
            deadline_ns=t0 + 50_000_000,
            payload=None,
            width=320,
            height=180,
        )
    )
    await scheduler.submit(
        FrameEnvelope(
            stream_id="a",
            frame_id=1,
            received_ns=t0,
            deadline_ns=t0 + 50_000_000,
            payload=None,
            width=320,
            height=180,
        )
    )

    batch = await scheduler.next_batch()

    assert batch.size == 2
    assert [frame.frame_id for frame in batch.frames] == [0, 1]
