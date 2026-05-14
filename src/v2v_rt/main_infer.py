from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from contextlib import suppress

import torch

from v2v_rt.config import AppConfig, load_config
from v2v_rt.egress.frame_sink import LoggingFrameSink
from v2v_rt.ingest.frame_source import FrameSource
from v2v_rt.ingest.websocket_source import WebSocketFrameSource
from v2v_rt.orchestration.pipeline import RealtimeV2VPipeline
from v2v_rt.orchestration.scheduler import SLOBatchScheduler


LOGGER = logging.getLogger(__name__)


async def ingest_worker(source: FrameSource, scheduler: SLOBatchScheduler, stop: asyncio.Event) -> None:
    while not stop.is_set():
        frame = await source.receive()
        await scheduler.submit(frame)


def choose_profile(config: AppConfig, current_index: int, last_latency_ms: float | None) -> int:
    """Move between profiles based on VRAM pressure and recent latency."""
    slow = (
        last_latency_ms is not None
        and last_latency_ms > config.slo.frame_deadline_ms * 0.9
    )

    if not torch.cuda.is_available():
        return current_index

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    used_gb = (total_bytes - free_bytes) / (1024**3)
    under_pressure = used_gb > config.optimization.max_vram_gb

    if under_pressure or slow:
        return min(current_index + 1, len(config.profiles) - 1)

    fast = (
        last_latency_ms is not None
        and last_latency_ms < config.slo.frame_deadline_ms * 0.5
    )
    has_headroom = used_gb < config.optimization.max_vram_gb * 0.8
    if current_index > 0 and fast and has_headroom:
        return current_index - 1

    return current_index


async def inference_loop(config: AppConfig) -> None:
    if config.models is not None and config.models.static_background_path:
        LOGGER.info(
            "Zero-flicker static background compositing configured: %s "
            "(MediaPipe model=%s, threshold=%.2f, feather=%.2f, temporal_smoothing=%.2f)",
            config.models.static_background_path,
            config.models.segmentation_model_selection,
            config.models.segmentation_mask_threshold,
            config.models.segmentation_feather_sigma,
            config.models.segmentation_temporal_smoothing,
        )
    if config.models is not None and config.models.identity_anchor_enabled:
        LOGGER.info(
            "Identity attention sink enabled with rolling KV window=%s strength=%.2f",
            config.models.identity_anchor_rolling_window,
            config.models.identity_anchor_strength,
        )

    pipeline = RealtimeV2VPipeline.build(config)
    LOGGER.info("StreamDiffusionV2 acceleration/precision report: %s", pipeline.precision_report)
    await pipeline.warmup()

    scheduler = SLOBatchScheduler(config.slo, config.default_profile)
    source = WebSocketFrameSource(config.server, config.slo)
    sink = LoggingFrameSink()

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    await source.start()
    producer = asyncio.create_task(ingest_worker(source, scheduler, stop))
    profile_index = 0
    last_latency_ms: float | None = None

    try:
        while not stop.is_set():
            profile_index = choose_profile(config, profile_index, last_latency_ms)
            profile = config.profiles[profile_index]
            pipeline.set_profile(profile)
            scheduler.set_profile(profile)

            batch = await scheduler.next_batch()
            try:
                frames = await pipeline.process_batch(batch)
            except torch.cuda.OutOfMemoryError:
                profile_index = min(profile_index + 1, len(config.profiles) - 1)
                pipeline.recover_after_oom(config.profiles[profile_index])
                scheduler.set_profile(config.profiles[profile_index])
                continue

            if frames:
                last_latency_ms = max(float(frame.metadata["latency_ms"]) for frame in frames)
            await sink.send_many(frames)
    finally:
        stop.set()
        producer.cancel()
        with suppress(asyncio.CancelledError):
            await producer
        await source.stop()
        pipeline.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run realtime V2V inference loop.")
    parser.add_argument("--config", default="configs/rtx3090.yaml", help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    asyncio.run(inference_loop(config))


if __name__ == "__main__":
    main()
