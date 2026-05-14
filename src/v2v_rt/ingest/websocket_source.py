from __future__ import annotations

import asyncio
from dataclasses import dataclass

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

from v2v_rt.config import ServerConfig, SLOConfig
from v2v_rt.types import FrameEnvelope, now_ns


@dataclass(slots=True)
class WebSocketFrameSource:
    """Binary JPEG/PNG frame ingest over WebSocket.

    This is a lightweight ingress bridge for the inference loop. A production
    deployment can replace it with aiortc while preserving the FrameSource API.
    """

    server: ServerConfig
    slo: SLOConfig
    queue_size: int = 8

    def __post_init__(self) -> None:
        self._queue: asyncio.Queue[FrameEnvelope] = asyncio.Queue(self.queue_size)
        self._app = FastAPI()
        self._server_task: asyncio.Task[None] | None = None
        self._frame_ids: dict[str, int] = {}
        self._install_routes()

    def _install_routes(self) -> None:
        @self._app.websocket("/ingest/{stream_id}")
        async def ingest(websocket: WebSocket, stream_id: str) -> None:
            await websocket.accept()
            try:
                while True:
                    data = await websocket.receive_bytes()
                    frame = self._decode_frame(stream_id, data)
                    if self._queue.full():
                        _ = self._queue.get_nowait()
                    await self._queue.put(frame)
            except WebSocketDisconnect:
                return

    def _decode_frame(self, stream_id: str, data: bytes) -> FrameEnvelope:
        received_ns = now_ns()
        array = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Unable to decode incoming frame bytes.")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frame_id = self._frame_ids.get(stream_id, 0)
        self._frame_ids[stream_id] = frame_id + 1
        deadline_ns = received_ns + self.slo.frame_deadline_ms * 1_000_000
        return FrameEnvelope(
            stream_id=stream_id,
            frame_id=frame_id,
            received_ns=received_ns,
            deadline_ns=deadline_ns,
            payload=rgb,
            width=int(rgb.shape[1]),
            height=int(rgb.shape[0]),
        )

    async def start(self) -> None:
        config = uvicorn.Config(
            self._app,
            host=self.server.host,
            port=self.server.port,
            log_level="info",
            loop="asyncio",
        )
        server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(server.serve())
        await asyncio.sleep(0)

    async def receive(self) -> FrameEnvelope:
        return await self._queue.get()

    async def stop(self) -> None:
        if self._server_task:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
