from __future__ import annotations

import pytest

pytest.importorskip("cv2")
pytest.importorskip("fastapi")

from v2v_rt.config import SLOConfig, ServerConfig
from v2v_rt.ingest.websocket_source import WebSocketFrameSource


def test_websocket_source_initializes_slotted_state() -> None:
    source = WebSocketFrameSource(ServerConfig(), SLOConfig())

    assert source._queue.maxsize == 8
    assert source._frame_ids == {}
