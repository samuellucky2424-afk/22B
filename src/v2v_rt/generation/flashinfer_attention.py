from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class AttentionRuntime:
    requested_backend: str
    active_backend: str
    available: bool
    detail: str = ""


class FlashInferAttention:
    """Small adapter around FlashInfer prefill attention with SDPA fallback.

    Expected tensor layout is [batch, heads, tokens, head_dim].
    """

    def __init__(self, backend: str = "flashinfer") -> None:
        self.runtime = AttentionRuntime(
            requested_backend=backend,
            active_backend="torch_sdpa",
            available=False,
        )
        self._prefill: Any | None = None
        if backend != "flashinfer":
            self.runtime.detail = "FlashInfer disabled by config."
            return

        try:
            from flashinfer import prefill

            self._prefill = prefill
            self.runtime = AttentionRuntime(
                requested_backend=backend,
                active_backend="flashinfer.prefill",
                available=True,
                detail="Using single_prefill_with_kv_cache_return_lse.",
            )
        except Exception as exc:  # pragma: no cover - depends on CUDA runtime.
            self.runtime.detail = f"FlashInfer unavailable, falling back to SDPA: {exc}"

    def self_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool) -> torch.Tensor:
        if self._prefill is None:
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

        outputs: list[torch.Tensor] = []
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
        for batch_idx in range(q.shape[0]):
            q_nhd = q[batch_idx].permute(1, 0, 2).contiguous()
            k_nhd = k[batch_idx].permute(1, 0, 2).contiguous()
            v_nhd = v[batch_idx].permute(1, 0, 2).contiguous()
            out = self._prefill.single_prefill_with_kv_cache_return_lse(
                q_nhd,
                k_nhd,
                v_nhd,
                causal=causal,
                kv_layout="NHD",
                sm_scale=sm_scale,
                backend="auto",
                return_lse=True,
            )
            if isinstance(out, tuple):
                out = out[0]
            outputs.append(out.permute(1, 0, 2).contiguous())

        return torch.stack(outputs, dim=0)
