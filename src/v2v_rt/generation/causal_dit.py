from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from v2v_rt.config import ModelConfig, OptimizationConfig
from v2v_rt.generation.flashinfer_attention import FlashInferAttention


@dataclass
class CausalDiTState:
    rolling_kv: dict[str, Any] = field(default_factory=dict)
    noise_scale: float = 1.0


class CausalDiTBackbone(torch.nn.Module):
    """Adapter boundary for a lightweight causal DiT V2V backbone."""

    def __init__(
        self,
        model: torch.nn.Module | None,
        attention: FlashInferAttention,
        state_window: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.attention = attention
        self.state_window = state_window

    @classmethod
    def load(cls, config: ModelConfig, optimization: OptimizationConfig, device: torch.device) -> "CausalDiTBackbone":
        checkpoint = Path(config.causal_dit_checkpoint)
        attention = FlashInferAttention(optimization.attention_backend)
        model: torch.nn.Module | None = None

        if checkpoint.exists():
            # Real implementation hook:
            #   model = StreamDiffusionV2Pipeline(...).denoiser
            # or a project-local Causal DiT module loaded from safetensors.
            # Keep this explicit until the checkpoint format is chosen.
            model = None

        backbone = cls(model=model, attention=attention, state_window=config.temporal_window)
        backbone.to(device)
        backbone.eval()
        return backbone

    def forward_stream(
        self,
        latents: torch.Tensor,
        control: torch.Tensor,
        state: CausalDiTState,
        *,
        prompt_embeds: torch.Tensor | None,
        denoise_steps: int,
    ) -> tuple[torch.Tensor, CausalDiTState]:
        if self.model is None:
            raise RuntimeError(
                "CausalDiTBackbone is a scaffold. Provide a real causal DiT or "
                "StreamDiffusionV2 denoiser adapter at models.causal_dit_checkpoint."
            )

        return self.model.forward_stream(
            latents=latents,
            control=control,
            state=state,
            prompt_embeds=prompt_embeds,
            denoise_steps=denoise_steps,
            attention_backend=self.attention,
        )
