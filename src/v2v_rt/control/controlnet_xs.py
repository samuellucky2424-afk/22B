from __future__ import annotations

from pathlib import Path

import kornia
import torch

from v2v_rt.config import ModelConfig


class ControlNetXSConditioner(torch.nn.Module):
    """ControlNet-XS boundary for background and geometry locking."""

    def __init__(self, adapter: torch.nn.Module | None, strength: float, background_lock_alpha: float) -> None:
        super().__init__()
        self.adapter = adapter
        self.strength = strength
        self.background_lock_alpha = background_lock_alpha

    @classmethod
    def load(cls, config: ModelConfig, device: torch.device, dtype: torch.dtype) -> "ControlNetXSConditioner":
        checkpoint = Path(config.controlnet_xs_checkpoint)
        adapter: torch.nn.Module | None = None
        if checkpoint.exists():
            # Real implementation hook:
            #   from diffusers import ControlNetXSAdapter
            #   adapter = ControlNetXSAdapter.from_pretrained(checkpoint, torch_dtype=dtype)
            adapter = None

        module = cls(
            adapter=adapter,
            strength=config.controlnet_strength,
            background_lock_alpha=config.background_lock_alpha,
        )
        module.to(device=device, dtype=dtype)
        module.eval()
        return module

    def encode_condition(self, frames: torch.Tensor) -> torch.Tensor:
        """Return a compact edge/geometry condition tensor.

        Input and output layout: [batch, 3, height, width].
        """
        gray = kornia.color.rgb_to_grayscale(frames.clamp(0, 1))
        _, edges = kornia.filters.canny(gray, low_threshold=0.1, high_threshold=0.2)
        return edges.repeat(1, 3, 1, 1).mul(self.strength)

    def apply(self, latents: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if self.adapter is None:
            raise RuntimeError(
                "ControlNetXSConditioner is a scaffold. Provide a real ControlNet-XS "
                "adapter at models.controlnet_xs_checkpoint before serving."
            )
        return self.adapter(latents, condition)
