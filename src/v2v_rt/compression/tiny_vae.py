from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from v2v_rt.config import ModelConfig


class TinyVideoVAE(torch.nn.Module):
    """Tiny VAE / 3D-VAE adapter for ultra-low-latency latent compression."""

    def __init__(self, impl: torch.nn.Module | None, latent_scale: float) -> None:
        super().__init__()
        self.impl = impl
        self.latent_scale = latent_scale

    @classmethod
    def load(cls, config: ModelConfig, device: torch.device, dtype: torch.dtype) -> "TinyVideoVAE":
        checkpoint = Path(config.tiny_vae_checkpoint)
        impl: torch.nn.Module | None = None
        if checkpoint.exists():
            # Real implementation hook:
            #   impl = TAEHV(...); impl.load_state_dict(torch.load(checkpoint))
            # or use a StreamDiffusionV2 TAEHV loader once checkpoint format is fixed.
            impl = None

        module = cls(impl=impl, latent_scale=config.latent_scale)
        module.to(device=device, dtype=dtype)
        module.eval()
        return module

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        if self.impl is None:
            raise RuntimeError(
                "TinyVideoVAE is a scaffold. Provide a real TAEHV/3D-VAE adapter "
                "at models.tiny_vae_checkpoint before serving."
            )
        if hasattr(self.impl, "encode"):
            return self.impl.encode(frames).mul(self.latent_scale)
        return F.interpolate(frames, scale_factor=0.125, mode="bilinear", align_corners=False)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if self.impl is None:
            raise RuntimeError(
                "TinyVideoVAE is a scaffold. Provide a real TAEHV/3D-VAE adapter "
                "at models.tiny_vae_checkpoint before serving."
            )
        latents = latents.div(self.latent_scale)
        if hasattr(self.impl, "decode"):
            return self.impl.decode(latents)
        return F.interpolate(latents, scale_factor=8.0, mode="bilinear", align_corners=False)
