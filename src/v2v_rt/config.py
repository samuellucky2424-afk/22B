from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cuda:0"
    dtype: str = "float16"
    compile_models: bool = True
    cuda_memory_fraction: float = 0.92
    warmup_batches: int = 3
    channels_last: bool = True
    allow_tf32: bool = True


@dataclass(frozen=True)
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    ingest_path: str = "/ingest/{stream_id}"
    egress_path: str = "/egress/{stream_id}"
    max_connections: int = 4


@dataclass(frozen=True)
class SLOConfig:
    ttff_ms: int = 500
    frame_deadline_ms: int = 50
    max_queue_ms: int = 12
    max_batch_size: int = 2
    drop_late_frames: bool = True
    target_fps: int = 20
    min_fps: int = 12


@dataclass(frozen=True)
class ProfileConfig:
    name: str
    width: int
    height: int
    fps: int
    max_batch_size: int
    denoise_steps: int


@dataclass(frozen=True)
class ModelConfig:
    streamdiffusion_checkpoint: str
    tiny_vae_checkpoint: str
    controlnet_xs_checkpoint: str
    causal_dit_checkpoint: str
    prompt: str
    negative_prompt: str = ""
    static_background_path: str | None = None
    segmentation_model_selection: int = 1
    segmentation_mask_threshold: float = 0.12
    segmentation_feather_sigma: float = 1.2
    segmentation_temporal_smoothing: float = 0.22
    identity_anchor_enabled: bool = True
    identity_anchor_strength: float = 0.85
    identity_anchor_rolling_window: int = 12
    controlnet_strength: float = 0.85
    background_lock_alpha: float = 0.95
    temporal_window: int = 8
    latent_scale: float = 0.18215


@dataclass(frozen=True)
class OptimizationConfig:
    attention_backend: str = "flashinfer"
    use_flashinfer_jit_cache: bool = True
    use_cuda_graphs: bool = False
    preallocate_workspace_mb: int = 512
    enable_streamdiffusion_acceleration: bool = True
    enable_taehv: bool = True
    vae_precision: str = "fp8_e4m3fn"
    dit_precision: str = "fp8_e4m3fn"
    strict_fp8: bool = False
    max_vram_gb: float = 22.0


@dataclass(frozen=True)
class AppConfig:
    runtime: RuntimeConfig
    server: ServerConfig
    slo: SLOConfig
    profiles: list[ProfileConfig] = field(default_factory=list)
    models: ModelConfig | None = None
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    @property
    def default_profile(self) -> ProfileConfig:
        if not self.profiles:
            raise ValueError("At least one runtime profile is required.")
        return self.profiles[0]


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Config section '{name}' must be a mapping.")
    return value


def load_config(path: str | Path) -> AppConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    profiles = [ProfileConfig(**item) for item in raw.get("profiles", [])]
    models = raw.get("models")
    return AppConfig(
        runtime=RuntimeConfig(**_section(raw, "runtime")),
        server=ServerConfig(**_section(raw, "server")),
        slo=SLOConfig(**_section(raw, "slo")),
        profiles=profiles,
        models=ModelConfig(**models) if models else None,
        optimization=OptimizationConfig(**_section(raw, "optimization")),
    )
