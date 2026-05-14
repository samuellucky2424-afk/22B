from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
import inspect
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from v2v_rt.config import AppConfig, ProfileConfig
from v2v_rt.types import BatchEnvelope, FrameEnvelope, GeneratedFrame, now_ns

try:
    from streamdiffusionv2 import StreamDiffusionV2Pipeline, VideoChunk
except ImportError as exc:  # pragma: no cover - exercised only before env setup.
    StreamDiffusionV2Pipeline = None  # type: ignore[assignment]
    VideoChunk = None  # type: ignore[assignment]
    _STREAMDIFFUSION_IMPORT_ERROR: ImportError | None = exc
else:
    _STREAMDIFFUSION_IMPORT_ERROR = None

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover - exercised only before env setup.
    mp = None  # type: ignore[assignment]
    _MEDIAPIPE_IMPORT_ERROR: ImportError | None = exc
else:
    _MEDIAPIPE_IMPORT_ERROR = None


LOGGER = logging.getLogger(__name__)


def torch_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


@dataclass(slots=True)
class PrecisionReport:
    requested_vae_precision: str
    requested_dit_precision: str
    fp8_requested: bool
    native_fp8_supported: bool
    taehv_enabled: bool
    tensorrt_enabled: bool
    vae_fp8_parameters: int
    dit_fp8_parameters: int


@dataclass
class ForegroundFrame:
    tensor: torch.Tensor
    mask: np.ndarray | None = None
    mask_coverage: float | None = None
    preprocess_ms: float = 0.0


@dataclass
class OutputContext:
    envelope: FrameEnvelope
    mask: np.ndarray | None = None
    mask_coverage: float | None = None
    preprocess_ms: float = 0.0


@dataclass
class IdentityAnchorState:
    initialized: bool = False
    anchor_frame_id: int | None = None
    anchor_kv_cache: Any | None = None
    anchor_reference_frame: torch.Tensor | None = None
    rolling_kv_cache: deque[Any] = field(default_factory=deque)
    source: str = "pending"

    @property
    def active_window_size(self) -> int:
        return int(self.anchor_kv_cache is not None) + len(self.rolling_kv_cache)

    @property
    def has_kv_anchor(self) -> bool:
        return self.anchor_kv_cache is not None

    def attention_window(self) -> list[Any]:
        if self.anchor_kv_cache is None:
            return list(self.rolling_kv_cache)
        return [self.anchor_kv_cache, *self.rolling_kv_cache]


@dataclass
class StreamState:
    pending: deque[tuple[FrameEnvelope, ForegroundFrame]] = field(default_factory=deque)
    awaiting_output: deque[OutputContext] = field(default_factory=deque)
    previous_frame: torch.Tensor | None = None
    previous_mask: np.ndarray | None = None
    identity_anchor: IdentityAnchorState = field(default_factory=IdentityAnchorState)
    noise_scale: float = 0.8
    initial_noise_scale: float = 0.8
    started: bool = False
    current_end: int = 0


class StaticBackgroundCompositor:
    """CPU-side matte extraction and static background compositing.

    StreamDiffusionV2 still receives RGB tensors, so we strip the live room by
    feeding foreground-only frames and preserve the alpha matte for the decoded
    frame that returns later from the streaming queue.
    """

    def __init__(
        self,
        background_path: str | None,
        width: int,
        height: int,
        *,
        model_selection: int = 1,
        mask_threshold: float = 0.12,
        feather_sigma: float = 1.2,
        temporal_smoothing: float = 0.22,
    ) -> None:
        self.background_path = background_path or None
        self.width = width
        self.height = height
        self.model_selection = model_selection
        self.mask_threshold = float(np.clip(mask_threshold, 0.0, 0.49))
        self.feather_sigma = max(0.0, float(feather_sigma))
        self.temporal_smoothing = float(np.clip(temporal_smoothing, 0.0, 1.0))
        self._segmenter: Any | None = None
        self._background_source_rgb: np.ndarray | None = None
        self._background_cache: dict[tuple[int, int], np.ndarray] = {}

        if self.background_path is None:
            return

        if mp is None:
            raise RuntimeError(
                "mediapipe is required for static background compositing. "
                "Install requirements.txt or add mediapipe to the runtime image."
            ) from _MEDIAPIPE_IMPORT_ERROR

        self._background_source_rgb = self._load_background(Path(self.background_path))
        self._background_cache[(height, width)] = self._resize_background(height, width)
        self._segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=self.model_selection,
        )
        LOGGER.info("Static background compositing enabled with %s", self.background_path)

    @property
    def enabled(self) -> bool:
        return self._segmenter is not None and self._background_source_rgb is not None

    def close(self) -> None:
        if self._segmenter is not None:
            close = getattr(self._segmenter, "close", None)
            if callable(close):
                close()
        self._segmenter = None

    def warmup(self) -> None:
        if not self.enabled:
            return
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        _, mask = self.extract_foreground(frame, previous_mask=None)
        self.composite(frame, mask)

    def extract_foreground(
        self,
        rgb: np.ndarray,
        previous_mask: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if not self.enabled:
            return rgb, None

        assert self._segmenter is not None
        rgb = np.ascontiguousarray(rgb.astype(np.uint8, copy=False))
        rgb.flags.writeable = False
        result = self._segmenter.process(rgb)
        raw_mask = getattr(result, "segmentation_mask", None)

        if raw_mask is None:
            mask = previous_mask if previous_mask is not None else np.ones(rgb.shape[:2], dtype=np.float32)
        else:
            mask = np.asarray(raw_mask, dtype=np.float32)

        mask = self._refine_mask(mask, previous_mask, rgb.shape[:2])
        foreground = rgb.astype(np.float32) * mask[..., None]
        return np.clip(foreground, 0, 255).astype(np.uint8), mask

    def composite(self, stylized_rgb: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        if not self.enabled or mask is None:
            return stylized_rgb

        stylized_rgb = np.ascontiguousarray(stylized_rgb.astype(np.uint8, copy=False))
        alpha = self._resize_mask(mask, stylized_rgb.shape[:2])
        background = self._background_for_shape(stylized_rgb.shape[:2])

        composited = (
            stylized_rgb.astype(np.float32) * alpha[..., None]
            + background.astype(np.float32) * (1.0 - alpha[..., None])
        )
        return np.clip(composited, 0, 255).astype(np.uint8)

    @staticmethod
    def _load_background(path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Static background image does not exist: {path}")

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Failed to read static background image: {path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    def _background_for_shape(self, shape_hw: tuple[int, int]) -> np.ndarray:
        height, width = shape_hw
        key = (height, width)
        background = self._background_cache.get(key)
        if background is None:
            background = self._resize_background(height, width)
            self._background_cache[key] = background
        return background

    def _resize_background(self, height: int, width: int) -> np.ndarray:
        assert self._background_source_rgb is not None
        if self._background_source_rgb.shape[:2] == (height, width):
            return self._background_source_rgb
        return cv2.resize(
            self._background_source_rgb,
            (width, height),
            interpolation=cv2.INTER_AREA,
        )

    def _refine_mask(
        self,
        mask: np.ndarray,
        previous_mask: np.ndarray | None,
        shape_hw: tuple[int, int],
    ) -> np.ndarray:
        mask = np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
        mask = self._resize_mask(mask, shape_hw)
        if self.feather_sigma > 0.0:
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=self.feather_sigma, sigmaY=self.feather_sigma)
        mask_span = max(1.0 - (2.0 * self.mask_threshold), 1e-6)
        mask = np.clip((mask - self.mask_threshold) / mask_span, 0.0, 1.0)

        if previous_mask is not None:
            previous = self._resize_mask(previous_mask, shape_hw)
            mask = ((1.0 - self.temporal_smoothing) * mask) + (self.temporal_smoothing * previous)

        return np.clip(mask, 0.0, 1.0).astype(np.float32, copy=False)

    @staticmethod
    def _resize_mask(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
        target_h, target_w = shape_hw
        if mask.shape[:2] != (target_h, target_w):
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return np.clip(mask.astype(np.float32, copy=False), 0.0, 1.0)


class IdentityAnchorAdapter:
    """Compatibility layer for attention-sink identity anchoring.

    StreamDiffusionV2 builds vary in how they expose KV caches. This adapter
    supports explicit encode_chunk kwargs when present and otherwise publishes
    the first-frame cache onto common stream/pipeline attributes.
    """

    CACHE_ATTRS = (
        "identity_anchor_kv_cache",
        "attention_sink_kv_cache",
        "reference_kv_cache",
        "identity_anchor_cache",
        "attention_sink_cache",
        "reference_cache",
        "reference_kv",
        "kv_cache",
        "past_key_values",
        "cached_key_values",
        "key_value_cache",
    )
    CACHE_METHODS = (
        "get_identity_anchor_cache",
        "get_attention_sink_cache",
        "export_identity_anchor",
        "export_kv_cache",
        "export_attention_sink",
        "get_kv_cache",
    )

    def __init__(self, config: AppConfig) -> None:
        model_config = config.models
        self.enabled = bool(model_config and model_config.identity_anchor_enabled)
        self.strength = float(model_config.identity_anchor_strength if model_config else 0.0)
        self.window_size = max(1, int(model_config.identity_anchor_rolling_window if model_config else 1))

    def configure_state(self, state: StreamState) -> None:
        state.identity_anchor.rolling_kv_cache = deque(maxlen=self.window_size)

    def prepare_encode(
        self,
        stream: Any,
        state: StreamState,
        full_video: torch.Tensor,
        anchor_frame: FrameEnvelope,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {}

        anchor = state.identity_anchor
        if anchor.anchor_reference_frame is None:
            anchor.anchor_reference_frame = self._first_reference_frame(full_video)
            anchor.anchor_frame_id = anchor_frame.frame_id

        candidates: dict[str, Any] = {
            "use_cache": True,
            "return_kv_cache": True,
            "preserve_kv_cache": True,
            "cache_identity_anchor": not anchor.has_kv_anchor,
            "cache_attention_sink": not anchor.has_kv_anchor,
            "return_identity_anchor": not anchor.has_kv_anchor,
            "return_attention_sink": not anchor.has_kv_anchor,
            "preserve_identity_anchor": True,
            "preserve_attention_sink": True,
            "lock_identity_anchor": True,
            "identity_anchor_frame": anchor.anchor_reference_frame,
            "attention_sink_frame": anchor.anchor_reference_frame,
            "reference_frame": anchor.anchor_reference_frame,
            "identity_anchor_frame_id": anchor.anchor_frame_id,
            "attention_sink_frame_id": anchor.anchor_frame_id,
            "identity_anchor_strength": self.strength,
        }

        if anchor.anchor_kv_cache is not None:
            rolling_window = anchor.attention_window()
            candidates.update(
                {
                    "identity_anchor_kv_cache": anchor.anchor_kv_cache,
                    "attention_sink_kv_cache": anchor.anchor_kv_cache,
                    "reference_kv_cache": anchor.anchor_kv_cache,
                    "identity_anchor_cache": anchor.anchor_kv_cache,
                    "attention_sink_cache": anchor.anchor_kv_cache,
                    "reference_cache": anchor.anchor_kv_cache,
                    "rolling_kv_cache": rolling_window,
                    "rolling_attention_window": rolling_window,
                    "rolling_identity_window": rolling_window,
                    "identity_anchor_window": rolling_window,
                    "attention_sink_window": rolling_window,
                    "identity_anchor_window_size": self.window_size,
                }
            )

        self.publish_to_stream(stream, anchor)
        return self._supported_kwargs(getattr(stream, "encode_chunk"), candidates)

    def capture_after_stage(self, stream: Any, state: StreamState, obj: Any, source: str) -> None:
        if not self.enabled:
            return

        anchor = state.identity_anchor
        cache = self._find_kv_cache(obj)
        if cache is None:
            cache = self._find_kv_cache(stream)

        if cache is None:
            if not anchor.initialized and anchor.anchor_reference_frame is not None:
                anchor.initialized = True
                anchor.source = "reference_frame"
                LOGGER.warning(
                    "Identity anchor is using the first foreground frame because no StreamDiffusionV2 KV cache "
                    "was exposed. Install a build that supports attention-sink KV export for strict identity lock."
                )
            return

        snapshot = self._detach_cache(cache)
        if anchor.anchor_kv_cache is None:
            anchor.anchor_kv_cache = snapshot
            anchor.initialized = True
            anchor.source = source
            LOGGER.info(
                "Captured identity attention sink from frame=%s source=%s.",
                anchor.anchor_frame_id,
                source,
            )
        else:
            anchor.rolling_kv_cache.append(snapshot)

        self.publish_to_stream(stream, anchor)

    def publish_to_stream(self, stream: Any, anchor: IdentityAnchorState) -> None:
        if not self.enabled or anchor.anchor_kv_cache is None:
            return

        rolling_window = anchor.attention_window()
        values = {
            "identity_anchor_kv_cache": anchor.anchor_kv_cache,
            "attention_sink_kv_cache": anchor.anchor_kv_cache,
            "reference_kv_cache": anchor.anchor_kv_cache,
            "identity_anchor_cache": anchor.anchor_kv_cache,
            "attention_sink_cache": anchor.anchor_kv_cache,
            "reference_cache": anchor.anchor_kv_cache,
            "rolling_kv_cache": rolling_window,
            "rolling_attention_window": rolling_window,
            "rolling_identity_window": rolling_window,
            "identity_anchor_window": rolling_window,
            "attention_sink_window": rolling_window,
            "identity_anchor_strength": self.strength,
        }

        for target in self._stream_targets(stream):
            for name, value in values.items():
                try:
                    setattr(target, name, value)
                except Exception:
                    continue

    @staticmethod
    def _first_reference_frame(full_video: torch.Tensor) -> torch.Tensor:
        # full_video is [B,C,T,H,W]; keep T=1 so StreamDiffusion adapters can
        # consume it as a video reference without reshaping on their side.
        return full_video[:, :, :1].detach().clone()

    @classmethod
    def _find_kv_cache(cls, obj: Any, seen: set[int] | None = None) -> Any | None:
        if obj is None:
            return None
        if seen is None:
            seen = set()
        object_id = id(obj)
        if object_id in seen:
            return None
        seen.add(object_id)

        if isinstance(obj, dict):
            for key in cls.CACHE_ATTRS:
                value = obj.get(key)
                if value is not None:
                    return value
            for nested_key in ("identity_anchor", "attention_sink", "cache", "metadata"):
                nested = obj.get(nested_key)
                if nested is not None and nested is not obj:
                    value = cls._find_kv_cache(nested, seen)
                    if value is not None:
                        return value

        for method_name in cls.CACHE_METHODS:
            method = getattr(obj, method_name, None)
            if callable(method):
                try:
                    value = method()
                except TypeError:
                    continue
                if value is not None:
                    return value

        for attr in cls.CACHE_ATTRS:
            value = getattr(obj, attr, None)
            if value is not None:
                return value

        for nested_name in ("pipeline_manager", "pipeline", "denoiser", "transformer"):
            nested = getattr(obj, nested_name, None)
            if nested is not None and nested is not obj:
                value = cls._find_kv_cache(nested, seen)
                if value is not None:
                    return value

        return None

    @staticmethod
    def _detach_cache(cache: Any) -> Any:
        if isinstance(cache, torch.Tensor):
            return cache.detach().clone()
        if isinstance(cache, dict):
            return {key: IdentityAnchorAdapter._detach_cache(value) for key, value in cache.items()}
        if isinstance(cache, list):
            return [IdentityAnchorAdapter._detach_cache(value) for value in cache]
        if isinstance(cache, tuple):
            return tuple(IdentityAnchorAdapter._detach_cache(value) for value in cache)
        return cache

    @staticmethod
    def _supported_kwargs(callable_obj: Any, candidates: dict[str, Any]) -> dict[str, Any]:
        try:
            signature = inspect.signature(callable_obj)
        except (TypeError, ValueError):
            return {}

        parameters = signature.parameters.values()
        accepts_var_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters)
        if accepts_var_kwargs:
            return {key: value for key, value in candidates.items() if value is not None}

        allowed = set(signature.parameters)
        return {key: value for key, value in candidates.items() if key in allowed and value is not None}

    @staticmethod
    def _stream_targets(stream: Any) -> tuple[Any, ...]:
        manager = getattr(stream, "pipeline_manager", None)
        pipeline = getattr(manager, "pipeline", None) if manager is not None else None
        denoiser = getattr(pipeline, "denoiser", None) if pipeline is not None else None
        transformer = getattr(pipeline, "transformer", None) if pipeline is not None else None
        return tuple(target for target in (stream, manager, pipeline, denoiser, transformer) if target is not None)


class RealtimeV2VPipeline:
    """Realtime adapter around StreamDiffusionV2's staged V2V API."""

    def __init__(self, config: AppConfig, stream: Any, profile: ProfileConfig) -> None:
        self.config = config
        self.stream = stream
        self.device = torch.device(config.runtime.device)
        self.dtype = torch_dtype(config.runtime.dtype)
        self.profile = profile
        self.streams: dict[str, StreamState] = {}
        self.compositor = self._create_compositor(profile)
        self.identity_anchor = IdentityAnchorAdapter(config)
        self.precision_report = self._verify_precision()

    @classmethod
    def build(cls, config: AppConfig) -> "RealtimeV2VPipeline":
        if config.models is None:
            raise ValueError("models config is required.")
        if StreamDiffusionV2Pipeline is None:
            raise RuntimeError(
                "streamdiffusionv2 is not installed. Install the RunPod environment "
                "with `python -m pip install -r requirements.txt`."
            ) from _STREAMDIFFUSION_IMPORT_ERROR

        device = torch.device(config.runtime.device)

        if config.runtime.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.cuda.set_per_process_memory_fraction(config.runtime.cuda_memory_fraction, device=device)

        stream = cls._create_stream(config, config.default_profile)
        return cls(config, stream, config.default_profile)

    @classmethod
    def _create_stream(cls, config: AppConfig, profile: ProfileConfig) -> Any:
        assert config.models is not None
        assert StreamDiffusionV2Pipeline is not None

        checkpoint_folder = Path(config.models.streamdiffusion_checkpoint)
        if not checkpoint_folder.exists():
            LOGGER.warning("StreamDiffusionV2 checkpoint folder does not exist yet: %s", checkpoint_folder)

        stream = StreamDiffusionV2Pipeline(
            checkpoint_folder=str(checkpoint_folder),
            mode="single",
            device=config.runtime.device,
            height=profile.height,
            width=profile.width,
            fps=profile.fps,
            step=profile.denoise_steps,
        )

        cls._enable_acceleration(stream, config)

        stream.prepare(config.models.prompt)
        return stream

    @staticmethod
    def _enable_acceleration(stream: Any, config: AppConfig) -> None:
        if not config.optimization.enable_streamdiffusion_acceleration:
            LOGGER.warning(
                "StreamDiffusionV2 acceleration is disabled; TAEHV/TensorRT fast path will not be used."
            )
            return

        try:
            stream.enable_acceleration(fast=True)
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

    async def warmup(self) -> None:
        """Warm CUDA allocator state after model construction.

        StreamDiffusionV2 loads weights and TensorRT/TAEHV paths during construction
        and acceleration. The first real chunk still drives shape-specific kernels.
        """
        self.compositor.warmup()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def close(self) -> None:
        close = getattr(self.stream, "close", None)
        if callable(close):
            close()
        self.compositor.close()

    def set_profile(self, profile: ProfileConfig) -> None:
        if (
            profile.name == self.profile.name
            and profile.width == self.profile.width
            and profile.height == self.profile.height
            and profile.fps == self.profile.fps
            and profile.denoise_steps == self.profile.denoise_steps
        ):
            return

        LOGGER.info("Rebuilding StreamDiffusionV2 for profile %s", profile.name)
        self.close()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.stream = self._create_stream(self.config, profile)
        self.profile = profile
        self.compositor = self._create_compositor(profile)
        self.identity_anchor = IdentityAnchorAdapter(self.config)
        self.streams.clear()
        self.precision_report = self._verify_precision()

    def recover_after_oom(self, profile: ProfileConfig) -> None:
        LOGGER.warning("Recovering StreamDiffusionV2 after CUDA OOM with profile %s", profile.name)
        self.close()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        self.stream = self._create_stream(self.config, profile)
        self.profile = profile
        self.compositor = self._create_compositor(profile)
        self.identity_anchor = IdentityAnchorAdapter(self.config)
        self.streams.clear()
        self.precision_report = self._verify_precision()

    def _create_compositor(self, profile: ProfileConfig) -> StaticBackgroundCompositor:
        model_config = self.config.models
        return StaticBackgroundCompositor(
            model_config.static_background_path if model_config is not None else None,
            profile.width,
            profile.height,
            model_selection=model_config.segmentation_model_selection if model_config is not None else 1,
            mask_threshold=model_config.segmentation_mask_threshold if model_config is not None else 0.12,
            feather_sigma=model_config.segmentation_feather_sigma if model_config is not None else 1.2,
            temporal_smoothing=model_config.segmentation_temporal_smoothing if model_config is not None else 0.22,
        )

    async def process_batch(self, batch: BatchEnvelope) -> list[GeneratedFrame]:
        return await asyncio.to_thread(self._process_batch_sync, batch)

    @torch.inference_mode()
    def _process_batch_sync(self, batch: BatchEnvelope) -> list[GeneratedFrame]:
        outputs: list[GeneratedFrame] = []
        for frame in batch.frames:
            state = self.streams.setdefault(frame.stream_id, self._new_stream_state())
            state.pending.append((frame, self._preprocess_frame(frame, state)))
            outputs.extend(self._drain_ready_chunks(frame.stream_id, state))
        return outputs

    def _new_stream_state(self) -> StreamState:
        noise_scale = float(getattr(self.stream, "noise_scale", 0.8))
        state = StreamState(noise_scale=noise_scale, initial_noise_scale=noise_scale)
        self.identity_anchor.configure_state(state)
        return state

    def _frames_needed(self, state: StreamState) -> int:
        chunk_size = int(getattr(self.stream, "chunk_size"))
        return 1 + chunk_size if not state.started else chunk_size

    def _drain_ready_chunks(self, stream_id: str, state: StreamState) -> list[GeneratedFrame]:
        outputs: list[GeneratedFrame] = []
        while len(state.pending) >= self._frames_needed(state):
            chunk_envelopes, foreground_frames, chunk_tensor, full_video = self._pop_video_chunk(state)
            state.awaiting_output.extend(
                OutputContext(
                    envelope=envelope,
                    mask=foreground.mask,
                    mask_coverage=foreground.mask_coverage,
                    preprocess_ms=foreground.preprocess_ms,
                )
                for envelope, foreground in zip(chunk_envelopes, foreground_frames, strict=False)
            )

            encoded_chunk = self._encode_chunk_with_identity_anchor(
                state,
                full_video,
                chunk_tensor,
                anchor_frame=chunk_envelopes[0],
            )
            state.noise_scale = float(encoded_chunk.noise_scale)
            self.identity_anchor.capture_after_stage(self.stream, state, encoded_chunk, "encoded_chunk")

            denoised_chunk = self.stream.denoise_chunk(encoded_chunk)
            if denoised_chunk is None:
                continue
            self.identity_anchor.capture_after_stage(self.stream, state, denoised_chunk, "denoised_chunk")

            decoded = self.stream.decode_chunk(denoised_chunk)
            outputs.extend(self._pack_decoded_frames(stream_id, state, decoded))

        return outputs

    def _encode_chunk_with_identity_anchor(
        self,
        state: StreamState,
        full_video: torch.Tensor,
        chunk_tensor: Any,
        *,
        anchor_frame: FrameEnvelope,
    ) -> Any:
        identity_kwargs = self.identity_anchor.prepare_encode(self.stream, state, full_video, anchor_frame)
        return self.stream.encode_chunk(
            full_video,
            chunk_tensor,
            previous_noise_scale=state.noise_scale,
            initial_noise_scale=state.initial_noise_scale,
            **identity_kwargs,
        )

    def _pop_video_chunk(
        self,
        state: StreamState,
    ) -> tuple[list[FrameEnvelope], list[ForegroundFrame], Any, torch.Tensor]:
        assert VideoChunk is not None

        needed = self._frames_needed(state)
        entries = [state.pending.popleft() for _ in range(needed)]
        envelopes = [entry[0] for entry in entries]
        foreground_frames = [entry[1] for entry in entries]
        frame_tensors = [frame.tensor for frame in foreground_frames]

        video_cthw = torch.stack(frame_tensors, dim=1)
        video_cthw = video_cthw.to(self.device, dtype=torch.bfloat16, non_blocking=True)

        chunk_size = int(getattr(self.stream, "chunk_size"))
        frame_seq_length = int(self.stream.pipeline_manager.pipeline.frame_seq_length)

        if not state.started:
            current_start = 0
            current_end = frame_seq_length * (1 + chunk_size // 4)
            full_video = video_cthw.unsqueeze(0)
            chunk_frames = full_video
            start_idx = 0
            end_idx = needed
            state.started = True
        else:
            if state.previous_frame is None:
                raise RuntimeError("Missing previous frame for motion-aware StreamDiffusionV2 chunk.")
            previous = state.previous_frame.to(self.device, dtype=torch.bfloat16, non_blocking=True)
            full_video = torch.cat([previous.unsqueeze(1), video_cthw], dim=1).unsqueeze(0)
            chunk_frames = video_cthw.unsqueeze(0)
            current_start = state.current_end
            current_end = current_start + (chunk_size // 4) * frame_seq_length
            start_idx = 1
            end_idx = 1 + chunk_size

        state.previous_frame = frame_tensors[-1]
        state.current_end = current_end

        video_chunk = VideoChunk(
            frames=chunk_frames,
            start_idx=start_idx,
            end_idx=end_idx,
            current_start=current_start,
            current_end=current_end,
        )
        return envelopes, foreground_frames, video_chunk, full_video

    def _preprocess_frame(self, frame: FrameEnvelope, state: StreamState) -> ForegroundFrame:
        rgb = frame.payload
        if not isinstance(rgb, np.ndarray):
            raise TypeError("Frame payload must be a decoded RGB numpy array.")

        resized = cv2.resize(
            rgb,
            (self.profile.width, self.profile.height),
            interpolation=cv2.INTER_AREA,
        )
        preprocess_start_ns = now_ns()
        foreground, mask = self.compositor.extract_foreground(resized, state.previous_mask)
        if mask is not None:
            state.previous_mask = mask

        tensor = torch.from_numpy(np.ascontiguousarray(foreground)).permute(2, 0, 1).float()
        mask_coverage = float(mask.mean()) if mask is not None else None
        preprocess_ms = (now_ns() - preprocess_start_ns) / 1_000_000.0
        return ForegroundFrame(
            tensor=tensor.div_(127.5).sub_(1.0),
            mask=mask,
            mask_coverage=mask_coverage,
            preprocess_ms=preprocess_ms,
        )

    def _pack_decoded_frames(
        self,
        stream_id: str,
        state: StreamState,
        decoded: np.ndarray,
    ) -> list[GeneratedFrame]:
        decoded_frames = self._normalize_decoded_video(decoded)
        produced_ns = now_ns()
        outputs: list[GeneratedFrame] = []

        for image in decoded_frames:
            compositing_start_ns = now_ns()
            if state.awaiting_output:
                context = state.awaiting_output.popleft()
                source_frame = context.envelope
                preprocess_ms = context.preprocess_ms
                mask_coverage = context.mask_coverage
                image = self.compositor.composite(image, context.mask)
            else:
                source_frame = FrameEnvelope(
                    stream_id=stream_id,
                    frame_id=-1,
                    received_ns=produced_ns,
                    deadline_ns=produced_ns,
                    payload=image,
                    width=self.profile.width,
                    height=self.profile.height,
                )
                preprocess_ms = 0.0
                mask_coverage = None
                image = self.compositor.composite(image, None)

            compositing_ms = (now_ns() - compositing_start_ns) / 1_000_000.0
            outputs.append(
                GeneratedFrame(
                    stream_id=stream_id,
                    frame_id=source_frame.frame_id,
                    encoded_payload=self._encode_jpeg(image),
                    produced_ns=produced_ns,
                    metadata={
                        "profile": self.profile.name,
                        "latency_ms": round((produced_ns - source_frame.received_ns) / 1_000_000.0, 2),
                        "noise_scale": round(state.noise_scale, 4),
                        "taehv": bool(getattr(self.stream, "use_taehv", False)),
                        "tensorrt": bool(getattr(self.stream, "use_tensorrt", False)),
                        "static_background": self.compositor.enabled,
                        "segmentation_ms": round(preprocess_ms, 2),
                        "compositing_ms": round(compositing_ms, 2),
                        "foreground_mask_coverage": (
                            round(mask_coverage, 4) if mask_coverage is not None else None
                        ),
                        "identity_anchor": state.identity_anchor.initialized,
                        "identity_anchor_has_kv": state.identity_anchor.has_kv_anchor,
                        "identity_anchor_frame_id": state.identity_anchor.anchor_frame_id,
                        "identity_anchor_source": state.identity_anchor.source,
                        "identity_anchor_window": state.identity_anchor.active_window_size,
                    },
                )
            )

        return outputs

    def _normalize_decoded_video(self, decoded: np.ndarray) -> np.ndarray:
        array = np.asarray(decoded)
        if array.ndim == 3:
            array = array[None, ...]
        if array.ndim != 4:
            raise ValueError(f"Decoded video must have [T,H,W,C] shape, got {array.shape}.")

        if array.dtype == np.uint8:
            return array

        array = array.astype(np.float32, copy=False)
        if array.min(initial=0.0) < -0.01:
            array = (array + 1.0) * 0.5
        if array.max(initial=0.0) <= 1.5:
            array = array * 255.0
        return np.clip(array, 0, 255).astype(np.uint8)

    def _encode_jpeg(self, image: np.ndarray) -> bytes:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError("Failed to encode generated frame.")
        return encoded.tobytes()

    def _verify_precision(self) -> PrecisionReport:
        requested_vae = self.config.optimization.vae_precision
        requested_dit = self.config.optimization.dit_precision
        fp8_requested = requested_vae.startswith("fp8") or requested_dit.startswith("fp8")
        native_fp8 = self._native_fp8_supported()
        taehv_enabled = bool(getattr(self.stream, "use_taehv", False))
        tensorrt_enabled = bool(getattr(self.stream, "use_tensorrt", False))
        vae_fp8 = self._count_float8_parameters(("vae", "taehv"))
        dit_fp8 = self._count_float8_parameters(("dit", "transformer", "block"))

        report = PrecisionReport(
            requested_vae_precision=requested_vae,
            requested_dit_precision=requested_dit,
            fp8_requested=fp8_requested,
            native_fp8_supported=native_fp8,
            taehv_enabled=taehv_enabled,
            tensorrt_enabled=tensorrt_enabled,
            vae_fp8_parameters=vae_fp8,
            dit_fp8_parameters=dit_fp8,
        )

        if not taehv_enabled:
            raise RuntimeError("TAEHV acceleration is required for this deployment but is not enabled.")
        if self.config.optimization.enable_streamdiffusion_acceleration and not tensorrt_enabled:
            raise RuntimeError("TensorRT acceleration was requested but StreamDiffusionV2 did not enable it.")

        if fp8_requested and not native_fp8:
            message = (
                "FP8 precision was requested, but this GPU does not expose native FP8 tensor cores. "
                "RTX 3090/Ampere should run BF16/FP16 with TAEHV/TensorRT instead."
            )
            if self.config.optimization.strict_fp8:
                raise RuntimeError(message)
            LOGGER.warning(message)
        elif fp8_requested and (vae_fp8 == 0 or dit_fp8 == 0):
            message = (
                "FP8 precision was requested, but loaded StreamDiffusionV2 modules do not expose "
                "float8 parameters for both VAE and DiT blocks."
            )
            if self.config.optimization.strict_fp8:
                raise RuntimeError(message)
            LOGGER.warning(message)

        LOGGER.info("StreamDiffusionV2 precision report: %s", report)
        return report

    def _native_fp8_supported(self) -> bool:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return False
        major, minor = torch.cuda.get_device_capability(self.device)
        return (major, minor) >= (8, 9)

    def _count_float8_parameters(self, name_tokens: tuple[str, ...]) -> int:
        manager = getattr(self.stream, "pipeline_manager", None)
        if manager is None:
            return 0
        root = getattr(manager, "pipeline", manager)
        if not hasattr(root, "named_parameters"):
            return 0

        float8_dtypes = {
            dtype
            for dtype in (
                getattr(torch, "float8_e4m3fn", None),
                getattr(torch, "float8_e5m2", None),
                getattr(torch, "float8_e4m3fnuz", None),
                getattr(torch, "float8_e5m2fnuz", None),
            )
            if dtype is not None
        }
        count = 0
        for name, parameter in root.named_parameters():
            lower_name = name.lower()
            if parameter.dtype in float8_dtypes and any(token in lower_name for token in name_tokens):
                count += parameter.numel()
        return count
