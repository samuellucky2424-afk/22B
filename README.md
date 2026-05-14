# v2v-rt-backend

Real-time video-to-video streaming backend scaffold for one RTX 3090 24 GB instance.

This first pass is intentionally the pipeline skeleton, not a full application. It defines the runtime envelope, deadline-aware orchestration, and model adapter boundaries needed to wire a Tiny VAE or compressed video VAE, causal DiT, ControlNet-XS, and FlashInfer attention into a StreamDiffusionV2-like online serving loop.

## Directory Structure

```text
v2v-rt-backend/
  README.md
  environment.yml
  requirements.txt
  pyproject.toml
  configs/
    rtx3090.yaml
  scripts/
    bootstrap_runpod.sh
  src/
    v2v_rt/
      __init__.py
      config.py
      main_infer.py
      types.py
      ingest/
        __init__.py
        frame_source.py
        websocket_source.py
      compression/
        __init__.py
        tiny_vae.py
      control/
        __init__.py
        controlnet_xs.py
      generation/
        __init__.py
        causal_dit.py
        flashinfer_attention.py
      orchestration/
        __init__.py
        pipeline.py
        scheduler.py
      egress/
        __init__.py
        frame_sink.py
  tests/
    test_scheduler.py
```

## Runtime Assumptions

- Linux RunPod image with NVIDIA driver new enough for CUDA 12.6 wheels.
- Python 3.10, matching the current StreamDiffusionV2 install guidance.
- PyTorch 2.6.0 CUDA 12.6 wheels, matching the StreamDiffusionV2 PyPI dependency set.
- FlashInfer installed with a CUDA 12.6 JIT cache to reduce first-request kernel compilation.
- TTFF target is interpreted as post-warmup session TTFF. The service warms model weights, kernels, and allocator state before accepting live streams.

## Intended Data Path

```text
WebRTC/WebSocket ingest
  -> deadline queue
  -> dynamic profile selection
  -> Tiny/3D VAE latent encode
  -> ControlNet-XS structural condition
  -> causal DiT denoise with rolling state
  -> Tiny/3D VAE decode
  -> WebRTC/WebSocket egress
```

## Bootstrap

```bash
cd v2v-rt-backend
bash deploy.sh
```

The current `main_infer.py` loop is model-ready but not checkpoint-complete. It will fail fast with clear errors when real model adapter load paths are missing.
