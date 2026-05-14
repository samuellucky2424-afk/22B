#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${CONDA_ENV_NAME:-v2v-rt-backend}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CONFIG_PATH="${CONFIG_PATH:-configs/rtx3090.yaml}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINICONDA_DIR="${MINICONDA_DIR:-${HOME}/miniconda3}"
MINICONDA_INSTALLER="${MINICONDA_INSTALLER:-Miniconda3-latest-Linux-x86_64.sh}"
MINICONDA_URL="${MINICONDA_URL:-https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER}}"
CONDA_CHANNEL="${CONDA_CHANNEL:-conda-forge}"
FLASHINFER_JIT_CACHE_INDEX="${FLASHINFER_JIT_CACHE_INDEX:-}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.7.4.post1}"
STREAMDIFFUSIONV2_VERSION="${STREAMDIFFUSIONV2_VERSION:-0.1.0}"
STREAMDIFFUSIONV2_GIT_URL="${STREAMDIFFUSIONV2_GIT_URL:-git+https://github.com/chenfengxu714/StreamDiffusionV2.git}"
DOWNLOAD_CHECKPOINTS="${DOWNLOAD_CHECKPOINTS:-1}"
WAN_MODEL_REPO="${WAN_MODEL_REPO:-Wan-AI/Wan2.1-T2V-1.3B}"
STREAMDIFFUSIONV2_HF_REPO="${STREAMDIFFUSIONV2_HF_REPO:-jerryfeng/StreamDiffusionV2}"
TAEHV_URL="${TAEHV_URL:-https://github.com/madebyollin/taehv/raw/main/taew2_1.pth}"
ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/.env.runpod}"

cd "${PROJECT_ROOT}"

log() {
  printf '[deploy] %s\n' "$*"
}

load_local_env() {
  if [[ ! -f "${ENV_FILE}" ]]; then
    return
  fi

  log "Loading local deployment environment from ${ENV_FILE}"
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
}

load_local_env

ensure_conda() {
  if command -v conda >/dev/null 2>&1; then
    return
  fi

  if [[ -x "${MINICONDA_DIR}/bin/conda" ]]; then
    export PATH="${MINICONDA_DIR}/bin:${PATH}"
    return
  fi

  log "Conda not found; installing Miniconda into ${MINICONDA_DIR}"
  curl -fsSL "${MINICONDA_URL}" -o "/tmp/${MINICONDA_INSTALLER}"
  bash "/tmp/${MINICONDA_INSTALLER}" -b -p "${MINICONDA_DIR}"
  rm -f "/tmp/${MINICONDA_INSTALLER}"
  export PATH="${MINICONDA_DIR}/bin:${PATH}"
}

ensure_nvidia_runtime() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "nvidia-smi was not found. Continue only if the RunPod image exposes CUDA through another runtime."
    return
  fi

  nvidia-smi
}

ensure_conda
ensure_nvidia_runtime

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  log "Creating Conda environment ${ENV_NAME} with Python ${PYTHON_VERSION} from ${CONDA_CHANNEL}"
  conda create -y --override-channels -c "${CONDA_CHANNEL}" \
    -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip git ffmpeg ninja cmake
else
  log "Using existing Conda environment ${ENV_NAME}"
fi

conda activate "${ENV_NAME}"

log "Installing pinned CUDA 12.6 / PyTorch 2.6.0 dependency stack"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url "${PYTORCH_INDEX_URL}" \
  "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0"

log "Installing application/model dependencies"
python -m pip install -r requirements.txt

export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-8.6}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export MAX_JOBS="${MAX_JOBS:-4}"

log "Installing flash-attn ${FLASH_ATTN_VERSION} with torch-visible build context"
python -m pip install --no-build-isolation "flash-attn==${FLASH_ATTN_VERSION}"

if [[ -n "${FLASHINFER_JIT_CACHE_INDEX}" ]]; then
  log "Attempting optional FlashInfer JIT cache install from ${FLASHINFER_JIT_CACHE_INDEX}"
  python -m pip install flashinfer-jit-cache --index-url "${FLASHINFER_JIT_CACHE_INDEX}" || \
    log "Optional flashinfer-jit-cache wheel unavailable; continuing with runtime cache generation"
else
  log "No FLASHINFER_JIT_CACHE_INDEX set; FlashInfer will generate/cache kernels on first use"
fi

log "Ensuring StreamDiffusionV2 is installed and importable"
python -m pip install --upgrade --no-deps "streamdiffusionv2==${STREAMDIFFUSIONV2_VERSION}"
if ! python - <<'PY'
from streamdiffusionv2 import StreamDiffusionV2Pipeline, VideoChunk

print("streamdiffusionv2 import: ok")
print("pipeline:", StreamDiffusionV2Pipeline)
print("video_chunk:", VideoChunk)
PY
then
  log "PyPI StreamDiffusionV2 install did not expose streamdiffusionv2; installing from official GitHub"
  python -m pip uninstall -y streamdiffusionv2 || true
  python -m pip install --no-deps "${STREAMDIFFUSIONV2_GIT_URL}"
  python - <<'PY'
from streamdiffusionv2 import StreamDiffusionV2Pipeline, VideoChunk

print("streamdiffusionv2 import: ok")
print("pipeline:", StreamDiffusionV2Pipeline)
print("video_chunk:", VideoChunk)
PY
fi

log "Installing v2v-rt-backend package in editable mode"
python -m pip install -e .

export PYTHONUNBUFFERED=1
export STREAMDIFFUSIONV2_ROOT="${STREAMDIFFUSIONV2_ROOT:-${PROJECT_ROOT}}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.hf-cache}"
export FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-${PROJECT_ROOT}/.flashinfer-cache}"
mkdir -p "${HF_HOME}"
mkdir -p "${FLASHINFER_CACHE_DIR}"

download_checkpoints() {
  if [[ "${DOWNLOAD_CHECKPOINTS}" != "1" ]]; then
    log "DOWNLOAD_CHECKPOINTS=${DOWNLOAD_CHECKPOINTS}; skipping checkpoint download"
    return
  fi

  log "Ensuring Wan base model exists under ${STREAMDIFFUSIONV2_ROOT}/wan_models"
  mkdir -p "${STREAMDIFFUSIONV2_ROOT}/wan_models" "${PROJECT_ROOT}/ckpts"

  if [[ ! -f "${STREAMDIFFUSIONV2_ROOT}/wan_models/Wan2.1-T2V-1.3B/config.json" ]]; then
    huggingface-cli download --resume-download "${WAN_MODEL_REPO}" \
      --local-dir "${STREAMDIFFUSIONV2_ROOT}/wan_models/Wan2.1-T2V-1.3B"
  else
    log "Wan base model already present"
  fi

  log "Ensuring StreamDiffusionV2 causal V2V checkpoint exists under ${PROJECT_ROOT}/ckpts"
  if [[ ! -d "${PROJECT_ROOT}/ckpts/wan_causal_dmd_v2v" ]]; then
    huggingface-cli download --resume-download "${STREAMDIFFUSIONV2_HF_REPO}" \
      --local-dir "${PROJECT_ROOT}/ckpts" \
      --include "wan_causal_dmd_v2v/*"
  else
    log "StreamDiffusionV2 causal V2V checkpoint already present"
  fi

  if [[ ! -f "${PROJECT_ROOT}/ckpts/taew2_1.pth" ]]; then
    log "Downloading TAEHV checkpoint"
    curl -L "${TAEHV_URL}" -o "${PROJECT_ROOT}/ckpts/taew2_1.pth"
  else
    log "TAEHV checkpoint already present"
  fi
}

download_checkpoints

log "Runtime sanity check"
python - <<'PY'
import importlib.metadata as metadata
import os

import torch

print("streamdiffusionv2_root:", os.environ.get("STREAMDIFFUSIONV2_ROOT"))
print("torch:", torch.__version__)
print("torch_cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print("vram_total_gb:", round(total_bytes / (1024**3), 2))
    print("vram_free_gb:", round(free_bytes / (1024**3), 2))

for package in ("flashinfer-python", "flashinfer-cubin", "streamdiffusionv2"):
    try:
        print(f"{package}:", metadata.version(package))
    except metadata.PackageNotFoundError:
        print(f"{package}: not installed")
PY

if command -v flashinfer >/dev/null 2>&1; then
  flashinfer show-config || true
fi

log "Launching realtime inference with ${CONFIG_PATH}"
exec python -m v2v_rt.main_infer --config "${CONFIG_PATH}"
