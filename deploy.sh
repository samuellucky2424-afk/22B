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

cd "${PROJECT_ROOT}"

log() {
  printf '[deploy] %s\n' "$*"
}

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

log "Installing pinned CUDA 12.6 / PyTorch 2.6.0 / FlashInfer dependency stack"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-8.6}"
if [[ -n "${FLASHINFER_JIT_CACHE_INDEX}" ]]; then
  log "Attempting optional FlashInfer JIT cache install from ${FLASHINFER_JIT_CACHE_INDEX}"
  python -m pip install flashinfer-jit-cache --index-url "${FLASHINFER_JIT_CACHE_INDEX}" || \
    log "Optional flashinfer-jit-cache wheel unavailable; continuing with runtime cache generation"
else
  log "No FLASHINFER_JIT_CACHE_INDEX set; FlashInfer will generate/cache kernels on first use"
fi

log "Ensuring StreamDiffusionV2 is installed explicitly"
python -m pip install --upgrade --no-deps "streamdiffusionv2[flash-attn]==0.1.0"

log "Installing v2v-rt-backend package in editable mode"
python -m pip install -e .

export PYTHONUNBUFFERED=1
export FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-${PROJECT_ROOT}/.flashinfer-cache}"
mkdir -p "${FLASHINFER_CACHE_DIR}"

log "Runtime sanity check"
python - <<'PY'
import importlib.metadata as metadata

import torch

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
