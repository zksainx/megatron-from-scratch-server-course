#!/usr/bin/env bash

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}

if [[ -z "${COURSE_ROOT:-}" ]]; then
  COMMON_ENV_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
  COURSE_ROOT=$(cd -- "$COMMON_ENV_DIR/.." && pwd)
fi
TRAIN_ROOT=${TRAIN_ROOT:-$(cd -- "$COURSE_ROOT/.." && pwd)}
MEGATRON_ROOT=${MEGATRON_ROOT:-$TRAIN_ROOT/Megatron-LM}
RUN_ROOT=${RUN_ROOT:-$COURSE_ROOT/runs}
export COURSE_ROOT TRAIN_ROOT MEGATRON_ROOT RUN_ROOT

RUNTIME_CACHE_DIR=${RUNTIME_CACHE_DIR:-$RUN_ROOT/cache/runtime}
COMPILE_CACHE_DIR=${COMPILE_CACHE_DIR:-/tmp/megatron-server-course-cache-${USER:-user}}
mkdir -p "$RUNTIME_CACHE_DIR"/{xdg,hf}
mkdir -p "$COMPILE_CACHE_DIR"/{torch_extensions,flashinfer,triton,home,pycache}

export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$RUNTIME_CACHE_DIR/xdg}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$COMPILE_CACHE_DIR/torch_extensions}
export FLASHINFER_CACHE_DIR=${FLASHINFER_CACHE_DIR:-$COMPILE_CACHE_DIR/flashinfer}
export HF_HOME=${HF_HOME:-$RUNTIME_CACHE_DIR/hf}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$COMPILE_CACHE_DIR/triton}
export PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX:-$COMPILE_CACHE_DIR/pycache}
export ORIGINAL_HOME=${ORIGINAL_HOME:-${HOME:-}}
export HOME=$COMPILE_CACHE_DIR/home
export PYTHONPATH=$MEGATRON_ROOT:${PYTHONPATH:-}

PYTHON_SITE_PACKAGES=$(python - <<'PY'
import site
paths = site.getsitepackages()
print(paths[0] if paths else "")
PY
)
CUDA13_PIP_LIB_DIR=$PYTHON_SITE_PACKAGES/nvidia/cu13/lib
if [[ -d "$CUDA13_PIP_LIB_DIR" ]]; then
  export LD_LIBRARY_PATH=$CUDA13_PIP_LIB_DIR:${LD_LIBRARY_PATH:-}
fi

if [[ -d /usr/include/python3.12 && -d /usr/include/pybind11 ]]; then
  export CPPFLAGS="-I/usr/include/python3.12 -I/usr/include ${CPPFLAGS:-}"
fi
