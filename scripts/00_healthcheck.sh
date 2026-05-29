#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
COURSE_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
source "$COURSE_ROOT/configs/4gpu_edu_pretrain.env"
source "$COURSE_ROOT/scripts/common_env.sh"

echo "[healthcheck] TRAIN_ROOT=$TRAIN_ROOT"
echo "[healthcheck] MEGATRON_ROOT=$MEGATRON_ROOT"
echo "[healthcheck] XDG_CACHE_HOME=$XDG_CACHE_HOME"
echo "[healthcheck] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

test -d "$MEGATRON_ROOT"
test -f "$MEGATRON_ROOT/pretrain_gpt.py"
test -f "$GPT2_VOCAB_FILE"
test -f "$GPT2_MERGE_FILE"

python - <<'PY'
import importlib.util
import os
import sys
import torch

print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
for idx in range(torch.cuda.device_count()):
    prop = torch.cuda.get_device_properties(idx)
    print(f"gpu[{idx}] name={prop.name} cc={prop.major}.{prop.minor} mem_gb={prop.total_memory/1024**3:.1f}")

for pkg in ["transformers", "datasets"]:
    print(f"{pkg}_installed", importlib.util.find_spec(pkg) is not None)
PY

python - <<'PY'
import transformer_engine
import transformer_engine_torch  # noqa: F401
print("transformer_engine", getattr(transformer_engine, "__version__", "unknown"))
print("transformer_engine_torch_ok True")
PY

python - <<'PY'
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core import parallel_state  # noqa: F401
print("megatron_import_ok True")
print("megatron_have_te", HAVE_TE)
print("megatron_te_spec_provider_ok", TESpecProvider is not None)
PY

echo "[healthcheck] ok"
