#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
scratch_root=${GBM_A100_SCRATCH:-/tmp/gbm-a100-${USER:-researcher}}
python_bin=${PYTHON_BIN:-python3.11}
pytorch_index_url=${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}

if ! command -v "$python_bin" >/dev/null 2>&1; then
  printf 'Required Python 3.11 executable is unavailable: %s\n' "$python_bin" >&2
  exit 2
fi

"$python_bin" - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"Python 3.11 is required; found {sys.version.split()[0]}")
PY

case "$scratch_root" in
  "$repo_root"|"$repo_root"/*)
    printf '%s\n' 'GBM_A100_SCRATCH must be outside the Git repository.' >&2
    exit 2
    ;;
esac

mkdir -p "$scratch_root/huggingface/datasets" "$scratch_root/huggingface/transformers" "$scratch_root/results"
export HF_HOME="$scratch_root/huggingface"
export HF_DATASETS_CACHE="$scratch_root/huggingface/datasets"
export TRANSFORMERS_CACHE="$scratch_root/huggingface/transformers"
export PIP_NO_CACHE_DIR=1

cd "$repo_root"
"$python_bin" scripts/install_repo_safety.py
nvidia-smi

venv_root="$scratch_root/venv-a100"
if [[ ! -d "$venv_root" ]]; then
  "$python_bin" -m venv "$venv_root"
fi

"$venv_root/bin/python" -m pip install --upgrade pip
"$venv_root/bin/python" -m pip install --no-cache-dir \
  --index-url "$pytorch_index_url" torch==2.3.0 torchtext==0.18.0
"$venv_root/bin/python" -m pip install --no-cache-dir scgpt==0.2.4
"$venv_root/bin/python" -m pip install --no-cache-dir -r requirements-a100.txt
"$venv_root/bin/python" -m pip install --no-cache-dir -e .
"$venv_root/bin/python" -m pip check
"$venv_root/bin/python" - <<'PY'
import torch
import torchtext
from scgpt.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab
if not torch.cuda.is_available():
    raise SystemExit("CUDA-enabled PyTorch is unavailable in the isolated A100 environment")
if not torch.__version__.startswith("2.3.") or not torchtext.__version__.startswith("0.18."):
    raise SystemExit(f"Expected torch 2.3.x + torchtext 0.18.x; found {torch.__version__} + {torchtext.__version__}")
print({
    "torch": torch.__version__,
    "torchtext": torchtext.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "scgpt_model_import": TransformerModel.__name__,
    "vocab_import": GeneVocab.__name__,
})
PY

printf 'A100 environment created. Scratch: %s\n' "$scratch_root"
printf 'Activate with: source %s/bin/activate\n' "$venv_root"
