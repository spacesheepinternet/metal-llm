#!/usr/bin/env bash
# One-time RunPod setup: install deps, pull the prepared dataset from HF Hub.
#
# Prereqs (set as pod environment variables when creating the pod):
#   HF_TOKEN  — HF access token with read+write (checkpoint pushes need write)
#   DATA_REPO — private HF *dataset* repo id holding the prepared data,
#               e.g. youruser/metal-llm-data
#
# Run from anywhere: bash scripts/runpod/setup_pod.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

: "${HF_TOKEN:?set HF_TOKEN pod env var}"
: "${DATA_REPO:?set DATA_REPO pod env var (e.g. youruser/metal-llm-data)}"

# torch ships with the RunPod PyTorch template — don't touch it.
# Pins mirror the laptop stack that pilot attempt #5 trained on.
# Ubuntu 24.04 images mark python externally-managed (PEP 668); the pod is
# disposable, so overriding is safe.
export PIP_BREAK_SYSTEM_PACKAGES=1
pip install -q "transformers==5.13.1" "peft==0.19.1" "bitsandbytes>=0.49" \
    accelerate "datasets>=5" "huggingface_hub>=1.24" numpy tqdm

mkdir -p data/prepared data/corpus/DadaGP-v1.1
hf download "$DATA_REPO" --repo-type dataset --local-dir data/prepared
# vocab json lives where train_qlora.py expects it by default
mv -f data/prepared/_DadaGP_all_tokens.json data/corpus/DadaGP-v1.1/

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python - <<'EOF'
import torch, transformers, peft, bitsandbytes
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("transformers", transformers.__version__, "| peft", peft.__version__,
      "| bnb", bitsandbytes.__version__)
EOF
echo "setup done — next: bash scripts/runpod/run_pilot.sh"
