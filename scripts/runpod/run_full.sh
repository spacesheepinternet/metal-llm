#!/usr/bin/env bash
# Full-corpus training run (all 20,676 prepared songs).
#
# Prereqs: setup_pod.sh done, HUB_REPO env var set (e.g. youruser/metal-llm-full).
# Override STEPS / BATCH / ACCUM via env if needed.
#
#   tmux new -s train
#   bash scripts/runpod/run_full.sh
# Resume after an interruption:
#   bash scripts/runpod/run_full.sh --resume
set -euo pipefail
cd "$(dirname "$0")/../.."

: "${HUB_REPO:?set HUB_REPO env var (e.g. youruser/metal-llm-full)}"
STEPS="${STEPS:-10000}"
BATCH="${BATCH:-2}"
ACCUM="${ACCUM:-4}"
mkdir -p outputs/full

python -u src/train_qlora.py \
    --pilot-songs 0 --steps "$STEPS" --seq-len 2048 \
    --batch-size "$BATCH" --grad-accum "$ACCUM" \
    --lr 2e-4 --lr-scheduler cosine --warmup-steps 100 \
    --eval-steps 500 \
    --extend-vocab --new-tokens-only \
    --window-cache outputs/full/windows.npy \
    --hub-repo "$HUB_REPO" \
    --out outputs/full \
    "$@" 2>&1 | tee -a outputs/full/train.log
