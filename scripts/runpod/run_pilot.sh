#!/usr/bin/env bash
# Pilot training run — identical config to the local attempts so loss curves
# stay comparable (500 songs, 500 steps, seq 2048, extend-vocab new-tokens-only).
#
# Prereqs: setup_pod.sh done, plus pod env var:
#   HUB_REPO — private HF *model* repo id for checkpoint pushes,
#              e.g. youruser/metal-llm-pilot (created automatically)
#
# Run inside tmux so an SSH drop doesn't kill training:
#   tmux new -s train
#   bash scripts/runpod/run_pilot.sh
# Resume after an interruption (pulls nothing — reuses outputs/pilot):
#   bash scripts/runpod/run_pilot.sh --resume
set -euo pipefail
cd "$(dirname "$0")/../.."

: "${HUB_REPO:?set HUB_REPO pod env var (e.g. youruser/metal-llm-pilot)}"
mkdir -p outputs/pilot

python -u src/train_qlora.py \
    --pilot-songs 500 --steps 500 --seq-len 2048 \
    --extend-vocab --new-tokens-only \
    --hub-repo "$HUB_REPO" \
    --out outputs/pilot \
    "$@" 2>&1 | tee -a outputs/pilot/train.log
