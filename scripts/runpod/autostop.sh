#!/bin/bash
# Stop this pod once training exits (checkpoints are on HF Hub; idle time is
# wasted money). Launch alongside training:
#   tmux new -d -s autostop bash scripts/runpod/autostop.sh
export $(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | xargs)
# wait up to 30 min for training to appear (safe to launch before it starts)
for _ in $(seq 1 180); do
    pgrep -f train_qlora >/dev/null && break
    sleep 10
done
while pgrep -f train_qlora >/dev/null; do sleep 60; done
sleep 180   # let background checkpoint uploads settle
runpodctl stop pod "$RUNPOD_POD_ID"
