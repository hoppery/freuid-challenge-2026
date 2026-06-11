#!/bin/bash
# Auto-resume: poll until CUDA can initialize again (user reset/rebooted to clear the
# GPU2 fault), then automatically launch the Stage-1 improvement sweep on GPUs 0,1
# (GPU3 = monitor excluded). Polls every 120s for up to ~24h.
cd /home/hoppery/ijcai_freuid_chanllenge || exit 1
unset PYTHONPATH
GUARD=checkpoints/.stage1_launched
if [ -f "$GUARD" ]; then echo "stage1 already launched ($GUARD exists at $(date)); skipping"; exit 0; fi
for i in $(seq 1 720); do
  if CUDA_VISIBLE_DEVICES=0 python3 -c "import torch; torch.zeros(1).cuda()" >/dev/null 2>&1; then
    echo "GPU RECOVERED at $(date) after $i checks"
    mkdir -p checkpoints && touch "$GUARD"
    CUDA_VISIBLE_DEVICES=0,1 python3 scripts/run_sweep.py \
      --configs exp_heavyaug exp_dinov2 exp_hpf exp_convnext_base baseline_indomain \
      --gpus 0 1
    echo "=== SWEEP DONE at $(date) ==="
    cat docs/experiments_results.csv 2>/dev/null
    exit 0
  fi
  sleep 120
done
echo "GPU NOT recovered within ~24h — manual attention needed"
exit 1
