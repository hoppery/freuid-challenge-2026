#!/bin/bash
# Poll until CUDA is available on GPU 0, then launch the Stage-1 improvement sweep.
# Guard file prevents re-launching on repeated boots.
cd /home/hoppery/ijcai_freuid_chanllenge || exit 1
unset PYTHONPATH
GUARD=checkpoints/.stage1_launched
if [ -f "$GUARD" ]; then echo "stage1 already launched ($GUARD exists at $(date)); skipping"; exit 0; fi
for i in $(seq 1 720); do
  if CUDA_VISIBLE_DEVICES=0 python3 -c "import torch; torch.zeros(1).cuda()" >/dev/null 2>&1; then
    echo "CUDA ready at $(date) after $i checks"
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
echo "CUDA not available within ~24h — manual attention needed"
exit 1
