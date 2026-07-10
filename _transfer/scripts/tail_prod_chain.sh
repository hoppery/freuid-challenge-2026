#!/bin/bash
# Chain: wait for tail_margin 5-fold → if good (LODO-CV < bce 0.0286) → retrain production
# with tail_margin → build the public submission CSV. Autonomous (independent of tick timing).
set -u
cd /home/hoppery/ijcai_freuid_chanllenge
export PATH=/home/hoppery/miniforge3/bin:$PATH
unset PYTHONPATH
LOG=/tmp/tail_prod_chain.log
BCE_REF=0.0286
echo "=== chain start $(date) ===" > $LOG

# 1) wait for the tail 5-fold to finish
while ! grep -q "TAIL5FOLD COMPLETE" /tmp/tail5fold.log 2>/dev/null; do sleep 60; done
CV=$(grep "LODO-CV FREUID" /tmp/tail5fold.log | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
echo "[$(date +%H:%M:%S)] tail 5-fold LODO-CV=$CV (bce ref $BCE_REF)" >> $LOG

# 2) good gate
good=$(python3 -c "
try:
    print(1 if float('$CV') < $BCE_REF else 0)
except Exception:
    print(0)
")
if [ "$good" != "1" ]; then
  echo "[$(date +%H:%M:%S)] NOT good (CV=$CV >= $BCE_REF) → keep bce production, no retrain" >> $LOG
  echo "=== CHAIN COMPLETE (no retrain) $(date) ===" >> $LOG
  exit 0
fi

# 3) retrain production with tail_margin (needs a free GPU; 5-fold done so GPU0 is free)
echo "[$(date +%H:%M:%S)] GOOD → retrain production (exp_prod896_tail) on GPU0" >> $LOG
CUDA_VISIBLE_DEVICES=0 python3 -m freuid.train --config configs/exp_prod896_tail.yaml \
    > checkpoints/exp_prod896_tail.train.log 2>&1
PB=$(python3 -c "import torch;c=torch.load('checkpoints/exp_prod896_tail/best.pt',map_location='cpu',weights_only=False);print(round(c['metrics']['freuid_score'],4))" 2>/dev/null)
echo "[$(date +%H:%M:%S)] production(tail) in-domain best=$PB" >> $LOG

# 4) build the public submission CSV from the tail production model
CUDA_VISIBLE_DEVICES=0 python3 scripts/ensemble_infer.py \
    --ckpts checkpoints/exp_prod896_tail --out submission_prod896_tail.csv >> $LOG 2>&1
echo "=== CHAIN COMPLETE (submission_prod896_tail.csv ready) $(date) ===" >> $LOG
