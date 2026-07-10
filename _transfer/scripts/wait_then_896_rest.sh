#!/bin/bash
# After the 896 validation (folds 0,2) finishes, run the remaining folds 1,3,4 and
# aggregate the full 5-fold LODO-CV at 896. Keeps GPUs busy to complete the winning run.
set -u
cd /home/hoppery/ijcai_freuid_chanllenge
export PATH=/home/hoppery/miniforge3/bin:$PATH
unset PYTHONPATH
LOG=/tmp/iter8_896_rest.log
echo "=== watcher start $(date) ===" > $LOG
# wait for validation (folds 0,2) to complete
while ! grep -q "896 validation COMPLETE" /tmp/val896.log 2>/dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] validation (folds 0,2) done; launching folds 1,3,4" >> $LOG
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/run_sweep.py \
    --configs exp_fda896_fold1 exp_fda896_fold3 exp_fda896_fold4 --gpus 0 1 >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] folds 1,3,4 done; aggregating full 5-fold" >> $LOG
python3 scripts/agg_folds.py exp_fda896 >> $LOG 2>&1
echo "=== Iter8 896 5-fold COMPLETE $(date) ===" >> $LOG
