#!/bin/bash
# Wait for (1) the 6-location IDNet extraction and (2) Iter6 (exp_idnet 5-fold) to finish,
# then rebuild the 10-location manifest, generate LODO folds, and launch Iter7 (exp_idnetall).
# Keeps GPUs busy without manual intervention. Run via nohup.
set -u
cd /home/hoppery/ijcai_freuid_chanllenge
export PATH=/home/hoppery/miniforge3/bin:$PATH
unset PYTHONPATH
LOG=/tmp/iter7_autolaunch.log
echo "=== watcher start $(date) ===" > $LOG

# 1) wait for extraction
while ! grep -q "extract complete" /tmp/idnet_extract6.log 2>/dev/null; do
  sleep 30
done
echo "[$(date +%H:%M:%S)] extraction complete" >> $LOG

# 2) wait for Iter6 sweep to finish (process gone AND 5 checkpoints present)
while pgrep -f "run_sweep.py --configs exp_idnet_fold" >/dev/null 2>&1; do
  sleep 60
done
echo "[$(date +%H:%M:%S)] Iter6 sweep process gone" >> $LOG
# record Iter6 aggregate for analysis
python3 scripts/agg_folds.py exp_idnet >> $LOG 2>&1 || true

# 3) rebuild 10-location manifest
python3 scripts/build_idnet_manifest.py --out manifests/idnet_all.parquet \
    --pos-cap 2500 --fraud-cap 1250 >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] manifest rebuilt" >> $LOG

# 4) generate LODO folds for Iter7
python3 scripts/make_folds.py exp_idnetall >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] folds generated" >> $LOG

# 5) launch Iter7 5-fold sweep on GPU0,1
echo "[$(date +%H:%M:%S)] launching Iter7 sweep" >> $LOG
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/run_sweep.py \
    --configs exp_idnetall_fold0 exp_idnetall_fold1 exp_idnetall_fold2 exp_idnetall_fold3 exp_idnetall_fold4 \
    --gpus 0 1 >> /tmp/iter7_5fold.log 2>&1
python3 scripts/agg_folds.py exp_idnetall >> /tmp/iter7_5fold.log 2>&1
echo "=== Iter7 complete $(date) ===" >> $LOG
