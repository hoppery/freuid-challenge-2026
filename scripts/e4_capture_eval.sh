#!/usr/bin/env bash
# E4 (MIDV-Holo capture data) completion monitor: wait for the run to finish 3 epochs, then eval
# every saved epoch on fantasyid_test (capture proxy, 1295 imgs). Baseline E2-early = 0.5466.
set -u
cd /home/hoppery/ijcai_freuid_chanllenge
PROXY=manifests/fantasyid_test.parquet
RUN=exp_e4_midvholo
PID=239450

for i in $(seq 1 200); do
  m=$(tail -n +2 checkpoints/$RUN/log.csv 2>/dev/null | wc -l)
  [ "$m" -ge 3 ] && { echo ">>> $RUN reached ep2"; break; }
  ps -p $PID >/dev/null 2>&1 || { echo ">>> proc exited (m=$m)"; break; }
  sleep 90
done

echo "================ E4 MIDV-Holo CAPTURE PROXY (fantasyid_test) vs E2-early 0.5466 ================"
echo "--- $RUN in-domain log ---"; cat checkpoints/$RUN/log.csv 2>/dev/null
for ck in epoch0.pt epoch1.pt epoch2.pt best.pt; do
  p="checkpoints/$RUN/$ck"
  [ -f "$p" ] || continue
  out=$(CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src OMP_NUM_THREADS=2 \
        python3 scripts/xeval_fantasyid.py --ckpt "$p" --manifest "$PROXY" 2>/dev/null | tail -1)
  echo "  $ck  ::  $out"
done
echo "================ DONE ================"
