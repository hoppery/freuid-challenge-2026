#!/usr/bin/env bash
set -u; cd /home/hoppery/ijcai_freuid_chanllenge
P0=354686; P1=354687
ready(){ [ "$(tail -n +2 checkpoints/$1/log.csv 2>/dev/null | wc -l)" -ge 3 ]; }
for i in $(seq 1 400); do
  r0=0;r1=0; ready exp_sam_fold0 && r0=1; ready exp_sam_fold2 && r1=1
  { [ "$r0" = 1 ] && [ "$r1" = 1 ]; } && { echo ">>> both SAM folds >=3ep"; break; }
  [ "$(ps -p $P0,$P1 -o pid= 2>/dev/null | wc -l)" = 0 ] && { echo ">>> both exited"; break; }
  sleep 150
done
echo "===== SAM vs DINOv2 LODO (held-type FREUID, lower=better) ====="
echo "DINOv2 baseline: EGYPT 0.0035 | BENIN 0.0165"
for f in fold0:EGYPT fold2:BENIN; do
  run=exp_sam_${f%%:*}; name=${f##*:}
  echo "--- SAM $name ---"; cat checkpoints/$run/log.csv 2>/dev/null
  best=$(tail -n +2 checkpoints/$run/log.csv 2>/dev/null | awk -F, 'NR==1||$5<m{m=$5} END{print m}')
  echo "  SAM $name BEST = $best"
done
echo "===== DONE ====="
