#!/usr/bin/env bash
set -u; cd /home/hoppery/ijcai_freuid_chanllenge
P0=395506; P1=395507
rdy(){ [ "$(tail -n +2 checkpoints/$1/log.csv 2>/dev/null | wc -l)" -ge 3 ]; }
for i in $(seq 1 400); do
  r0=0;r1=0; rdy exp_sam02_fold1 && r0=1; rdy exp_sam02_fold2 && r1=1
  { [ "$r0" = 1 ] && [ "$r1" = 1 ]; } && { echo ">>> both >=3ep"; break; }
  [ "$(ps -p $P0,$P1 -o pid= 2>/dev/null | wc -l)" = 0 ] && { echo ">>> both exited"; break; }
  sleep 150
done
echo "===== SAM rho=0.02 SALVAGE: the 2 collapse folds vs DINOv2 & rho=0.05 ====="
echo "DINOv2: GUINEA 0.0216 BENIN 0.0165 | rho0.05(collapsed): GUINEA 0.2517 BENIN 0.0494"
for f in fold1:GUINEA fold2:BENIN; do
  run=exp_sam02_${f%%:*}; name=${f##*:}
  echo "--- rho0.02 $name ---"; cat checkpoints/$run/log.csv 2>/dev/null
  echo "  rho0.02 $name BEST = $(tail -n +2 checkpoints/$run/log.csv 2>/dev/null | awk -F, 'NR==1||$5<m{m=$5} END{print m}')"
done
echo "===== DONE ====="
