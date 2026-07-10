#!/usr/bin/env bash
set -u; cd /home/hoppery/ijcai_freuid_chanllenge
P0=364259; P1=364260
ready(){ [ "$(tail -n +2 checkpoints/$1/log.csv 2>/dev/null | wc -l)" -ge 3 ]; }
for i in $(seq 1 400); do
  r0=0;r1=0; ready exp_sam_fold3 && r0=1; ready exp_sam_fold4 && r1=1
  { [ "$r0" = 1 ] && [ "$r1" = 1 ]; } && { echo ">>> both SAM folds >=3ep"; break; }
  [ "$(ps -p $P0,$P1 -o pid= 2>/dev/null | wc -l)" = 0 ] && { echo ">>> both exited"; break; }
  sleep 150
done
echo "===== SAM 5-fold completion vs DINOv2 (held FREUID, lower=better) ====="
echo "DINOv2: EGYPT 0.0035 MAURITIUS 0.0052 BENIN 0.0165 GUINEA 0.0216 MOZ 0.0602 (mean 0.0214)"
echo "SAM so far: EGYPT 0.00211(better) BENIN 0.0494(worse)"
for f in fold3:MOZAMBIQUE fold4:MAURITIUS; do
  run=exp_sam_${f%%:*}; name=${f##*:}
  echo "--- SAM $name ---"; cat checkpoints/$run/log.csv 2>/dev/null
  best=$(tail -n +2 checkpoints/$run/log.csv 2>/dev/null | awk -F, 'NR==1||$5<m{m=$5} END{print m}')
  echo "  SAM $name BEST = $best"
done
echo "===== DONE ====="
