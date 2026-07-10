#!/usr/bin/env bash
set -u; cd /home/hoppery/ijcai_freuid_chanllenge
P0=341832; P1=341833
# report once both have >=4 epochs (trend clear) or both procs exit (early-stop)
ready(){ [ "$(tail -n +2 checkpoints/$1/log.csv 2>/dev/null | wc -l)" -ge 4 ]; }
for i in $(seq 1 320); do
  r0=0;r1=0; ready exp_eva02_benin && r0=1; ready exp_aimv2_benin && r1=1
  { [ "$r0" = 1 ] && [ "$r1" = 1 ]; } && { echo ">>> both backbones >=4ep"; break; }
  [ "$(ps -p $P0,$P1 -o pid= 2>/dev/null | wc -l)" = 0 ] && { echo ">>> both procs exited"; break; }
  sleep 120
done
echo "===== BACKBONE SCREEN: BENIN-held LODO (held-type FREUID, lower=better) ====="
echo "DINOv2 baseline BENIN = 0.0165 (DINOv3 was 0.1736 = rejected)"
for r in exp_eva02_benin:EVA-02 exp_aimv2_benin:AIMv2; do
  run=${r%%:*}; name=${r##*:}
  echo "--- $name ($run) ---"; cat checkpoints/$run/log.csv 2>/dev/null
  best=$(tail -n +2 checkpoints/$run/log.csv 2>/dev/null | awk -F, 'NR==1||$5<m{m=$5} END{print m}')
  echo "  $name BEST BENIN FREUID = $best"
done
echo "===== DONE ====="
