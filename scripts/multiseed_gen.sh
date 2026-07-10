#!/usr/bin/env bash
# When s45,s46 finish, generate ep2 + best CSVs (candidates for the 0.043-recovery lottery). NO auto-submit.
set -u; cd /home/hoppery/ijcai_freuid_chanllenge
S45=66392; S46=66393
gen(){ # $1=ckpt-dir-or-file label $2=out
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=_transfer/src OMP_NUM_THREADS=2 python3 _transfer/scripts/ensemble_infer.py --ckpts "$1" --out "$2" 2>&1 | grep -E "wrote|predicted" | tail -1
}
until ! ps -p $S45 >/dev/null 2>&1 && ! ps -p $S46 >/dev/null 2>&1; do sleep 180; done
echo "=== s45 log ==="; cat checkpoints/exp_prod896_large_s45/log.csv 2>/dev/null
echo "=== s46 log ==="; cat checkpoints/exp_prod896_large_s46/log.csv 2>/dev/null
# best.pt of each seed (the candidates). also make per-epoch dirs for ep2.
for S in s45 s46; do
  D=checkpoints/exp_prod896_large_$S
  [ -f $D/best.pt ] && gen $D submission_large_${S}_best.csv
  if [ -f $D/epoch2.pt ]; then mkdir -p ${D}_ep2; cp $D/epoch2.pt ${D}_ep2/best.pt; cp $D/config.yaml ${D}_ep2/config.yaml; gen ${D}_ep2 submission_large_${S}_ep2.csv; fi
done
echo "===== 0.043-recovery candidate CSVs ready ====="
ls submission_large_s45*.csv submission_large_s46*.csv 2>/dev/null
echo "===== DONE ====="
