#!/bin/bash
# Multiseed LODO ensemble eval: wait egypt dumps -> fuse -> benin dumps -> fuse. Reports s42 vs s42+s43.
cd /home/hoppery/ijcai_freuid_chanllenge

fuse() {  # $1=fold label $2=s42.csv $3=s43.csv $4=single_baseline
python3 - "$1" "$2" "$3" "$4" <<'PY'
import sys, pandas as pd, numpy as np
from scipy.stats import rankdata
sys.path.insert(0,'src')
from freuid.metrics import compute_metrics
lbl,f42,f43,base=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]
a=pd.read_csv(f42).rename(columns={'score':'s42'}); b=pd.read_csv(f43).rename(columns={'score':'s43'})
d=a.merge(b[['path','s43']],on='path'); y=d['label'].to_numpy()
def F(c): m=compute_metrics(y,d[c].to_numpy()); return m['freuid_score']
d['ens']=rankdata(d['s42'])+rankdata(d['s43'])
mens=compute_metrics(y,d['ens'].to_numpy())
print(f"  {lbl}: s42={F('s42'):.4f}  s43={F('s43'):.4f}  ENS(s42+s43)={mens['freuid_score']:.4f}  (single baseline {base})")
PY
}

# wait egypt dumps
for i in $(seq 1 40); do [ -f /tmp/eg_s42.csv ] && [ -f /tmp/eg_s43.csv ] && break; sleep 15; done
echo "=== EGYPT 멀티시드 ==="
fuse "EGYPT" /tmp/eg_s42.csv /tmp/eg_s43.csv 0.0027

# benin dumps (GPUs now free)
setsid env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=_transfer/src OMP_NUM_THREADS=2 python3 scripts/score_dump.py --ckpt checkpoints/exp_fdab12_benin/best.pt --holdout "BENIN/DL" --out /tmp/bn_s42.csv < /dev/null > .reports/dump_bn_s42.out 2>&1 &
setsid env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=_transfer/src OMP_NUM_THREADS=2 python3 scripts/score_dump.py --ckpt checkpoints/exp_fdab12_benin_s43/best.pt --holdout "BENIN/DL" --out /tmp/bn_s43.csv < /dev/null > .reports/dump_bn_s43.out 2>&1 &
for i in $(seq 1 40); do [ -f /tmp/bn_s42.csv ] && [ -f /tmp/bn_s43.csv ] && break; sleep 15; done
echo "=== BENIN 멀티시드 ==="
fuse "BENIN" /tmp/bn_s42.csv /tmp/bn_s43.csv 0.0140
echo "MULTISEED EVAL DONE"
