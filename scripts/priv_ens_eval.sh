#!/bin/bash
# private architectural-diversity: ViT-L FDA + ConvNeXt FDA ensemble on LODO folds.
cd /home/hoppery/ijcai_freuid_chanllenge
D=_transfer/src
# EGYPT dumps (parallel)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$D OMP_NUM_THREADS=2 python3 scripts/score_dump.py --ckpt checkpoints/exp_fdab12_egypt/best.pt --holdout "EGYPT/DL" --out /tmp/vl_eg.csv > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$D OMP_NUM_THREADS=2 python3 scripts/score_dump.py --ckpt checkpoints/exp_cnxfda_egypt/best.pt --holdout "EGYPT/DL" --out /tmp/cx_eg.csv > /dev/null 2>&1 &
wait
# BENIN dumps (parallel)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$D OMP_NUM_THREADS=2 python3 scripts/score_dump.py --ckpt checkpoints/exp_fdab12_benin/best.pt --holdout "BENIN/DL" --out /tmp/vl_bn.csv > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$D OMP_NUM_THREADS=2 python3 scripts/score_dump.py --ckpt checkpoints/exp_cnxfda_benin/best.pt --holdout "BENIN/DL" --out /tmp/cx_bn.csv > /dev/null 2>&1 &
wait
python3 - <<'PY'
import pandas as pd, numpy as np, sys
from scipy.stats import rankdata
sys.path.insert(0,'src'); from freuid.metrics import compute_metrics
def F(f_vl,f_cx,lbl,vlbase,cxbase):
    vl=pd.read_csv(f_vl).rename(columns={'score':'vl'}); cx=pd.read_csv(f_cx).rename(columns={'score':'cx'})
    d=vl.merge(cx[['path','cx']],on='path'); y=d['label'].to_numpy()
    fvl=compute_metrics(y,d['vl'].to_numpy())['freuid_score']; fcx=compute_metrics(y,d['cx'].to_numpy())['freuid_score']
    corr=pd.Series(d['vl']).corr(pd.Series(d['cx']),method='spearman')
    best=(fvl,'50/50')
    for w in [0.3,0.4,0.5,0.6,0.7]:
        ens=(1-w)*rankdata(d['vl'])+w*rankdata(d['cx'])
        fe=compute_metrics(y,ens)['freuid_score']
        if fe<best[0]: best=(fe,f'{w:.0%}cx')
    print(f"  {lbl}: ViT-L={fvl:.4f} ConvNeXt={fcx:.4f} | 상관={corr:.4f} | ★앙상블 best={best[0]:.4f} ({best[1]})")
    return best[0]
print("=== private 아키텍처-다양성 앙상블 (ViT-L FDA + ConvNeXt FDA) ===")
e=F('/tmp/vl_eg.csv','/tmp/cx_eg.csv','EGYPT',0.0027,0.0155)
b=F('/tmp/vl_bn.csv','/tmp/cx_bn.csv','BENIN',0.0140,0.0023)
print(f"  >>> 앙상블 mean={ (e+b)/2:.4f}  (ViT-L FDA 단독 mean 0.0084)")
PY
echo "PRIV ENS DONE"
