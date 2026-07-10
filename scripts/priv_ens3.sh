#!/bin/bash
# private 3-member per-fold ensemble: 896 ViT-L + 896 ConvNeXt + 1120 member. Finds best per-fold combo.
cd /home/hoppery/ijcai_freuid_chanllenge
D=_transfer/src
dump(){ CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$D OMP_NUM_THREADS=2 python3 scripts/score_dump.py --ckpt "$2" --holdout "$3" --out "$4" >/dev/null 2>&1; }

# EGYPT trio
dump 0 checkpoints/exp_fdab12_egypt/best.pt   "EGYPT/DL" /tmp/e_vl896.csv &
dump 1 checkpoints/exp_cnxfda_egypt/best.pt   "EGYPT/DL" /tmp/e_cx896.csv &
wait
dump 0 checkpoints/exp_fda1120_egypt/best.pt  "EGYPT/DL" /tmp/e_vl1120.csv &
# BENIN trio (GPU1 병렬)
dump 1 checkpoints/exp_fdab12_benin/best.pt   "BENIN/DL" /tmp/b_vl896.csv &
wait
dump 0 checkpoints/exp_cnxfda_benin/best.pt     "BENIN/DL" /tmp/b_cx896.csv &
dump 1 checkpoints/exp_cnxfda1120_benin/best.pt "BENIN/DL" /tmp/b_cx1120.csv &
wait

python3 - <<'PY'
import pandas as pd, numpy as np, sys, itertools
from scipy.stats import rankdata
sys.path.insert(0,'src'); from freuid.metrics import compute_metrics
def fold(name, files):
    dfs=[pd.read_csv(f).rename(columns={'score':n}) for n,f in files.items()]
    d=dfs[0]
    for x in dfs[1:]: d=d.merge(x[['path',x.columns[-1]]],on='path')
    y=d['label'].to_numpy(); names=list(files)
    print(f"  [{name}] 단독:", {n:round(compute_metrics(y,d[n].to_numpy())['freuid_score'],4) for n in names})
    best=(9,'');
    # 모든 부분집합 + 균등 rank 앙상블
    for r in range(2,len(names)+1):
        for combo in itertools.combinations(names,r):
            ens=sum(rankdata(d[n]) for n in combo)
            fe=compute_metrics(y,ens)['freuid_score']
            if fe<best[0]: best=(fe,'+'.join(combo))
    print(f"  [{name}] ★best 앙상블 = {best[0]:.4f} ({best[1]})")
    return best[0]
print("=== private 3-멤버 앙상블 (현 best: EGYPT 0.0002 / BENIN 0.0062 / mean 0.0032) ===")
e=fold("EGYPT", {'vl896':'/tmp/e_vl896.csv','cx896':'/tmp/e_cx896.csv','vl1120':'/tmp/e_vl1120.csv'})
b=fold("BENIN", {'vl896':'/tmp/b_vl896.csv','cx896':'/tmp/b_cx896.csv','cx1120':'/tmp/b_cx1120.csv'})
print(f"  >>> 새 mean = {(e+b)/2:.4f}")
PY
echo "PRIV ENS3 DONE"
