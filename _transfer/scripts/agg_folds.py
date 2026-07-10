"""Aggregate 5 LODO folds into a single CV estimate (mean ± std of FREUID Score).
Usage: python3 scripts/agg_folds.py <base_config_stem>"""
import sys
import numpy as np
import torch
from freuid.metrics import freuid_score

DOC_TYPES = ["EGYPT/DL", "GUINEA/DL", "BENIN/DL", "MOZAMBIQUE/DL", "MAURITIUS/ID"]


def main():
    base = sys.argv[1]
    rows = []
    for i, dt in enumerate(DOC_TYPES):
        try:
            m = torch.load(f"checkpoints/{base}_fold{i}/best.pt",
                           map_location="cpu", weights_only=False)["metrics"]
            fs = m.get("freuid_score", freuid_score(m["audet"], m["apcer_at_1pct_bpcer"]))
            rows.append((dt, fs, m["audet"], m["apcer_at_1pct_bpcer"], m["roc_auc"]))
        except FileNotFoundError:
            rows.append((dt, None, None, None, None))
    print(f"{'fold (held-out)':16s} {'FREUID':>7s} {'AuDET':>7s} {'APCER@1%':>9s} {'AUC':>7s}")
    fss = []
    for dt, fs, au, ap, auc in rows:
        if fs is None:
            print(f"{dt:16s}  NO CKPT"); continue
        fss.append(fs)
        print(f"{dt:16s} {fs:7.4f} {au:7.4f} {ap:9.4f} {auc:7.4f}")
    if fss:
        print(f"\nLODO-CV FREUID = {np.mean(fss):.4f} ± {np.std(fss):.4f}  (n={len(fss)}/5 folds)")


if __name__ == "__main__":
    main()
