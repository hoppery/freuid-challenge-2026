"""Rank-average multiple submission CSVs into a diverse ensemble. Rank-averaging is robust to per-model
score-scale differences (better than raw mean for heterogeneous architectures). Only ranks the rows whose
image is on disk (real predictions); the 0.5-filled private rows are left at 0.5.

Usage: python3 scripts/rank_ensemble.py --out submission_ens.csv a.csv b.csv c.csv
"""
from __future__ import annotations
import argparse, pandas as pd, numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("csvs", nargs="+")
    a = ap.parse_args()

    dfs = [pd.read_csv(c) for c in a.csvs]
    base = dfs[0][["id"]].copy()
    # rows with a real (non-0.5) prediction in the FIRST csv define the public set
    real_mask = dfs[0]["label"] != 0.5
    ids_real = dfs[0].loc[real_mask, "id"]

    rank_sum = np.zeros(len(ids_real))
    for c, df in zip(a.csvs, dfs):
        df = df.set_index("id").loc[ids_real, "label"]
        r = df.rank(method="average").to_numpy() / len(df)   # normalized rank in [0,1]
        rank_sum += r
        print(f"  {c}: real preds used = {len(df)}")
    ens_rank = rank_sum / len(a.csvs)

    out = pd.read_csv(a.csvs[0]).set_index("id")
    out.loc[ids_real, "label"] = ens_rank          # real rows = ensemble rank score
    out = out.reset_index()
    out.to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(out)} rows, {len(ids_real)} ensembled)")


if __name__ == "__main__":
    main()
