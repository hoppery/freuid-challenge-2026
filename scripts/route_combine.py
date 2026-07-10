"""Combine domain-router + capture-card + field-tamper-card CSVs into a routed submission.
Each row is scored by the card matching its predicted DOMAIN:
  P(captured) > threshold -> capture card ; else -> field-tamper card.
Modes: hard (raw score of assigned card), ranknorm (percentile of assigned card, scale-safe),
soft (P*capture + (1-P)*ft). If --labels given (a parquet with path/id + fraud label), reports
FREUID for routed vs single cards vs oracle (route by domain guess) to validate the hedge.

CSV format for --router/--capture/--ft: columns id,label  (freuid.infer / infer_tta_ens output).
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _load(p, name):
    d = pd.read_csv(p)[["id", "label"]].rename(columns={"label": name})
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", required=True, help="csv id,label=P(captured)")
    ap.add_argument("--capture", required=True)
    ap.add_argument("--ft", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--mode", choices=["hard", "ranknorm", "soft"], default="hard")
    ap.add_argument("--labels", default=None, help="optional parquet with id,label(fraud) for FREUID eval")
    a = ap.parse_args()

    r = _load(a.router, "pcap").merge(_load(a.capture, "cap"), on="id").merge(_load(a.ft, "ft"), on="id")
    N = len(r)
    is_cap = (r["pcap"] > a.threshold).to_numpy()
    capR = rankdata(r["cap"]) / N
    ftR = rankdata(r["ft"]) / N
    if a.mode == "hard":
        final = np.where(is_cap, r["cap"], r["ft"])
    elif a.mode == "ranknorm":
        final = np.where(is_cap, capR, ftR)
    else:  # soft
        p = r["pcap"].to_numpy()
        final = p * r["cap"].to_numpy() + (1 - p) * r["ft"].to_numpy()
    r["label"] = final
    r[["id", "label"]].to_csv(a.out, index=False)
    print(f"wrote {a.out} ({N} rows) | routed captured={int(is_cap.sum())} bd={int((~is_cap).sum())} "
          f"| mode={a.mode} thr={a.threshold}")

    if a.labels:
        import sys
        sys.path.insert(0, "src")
        from freuid.metrics import compute_metrics
        lab = pd.read_parquet(a.labels)
        idcol = "id" if "id" in lab.columns else ("image_id" if "image_id" in lab.columns else None)
        if idcol is None:  # derive id from path stem
            lab["id"] = lab["path"].str.split("/").str[-1].str.split(".").str[0]
            idcol = "id"
        lab = lab.rename(columns={idcol: "id"})[["id", "label"]].rename(columns={"label": "y"})
        m = r.merge(lab, on="id")
        y = m["y"].to_numpy()
        def fr(s): return compute_metrics(y, s)["freuid_score"]
        print(f"\n=== FREUID on labeled mix (n={len(m)}, fraud {int(y.sum())}/{len(m)}) ===")
        print(f"  single capture           : {fr(m['cap']):.4f}")
        print(f"  single field-tamper      : {fr(m['ft']):.4f}")
        ic = (m['pcap'] > a.threshold).to_numpy()
        capRm = rankdata(m['cap']) / len(m); ftRm = rankdata(m['ft']) / len(m)
        print(f"  ROUTED hard              : {fr(np.where(ic, m['cap'], m['ft'])):.4f}")
        print(f"  ROUTED ranknorm          : {fr(np.where(ic, capRm, ftRm)):.4f}")
        pp = m['pcap'].to_numpy()
        print(f"  ROUTED soft              : {fr(pp*m['cap'].to_numpy()+(1-pp)*m['ft'].to_numpy()):.4f}")
        print(f"  router acc (if is_digital in labels-source): route captured={int(ic.sum())}/{len(m)}")


if __name__ == "__main__":
    main()
