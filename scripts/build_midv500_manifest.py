"""E5 capture manifest = E2 base + MAX-diversity real captured GENUINE.

E4g showed MIDV-Holo's NARROW genuine (holographic passports, 1 type) doesn't tighten the FantasyID
genuine distribution. MIDV-500 = 4214 real smartphone-captured genuine across 18 DIVERSE doc types
(IDs/licenses/passports, many countries) × 10 capture conditions — directly targets that weakness.
Combine MIDV-500 + MIDV-Holo genuine (both captured, label 0), upweighted, onto the E2 base. NO fraud
(E4: holographic fraud hurt the proxy). Goal: diverse captured-genuine tightens the op-point (proxy weak spot).
"""
from __future__ import annotations
import glob, os
import pandas as pd

BASE = "manifests/freuid_fid_upweighted.parquet"
UPW = 2
COLS = ["path", "label", "attack_type", "doc_type", "source", "split", "is_digital"]


def _row(path, doc_type, source):
    return {"path": path, "label": 0, "attack_type": "none", "doc_type": doc_type,
            "source": source, "split": "train", "is_digital": False}


def main():
    base = pd.read_parquet(BASE)[COLS]

    # MIDV-500: data/raw/midv500/ex/<NN_xxx>/images/<COND>/<F>.tif
    m500 = sorted(glob.glob("data/raw/midv500/ex/*/images/*/*.tif"))
    r500 = [_row(p, "MIDV500/" + p.split("/ex/")[1].split("/")[0], "midv500") for p in m500]
    df500 = pd.DataFrame(r500)

    # MIDV-Holo genuine (already extracted)
    mholo = sorted(glob.glob("data/raw/midv_holo/ex/images/origins/**/*.jpg", recursive=True))
    rholo = [_row(p, "MIDVHOLO/passport", "midvholo") for p in mholo]
    dfholo = pd.DataFrame(rholo)

    cap = pd.concat([df500, dfholo], ignore_index=True)
    miss = (~cap["path"].map(os.path.exists)).sum()
    print(f"captured-genuine: midv500={len(df500)} midvholo={len(dfholo)} total={len(cap)} missing={miss}")

    cap_up = pd.concat([cap] * UPW, ignore_index=True)
    out = pd.concat([base, cap_up], ignore_index=True)
    out.to_parquet("manifests/freuid_fid_capgen.parquet")
    cg = out[out["source"].isin(["midv500", "midvholo"])]
    print(f"freuid_fid_capgen: n={len(out)} labels={dict(out['label'].value_counts())} "
          f"capgen={len(cg)} (×{UPW} upweight) doc_types={cg['doc_type'].nunique()}")


if __name__ == "__main__":
    main()
