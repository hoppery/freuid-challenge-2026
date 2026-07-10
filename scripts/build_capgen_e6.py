"""E6 capture manifest = E5 captured-genuine + MORE diverse captured sources (per user: expand sources).

E5 (MIDV-500 18 types + MIDV-Holo) beat the ceiling (0.4548). The lever is captured-genuine DIVERSITY,
and upweight is saturated (×4 ≯ ×2), so add more SOURCES:
  + MIDV-2019 (3 captured doc types, smartphone video frames, .tif)
  + BID (Brazilian IDs: CNH/RG/CPF, 8 doc types, real document photos — '_in.jpg' only, masks excluded),
    subsampled ~SUB_BID across the 8 types.
All captured genuine (label 0, is_digital False), ×2 upweight (E5-optimal), NO fraud.
"""
from __future__ import annotations
import glob, os, random
import pandas as pd

BASE = "manifests/freuid_fid_upweighted.parquet"
UPW = 2
SUB_BID = 3200           # BID subsample (diverse across 8 types)
COLS = ["path", "label", "attack_type", "doc_type", "source", "split", "is_digital"]
random.seed(42)


def _row(path, doc_type, source):
    return {"path": path, "label": 0, "attack_type": "none", "doc_type": doc_type,
            "source": source, "split": "train", "is_digital": False}


def main():
    base = pd.read_parquet(BASE)[COLS]

    m500 = sorted(glob.glob("data/raw/midv500/ex/*/images/*/*.tif"))
    r500 = [_row(p, "MIDV500/" + p.split("/ex/")[1].split("/")[0], "midv500") for p in m500]

    mholo = sorted(glob.glob("data/raw/midv_holo/ex/images/origins/**/*.jpg", recursive=True))
    rholo = [_row(p, "MIDVHOLO/passport", "midvholo") for p in mholo]

    m2019 = sorted(glob.glob("data/raw/midv2019/ex/*/images/*/*.tif"))
    r2019 = [_row(p, "MIDV2019/" + p.split("/ex/")[1].split("/")[0], "midv2019") for p in m2019]

    # BID: real document PHOTOS only (_in.jpg); exclude _gt_segmentation masks. Subsample per doc-type.
    bid_all = [p for p in glob.glob("data/raw/bid/BID Dataset/*/*_in.jpg")]
    by_type = {}
    for p in bid_all:
        by_type.setdefault(p.split("/")[-2], []).append(p)
    per = max(1, SUB_BID // max(1, len(by_type)))
    bid_keep = []
    for t, ps in by_type.items():
        random.shuffle(ps)
        bid_keep += ps[:per]
    rbid = [_row(p, "BID/" + p.split("/")[-2], "bid") for p in bid_keep]

    cap = pd.DataFrame(r500 + rholo + r2019 + rbid)
    miss = (~cap["path"].map(os.path.exists)).sum()
    from collections import Counter
    print("captured-genuine sources:", dict(Counter(cap["source"])), "missing=", miss,
          "doc_types=", cap["doc_type"].nunique())

    out = pd.concat([base] + [cap] * UPW, ignore_index=True)
    out.to_parquet("manifests/freuid_fid_capgen_e6.parquet")
    print(f"freuid_fid_capgen_e6: n={len(out)} labels={dict(out['label'].value_counts())} "
          f"capgen_unique={len(cap)} (×{UPW})")


if __name__ == "__main__":
    main()
