"""E9 capture manifest = E6 + MIDV-2020 captured-genuine (NEW source, 10 new doc types, 2 modalities).
MIDV-2020 photo (phone) + scan_upright = 2000 genuine captured images across 10 European/Asian ID types
absent from E6. Added at the same x2 upweight as the other captured-genuine sources (per build_capgen_e6).
"""
import glob
import pandas as pd
from collections import Counter

UPW = 2
COLS = ["path", "label", "attack_type", "doc_type", "source", "split", "is_digital"]
BASE = "manifests/freuid_fid_capgen_e6.parquet"

def main():
    e6 = pd.read_parquet(BASE)[COLS]
    rows = []
    for modality, root in [("photo", "data/raw/midv2020/photo_ex/images"),
                           ("scan",  "data/raw/midv2020/dataset/images")]:
        for p in sorted(glob.glob(f"{root}/*/*.jpg")):
            dtype = p.split("/images/")[1].split("/")[0]
            rows.append({"path": p, "label": 0, "attack_type": "none",
                         "doc_type": f"MIDV2020/{dtype}", "source": "midv2020",
                         "split": "train", "is_digital": False})
    m2020 = pd.DataFrame(rows)[COLS]
    out = pd.concat([e6] + [m2020] * UPW, ignore_index=True)
    out.to_parquet("manifests/freuid_fid_capgen_e9.parquet")
    print(f"E9 total {len(out)} | E6 {len(e6)} + MIDV2020 {len(m2020)} x{UPW}")
    print("MIDV2020 doc_types:", sorted(m2020['doc_type'].unique()))
    print("sources:", dict(Counter(out['source'])))

if __name__ == "__main__":
    main()
