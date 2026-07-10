"""Build a capture-axis manifest = E2 base (freuid_fid_upweighted) + MIDV-Holo CAPTURED data.

MIDV-Holo = real smartphone-captured holographic passports: 790 GENUINE views (origins, ~26 docs ×
~30 capture conditions) + ~17k captured FORGERIES (photo_replacement / photo_holo_copy /
copy_without_holo / pseudo_holo_copy). This is the FIRST real CAPTURED data on the attack side (the
E2 recipe only had FantasyID-physical genuine + GenAI attacks). The capture axis's only proven lever
is real physical data, so we add it here.

Two manifests are written:
  freuid_fid_midvholo_g.parquet   = E2 base + MIDV genuine (upweighted ×UPW)            [genuine-only]
  freuid_fid_midvholo_gf.parquet  = E2 base + MIDV genuine (×UPW) + fraud (FR/doc frames) [genuine+fraud]
Genuine is upweighted (small but precious); fraud is frame-subsampled per document to avoid the 17k
holographic-copy frames swamping the mix / overfitting a hologram-specific artifact.
"""
from __future__ import annotations
import glob, os, re
import pandas as pd

BASE = "manifests/freuid_fid_upweighted.parquet"
ROOT = "data/raw/midv_holo/ex/images"
UPW = 4           # genuine upweight factor
FR_PER_DOC = 3    # fraud frames kept per document clip
COLS = ["path", "label", "attack_type", "doc_type", "source", "split", "is_digital"]


def _row(path, label, attack_type):
    return {"path": path, "label": label, "attack_type": attack_type,
            "doc_type": "MIDVHOLO/passport", "source": "midvholo",
            "split": "train", "is_digital": False}


def _docid(p):
    # .../<category>/passport/<pspXX_YY_ZZ>/img_NNNN.jpg -> the psp clip id
    m = re.search(r"/(psp\d+_\d+_\d+)/", p)
    return m.group(1) if m else os.path.dirname(p)


def main():
    base = pd.read_parquet(BASE)
    base = base[COLS]

    gen = sorted(glob.glob(f"{ROOT}/origins/**/*.jpg", recursive=True))
    gen_rows = [_row(p, 0, "none") for p in gen]
    gen_df = pd.DataFrame(gen_rows)
    gen_up = pd.concat([gen_df] * UPW, ignore_index=True)

    fraud = sorted(glob.glob(f"{ROOT}/fraud/**/*.jpg", recursive=True))
    # subsample FR_PER_DOC evenly-spaced frames per (category, document)
    by = {}
    for p in fraud:
        cat = p.split("/fraud/")[1].split("/")[0]
        by.setdefault((cat, _docid(p)), []).append(p)
    fr_keep = []
    for k, frames in by.items():
        frames = sorted(frames)
        step = max(1, len(frames) // FR_PER_DOC)
        fr_keep += frames[::step][:FR_PER_DOC]
    # MIDV-Holo forgeries are real captured (print/photo-recapture) attacks -> "print_capture".
    # keep the holographic subcategory in doc_type for traceability.
    fr_rows = []
    for p in fr_keep:
        cat = p.split("/fraud/")[1].split("/")[0]
        r = _row(p, 1, "print_capture"); r["doc_type"] = f"MIDVHOLO/{cat}"
        fr_rows.append(r)
    fr_df = pd.DataFrame(fr_rows)

    # verify files exist (extraction had a benign tar-trailer error)
    for name, df in [("genuine", gen_df), ("fraud", fr_df)]:
        miss = (~df["path"].map(os.path.exists)).sum()
        print(f"MIDV {name}: {len(df)} rows, {miss} missing")

    g_only = pd.concat([base, gen_up], ignore_index=True)
    g_only.to_parquet("manifests/freuid_fid_midvholo_g.parquet")
    gf = pd.concat([base, gen_up, fr_df], ignore_index=True)
    gf.to_parquet("manifests/freuid_fid_midvholo_gf.parquet")

    for name, df in [("freuid_fid_midvholo_g", g_only), ("freuid_fid_midvholo_gf", gf)]:
        midv = df[df["source"] == "midvholo"]
        print(f"{name}: n={len(df)}  labels={dict(df['label'].value_counts())}  "
              f"midv={len(midv)} (g={int((midv['label']==0).sum())}, f={int((midv['label']==1).sum())})")


if __name__ == "__main__":
    main()
