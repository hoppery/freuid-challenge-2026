from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
from freuid.data.schema import COLUMNS, Label, AttackType

ROOT = Path("data/raw/freuid")
_TRAIN_CSV = ROOT / "train_labels.csv"
# RELEASE-DAY OVERRIDE: point at the private test on release via env vars (no code edit needed):
#   FREUID_SAMPLE_SUB=<path to private sample_submission.csv>  FREUID_TEST_DIR=<dir with <id>.jpeg>
_SAMPLE_SUB = Path(os.environ.get("FREUID_SAMPLE_SUB", str(ROOT / "sample_submission.csv")))
# NOTE: the archive uses DOUBLED folders: images live at train/train/<id>.jpeg and
# public_test/public_test/<id>.jpeg, while the CSV image_path says only "train/<id>.jpeg".
_TRAIN_IMG_DIR = ROOT / "train" / "train"
_TEST_DIR = Path(os.environ.get("FREUID_TEST_DIR", str(ROOT / "public_test" / "public_test")))


def build_freuid_manifest() -> pd.DataFrame:
    """Official FREUID train_labels.csv -> unified manifest rows."""
    if not _TRAIN_CSV.exists():
        raise FileNotFoundError(f"{_TRAIN_CSV} missing — unzip the FREUID archive first.")
    df = pd.read_csv(_TRAIN_CSV)
    for c in ("id", "image_path", "label", "type"):
        if c not in df.columns:
            raise ValueError(f"column {c!r} not in {list(df.columns)} — FREUID schema changed")
    out = pd.DataFrame({
        "path": df["id"].astype(str).map(lambda i: str(_TRAIN_IMG_DIR / f"{i}.jpeg")),
        "label": df["label"].astype(int),
        "attack_type": df["label"].map(
            lambda v: AttackType.NONE if int(v) == Label.BONA_FIDE else AttackType.UNKNOWN),
        "doc_type": df["type"].astype(str),
        "source": "freuid",
        "split": "train",
    })
    return out[COLUMNS]


def build_freuid_test_index(existing_only: bool = False) -> pd.DataFrame:
    """Authoritative test ids from sample_submission.csv -> (image_id, path).

    existing_only=True keeps only files present on disk (for smoke tests on a subset).
    The real submission run must have every image present so all ids are covered.
    """
    if not _SAMPLE_SUB.exists():
        raise FileNotFoundError(f"{_SAMPLE_SUB} missing — download sample_submission.csv first.")
    sub = pd.read_csv(_SAMPLE_SUB)
    id_col = "id" if "id" in sub.columns else sub.columns[0]
    idx = pd.DataFrame({
        "image_id": sub[id_col].astype(str),
        "path": sub[id_col].astype(str).map(lambda i: str(_TEST_DIR / f"{i}.jpeg")),
    })
    if existing_only:
        mask = idx["path"].map(lambda p: Path(p).exists())
        idx = idx[mask].reset_index(drop=True)
    return idx
