"""Adapter: FantasyID (Idiap) -> unified manifest.

FantasyID = fantasy ID cards, 10-country styles, bona-fide (printed+captured) vs
GenAI manipulations (face-swap + text-inpaint). Directly relevant to FREUID's
GenAI-digital-edit axis. Layout: data/raw/fantasyid/FantasyID/{train,test}.csv with
columns path,is_attack,attack_type and images under train|test/{attack,bonafide}/...
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from freuid.data.schema import COLUMNS, Label, AttackType

ROOT = Path("data/raw/fantasyid/FantasyID")

# FantasyID attack_type -> our taxonomy. face/face_text/text are all GenAI digital edits.
_ATTACK_MAP = {
    "none": AttackType.NONE,
    "face": AttackType.GENAI_DIGITAL,
    "text": AttackType.GENAI_DIGITAL,
    "face_text": AttackType.GENAI_DIGITAL,
}


def _doc_type(path: str) -> str:
    """Country prefix from the filename (e.g. 'arabic-003_03.jpg' -> 'fid_arabic')."""
    stem = Path(path).name
    country = stem.split("-")[0].split("_")[0]
    return f"fid_{country.lower()}"


def build_fantasyid_manifest() -> pd.DataFrame:
    frames = []
    for split in ("train", "test"):
        csv = ROOT / f"{split}.csv"
        if not csv.exists():
            raise FileNotFoundError(f"{csv} missing — download FantasyID first.")
        df = pd.read_csv(csv)
        for c in ("path", "is_attack", "attack_type"):
            if c not in df.columns:
                raise ValueError(f"{csv}: column {c!r} not in {list(df.columns)}")
        at = df["attack_type"].map(lambda v: _ATTACK_MAP.get(str(v), AttackType.UNKNOWN))
        frames.append(pd.DataFrame({
            "path": df["path"].map(lambda p: str(ROOT / p)),
            "label": df["is_attack"].map(lambda b: int(Label.ATTACK if bool(b) else Label.BONA_FIDE)),
            "attack_type": at,
            "doc_type": df["path"].map(_doc_type),
            "source": "fantasyid",
            "split": split,
        }))
    return pd.concat(frames, ignore_index=True)[COLUMNS]
