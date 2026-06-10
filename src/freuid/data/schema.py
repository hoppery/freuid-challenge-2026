from __future__ import annotations
from enum import IntEnum
from pathlib import Path
import pandas as pd


class Label(IntEnum):
    BONA_FIDE = 0
    ATTACK = 1


class AttackType(str):
    """String constants for the attack taxonomy (subclasses str so values ARE strings)."""
    NONE = "none"
    PHYSICAL = "physical"
    GENAI_DIGITAL = "genai_digital"
    PRINT_CAPTURE = "print_capture"
    OTHER_DIGITAL = "other_digital"
    UNKNOWN = "unknown"


ATTACK_TYPES = {
    "none", "physical", "genai_digital", "print_capture", "other_digital", "unknown",
}
COLUMNS = ["path", "label", "attack_type", "doc_type", "source", "split"]


def validate_manifest(df: pd.DataFrame) -> None:
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"manifest missing columns: {missing}")
    bad = {int(v) for v in df["label"].unique()} - {0, 1}
    if bad:
        raise ValueError(f"invalid label values: {bad}")
    bad_at = set(df["attack_type"].astype(str).unique()) - ATTACK_TYPES
    if bad_at:
        raise ValueError(f"invalid attack_type values: {bad_at}")
    if df["split"].isna().any():
        raise ValueError("split column has nulls")


def write_manifest(df: pd.DataFrame, path: str | Path) -> None:
    validate_manifest(df)
    out = df[COLUMNS].copy()
    out["label"] = out["label"].astype(int)
    out["attack_type"] = out["attack_type"].astype(str)
    out["doc_type"] = out["doc_type"].astype(str)
    out["source"] = out["source"].astype(str)
    out["split"] = out["split"].astype(str)
    out["path"] = out["path"].astype(str)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)


def read_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    validate_manifest(df)
    return df[COLUMNS]
