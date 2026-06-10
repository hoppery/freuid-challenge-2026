import pandas as pd
import pytest
from freuid.data.schema import (
    COLUMNS, Label, AttackType, validate_manifest, write_manifest, read_manifest,
)


def _row(**kw):
    base = dict(path="a.jpg", label=Label.ATTACK, attack_type=AttackType.PHYSICAL,
                doc_type="doc_a", source="freuid", split="train")
    base.update(kw)
    return base


def test_columns_constant():
    assert COLUMNS == ["path", "label", "attack_type", "doc_type", "source", "split"]


def test_validate_accepts_good_frame():
    df = pd.DataFrame([_row(), _row(label=Label.BONA_FIDE, attack_type=AttackType.NONE)])
    validate_manifest(df)  # should not raise


def test_validate_rejects_bad_label():
    df = pd.DataFrame([_row(label=2)])
    with pytest.raises(ValueError, match="label"):
        validate_manifest(df)


def test_validate_rejects_missing_column():
    df = pd.DataFrame([{"path": "a.jpg"}])
    with pytest.raises(ValueError, match="missing columns"):
        validate_manifest(df)


def test_roundtrip_parquet(tmp_path):
    df = pd.DataFrame([_row(), _row(label=Label.BONA_FIDE, attack_type=AttackType.NONE)])
    p = tmp_path / "m.parquet"
    write_manifest(df, p)
    out = read_manifest(p)
    assert list(out.columns) == COLUMNS
    assert len(out) == 2
