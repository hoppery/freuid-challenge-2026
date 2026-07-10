"""Adapters for external ID-document datasets → unified manifest rows (plan_v3).

Goal: maximize document-TYPE diversity so the model learns type-invariant fraud cues and
generalizes to the private test's 2 unseen doc types. Each adapter emits the standard
columns (path,label,attack_type,doc_type,source,split) + is_digital where known.

Label convention: 0 = bona-fide / genuine, 1 = attack / fraud.
doc_type is source-prefixed and unique per dataset×layout so leave-type-out is meaningful.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

from freuid.data.schema import Label, AttackType

ROOT = Path("data/raw")
_EXT = (".jpg", ".jpeg", ".png")


def _imgs(d: Path):
    return [p for p in d.rglob("*") if p.suffix.lower() in _EXT]


def build_idnet_manifest(extracted=ROOT / "idnet" / "extracted") -> pd.DataFrame:
    """IDNet: <COUNTRY>/positive (genuine) + <COUNTRY>/fraud*_* (attacks). 20 countries,
    real manipulation attacks (inpaint-and-rewrite, crop-and-replace) matching the private
    threat model. Each country = a doc_type."""
    rows = []
    if not extracted.exists():
        return pd.DataFrame(columns=["path", "label", "attack_type", "doc_type", "source", "split"])
    for country_dir in sorted(extracted.iterdir()):
        if not country_dir.is_dir():
            continue
        c = country_dir.name
        for sub in country_dir.iterdir():
            if not sub.is_dir():
                continue
            name = sub.name.lower()
            if name == "positive":
                lab, at = Label.BONA_FIDE, AttackType.NONE
            elif name.startswith("fraud"):
                lab = Label.ATTACK
                at = (AttackType.GENAI_DIGITAL if "inpaint" in name else
                      AttackType.OTHER_DIGITAL)
            else:
                continue  # meta/ etc.
            for p in _imgs(sub):
                rows.append({"path": str(p), "label": int(lab), "attack_type": at,
                             "doc_type": f"idnet_{c}", "source": "idnet", "split": "train",
                             "is_digital": True})
    return pd.DataFrame(rows)


def build_docxpand_manifest(root=ROOT / "docxpand" / "DocXPand-25k" / "documents") -> pd.DataFrame:
    """DocXPand-25k: 9 synthetic ID templates, all GENUINE. Genuine-type diversity;
    pseudo-attacks are synthesized at train time (SBD). doc_type = template."""
    rows = []
    if not root.exists():
        return pd.DataFrame(columns=["path", "label", "attack_type", "doc_type", "source", "split"])
    for tpl in sorted(root.iterdir()):
        if not tpl.is_dir():
            continue
        for p in _imgs(tpl):
            rows.append({"path": str(p), "label": int(Label.BONA_FIDE),
                         "attack_type": AttackType.NONE, "doc_type": f"docxpand_{tpl.name}",
                         "source": "docxpand", "split": "train", "is_digital": True})
    return pd.DataFrame(rows)


def build_bid_manifest(root=ROOT / "bid" / "BID Dataset") -> pd.DataFrame:
    """BID: Brazilian IDs (CPF/RG/CNH, front/back/open). Genuine layouts (fake PII).
    Genuine Brazilian-type diversity. doc_type = document kind."""
    rows = []
    if not root.exists():
        return pd.DataFrame(columns=["path", "label", "attack_type", "doc_type", "source", "split"])
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        kind = sub.name.split("_")[0]  # CPF / RG / CNH
        for p in _imgs(sub):
            rows.append({"path": str(p), "label": int(Label.BONA_FIDE),
                         "attack_type": AttackType.NONE, "doc_type": f"bid_{kind}",
                         "source": "bid", "split": "train", "is_digital": False})
    return pd.DataFrame(rows)
