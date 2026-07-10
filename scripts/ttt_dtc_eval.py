"""TTT-DTC (failure-analysis-driven, 2026-06-15): Test-Time Training via the DTC self-supervised
consistency task. Failure analysis: every TRAINING-time change fails; only "what the model sees"
(resolution) and TEST-TIME adaptation (TTA) work, and EGYPT/BENIN need opposite training-time
remedies — so the only cross-fold lever is test-time. Our DTC has a built-in self-supervised task
(TraceMix consistency). TTT (Sun et al.) adapts the model at test time by minimizing the self-sup
loss on test images → improves the main (fraud) task under distribution shift, more forensic-
targeted than TENT's entropy. Adapts LayerNorm only (conservative). Compares no-TTA / TENT / TTT-DTC
on a held-out LODO fold.

Usage:
  python3 scripts/ttt_dtc_eval.py --ckpt checkpoints/exp_d9_res896_egypt/best.pt --holdout EGYPT/DL
"""
from __future__ import annotations
import argparse
import copy
import numpy as np
import pandas as pd
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from freuid.config import Config
from freuid.data.transforms import _MEAN, _STD
from freuid.data.tracemix import TraceMix
from freuid.models.classifier import build_classifier
from freuid.metrics import compute_metrics


class TTTDataset(Dataset):
    """Returns (clean_x, tracemix_x, y_cons, y_fraud). clean = deterministic resize+norm (for
    prediction); tracemix = self-supervised consistency view of the SAME framing (for adaptation)."""
    def __init__(self, df, size, p_incoherent=0.5):
        self.df = df.reset_index(drop=True)
        self.geom = A.Compose([A.LongestMaxSize(max_size=size),
                               A.PadIfNeeded(size, size, border_mode=0)])
        self.norm = A.Compose([A.Normalize(_MEAN, _STD), ToTensorV2()])
        self.tm = TraceMix(p_incoherent=p_incoherent)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.imread(r["path"], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(r["path"])
        g = self.geom(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))["image"]
        clean = self.norm(image=g)["image"]
        tm_img, y_cons = self.tm(g)
        tm = self.norm(image=tm_img)["image"]
        return clean, tm, float(y_cons), float(r["label"])


def load_model(ckpt):
    ck = torch.load(ckpt, map_location="cuda", weights_only=False)
    cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in Config.__dataclass_fields__})
    m = build_classifier(cfg.model_type, cfg.backbone, pretrained=False, img_size=cfg.img_size,
                         freeze_backbone=cfg.freeze_backbone,
                         use_prototype=getattr(cfg, "use_prototype", False),
                         use_clip=getattr(cfg, "use_clip", False),
                         use_chroma=getattr(cfg, "use_chroma", False),
                         use_spectral=getattr(cfg, "use_spectral", False),
                         use_gsd=getattr(cfg, "use_gsd", False),
                         gsd_r=getattr(cfg, "gsd_r", 3)).cuda()
    m.load_state_dict(ck["model"])
    return m, cfg


def ln_params(m):
    return [p for mod in m.modules() if isinstance(mod, torch.nn.LayerNorm) for p in mod.parameters()]


def trace_params(m):
    # shared trace-branch params (also feed the fraud head) — the consistency self-sup loss
    # flows through these, so adapting them via consistency improves the fraud task too.
    return [p for n, p in m.named_parameters()
            if any(k in n for k in ("trace.", "patch_proj", "cons_head"))]


def rep(name, y, s):
    mm = compute_metrics(y, s)
    print(f"  {name:14s} FREUID={mm['freuid_score']:.4f}  AuDET={mm['audet']:.4f}  "
          f"APCER@1%={mm['apcer_at_1pct_bpcer']:.4f}  AUC={mm['roc_auc']:.4f}")
    return mm['freuid_score']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--manifest", default="manifests/freuid.parquet")
    a = ap.parse_args()
    df = pd.read_parquet(a.manifest)
    fold = df[df["doc_type"].astype(str) == a.holdout].reset_index(drop=True)
    y = fold["label"].to_numpy()
    m0, cfg = load_model(a.ckpt)
    bs = cfg.batch_size
    dl = DataLoader(TTTDataset(fold, cfg.img_size), batch_size=bs, shuffle=False,
                    num_workers=16, pin_memory=True)
    print(f"holdout={a.holdout} n={len(fold)} (bona {(y==0).sum()}/attack {(y==1).sum()}) bs={bs}")
    bce = torch.nn.BCEWithLogitsLoss()

    # --- no-TTA ---
    m = copy.deepcopy(m0).eval()
    s = []
    with torch.no_grad():
        for clean, tm, yc, yf in dl:
            with torch.autocast("cuda", enabled=cfg.amp):
                s.append(torch.sigmoid(m(clean.cuda()).squeeze(1)).float().cpu().numpy())
    s_no = np.concatenate(s)

    # --- TENT (entropy on clean) ---
    m = copy.deepcopy(m0)
    for p in m.parameters(): p.requires_grad_(False)
    for p in ln_params(m): p.requires_grad_(True)
    opt = torch.optim.Adam(ln_params(m), lr=a.lr); m.train()
    s = []
    for clean, tm, yc, yf in dl:
        x = clean.cuda(); opt.zero_grad()
        with torch.autocast("cuda", enabled=cfg.amp):
            p = torch.sigmoid(m(x).squeeze(1))
            ent = -(p*torch.log(p+1e-6)+(1-p)*torch.log(1-p+1e-6)).mean()
            pm = p.mean().clamp(1e-6, 1-1e-6)
            marg = -(pm*torch.log(pm)+(1-pm)*torch.log(1-pm))
            loss = ent - 1.0*marg
        if torch.isfinite(loss): loss.backward(); opt.step()
        with torch.no_grad():
            s.append(torch.sigmoid(m(x).squeeze(1)).float().cpu().numpy())
    s_tent = np.concatenate(s)

    # --- TTT-DTC (TENT entropy on LN + self-sup consistency on the trace branch) ---
    m = copy.deepcopy(m0)
    for p in m.parameters(): p.requires_grad_(False)
    adapt = ln_params(m) + trace_params(m)
    for p in adapt: p.requires_grad_(True)
    opt = torch.optim.Adam(adapt, lr=a.lr); m.train()
    s = []
    for clean, tm, yc, yf in dl:
        opt.zero_grad()
        with torch.autocast("cuda", enabled=cfg.amp):
            p = torch.sigmoid(m(clean.cuda()).squeeze(1))
            ent = -(p*torch.log(p+1e-6)+(1-p)*torch.log(1-p+1e-6)).mean()
            pm = p.mean().clamp(1e-6, 1-1e-6)
            marg = -(pm*torch.log(pm)+(1-pm)*torch.log(1-pm))
            _, cons = m(tm.cuda(), return_consistency=True)
            loss = (ent - 1.0*marg) + bce(cons.squeeze(1), yc.cuda())
        if torch.isfinite(loss): loss.backward(); opt.step()
        with torch.no_grad():
            s.append(torch.sigmoid(m(clean.cuda()).squeeze(1)).float().cpu().numpy())
    s_ttt = np.concatenate(s)

    print("=== results ===")
    rep("no-TTA", y, s_no)
    rep("TENT", y, s_tent)
    rep("TTT-DTC", y, s_ttt)
    from scipy.stats import rankdata
    rep("TENT+TTT fuse", y, rankdata(s_tent)+rankdata(s_ttt))


if __name__ == "__main__":
    main()
