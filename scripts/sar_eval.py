"""GEN-4: SAR (Sharpness-Aware + Reliable entropy minimization, ICLR'23) vs our TENT vs no-TTA.
SAR upgrades TENT with: (1) RELIABLE-sample filtering (only adapt on low-entropy/confident samples
— confident-wrong attacks that poison the APCER@1% tail are skipped); (2) SAM sharpness-aware step
(flat minimum, robust to the noisy OOD samples that remain); (3) anti-COLLAPSE reset to snapshot if
the moving-average entropy degenerates. ViT+LayerNorm-native = our exact setup. Adapts LayerNorm
params only (like our TENT). Metric = official FREUID (lower=better).

Usage:
  python3 scripts/sar_eval.py --ckpt checkpoints/exp_gen3_cma_egypt/best.pt --holdout EGYPT/DL
  python3 scripts/sar_eval.py --ckpt checkpoints/exp_e2early_all/epoch0.pt --manifest manifests/fantasyid_test.parquet
"""
from __future__ import annotations
import argparse, copy, math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from freuid.config import Config
from freuid.data.dataset import ManifestDataset
from freuid.data.transforms import build_transforms
from freuid.models.classifier import build_classifier
from freuid.metrics import compute_metrics

E0 = 0.4 * math.log(2)   # reliable-entropy threshold for binary (SAR uses 0.4*ln C)


def _build(ckpt):
    ck = torch.load(ckpt, map_location="cuda", weights_only=False)
    cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in Config.__dataclass_fields__})
    m = build_classifier(cfg.model_type, cfg.backbone, pretrained=False, img_size=cfg.img_size,
                         freeze_backbone=cfg.freeze_backbone,
                         use_prototype=getattr(cfg, "use_prototype", False),
                         use_clip=getattr(cfg, "use_clip", False),
                         use_chroma=getattr(cfg, "use_chroma", False),
                         use_spectral=getattr(cfg, "use_spectral", False),
                         use_gsd=getattr(cfg, "use_gsd", False), gsd_r=getattr(cfg, "gsd_r", 3),
                         lora_rank=getattr(cfg, "lora_rank", 0),
                         lora_alpha=getattr(cfg, "lora_alpha", 16),
                         cma_chroma=getattr(cfg, "cma_chroma", False)).cuda()
    m.load_state_dict(ck["model"])
    return m, cfg


def _loader(df, cfg):
    return DataLoader(ManifestDataset(df.assign(image_id=np.arange(len(df))),
                                      build_transforms("eval", cfg.img_size), with_label=False),
                      batch_size=cfg.batch_size, shuffle=False, num_workers=12, pin_memory=True)


def _ln_params(m):
    ps = [p for mod in m.modules() if isinstance(mod, torch.nn.LayerNorm) for p in mod.parameters()]
    for p in m.parameters():
        p.requires_grad_(False)
    for p in ps:
        p.requires_grad_(True)
    return ps


def _ent(p):                       # per-sample binary entropy of sigmoid prob p
    return -(p * torch.log(p + 1e-6) + (1 - p) * torch.log(1 - p + 1e-6))


@torch.no_grad()
def _infer(m, dl, cfg):
    m.eval(); s = []
    for x, _ in dl:
        with torch.autocast("cuda", enabled=cfg.amp):
            s.append(torch.sigmoid(m(x.cuda()).squeeze(1)).float().cpu().numpy())
    return np.concatenate(s)


def tent(ckpt, df, lr=2e-5):
    m, cfg = _build(ckpt); ps = _ln_params(m)
    opt = torch.optim.Adam(ps, lr=lr); m.train(); s = []
    for x, _ in _loader(df, cfg):
        x = x.cuda(); opt.zero_grad()
        with torch.autocast("cuda", enabled=cfg.amp):
            p = torch.sigmoid(m(x).squeeze(1))
            ent = _ent(p).mean()
            pm = p.mean().clamp(1e-6, 1 - 1e-6); marg = -(pm * torch.log(pm) + (1 - pm) * torch.log(1 - pm))
            loss = ent - 1.0 * marg
        if torch.isfinite(loss): loss.backward(); opt.step()
        with torch.no_grad():
            s.append(torch.sigmoid(m(x).squeeze(1)).float().cpu().numpy())
    del m; torch.cuda.empty_cache(); return np.concatenate(s)


def sar(ckpt, df, lr=2e-5, rho=0.05, reset_thresh=0.1):
    m, cfg = _build(ckpt); ps = _ln_params(m)
    snapshot = copy.deepcopy({k: v.detach().clone() for k, v in m.state_dict().items()})
    opt = torch.optim.SGD(ps, lr=lr, momentum=0.9); m.train(); s = []
    ema = None
    for x, _ in _loader(df, cfg):
        x = x.cuda()
        # ---- SAM step 1: gradient of RELIABLE-sample entropy at w ----
        opt.zero_grad()
        with torch.autocast("cuda", enabled=cfg.amp):
            p = torch.sigmoid(m(x).squeeze(1)); e = _ent(p)
            mask = (e < E0).float()
            loss = (e * mask).sum() / (mask.sum() + 1e-6)
        if not torch.isfinite(loss):
            with torch.no_grad(): s.append(torch.sigmoid(m(x).squeeze(1)).float().cpu().numpy())
            continue
        loss.backward()
        with torch.no_grad():
            gnorm = torch.sqrt(sum((p.grad ** 2).sum() for p in ps if p.grad is not None) + 1e-12)
            eps = [rho * p.grad / (gnorm + 1e-12) if p.grad is not None else None for p in ps]
            for p, ew in zip(ps, eps):
                if ew is not None: p.add_(ew)
        # ---- SAM step 2: gradient at perturbed w', then restore w and step ----
        opt.zero_grad()
        with torch.autocast("cuda", enabled=cfg.amp):
            p2 = torch.sigmoid(m(x).squeeze(1)); e2 = _ent(p2)
            mask2 = (e2 < E0).float()
            loss2 = (e2 * mask2).sum() / (mask2.sum() + 1e-6)
        if torch.isfinite(loss2): loss2.backward()
        with torch.no_grad():
            for p, ew in zip(ps, eps):
                if ew is not None: p.sub_(ew)
        opt.step()
        # ---- anti-collapse reset ----
        cur = loss.item(); ema = cur if ema is None else 0.9 * ema + 0.1 * cur
        if ema is not None and ema < reset_thresh:
            m.load_state_dict(snapshot); ema = None
        with torch.no_grad():
            s.append(torch.sigmoid(m(x).squeeze(1)).float().cpu().numpy())
    del m; torch.cuda.empty_cache(); return np.concatenate(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--holdout", default=None, help="doc_type fold, e.g. EGYPT/DL")
    ap.add_argument("--manifest", default="manifests/freuid.parquet")
    a = ap.parse_args()
    df = pd.read_parquet(a.manifest)
    if a.holdout:
        df = df[df["doc_type"].astype(str) == a.holdout].reset_index(drop=True)
    y = df["label"].to_numpy()
    print(f"ckpt={a.ckpt} holdout={a.holdout} manifest={a.manifest} n={len(df)} "
          f"(bona {(y==0).sum()}/attack {(y==1).sum()})")
    m, cfg = _build(a.ckpt)
    no = _infer(m, _loader(df, cfg), cfg); del m; torch.cuda.empty_cache()
    for name, sc in [("no-TTA", no), ("TENT", tent(a.ckpt, df)), ("SAR", sar(a.ckpt, df))]:
        mm = compute_metrics(y, sc)
        print(f"  {name:8s} FREUID={mm['freuid_score']:.4f}  APCER@1%={mm['apcer_at_1pct_bpcer']:.4f}  AUC={mm['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
