"""Test-time adaptation (TENT, Wang et al. ICLR'21) — iteration #26.

Root cause of the hard-fold wall (docs/iterations.md): the held-out doc type is OUTSIDE
the convex hull of the training types, so no train-time trick (MixStyle/GRL/compactness)
can synthesize it. The ONLY way to actually see the unseen type is at test time — and in
FREUID the test images are published before scoring. TENT adapts the model to the
unlabelled test distribution by minimizing prediction entropy, updating ONLY the
LayerNorm affine params (cheap, stable). We evaluate it honestly on the leave-one-doc-
type-out val set (the unseen type), comparing adapted vs frozen.

Usage:
  python3 scripts/tta_eval.py --ckpt checkpoints/exp_dtc_dinov2_egypt_518/best.pt \
                              [--steps 1 --lr 1e-3]
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from freuid.config import Config
from freuid.data.schema import read_manifest
from freuid.data.dataset import ManifestDataset
from freuid.data.transforms import build_transforms
from freuid.models.classifier import build_classifier
from freuid.train import split
from freuid.metrics import compute_metrics


def _collect_ln_params(model):
    params = []
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.requires_grad_(True)
            params += [p for p in m.parameters() if p.requires_grad]
    return params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=1, help="adaptation steps per batch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--div", type=float, default=0.0,
                    help="diversity weight: maximize batch-marginal entropy to prevent "
                         "the TENT collapse-to-one-class failure (iter #27, SHOT/IM)")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cuda", weights_only=False)
    cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in Config.__dataclass_fields__})
    df = read_manifest(cfg.manifest)
    _, val = split(df, cfg)
    y = val["label"].to_numpy()
    dl = DataLoader(ManifestDataset(val, build_transforms("eval", cfg.img_size)),
                    batch_size=cfg.batch_size, shuffle=False,
                    num_workers=cfg.num_workers, pin_memory=True)

    def fresh():
        m = build_classifier(cfg.model_type, cfg.backbone, pretrained=False,
                             img_size=cfg.img_size, freeze_backbone=cfg.freeze_backbone,
                             use_prototype=getattr(cfg, "use_prototype", False)).cuda()
        m.load_state_dict(ck["model"])
        return m

    # ---- frozen baseline ----
    m = fresh(); m.eval()
    base = []
    with torch.no_grad():
        for x, _ in dl:
            with torch.autocast("cuda", enabled=cfg.amp):
                base.append(torch.sigmoid(m(x.cuda())).float().squeeze(1).cpu().numpy())
    base = np.concatenate(base)

    # ---- TENT: minimize prediction entropy, update only LayerNorm affine ----
    m = fresh()
    for p in m.parameters():
        p.requires_grad_(False)
    ln_params = _collect_ln_params(m)
    opt = torch.optim.Adam(ln_params, lr=a.lr)
    m.train()                       # LN uses batch input; no BN running stats in ViT
    adapted = []
    for x, _ in dl:
        x = x.cuda()
        for _ in range(a.steps):
            opt.zero_grad()
            with torch.autocast("cuda", enabled=cfg.amp):
                logit = m(x).squeeze(1)
                p = torch.sigmoid(logit)
                # binary entropy of each prediction; minimize -> confident predictions
                ent = -(p * torch.log(p + 1e-6) + (1 - p) * torch.log(1 - p + 1e-6))
                loss = ent.mean()
                if a.div > 0:        # anti-collapse: MAXIMIZE batch-marginal entropy
                    pm = p.mean().clamp(1e-6, 1 - 1e-6)
                    marg = -(pm * torch.log(pm) + (1 - pm) * torch.log(1 - pm))
                    loss = loss - a.div * marg
            loss.backward()
            opt.step()
        with torch.no_grad():
            adapted.append(torch.sigmoid(m(x).squeeze(1)).float().cpu().numpy())
    adapted = np.concatenate(adapted)

    bm, am = compute_metrics(y, base), compute_metrics(y, adapted)
    print(f"ckpt={a.ckpt}  holdout={cfg.holdout_groups!r}  steps={a.steps} lr={a.lr}")
    print("  frozen :", {k: round(v, 4) for k, v in bm.items()})
    print("  TENT   :", {k: round(v, 4) for k, v in am.items()})
    print(f"  ΔFREUID = {am['freuid_score'] - bm['freuid_score']:+.4f} "
          f"({'BETTER' if am['freuid_score'] < bm['freuid_score'] else 'worse'})")


if __name__ == "__main__":
    main()
