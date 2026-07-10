"""Cross-dataset eval: a FREUID-trained model scored on FantasyID (plan_v2 §B).

FantasyID = 933 PHYSICAL printed-and-captured bona-fides + 2,351 GenAI attacks. This is
the closest available analogue of the FREUID test's capture-style + GenAI shift, which
the leave-one-doc-type-out folds CANNOT measure (train is 99.97% born-digital). Use this
as the honest proxy for the public/private gap.

Usage: python3 scripts/xeval_fantasyid.py --ckpt checkpoints/<run>/best.pt [--tta]
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
from freuid.metrics import compute_metrics


def _build(cfg):
    return build_classifier(cfg.model_type, cfg.backbone, pretrained=False,
                            img_size=cfg.img_size, freeze_backbone=cfg.freeze_backbone,
                            use_prototype=getattr(cfg, "use_prototype", False),
                            use_clip=getattr(cfg, "use_clip", False),
                            use_chroma=getattr(cfg, "use_chroma", False),
                            use_spectral=getattr(cfg, "use_spectral", False),
                            use_gsd=getattr(cfg, "use_gsd", False),
                            gsd_r=getattr(cfg, "gsd_r", 3),
                            lora_rank=getattr(cfg, "lora_rank", 0),
                            lora_alpha=getattr(cfg, "lora_alpha", 16),
                            cma_chroma=getattr(cfg, "cma_chroma", False),
                            cdc_theta=getattr(cfg, "cdc_theta", 0.0)).cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="manifests/fantasyid.parquet")
    ap.add_argument("--tta", action="store_true", help="conservative TENT (lr 2e-5, div 1.0)")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cuda", weights_only=False)
    cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in Config.__dataclass_fields__})
    val = read_manifest(a.manifest)
    y = val["label"].to_numpy()
    dl = DataLoader(ManifestDataset(val, build_transforms("eval", cfg.img_size)),
                    batch_size=cfg.batch_size, shuffle=False,
                    num_workers=cfg.num_workers, pin_memory=True)

    m = _build(cfg); m.load_state_dict(ck["model"])
    if not a.tta:
        m.eval()
        sc = []
        with torch.no_grad():
            for x, _ in dl:
                with torch.autocast("cuda", enabled=cfg.amp):
                    sc.append(torch.sigmoid(m(x.cuda())).float().squeeze(1).cpu().numpy())
        s = np.concatenate(sc)
    else:
        for p in m.parameters():
            p.requires_grad_(False)
        ln = [p for mod in m.modules() if isinstance(mod, torch.nn.LayerNorm)
              for p in mod.parameters()]
        for p in ln:
            p.requires_grad_(True)
        opt = torch.optim.Adam(ln, lr=2e-5)
        m.train()
        sc = []
        for x, _ in dl:
            x = x.cuda()
            opt.zero_grad()
            with torch.autocast("cuda", enabled=cfg.amp):
                p = torch.sigmoid(m(x).squeeze(1))
                ent = -(p * torch.log(p + 1e-6) + (1 - p) * torch.log(1 - p + 1e-6)).mean()
                pm = p.mean().clamp(1e-6, 1 - 1e-6)
                marg = -(pm * torch.log(pm) + (1 - pm) * torch.log(1 - pm))
                (ent - 1.0 * marg).backward()
            opt.step()
            with torch.no_grad():
                sc.append(torch.sigmoid(m(x).squeeze(1)).float().cpu().numpy())
        s = np.concatenate(sc)

    metr = compute_metrics(y, s)
    print(f"ckpt={a.ckpt}  FantasyID xeval{'  +TTA' if a.tta else ''}")
    print(f"  n={len(val)} (bona {int((y==0).sum())} / attack {int((y==1).sum())})")
    print("  ", {k: round(v, 4) for k, v in metr.items()})


if __name__ == "__main__":
    main()
