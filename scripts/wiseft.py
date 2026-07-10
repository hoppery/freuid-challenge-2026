"""WiSE-FT (Wortsman et al. 2021): interpolate fine-tuned backbone weights back toward the PRETRAINED
init to recover OOD robustness without losing in-domain accuracy. theta = (1-a)*pretrained + a*finetuned
for all BACKBONE params (head kept from finetuned). Anti-overfit lever for the public generalization gap.

Usage: python3 scripts/wiseft.py --ckpt checkpoints/exp_prod896_large_bce/best.pt --alpha 0.5 --out checkpoints/wiseft_a05.pt
"""
from __future__ import annotations
import argparse, torch, timm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight on finetuned (1=pure FT, 0=pretrained)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ft = ck["model"]
    backbone = ck["cfg"].get("backbone") if isinstance(ck.get("cfg"), dict) else ck["cfg"].backbone
    # pretrained backbone weights (no head; num_classes=0)
    pre_model = timm.create_model(backbone, pretrained=True, num_classes=0)
    pre = pre_model.state_dict()

    new = {}
    n_interp = 0
    for k, v in ft.items():
        if k.startswith("head."):
            new[k] = v                      # keep the trained fraud head
        elif k in pre and pre[k].shape == v.shape:
            new[k] = (1 - a.alpha) * pre[k].float() + a.alpha * v.float()
            new[k] = new[k].to(v.dtype)
            n_interp += 1
        else:
            new[k] = v                      # param absent in pretrained (e.g. head) — keep FT
    print(f"WiSE-FT alpha={a.alpha}: interpolated {n_interp}/{len(ft)} backbone params toward pretrained")
    ck["model"] = new
    torch.save(ck, a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
