"""Probe the largest SAFE micro-batch for a backbone at a given resolution.
Runs forward+backward with grad-checkpointing + partial unfreeze (the training config),
measures peak VRAM, and reports the batch that fits within a target fraction of the GPU.
Usage: python3 scripts/probe_batch.py --img 728 --batches 4 8 12 16 --target 0.82
"""
from __future__ import annotations
import argparse
import torch
from freuid.models.classifier import build_classifier, unfreeze_last_k_blocks, enable_grad_checkpointing


def try_batch(backbone, img, bs, unfreeze_k, ckpt):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_classifier("rgb", backbone, pretrained=False, img_size=img).cuda()
    if unfreeze_k:
        unfreeze_last_k_blocks(model, unfreeze_k)
    if ckpt:
        enable_grad_checkpointing(model)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    x = torch.randn(bs, 3, img, img, device="cuda")
    y = torch.zeros(bs, 1, device="cuda")
    lossfn = torch.nn.BCEWithLogitsLoss()
    for _ in range(2):  # 2 steps to capture optimizer-state memory
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = lossfn(model(x), y)
        loss.backward()
        opt.step()
    peak = torch.cuda.max_memory_allocated() / 1e9
    del model, opt, x, y
    torch.cuda.empty_cache()
    return peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vit_base_patch14_reg4_dinov2.lvd142m")
    ap.add_argument("--img", type=int, default=728)
    ap.add_argument("--batches", type=int, nargs="+", default=[4, 8, 12, 16])
    ap.add_argument("--unfreeze-k", type=int, default=2)
    ap.add_argument("--ckpt", action="store_true", default=True)
    ap.add_argument("--no-ckpt", dest="ckpt", action="store_false")
    ap.add_argument("--target", type=float, default=0.82, help="target VRAM fraction (headroom below 1.0)")
    a = ap.parse_args()
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    budget = total * a.target
    print(f"GPU total={total:.1f}GB  target={a.target}  budget={budget:.1f}GB  "
          f"img={a.img}  ckpt={a.ckpt}  unfreeze_k={a.unfreeze_k}")
    best_fit = None
    for bs in a.batches:
        try:
            peak = try_batch(a.backbone, a.img, bs, a.unfreeze_k, a.ckpt)
            fits = peak <= budget
            print(f"  bs={bs:3d}  peak={peak:5.1f}GB  {'OK' if fits else 'OVER'}")
            if fits:
                best_fit = bs
        except torch.cuda.OutOfMemoryError:
            print(f"  bs={bs:3d}  OOM")
            torch.cuda.empty_cache()
            break
    print(f"RECOMMENDED micro_bs (img={a.img}) = {best_fit}")


if __name__ == "__main__":
    main()
