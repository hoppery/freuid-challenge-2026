from __future__ import annotations
import argparse
import csv
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from freuid.config import Config
from freuid.data.schema import read_manifest
from freuid.data.dataset import ManifestDataset, DTCTrainDataset
from freuid.data.transforms import build_transforms
from freuid.models.classifier import build_classifier
from freuid.metrics import compute_metrics, metrics_by_group, apcer_surrogate_loss


@torch.no_grad()
def _sam_ascent(params, rho):
    """SAM step-1: move weights to the worst-case point in the rho-ball (w += rho * g/||g||)."""
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return [None] * len(params)
    norm = torch.norm(torch.stack([g.detach().norm(2) for g in grads]), 2)
    scale = rho / (norm + 1e-12)
    eps = []
    for p in params:
        if p.grad is None:
            eps.append(None); continue
        e = p.grad.detach() * scale
        p.add_(e); eps.append(e)
    return eps


@torch.no_grad()
def _sam_restore(params, eps):
    for p, e in zip(params, eps):
        if e is not None:
            p.sub_(e)


def focal_bce_with_logits(logits, targets, gamma: float = 2.0, alpha: float = 0.5):
    """Focal BCE (Lin et al.) on the fraud logit (E3 capture lever). Down-weights easy,
    confidently-correct samples by (1−p_t)^γ so the gradient concentrates on the hard
    genuine/attack tail — directly the APCER@1%BPCER operating point the capture axis is
    weak at. alpha balances the positive (attack) class. Reduces to BCE at gamma=0."""
    p = torch.sigmoid(logits)
    ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    a_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (a_t * (1 - p_t).clamp(min=1e-6) ** gamma * ce).mean()


def _worker_init(_):
    # Each dataloader worker must NOT spawn a full cv2/torch thread pool — with many workers
    # (esp. two trainings sharing a box) that oversubscribes cores (load avg >1000) and starves
    # the GPU. Pin every worker to single-threaded CPU ops.
    try:
        import cv2
        cv2.setNumThreads(0)
    except Exception:
        pass
    torch.set_num_threads(1)


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def split(df, cfg: Config):
    """Domain-holdout split by cfg.holdout_by, else random."""
    rng = np.random.default_rng(cfg.seed)
    if cfg.holdout_by and cfg.holdout_by in df and df[cfg.holdout_by].nunique() > 1:
        if cfg.holdout_groups:
            val_groups = {g.strip() for g in cfg.holdout_groups.split(",") if g.strip()}
            unknown = val_groups - set(df[cfg.holdout_by].astype(str).unique())
            if unknown:
                raise ValueError(f"holdout_groups not in manifest: {unknown}")
        else:
            groups = np.asarray(df[cfg.holdout_by].astype(str).unique(), dtype=object)
            rng.shuffle(groups)
            n_val = max(1, int(len(groups) * cfg.val_fraction))
            val_groups = set(groups[:n_val])
        val = df[df[cfg.holdout_by].isin(val_groups)]
        tr = df[~df[cfg.holdout_by].isin(val_groups)]
        # guard: validation needs both classes; fall back to random if not
        if val["label"].nunique() < 2 or tr["label"].nunique() < 2:
            return _random_split(df, cfg, rng)
    else:
        return _random_split(df, cfg, rng)
    return tr.reset_index(drop=True), val.reset_index(drop=True)


def _random_split(df, cfg: Config, rng):
    idx = rng.permutation(len(df))
    n_val = max(1, int(len(df) * cfg.val_fraction))
    val = df.iloc[idx[:n_val]]
    tr = df.iloc[idx[n_val:]]
    return tr.reset_index(drop=True), val.reset_index(drop=True)


@torch.no_grad()
def evaluate(model, loader, device, val_df, amp: bool):
    model.eval()
    scores = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        multi_crop = x.dim() == 5            # (B,N,C,H,W) from PatchEvalDataset
        if multi_crop:
            b, n = x.shape[:2]
            x = x.flatten(0, 1)
        with torch.autocast("cuda", enabled=amp):
            p = torch.sigmoid(model(x)).float().squeeze(1)
        if multi_crop:
            p = p.view(b, n).mean(dim=1)     # average over crops per image
        scores.append(p.cpu().numpy())
    s = np.concatenate(scores)
    y = val_df["label"].to_numpy()
    m = compute_metrics(y, s)
    m["by_attack_type"] = metrics_by_group(y, s, val_df["attack_type"].to_numpy())
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    cfg = Config.load(ap.parse_args().config)
    set_seed(cfg.seed)
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    cfg.save(f"{cfg.out_dir}/config.yaml")
    device = "cuda"

    df = read_manifest(cfg.manifest)
    tr, val = split(df, cfg)
    print(f"train={len(tr)} val={len(val)} holdout_by={cfg.holdout_by!r}")

    is_dtc = cfg.model_type == "dtc"
    use_adv = is_dtc and cfg.adv_lambda > 0
    use_loc = is_dtc and cfg.loc_lambda > 0
    loc_grid = 8 if use_loc else 0   # matches DTCNet.patch_grid default
    train_ds = (DTCTrainDataset(tr, cfg.img_size, p_incoherent=cfg.tracemix_p,
                                sbd_p=cfg.sbd_p, patch_mode=cfg.patch_mode,
                                return_doc=use_adv, fda_p=cfg.fda_p,
                                heavy_recapture=cfg.heavy_recapture,
                                loc_grid=loc_grid,
                                appearance_inv=cfg.appearance_inv,
                                appearance_inv_mild=cfg.appearance_inv_mild,
                                patch_scale=(cfg.patch_scale_min, cfg.patch_scale_max),
                                fhag=cfg.fhag, field_tamper_p=cfg.field_tamper_p) if is_dtc
                else ManifestDataset(tr, build_transforms(cfg.train_aug, cfg.img_size)))
    tl = DataLoader(train_ds,
                    batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                    pin_memory=True, drop_last=True, worker_init_fn=_worker_init,
                    persistent_workers=cfg.num_workers > 0)
    if cfg.patch_mode:
        from freuid.data.dataset import PatchEvalDataset
        val_ds = PatchEvalDataset(val, cfg.img_size, frac=cfg.patch_eval_frac)
        val_bs = max(1, cfg.batch_size // 5)   # 5 crops per sample
    else:
        val_ds = ManifestDataset(val, build_transforms("eval", cfg.img_size))
        val_bs = cfg.batch_size
    vl = DataLoader(val_ds, batch_size=val_bs, shuffle=False,
                    num_workers=cfg.num_workers, pin_memory=True, worker_init_fn=_worker_init)

    if is_dtc and (use_adv or cfg.use_prototype or cfg.mixstyle_p > 0 or cfg.use_clip
                   or cfg.use_chroma or cfg.use_spectral or cfg.use_gsd):
        from freuid.models.dtc import build_dtc_model
        model = build_dtc_model(cfg.backbone, pretrained=cfg.pretrained,
                                img_size=cfg.img_size, freeze_backbone=cfg.freeze_backbone,
                                unfreeze_blocks=cfg.unfreeze_blocks,
                                n_doc_types=int(tr["doc_type"].nunique()) if use_adv else 0,
                                grl_lambda=cfg.grl_lambda,
                                use_prototype=cfg.use_prototype,
                                mixstyle_p=cfg.mixstyle_p, use_clip=cfg.use_clip,
                                use_chroma=cfg.use_chroma,
                                use_spectral=cfg.use_spectral,
                                use_gsd=cfg.use_gsd, gsd_r=cfg.gsd_r,
                                lora_rank=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
                                cma_chroma=cfg.cma_chroma, cdc_theta=cfg.cdc_theta).to(device)
    else:
        model = build_classifier(cfg.model_type, cfg.backbone, pretrained=cfg.pretrained,
                                 img_size=cfg.img_size, freeze_backbone=cfg.freeze_backbone,
                                 unfreeze_blocks=cfg.unfreeze_blocks,
                                 lora_rank=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
                                 cdc_theta=cfg.cdc_theta).to(device)
    # warm-start backbone from a checkpoint (e.g. public-best field-tamper -> capture adaptation test).
    # Loads matching-shape tensors; auto-adds the DTC 'rgb.' host prefix when the source is a bare backbone.
    if getattr(cfg, "init_ckpt", ""):
        src_sd = torch.load(cfg.init_ckpt, map_location="cpu", weights_only=False)["model"]
        tgt_sd = model.state_dict()
        remap = {}
        for k, v in src_sd.items():
            for cand in (k, f"rgb.{k}"):
                if cand in tgt_sd and tgt_sd[cand].shape == v.shape:
                    remap[cand] = v
                    break
        model.load_state_dict(remap, strict=False)
        print(f"[init_ckpt] warm-started {len(remap)}/{len(src_sd)} tensors from {cfg.init_ckpt}")
    # L2-SP anchor (anti-erosion): snapshot pretrained values of the UNFROZEN backbone params so the
    # loss can pull them back toward init — preserves DINOv2's OOD generalization while still adapting.
    anchor_pairs = []
    if getattr(cfg, "anchor_lambda", 0.0) > 0:
        rgb = getattr(model, "rgb", None)
        if rgb is not None:
            for _n, p in rgb.named_parameters():
                if p.requires_grad:
                    anchor_pairs.append((p, p.detach().clone()))
        print(f"L2-SP anchor on {len(anchor_pairs)} backbone tensors "
              f"({sum(p.numel() for p, _ in anchor_pairs)/1e6:.1f}M params), lambda={cfg.anchor_lambda}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    # SAM needs bf16 (no GradScaler): with fp16, an overflowing grad becomes inf and SAM's
    # normalization does inf*0=NaN → divergence. bf16 has no overflow, so disable the scaler for SAM.
    _sam_on = getattr(cfg, "sam_rho", 0.0) > 0
    scaler = torch.amp.GradScaler(enabled=cfg.amp and not _sam_on)
    lossfn = torch.nn.BCEWithLogitsLoss()
    # FRAUD-head loss: focal (E3 capture lever) when focal_gamma>0, else plain BCE. The
    # consistency/aux heads always use plain BCE (lossfn) — focal targets the fraud op-point only.
    if cfg.focal_gamma > 0:
        print(f"focal BCE on fraud head (gamma={cfg.focal_gamma}, alpha={cfg.focal_alpha})")
        fraud_loss = lambda l, t: focal_bce_with_logits(l, t, cfg.focal_gamma, cfg.focal_alpha)
    else:
        fraud_loss = lossfn
    use_sam = getattr(cfg, "sam_rho", 0.0) > 0
    if use_sam:
        print(f"SAM enabled (rho={cfg.sam_rho}) — flat-minima for OOD/unseen generalization")
    use_compact = is_dtc and cfg.compact_lambda > 0
    bona_center = (torch.zeros(512, device=device, requires_grad=True)
                   if use_compact else None)
    if use_compact:
        opt.add_param_group({"params": [bona_center]})
    ema = torch.optim.swa_utils.AveragedModel(
        model, avg_fn=torch.optim.swa_utils.get_ema_avg_fn(cfg.ema_decay)
    ) if cfg.ema_decay > 0 else None

    with open(f"{cfg.out_dir}/log.csv", "w", newline="") as log:
        w = csv.writer(log)
        w.writerow(["epoch", "train_loss", "audet", "apcer@1%bpcer", "roc_auc", "freuid_score"])
        log.flush()
        best = 1e9
        for ep in range(cfg.epochs):
            model.train()
            tot = 0.0
            for batch in tl:
                x = batch[0].to(device, non_blocking=True)
                y = batch[1].to(device).unsqueeze(1)

                def _compute_loss():
                    with torch.autocast("cuda", enabled=cfg.amp,
                                        dtype=(torch.bfloat16 if use_sam else torch.float16)):
                        if use_adv:
                            y_cons = batch[2].to(device).unsqueeze(1)
                            y_doc = batch[3].to(device)
                            fraud, cons, doc = model(x, return_consistency=True,
                                                     return_doc=True)
                            loss = (fraud_loss(fraud, y)
                                    + cfg.dtc_lambda * lossfn(cons, y_cons)
                                    + cfg.adv_lambda * torch.nn.functional.cross_entropy(doc, y_doc))
                        elif use_compact:
                            y_cons = batch[2].to(device).unsqueeze(1)
                            fraud, cons, h = model(x, return_consistency=True,
                                                   return_embed=True)
                            loss = fraud_loss(fraud, y) + cfg.dtc_lambda * lossfn(cons, y_cons)
                            bona = (y.squeeze(1) == 0)
                            if bona.any():       # compactness: genuine embeddings → center
                                d = ((h[bona] - bona_center) ** 2).sum(1).mean()
                                loss = loss + cfg.compact_lambda * d
                        elif use_loc:
                            y_cons = batch[2].to(device).unsqueeze(1)
                            y_loc = batch[3].to(device)            # (B, g*g)
                            fraud, cons, loc = model(x, return_consistency=True, return_loc=True)
                            loss = (fraud_loss(fraud, y) + cfg.dtc_lambda * lossfn(cons, y_cons)
                                    + cfg.loc_lambda * lossfn(loc, y_loc))
                        elif is_dtc:
                            y_cons = batch[2].to(device).unsqueeze(1)
                            fraud, cons = model(x, return_consistency=True)
                            loss = fraud_loss(fraud, y) + cfg.dtc_lambda * lossfn(cons, y_cons)
                        else:
                            fraud = model(x)
                            loss = fraud_loss(fraud, y)
                        if cfg.apcer_lambda > 0:    # operating-point surrogate (iter #24)
                            loss = loss + cfg.apcer_lambda * apcer_surrogate_loss(fraud, y)
                        if anchor_pairs:            # L2-SP anti-erosion (mean sq-drift from pretrained)
                            anc = sum(((p - p0) ** 2).sum() for p, p0 in anchor_pairs)
                            anc = anc / sum(p.numel() for p, _ in anchor_pairs)
                            loss = loss + cfg.anchor_lambda * anc
                    return loss

                opt.zero_grad()
                loss = _compute_loss()
                scaler.scale(loss).backward()
                if use_sam:                          # SAM: re-grad at the worst-case point in rho-ball
                    _sp = [p for p in model.parameters() if p.requires_grad]
                    _eps = _sam_ascent(_sp, cfg.sam_rho)
                    opt.zero_grad()
                    scaler.scale(_compute_loss()).backward()
                    _sam_restore(_sp, _eps)
                scaler.step(opt)
                scaler.update()
                if ema is not None:
                    ema.update_parameters(model)
                tot += loss.item() * len(x)
            sched.step()
            eval_model = ema.module if ema is not None else model
            m = evaluate(eval_model, vl, device, val, cfg.amp)
            tl_loss = tot / max(1, len(tr))
            print(f"ep{ep} loss={tl_loss:.4f} AuDET={m['audet']:.4f} "
                  f"APCER@1%={m['apcer_at_1pct_bpcer']:.4f} AUC={m['roc_auc']:.4f} "
                  f"FREUID={m['freuid_score']:.4f}")
            print("  by attack_type:", m["by_attack_type"])
            w.writerow([ep, tl_loss, m["audet"], m["apcer_at_1pct_bpcer"], m["roc_auc"],
                        m["freuid_score"]])
            log.flush()
            score = m["freuid_score"]   # official competition metric (lower better)
            save_sd = (ema.module if ema is not None else model).state_dict()
            ckpt = {"model": save_sd, "cfg": cfg.__dict__,
                    "metrics": {k: v for k, v in m.items() if k != "by_attack_type"}}
            if score < best:
                best = score
                torch.save(ckpt, f"{cfg.out_dir}/best.pt")
                print(f"  saved best (FREUID score={best:.4f})")
            # Per-epoch snapshots: needed when the held-out/random-val metric does NOT reflect the
            # deployment objective (e.g. E2 capture recipe — appearance-OOD collapses after ~1 epoch
            # while random-val keeps improving), so the best epoch must be chosen post-hoc by an
            # external proxy (capture/LODO) rather than by val FREUID.
            if getattr(cfg, "save_every_epoch", False):
                torch.save(ckpt, f"{cfg.out_dir}/epoch{ep}.pt")
                print(f"  saved epoch{ep}.pt")


if __name__ == "__main__":
    main()
