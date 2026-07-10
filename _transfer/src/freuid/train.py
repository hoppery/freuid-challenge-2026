from __future__ import annotations
import argparse
import copy
import csv
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from freuid.config import Config
from freuid.data.schema import read_manifest
from freuid.data.dataset import ManifestDataset
from freuid.data.transforms import build_transforms
from freuid.models.classifier import (build_classifier, freeze_backbone,
                                       unfreeze_last_k_blocks, enable_grad_checkpointing)
from freuid.losses import build_loss
from freuid.metrics import compute_metrics, metrics_by_group


@torch.no_grad()
def _sam_ascent(params, rho):
    """SAM step-1: move weights to the worst-case point in the rho-ball (w += rho * g/||g||).
    Returns the perturbation list so it can be undone before the descent step."""
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
        p.add_(e)
        eps.append(e)
    return eps


@torch.no_grad()
def _sam_restore(params, eps):
    for p, e in zip(params, eps):
        if e is not None:
            p.sub_(e)


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class EMA:
    """Exponential moving average of model weights (stabilizes the high-variance
    cross-domain metric and usually improves generalization)."""

    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                s.copy_(v)

    def copy_to(self, model):
        sd = model.state_dict()
        model.load_state_dict({k: self.shadow[k].to(sd[k].dtype) for k in sd})


def split(df, cfg: Config):
    """Domain-holdout split. Priority: explicit holdout_values > holdout_by random > random."""
    rng = np.random.default_rng(cfg.seed)
    if cfg.holdout_values:
        col = cfg.holdout_by or "doc_type"
        vals = {v.strip() for v in cfg.holdout_values.split(",") if v.strip()}
        mask = df[col].astype(str).isin(vals)
        val, tr = df[mask], df[~mask]
        if cfg.val_sources:
            srcs = {s.strip() for s in cfg.val_sources.split(",") if s.strip()}
            val = val[val["source"].astype(str).isin(srcs)]
        return tr.reset_index(drop=True), val.reset_index(drop=True)
    if cfg.holdout_by and cfg.holdout_by in df and df[cfg.holdout_by].nunique() > 1:
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
def evaluate(model, loader, device, val_df, amp: bool, amp_dtype=torch.bfloat16):
    model.eval()
    scores = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp):
            scores.append(torch.sigmoid(model(x)).float().squeeze(1).cpu().numpy())
    s = np.concatenate(scores)
    n_nan = int(np.isnan(s).sum())
    if n_nan:
        print(f"  WARNING: {n_nan}/{len(s)} NaN scores in eval — replacing with 0.5 "
              f"(model numerically unstable; investigate)")
        s = np.nan_to_num(s, nan=0.5)
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
    if cfg.extra_manifests:
        extra = [read_manifest(p.strip()) for p in cfg.extra_manifests.split(",") if p.strip()]
        df = pd.concat([df] + extra, ignore_index=True)
        print(f"combined manifest: {len(df)} rows from {1+len(extra)} sources "
              f"{dict(df['source'].value_counts())}")
    tr, val = split(df, cfg)
    print(f"train={len(tr)} val={len(val)} holdout_by={cfg.holdout_by!r} "
          f"holdout_values={cfg.holdout_values!r} val_sources={cfg.val_sources!r}")
    print(f"  train doc_types={tr['doc_type'].nunique()} sources={dict(tr['source'].value_counts())}")

    dtc_mode = (cfg.model_type == "dtc")
    if dtc_mode:
        doctypes = sorted(tr["doc_type"].astype(str).unique())
        dt2id = {d: i for i, d in enumerate(doctypes)}
        tr = tr.copy()
        tr["doctype_id"] = tr["doc_type"].astype(str).map(dt2id).astype(int)
        cfg.n_doctypes = len(doctypes)
        print(f"  DTC: {cfg.n_doctypes} doc-type domains {doctypes}")

    extra_ops = None
    if cfg.fda:
        from freuid.data.fda import FDATransform
        rng2 = np.random.default_rng(cfg.seed)
        ref_paths = tr["path"].sample(min(96, len(tr)), random_state=int(cfg.seed)).tolist()
        extra_ops = [FDATransform(ref_paths, cfg.img_size, beta=cfg.fda_beta, p=0.5)]
        print(f"FDA enabled (beta={cfg.fda_beta}, {len(ref_paths)} refs)")
    tl = DataLoader(ManifestDataset(tr, build_transforms(cfg.train_aug, cfg.img_size, extra=extra_ops),
                                    with_doctype=dtc_mode,
                                    self_blend_p=getattr(cfg, "self_blend_p", 0.0),
                                    field_tamper_p=getattr(cfg, "field_tamper_p", 0.0)),
                    batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                    pin_memory=True, drop_last=True)
    vl = DataLoader(ManifestDataset(val, build_transforms("eval", cfg.img_size)),
                    batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                    pin_memory=True)

    model = build_classifier(cfg.model_type, cfg.backbone, pretrained=cfg.pretrained,
                             img_size=cfg.img_size, n_doctypes=cfg.n_doctypes).to(device)
    if getattr(cfg, "init_ckpt", ""):
        # WARM-START from a strong checkpoint (e.g. snapshot_ep2 ViT-Large) before fine-tuning —
        # preserves its seen+unseen-type knowledge while a short capture-aug fine-tune adds capture
        # robustness (private = unseen types + captured). Load weights, THEN apply unfreeze/freeze.
        _ick = torch.load(cfg.init_ckpt, map_location=device, weights_only=False)
        _miss, _unexp = model.load_state_dict(_ick["model"], strict=False)
        print(f"warm-start init from {cfg.init_ckpt} (missing={len(_miss)} unexpected={len(_unexp)})")
    if cfg.unfreeze_last_k and cfg.unfreeze_last_k > 0:
        n = unfreeze_last_k_blocks(model, cfg.unfreeze_last_k)
        print(f"partial unfreeze (last {cfg.unfreeze_last_k} blocks) -> {n} trainable params")
    elif cfg.freeze_backbone:
        n = freeze_backbone(model)
        print(f"frozen backbone -> {n} trainable params")
    if getattr(cfg, "grad_checkpointing", False):
        ok = enable_grad_checkpointing(model)
        print(f"grad_checkpointing = {ok}")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg.lr, weight_decay=cfg.weight_decay)
    accum = max(1, getattr(cfg, "grad_accum", 1))
    steps_per_epoch = max(1, len(tr) // cfg.batch_size // accum)   # OPTIMIZER steps/epoch
    total_steps = steps_per_epoch * cfg.epochs
    warmup = min(cfg.warmup_steps, max(1, total_steps // 10))

    def _lr_lambda(step):  # linear warmup -> cosine decay (per-step)
        if step < warmup:
            return (step + 1) / warmup
        import math
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
    use_scaler = cfg.amp and amp_dtype == torch.float16  # bf16 needs no GradScaler
    scaler = torch.amp.GradScaler(enabled=use_scaler)
    print(f"amp={cfg.amp} dtype={cfg.amp_dtype} scaler={use_scaler} warmup={warmup} "
          f"steps/ep={steps_per_epoch} micro_bs={cfg.batch_size} accum={accum} "
          f"eff_bs={cfg.batch_size*accum} img={cfg.img_size}")
    ema = EMA(model, cfg.ema_decay) if cfg.ema else None
    ema_model = copy.deepcopy(model) if cfg.ema else None
    if ema:
        print(f"EMA enabled (decay={cfg.ema_decay})")
    if cfg.loss == "tail_margin":
        lossfn = build_loss("tail_margin", quantile=cfg.tail_quantile, margin=cfg.tail_margin,
                            lambda_tail=cfg.tail_lambda, mu_bona=cfg.tail_mu,
                            space=cfg.tail_space, warmup_steps=cfg.tail_warmup).to(device)
    else:
        lossfn = build_loss(cfg.loss)
    print(f"loss = {cfg.loss}")
    use_sam = getattr(cfg, "sam_rho", 0.0) > 0
    if use_sam:
        assert accum == 1, "SAM requires grad_accum=1 (two fwd-bwd on the SAME batch)"
        print(f"SAM enabled (rho={cfg.sam_rho}) — flat-minima for OOD/unseen generalization")

    with open(f"{cfg.out_dir}/log.csv", "w", newline="") as log:
        w = csv.writer(log)
        w.writerow(["epoch", "train_loss", "audet", "apcer@1%bpcer", "freuid_score", "roc_auc"])
        log.flush()
        import math
        ce = torch.nn.CrossEntropyLoss()
        best = 1e9
        gstep = 0
        since_improve = 0
        patience = max(0, getattr(cfg, "early_stop_patience", 0))
        for ep in range(cfg.epochs):
            model.train()
            tot = 0.0
            micro = 0
            opt.zero_grad(set_to_none=True)
            for batch in tl:
                if dtc_mode:
                    x, y, dtid = batch
                    x = x.to(device, non_blocking=True)
                    y = y.to(device).unsqueeze(1)
                    dtid = dtid.to(device, non_blocking=True)
                    p = gstep / max(1, total_steps)
                    alpha = cfg.dtc_alpha * (2.0 / (1.0 + math.exp(-10 * p)) - 1.0)
                    with torch.autocast("cuda", dtype=amp_dtype, enabled=cfg.amp):
                        fraud, dlog = model.forward_dtc(x, alpha)
                        loss = lossfn(fraud, y) + ce(dlog, dtid)
                else:
                    x, y = batch
                    x = x.to(device, non_blocking=True)
                    y = y.to(device).unsqueeze(1)
                    with torch.autocast("cuda", dtype=amp_dtype, enabled=cfg.amp):
                        loss = lossfn(model(x), y)
                tot += loss.item() * len(x)
                scaler.scale(loss / accum).backward()   # scale for accumulation
                micro += 1
                if micro % accum == 0:                   # optimizer step boundary
                    if use_sam:                          # SAM: ascend to worst-case in rho-ball, re-grad there
                        _sp = [p for p in model.parameters() if p.requires_grad]
                        _eps = _sam_ascent(_sp, cfg.sam_rho)
                        opt.zero_grad(set_to_none=True)
                        with torch.autocast("cuda", dtype=amp_dtype, enabled=cfg.amp):
                            if dtc_mode:
                                _f2, _d2 = model.forward_dtc(x, alpha)
                                loss2 = lossfn(_f2, y) + ce(_d2, dtid)
                            else:
                                loss2 = lossfn(model(x), y)
                        scaler.scale(loss2).backward()   # gradient at the perturbed (sharp) point
                        _sam_restore(_sp, _eps)          # move weights back before the actual update
                    if cfg.grad_clip and cfg.grad_clip > 0:
                        if use_scaler:
                            scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    sched.step()   # per-OPTIMIZER-step schedule (warmup -> cosine)
                    gstep += 1
                    if ema:
                        ema.update(model)
            if micro % accum != 0:                       # flush trailing partial accum group
                if cfg.grad_clip and cfg.grad_clip > 0:
                    if use_scaler:
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                gstep += 1
                if ema:
                    ema.update(model)
            if ema:
                ema.copy_to(ema_model)
                m = evaluate(ema_model, vl, device, val, cfg.amp, amp_dtype)
            else:
                m = evaluate(model, vl, device, val, cfg.amp, amp_dtype)
            tl_loss = tot / max(1, len(tr))
            print(f"ep{ep} loss={tl_loss:.4f} AuDET={m['audet']:.4f} "
                  f"APCER@1%={m['apcer_at_1pct_bpcer']:.4f} FREUID={m['freuid_score']:.4f} "
                  f"AUC={m['roc_auc']:.4f}")
            print("  by attack_type:", m["by_attack_type"])
            w.writerow([ep, tl_loss, m["audet"], m["apcer_at_1pct_bpcer"],
                        m["freuid_score"], m["roc_auc"]])
            log.flush()
            score = m["freuid_score"]   # OFFICIAL competition metric (lower better)
            _sd = ema_model.state_dict() if ema else model.state_dict()
            if score < best:
                best = score
                since_improve = 0
                torch.save({"model": _sd, "cfg": cfg.__dict__,
                            "metrics": {k: v for k, v in m.items() if k != "by_attack_type"}},
                           f"{cfg.out_dir}/best.pt")
                print(f"  saved best (FREUID={best:.4f})")
            else:
                since_improve += 1
            # per-epoch snapshot (capture card: val FREUID is in-distribution, does NOT reflect the
            # capture objective — pick the deployment epoch post-hoc by fantasyid_test proxy).
            torch.save({"model": _sd, "cfg": cfg.__dict__,
                        "metrics": {k: v for k, v in m.items() if k != "by_attack_type"}},
                       f"{cfg.out_dir}/epoch{ep}.pt")
            if patience and since_improve >= patience:
                print(f"  early stop @ ep{ep} (no improve {since_improve} epochs; best={best:.4f})")
                break


if __name__ == "__main__":
    main()
