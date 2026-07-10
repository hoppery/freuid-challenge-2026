from __future__ import annotations
from dataclasses import dataclass, asdict
import yaml


@dataclass
class Config:
    manifest: str = "manifests/freuid.parquet"
    backbone: str = "convnextv2_tiny.fcmae_ft_in22k_in1k"
    model_type: str = "rgb"   # "rgb" (baseline) | "freq_dual"
    train_aug: str = "train"  # "train" (light) | "heavy" (forensic recapture aug)
    img_size: int = 384
    batch_size: int = 32
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 0.05
    num_workers: int = 8
    grad_accum: int = 1       # micro-batches accumulated per optimizer step (high-res: keep effective batch up while micro-batch fits VRAM)
    grad_checkpointing: bool = False  # activation checkpointing (trades compute for VRAM; required for hi-res ViT)
    early_stop_patience: int = 0  # stop after this many epochs w/o val improvement (0=off); cuts wasted post-collapse epochs (best is ep0-1)
    val_fraction: float = 0.15
    holdout_by: str = "doc_type"   # domain-holdout column ("" = random split)
    holdout_values: str = ""       # explicit doc_types to hold out for val (comma-sep); overrides random pick
    extra_manifests: str = ""      # extra manifest parquet paths to concat for TRAINING (comma-sep); multi-source DG
    val_sources: str = ""          # restrict val to these `source` values (comma-sep); "" = any
    seed: int = 42
    out_dir: str = "checkpoints/baseline"
    amp: bool = True
    amp_dtype: str = "bf16"   # "bf16" (stable, no GradScaler) | "fp16" (legacy — caused NaN collapses)
    warmup_steps: int = 500   # linear LR warmup before cosine (prevents early AdamW blowup)
    grad_clip: float = 1.0    # max grad-norm (0=off); stabilizes training, prevents collapse-to-prior
    ema: bool = False         # weight EMA (stabilizes the high-variance cross-domain metric + helps DG)
    ema_decay: float = 0.999
    pretrained: bool = True
    freeze_backbone: bool = False   # linear-probe / adapter mode (foundation backbones generalize better frozen)
    unfreeze_last_k: int = 0        # ViT: train only last-k blocks + norm + head (partial unfreeze = DG sweet spot)
    n_doctypes: int = 0             # DTC: number of doc-type classes (auto-set from train data)
    dtc_alpha: float = 1.0          # DTC: gradient-reversal strength (domain-invariance weight)
    fda: bool = False               # Fourier Domain Adaptation aug (cross-doc-type style randomization)
    fda_beta: float = 0.05          # FDA low-freq band size (fraction); smaller = subtler style swap
    loss: str = "bce"               # "bce" | "tail_margin" (operating-point-aligned, see losses.py)
    tail_lambda: float = 1.0        # weight of the attack-below-tau hinge
    tail_mu: float = 0.2            # weight of the bona-fide right-tail compression
    tail_margin: float = 1.0        # logit margin above tau attacks must clear
    tail_quantile: float = 0.99     # bona-fide quantile defining tau (matches 1% BPCER)
    tail_space: str = "prob"        # "prob" (bounded, safe) | "logit" (DIVERGES — iter-1 failure)
    tail_warmup: int = 300          # steps before tail terms activate (bank settles first)
    init_ckpt: str = ""             # warm-start: load this checkpoint's weights before fine-tuning
                                    # (snapshot_ep2 ViT-Large → add capture aug, keep unseen-type)
    self_blend_p: float = 0.0       # prob a GENUINE train sample → synthetic self-blend forgery
                                    # (doc-agnostic forgery cue → UNSEEN-doc-type generalization)
    field_tamper_p: float = 0.0     # prob a GENUINE train sample → targeted field-tamper forgery
                                    # (localized low-QF JPEG + alpha-feather; ROOT-line ft03 → public 0.0103,
                                    # rank 9). On ViT-L: test if capacity absorbs it w/o the ViT-B EGYPT collapse.
    sam_rho: float = 0.0            # >0 = Sharpness-Aware Minimization (seek FLAT minima → better OOD/
                                    # unseen-doc-type generalization). 2 fwd-bwd/step; requires grad_accum=1.

    @staticmethod
    def load(path: str) -> "Config":
        with open(path) as f:
            return Config(**(yaml.safe_load(f) or {}))

    def save(self, path: str):
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f)
