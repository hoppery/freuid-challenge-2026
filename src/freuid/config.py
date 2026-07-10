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
    val_fraction: float = 0.15
    holdout_by: str = "doc_type"   # domain-holdout column ("" = random split)
    holdout_groups: str = ""       # comma-separated group names to force as val
                                   # (e.g. "EGYPT/DL"); empty = seeded random choice
    seed: int = 42
    out_dir: str = "checkpoints/baseline"
    amp: bool = True
    pretrained: bool = True
    init_ckpt: str = ""             # warm-start backbone from this checkpoint (strict=False, remaps rgb.* host)
    freeze_backbone: bool = False   # frozen backbone + adapter head (DINOv2 frozen stage)
    unfreeze_blocks: int = 0        # with freeze_backbone: unfreeze last N ViT blocks (+norm)
    # DTC (model_type "dtc"): TraceMix self-supervised consistency auxiliary
    dtc_lambda: float = 0.5         # weight of the consistency BCE loss
    tracemix_p: float = 0.5         # probability a train sample is made trace-incoherent
    sbd_p: float = 0.0              # probability a bona-fide becomes a self-blended
                                    # pseudo-attack (SBD, iteration #10)
    patch_mode: bool = False        # patch-based train (RandomResizedCrop) + 5-crop
                                    # eval — destroys layout/script identity (iter #11)
    adv_lambda: float = 0.0         # weight of the adversarial doc-type CE loss
                                    # (GRL/DANN head, iteration #13); 0 = off
    grl_lambda: float = 1.0         # gradient-reversal strength inside the GRL
    fda_p: float = 0.0              # FDA cross-type amplitude-swap probability (iter #16)
    ema_decay: float = 0.0          # weight-EMA decay (e.g. 0.999); 0 = off (iter #21)
    compact_lambda: float = 0.0     # bona-fide compactness aux loss weight (iter #22):
                                    # pull genuine-doc embeddings to a learned center so
                                    # unseen-type bona-fides don't drift to "attack"
    use_prototype: bool = False     # prototype distance fraud head (iter #23, OOD-robust)
    apcer_lambda: float = 0.0       # weight of the APCER@1%BPCER surrogate loss (iter #24)
    mixstyle_p: float = 0.0         # MixStyle doc-type feature randomization prob (iter #25)
    heavy_recapture: bool = False   # add print-and-capture aug to DTC path (plan_v2 E1)
    # CAPTURE-axis levers (E3, 2026-06-18): target the print/screen-recapture private axis,
    # which is op-point/texture-bound (NOT appearance) so they avoid the EGYPT antagonism.
    focal_gamma: float = 0.0        # >0 = focal BCE on the FRAUD head (γ); down-weights easy
                                    # samples, sharpens the hard genuine/attack tail (APCER@1%)
    focal_alpha: float = 0.5        # focal class-balance weight for the positive (attack) class
    cdc_theta: float = 0.0          # >0 = Central Difference Conv in the trace branch (CDCN,
                                    # Yu'20): encodes recapture gradient texture; θ blends
                                    # vanilla(0)→pure-difference(1) conv on the high-pass residual
    fhag: bool = False              # Frequency-band amplitude augmentation (FHAG): perturb radial
                                    # spectrum bands so the trace branch is invariant to the
                                    # born-digital↔captured band-energy shift (both classes)
    save_every_epoch: bool = False  # also snapshot epoch{N}.pt each epoch — pick best epoch post-hoc
                                    # by an external proxy when val FREUID ≠ deployment objective (E2)
    anchor_lambda: float = 0.0      # L2-SP: penalize MEAN sq-drift of unfrozen backbone params from
                                    # pretrained init → preserve DINOv2 generalization (anti-erosion,
                                    # deep-research lever #1; stops appearance-OOD collapse on E2)
    lora_rank: int = 0              # >0 = freeze backbone + LoRA adapters on attn qkv/proj (rank r);
                                    # ADD capacity without ERODING DINOv2 OOD generalization (lever #1)
    lora_alpha: int = 16            # LoRA scaling = alpha/rank
    cma_chroma: bool = False        # CMA chromaticity stream (GEN-3): use_chroma front-end keeps
                                    # recapture-robust low-mid chromaticity (no high-pass) → BENIN
    loc_lambda: float = 0.0         # per-patch trace-inconsistency localization loss (D6)
    use_clip: bool = False          # frozen CLIP semantic stream for GenAI detection (D7)
    appearance_inv: bool = False    # strong color/palette randomization to break doc-type
                                    # color identity → unseen-type generalization (D10)
    appearance_inv_mild: bool = False  # gentler variant (no ToGray/ChannelShuffle, lower p):
                                    # keep EGYPT gain without destroying BENIN's chroma cue (D10b)
    use_chroma: bool = False        # parallel chroma-consistency stream targeting BENIN's
                                    # clean-GenAI tail (local chroma/white-balance anomalies) (D11)
    use_spectral: bool = False      # GenAI frequency-fingerprint branch: layout-invariant radial
                                    # power spectrum catches upsampling artifacts (BENIN tail, D12)
    use_gsd: bool = False           # Geometric Semantic Decoupling: suppress dominant semantic
                                    # subspace of frozen CLIP → forensic residual (EGYPT, D13)
    gsd_r: int = 3                  # number of dominant semantic directions removed
    # Resolution TILING (D14): with patch_mode, train on LARGE native-res tiles (preserve
    # forensic detail past the whole-doc-896 ceiling without 1036's single-pass instability) +
    # multi-crop eval aggregation. #11 failed with TINY patches (0.08-0.35); large tiles fix it.
    patch_scale_min: float = 0.08
    patch_scale_max: float = 0.35
    patch_eval_frac: float = 0.45   # crop = frac*short-side, resized to img_size, 5-crop avg
    sam_rho: float = 0.0            # >0 = Sharpness-Aware Minimization (flat minima → OOD/unseen-type
                                    # generalization). Two fwd-bwd per step (root loop is already per-batch).
    field_tamper_p: float = 0.0     # prob a GENUINE train sample → targeted text-field tampering attack
                                    # (copy-move/overwrite/strikethrough + alpha-feather + localized low-QF
                                    # JPEG = the DeepID-winner compression signal). data/field_tamper.py (research)

    @staticmethod
    def load(path: str) -> "Config":
        with open(path) as f:
            return Config(**(yaml.safe_load(f) or {}))

    def save(self, path: str):
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f)
