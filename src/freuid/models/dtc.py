"""DTC — Double-Trace Consistency network (the novel method; docs/winning-strategy.md §3).

Builds on HPFDualStream (RGB backbone + high-pass residual forensic branch) and adds:
1. PATCH-LEVEL trace embeddings from the residual branch (spatial map kept, not pooled).
2. A consistency module: pairwise cosine statistics over patch embeddings distill
   "is the acquisition trace internally coherent?" into a small stats vector.
3. An auxiliary consistency head trained on self-supervised TraceMix labels
   (data/tracemix.py), plus a 3-way fusion head [RGB ; global trace ; consistency stats]
   for the fraud logit.

Rationale: trace CONSISTENCY is doc-type-agnostic by construction — a genuine capture
(even a legitimately recaptured one) is uniformly traced, while a locally edited +
reprinted forgery is not. This targets the unseen-doc-type private test directly.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from freuid.models.freq_classifier import HighPassResidual

_N_STATS = 7


class _GradReverse(torch.autograd.Function):
    """Gradient reversal (DANN). Forward = identity; backward = -lambda * grad."""

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grad_reverse(x, lam: float = 1.0):
    return _GradReverse.apply(x, lam)


def _sim_stats(z: torch.Tensor) -> torch.Tensor:
    """z: (B, N, D) L2-normalized patch embeddings -> (B, 7) consistency statistics.

    Off-diagonal pairwise cosine similarities, summarized as: mean, std, min,
    5th/25th percentiles, mean and min of each patch's max-similarity-to-others
    (an inconsistent patch is dissimilar from ALL others, so its max similarity drops).
    """
    b, n, _ = z.shape
    sim = z @ z.transpose(1, 2)                                   # (B, N, N)
    eye = torch.eye(n, device=z.device, dtype=torch.bool)
    off = sim.masked_select(~eye).view(b, n * (n - 1))            # (B, N(N-1))
    per_patch_max = sim.masked_fill(eye, -1.0).amax(dim=2)        # (B, N)
    q = torch.quantile(off.float(), torch.tensor([0.05, 0.25], device=z.device), dim=1)
    return torch.stack([
        off.mean(1), off.std(1), off.amin(1), q[0], q[1],
        per_patch_max.mean(1), per_patch_max.amin(1),
    ], dim=1).to(z.dtype)


class MixStyle(nn.Module):
    """MixStyle (Zhou et al., ICLR'21) for doc-type domain randomization (iteration #25).

    Doc-type identity lives in feature channel statistics. During training, normalize each
    sample's tokens, then re-inject a Beta-mixed blend of its own and a shuffled sample's
    channel mean/std — randomizing the 'document-type style' while keeping content. The
    fraud head can no longer rely on type-appearance, so unseen types stop reading as
    anomalies. No-op at eval. Operates on (B, N, C) tokens (stats over the token dim).
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p, self.beta, self.eps = p, torch.distributions.Beta(alpha, alpha), eps

    def forward(self, x):                                  # x: (B, N, C)
        if not self.training or torch.rand(1).item() > self.p or x.size(0) < 2:
            return x
        mu = x.mean(dim=1, keepdim=True)
        sig = (x.var(dim=1, keepdim=True) + self.eps).sqrt()
        x_norm = (x - mu) / sig
        lam = self.beta.sample((x.size(0), 1, 1)).to(x.device)
        perm = torch.randperm(x.size(0), device=x.device)
        mu_mix = lam * mu + (1 - lam) * mu[perm]
        sig_mix = lam * sig + (1 - lam) * sig[perm]
        return x_norm * sig_mix + mu_mix


class CDCConv2d(nn.Module):
    """Central Difference Convolution (Yu et al., CDCN — CVPR'20 face anti-spoofing, E3 lever).

    y = vanilla_conv(x) − θ · (center_pixel · Σ kernel_weights). The central-difference term
    encodes local intensity-GRADIENT structure, which is the discriminative cue for
    print/screen RECAPTURE (the analog-hole micro-texture genuine physical captures carry and
    born-digital forgeries lack — the same texture face-anti-spoofing exploits for live-vs-spoof).
    θ blends vanilla (θ=0) ↔ pure-difference (θ=1). Implemented as the standard CDCN form: a
    vanilla conv minus a 1×1 conv (stride-aligned to the same window centers) of the per-channel
    summed kernel — exact, no kernel materialization."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, theta=0.7):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)
        if abs(self.theta) < 1e-8:
            return out_normal
        kernel_diff = self.conv.weight.sum(dim=(2, 3), keepdim=True)        # (out,in,1,1)
        out_diff = F.conv2d(x, kernel_diff, bias=None, stride=self.conv.stride, padding=0)
        return out_normal - self.theta * out_diff


class TraceEncoder(nn.Module):
    """CNN over the 3-channel high-pass residual; keeps the spatial map so both a global
    trace feature and per-patch embeddings can be extracted. cdc_theta>0 swaps the convs for
    Central Difference Convs (CDCN) — gradient-texture sensitive, targets recapture (E3)."""

    def __init__(self, width: int = 256, in_ch: int = 3, cdc_theta: float = 0.0):
        super().__init__()

        def _conv(i, o):
            if cdc_theta > 0:
                return CDCConv2d(i, o, 3, stride=2, padding=1, theta=cdc_theta)
            return nn.Conv2d(i, o, 3, stride=2, padding=1)

        self.net = nn.Sequential(
            _conv(in_ch, 32), nn.BatchNorm2d(32), nn.GELU(),
            _conv(32, 64), nn.BatchNorm2d(64), nn.GELU(),
            _conv(64, 128), nn.BatchNorm2d(128), nn.GELU(),
            _conv(128, width), nn.BatchNorm2d(width), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)   # (B, width, H/16, W/16)


class ChromaResidual(nn.Module):
    """Param-free chroma-inconsistency front-end (D11): denormalize -> opponent-color
    channels (RGB minus luminance) -> local high-pass. Isolates the COLOR signal that the
    luminance high-pass (HighPassResidual) discards. Motivated by the appearance-invariance
    ablation (D10): BENIN's genuine/attack separation lives in chroma, and clean-GenAI edits
    (face-swap / inpaint) that leave NO texture trace still leave subtle LOCAL chroma /
    white-balance discontinuities. Feeds the same TraceEncoder + consistency-stats path, so
    the consistency module flags a region whose chroma is internally inconsistent."""

    def __init__(self, cma: bool = False):
        super().__init__()
        # cma=True (GEN-3, Chromaticity Map Adapter): KEEP the low-mid-frequency chromaticity that
        # SURVIVES print/screen recapture (deep-research: high-pass/forensic DIES, chroma survives).
        # cma=False (D11 legacy): high-pass the chroma → fragile, recapture-killed.
        self.cma = cma
        self.hp = None if cma else HighPassResidual(ksize=5, sigma=1.0)
        self.register_buffer("_m", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_s", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        img = x * self._s + self._m                                   # denorm to ~[0,1] RGB
        lum = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        chroma = img - lum                                            # opponent chroma (luma removed)
        if not self.cma:
            return self.hp(chroma)                                    # legacy: local high-pass
        # CMA: intensity-normalized chromaticity, palette-removed (per-image spatial mean) so the
        # encoder sees LOCAL color-pipeline inconsistency, not the doc-type GLOBAL palette (which
        # sank earlier appearance approaches / unseen-type). Low-mid freq retained → recapture-robust.
        s = img.sum(dim=1, keepdim=True) + 1e-4
        chromaticity = img / s                                        # (B,3) rg-chromaticity
        feat = torch.cat([chroma, chromaticity], dim=1)              # (B,6) color geometry
        return feat - feat.mean(dim=(-2, -1), keepdim=True)          # drop global cast, keep local


def suppress_dominant_subspace(C: torch.Tensor, r: int = 3) -> torch.Tensor:
    """Geometric Semantic Decoupling (D13, GSD — arXiv 2603.09242): remove the dominant
    SEMANTIC subspace from frozen-foundation features so the head cannot ride the semantic
    shortcut (doc-type appearance) that makes unseen-type genuine docs read as attacks (EGYPT).
    F' = F(I − UUᵀ), U = top-r right singular vectors of the (centered) batch. Operates on a
    FROZEN CLIP stream → the residual is the non-semantic (forensic) part of CLIP."""
    Cf = C.float()
    Cc = Cf - Cf.mean(0, keepdim=True)
    try:
        _, _, Vh = torch.linalg.svd(Cc, full_matrices=False)
    except Exception:
        return C
    rr = min(r, Vh.shape[0])
    V = Vh[:rr].transpose(0, 1)                       # (d, r) dominant directions
    return (Cf - (Cf @ V) @ V.transpose(0, 1)).to(C.dtype)


class SpectralFingerprint(nn.Module):
    """Layout-invariant GenAI frequency-fingerprint branch (D12). Clean GAN/diffusion edits
    leave periodic upsampling artifacts in the Fourier spectrum even with NO spatial/texture
    trace (so the high-pass trace branch & resolution miss them — BENIN's clean-GenAI tail).
    The AZIMUTHALLY-AVERAGED 1D power spectrum (radial profile of |FFT|) captures this while
    being invariant to spatial layout/translation — avoiding the doc-type-layout shortcut that
    sank the raw 2D global-FFT branch (iter #2 / D2). Per-sample shape-normalized so absolute
    brightness/contrast doesn't leak; outputs a small spectral embedding for the fusion head."""

    def __init__(self, size: int = 256, n_bins: int = 128, out_dim: int = 64):
        super().__init__()
        self.size, self.n_bins = size, n_bins
        c = size // 2
        yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
        r = torch.sqrt((xx - c).float() ** 2 + (yy - c).float() ** 2)
        bin_idx = (r / r.max() * (n_bins - 1)).round().long().flatten()      # (size*size,)
        counts = torch.zeros(n_bins).index_add_(
            0, bin_idx, torch.ones(size * size))
        self.register_buffer("bin_idx", bin_idx)
        self.register_buffer("counts", counts.clamp(min=1))
        self.register_buffer("_m", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_s", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.enc = nn.Sequential(
            nn.Linear(n_bins, 128), nn.GELU(), nn.Linear(128, out_dim), nn.GELU())

    def forward(self, x):
        img = x * self._s + self._m
        gray = (0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]).float()
        if gray.shape[-1] != self.size:
            gray = F.interpolate(gray, size=self.size, mode="bilinear", align_corners=False)
        mag = torch.fft.fftshift(torch.abs(torch.fft.fft2(gray.squeeze(1))), dim=(-2, -1))
        mag = torch.log1p(mag).flatten(1)                                    # (B, size*size)
        radial = torch.zeros(mag.shape[0], self.n_bins, device=x.device, dtype=mag.dtype)
        radial = radial.index_add(1, self.bin_idx, mag) / self.counts        # (B, n_bins)
        radial = (radial - radial.mean(1, keepdim=True)) / (radial.std(1, keepdim=True) + 1e-5)
        return self.enc(radial)                                              # (B, out_dim)


class LoRALinear(nn.Module):
    """Low-rank adapter wrapping a frozen nn.Linear: y = W0 x + (alpha/r) (B A) x, with B init 0
    so the adapted model starts == pretrained. Deep-research lever #1: ADD capacity without ERODING
    the backbone (full/partial unfreeze destroys DINOv2's OOD generalization; LoRA preserves it).
    Only A,B train; W0 stays frozen."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scaling = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scaling


def inject_lora(backbone, r: int = 8, alpha: int = 16) -> int:
    """Replace each transformer block's attention qkv (and proj) Linear with a LoRA-wrapped version.
    Backbone must already be fully frozen. Returns the number of layers adapted."""
    n = 0
    if not hasattr(backbone, "blocks"):
        raise ValueError("inject_lora needs a ViT-style backbone with .blocks")
    for blk in backbone.blocks:
        attn = getattr(blk, "attn", None)
        if attn is None:
            continue
        for name in ("qkv", "proj"):
            lin = getattr(attn, name, None)
            if isinstance(lin, nn.Linear):
                setattr(attn, name, LoRALinear(lin, r=r, alpha=alpha))
                n += 1
    return n


class DTCNet(nn.Module):
    """RGB backbone + trace branch + consistency module, fused to a fraud logit.

    forward(x) -> fraud logit (B,1): drop-in compatible with eval/inference code.
    forward(x, return_consistency=True) -> (fraud logit, consistency logit) for the
    auxiliary TraceMix loss during training.
    """

    def __init__(self, backbone: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
                 pretrained: bool = True, drop_rate: float = 0.1,
                 patch_grid: int = 8, trace_width: int = 256, trace_dim: int = 128,
                 img_size: int | None = None, freeze_backbone: bool = False,
                 unfreeze_blocks: int = 0, n_doc_types: int = 0, grl_lambda: float = 1.0,
                 use_prototype: bool = False, mixstyle_p: float = 0.0,
                 use_clip: bool = False, use_chroma: bool = False,
                 use_spectral: bool = False, use_gsd: bool = False, gsd_r: int = 3,
                 lora_rank: int = 0, lora_alpha: int = 16, cma_chroma: bool = False,
                 cdc_theta: float = 0.0):
        super().__init__()
        kw = dict(pretrained=pretrained, num_classes=0, drop_rate=drop_rate)
        if img_size is not None:
            try:
                self.rgb = timm.create_model(backbone, img_size=img_size, **kw)
            except TypeError:
                self.rgb = timm.create_model(backbone, **kw)
        else:
            self.rgb = timm.create_model(backbone, **kw)
        if freeze_backbone:
            # partial-unfreeze host (iteration #7): the winning DINOv2 recipe — freeze
            # all, then re-enable the last N blocks + final norm (see docs/iterations.md)
            for p in self.rgb.parameters():
                p.requires_grad = False
            if unfreeze_blocks > 0:
                if not hasattr(self.rgb, "blocks"):
                    raise ValueError(f"unfreeze_blocks needs ViT-style .blocks; "
                                     f"{backbone!r} has none")
                for p in self.rgb.blocks[-unfreeze_blocks:].parameters():
                    p.requires_grad = True
                if hasattr(self.rgb, "norm"):
                    for p in self.rgb.norm.parameters():
                        p.requires_grad = True
            if lora_rank > 0:
                # LoRA adapters (frozen backbone + trainable low-rank qkv/proj) — preserves DINOv2
                # OOD generalization while adding adaptation capacity (deep-research lever #1).
                n_lora = inject_lora(self.rgb, r=lora_rank, alpha=lora_alpha)
                if hasattr(self.rgb, "norm"):     # let final norm adapt too (cheap, helps calibration)
                    for p in self.rgb.norm.parameters():
                        p.requires_grad = True
                print(f"LoRA injected on {n_lora} attn layers (r={lora_rank}, alpha={lora_alpha})")
        rgb_dim = self.rgb.num_features
        # MixStyle on RGB tokens (iteration #25): requires a ViT-style backbone with
        # forward_features returning (B, N, C); no-op otherwise.
        self.mixstyle = MixStyle(p=mixstyle_p) if mixstyle_p > 0 else None
        self.hpf = HighPassResidual(ksize=5, sigma=1.0)
        self.trace = TraceEncoder(trace_width, cdc_theta=cdc_theta)
        self.patch_grid = patch_grid
        self.patch_proj = nn.Conv2d(trace_width, trace_dim, 1)
        self.global_proj = nn.Linear(trace_width, trace_dim)
        # iteration #D11: parallel CHROMA-consistency stream. Mirrors the trace stream on a
        # color-opponent residual, so the consistency module also measures whether the
        # document's CHROMA is internally coherent — targets BENIN's clean-GenAI tail (no
        # texture trace, but local chroma/white-balance anomalies). Extends "Double-Trace" to
        # trace+chroma consistency. cons_head sees both stat vectors (self-supervised on the
        # same TraceMix label); fusion head gets the chroma global feat + chroma stats.
        self.use_chroma = use_chroma
        if use_chroma:
            self.chroma = ChromaResidual(cma=cma_chroma)
            self.chroma_enc = TraceEncoder(trace_width, in_ch=6 if cma_chroma else 3)
            self.chroma_global = nn.Linear(trace_width, trace_dim)
            self.chroma_patch = nn.Conv2d(trace_width, trace_dim, 1)
        chroma_extra = (trace_dim + _N_STATS) if use_chroma else 0
        # iteration #D12: GenAI frequency-fingerprint branch (layout-invariant radial spectrum)
        self.use_spectral = use_spectral
        spectral_dim = 64
        if use_spectral:
            self.spectral = SpectralFingerprint(size=256, n_bins=128, out_dim=spectral_dim)
        spectral_extra = spectral_dim if use_spectral else 0
        self.cons_head = nn.Sequential(
            nn.Linear(_N_STATS * (2 if use_chroma else 1), 32), nn.GELU(), nn.Linear(32, 1),
        )
        # iteration #D7: frozen CLIP semantic stream — catches GenAI manipulation (face-
        # swap / whole-image diffusion) that the forensic trace features miss (FantasyID
        # AUC ~0.5 across all our models). FatFormer-lineage: CLIP generalizes to unseen
        # GAN/diffusion. Fed a 224-resized, CLIP-normalized copy of the input; frozen.
        # frozen CLIP stream — built if use_clip (raw semantic, D7) OR use_gsd (semantic-
        # suppressed forensic residual, D13). iteration #D7: CLIP catches GenAI manipulation
        # the trace features miss. iteration #D13 (GSD): suppress dominant semantic subspace.
        self.use_clip = use_clip
        self.use_gsd = use_gsd
        self.gsd_r = gsd_r
        need_clip = use_clip or use_gsd
        clip_dim = 0
        if need_clip:
            self.clip = timm.create_model("vit_base_patch16_clip_224.openai",
                                          pretrained=pretrained, num_classes=0)
            for p in self.clip.parameters():
                p.requires_grad = False
            clip_dim = self.clip.num_features
            imn_m = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            imn_s = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            clip_m = torch.tensor([0.4815, 0.4578, 0.4082]).view(1, 3, 1, 1)
            clip_s = torch.tensor([0.2686, 0.2613, 0.2758]).view(1, 3, 1, 1)
            self.register_buffer("_imn_m", imn_m); self.register_buffer("_imn_s", imn_s)
            self.register_buffer("_clip_m", clip_m); self.register_buffer("_clip_s", clip_s)
        clip_extra = clip_dim * (int(use_clip) + int(use_gsd))
        self.head = nn.Sequential(
            nn.Linear(rgb_dim + trace_dim + _N_STATS + clip_extra + chroma_extra + spectral_extra,
                      512), nn.GELU(),
            nn.Dropout(drop_rate), nn.Linear(512, 1),
        )
        # iteration #23: prototype distance head. Instead of a linear hyperplane (which
        # an unseen doc type can fall on the wrong side of wholesale), score by distance
        # to a learned BONA-FIDE prototype vs an ATTACK prototype in the 512-d embedding.
        # Distance-based decisions are inherently more OOD-robust: a novel-type genuine
        # doc is "near genuine, far from attack" regardless of its appearance.
        self.use_proto = use_prototype
        if use_prototype:
            self.proto = nn.Parameter(torch.randn(2, 512) * 0.1)   # [bona, attack]
            self.proto_scale = nn.Parameter(torch.tensor(1.0))
        # iteration #13: adversarial doc-type head (DANN). Gradient reversal on the
        # fused decision embedding removes doc-type identity from the decision space —
        # targets the EGYPT-mode failure (unseen-type bona-fides drifting to "attack").
        self.doc_head = (nn.Linear(512, n_doc_types) if n_doc_types > 0 else None)
        self.grl_lambda = grl_lambda
        # iteration #D6: localization head — per-patch trace-inconsistency logit from the
        # patch embedding + its deviation from the document's consensus trace. Supervised
        # by TraceMix paste masks. Spatially grounds the consistency signal (type-invariant)
        # and yields the paper's localization figure.
        self.loc_head = nn.Sequential(
            nn.Linear(trace_dim * 2, 128), nn.GELU(), nn.Linear(128, 1),
        )

    def _trace_features(self, x, return_z=False):
        fmap = self.trace(self.hpf(x))                                  # (B,W,h,w)
        g = self.global_proj(fmap.mean(dim=(-2, -1)))                   # (B,trace_dim)
        p = self.patch_proj(F.adaptive_avg_pool2d(fmap, self.patch_grid))
        z = F.normalize(p.flatten(2).transpose(1, 2), dim=-1)           # (B,N,D)
        if return_z:
            return g, _sim_stats(z), z
        return g, _sim_stats(z)

    def _chroma_features(self, x):
        """Mirror of _trace_features on the chroma residual -> (global chroma feat, stats)."""
        fmap = self.chroma_enc(self.chroma(x))                          # (B,W,h,w)
        g = self.chroma_global(fmap.mean(dim=(-2, -1)))                 # (B,trace_dim)
        p = self.chroma_patch(F.adaptive_avg_pool2d(fmap, self.patch_grid))
        z = F.normalize(p.flatten(2).transpose(1, 2), dim=-1)           # (B,N,D)
        return g, _sim_stats(z)

    def _loc_logits(self, z):
        """z: (B,N,D) -> (B,N) per-patch inconsistency logits."""
        consensus = z.mean(dim=1, keepdim=True)            # document consensus trace
        feat = torch.cat([z, z - consensus], dim=-1)       # patch + deviation
        return self.loc_head(feat).squeeze(-1)

    def _clip_feat(self, x):
        # x is ImageNet-normalized; denorm -> CLIP-norm -> resize 224, frozen forward
        img = x * self._imn_s + self._imn_m
        img = (img - self._clip_m) / self._clip_s
        if img.shape[-1] != 224:
            img = F.interpolate(img, size=224, mode="bilinear", align_corners=False)
        with torch.no_grad():
            return self.clip(img)

    def _rgb_feat(self, x):
        if self.mixstyle is None:
            return self.rgb(x)
        # token-level MixStyle then mean-pool (matches num_classes=0 pooled output)
        t = self.rgb.forward_features(x)              # (B, N, C)
        t = self.mixstyle(t)
        return t.mean(dim=1)

    def forward(self, x, return_consistency: bool = False, return_doc: bool = False,
                return_embed: bool = False, return_loc: bool = False):
        f_rgb = self._rgb_feat(x)
        if return_loc:
            g, stats, z = self._trace_features(x, return_z=True)
        else:
            g, stats = self._trace_features(x)
        parts = [f_rgb, g, stats]
        cons_stats = stats
        if self.use_chroma:
            g_c, stats_c = self._chroma_features(x)
            parts += [g_c, stats_c]
            cons_stats = torch.cat([stats, stats_c], dim=1)
        if self.use_clip or self.use_gsd:
            c = self._clip_feat(x)
            if self.use_clip:
                parts.append(c)
            if self.use_gsd:
                parts.append(suppress_dominant_subspace(c, self.gsd_r).to(f_rgb.dtype))
        if self.use_spectral:
            parts.append(self.spectral(x).to(f_rgb.dtype))
        fused = torch.cat(parts, dim=1)
        h = self.head[:3](fused)          # shared decision embedding (512)
        if self.use_proto:
            d = torch.cdist(h, self.proto)            # (B,2): dist to [bona, attack]
            fraud = (d[:, 0] - d[:, 1]).unsqueeze(1) * self.proto_scale
        else:
            fraud = self.head[3](h)
        outs = [fraud]
        if return_consistency:
            outs.append(self.cons_head(cons_stats))
        if return_loc:
            outs.append(self._loc_logits(z))          # (B, N) per-patch logits
        if return_doc:
            assert self.doc_head is not None, "model built without n_doc_types"
            outs.append(self.doc_head(grad_reverse(h, self.grl_lambda)))
        if return_embed:
            outs.append(h)                # for bona-fide compactness loss (iter #22)
        return outs[0] if len(outs) == 1 else tuple(outs)


def build_dtc_model(backbone: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
                    pretrained: bool = True, drop_rate: float = 0.1,
                    patch_grid: int = 8, img_size: int | None = None,
                    freeze_backbone: bool = False, unfreeze_blocks: int = 0,
                    n_doc_types: int = 0, grl_lambda: float = 1.0,
                    use_prototype: bool = False, mixstyle_p: float = 0.0,
                    use_clip: bool = False, use_chroma: bool = False,
                    use_spectral: bool = False, use_gsd: bool = False, gsd_r: int = 3,
                    lora_rank: int = 0, lora_alpha: int = 16, cma_chroma: bool = False,
                    cdc_theta: float = 0.0):
    return DTCNet(backbone, pretrained=pretrained, drop_rate=drop_rate,
                  patch_grid=patch_grid, img_size=img_size,
                  freeze_backbone=freeze_backbone, unfreeze_blocks=unfreeze_blocks,
                  n_doc_types=n_doc_types, grl_lambda=grl_lambda,
                  use_prototype=use_prototype, mixstyle_p=mixstyle_p, use_clip=use_clip,
                  use_chroma=use_chroma, use_spectral=use_spectral, use_gsd=use_gsd, gsd_r=gsd_r,
                  lora_rank=lora_rank, lora_alpha=lora_alpha, cma_chroma=cma_chroma,
                  cdc_theta=cdc_theta)
