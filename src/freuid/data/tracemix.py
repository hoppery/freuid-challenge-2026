"""TraceMix: self-supervised trace-consistency label generation (DTC method).

Core idea (docs/winning-strategy.md §3): a genuine capture carries ONE coherent
acquisition trace; a print-and-capture forgery of a locally edited document carries
regionally INCONSISTENT traces. We synthesize both cases from any image, for free:

- coherent  (y_cons=1): apply ONE recapture chain uniformly (or leave clean).
- incoherent(y_cons=0): apply chain A globally, then paste region(s) processed with a
  DELIBERATELY DIFFERENT chain B (different JPEG-QF chain / moiré freq / blur), alpha-
  feathered so the model must read trace statistics, not paste seams.

Applied to BOTH fraud and bona-fide images (class-agnostic), AFTER geometric augs and
BEFORE Normalize, on uint8 RGB. The consistency label supervises an auxiliary head;
it is independent of the fraud label.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Single-trace operations (pure functions on uint8 RGB)
# ---------------------------------------------------------------------------

def apply_jpeg_chain(img: np.ndarray, qualities: list[int]) -> np.ndarray:
    out = img
    for q in qualities:
        ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out


def apply_moire(img: np.ndarray, amp: float, freq: float, theta: float, phase: float) -> np.ndarray:
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    pattern = amp * np.sin(2 * np.pi * freq * (xx * np.cos(theta) + yy * np.sin(theta)) + phase)
    out = img.astype(np.float32) + pattern[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    k = max(3, int(2 * round(2 * sigma) + 1))
    return cv2.GaussianBlur(img, (k, k), sigma)


def apply_noise(img: np.ndarray, std: float, rng: random.Random) -> np.ndarray:
    noise = np.random.default_rng(rng.randrange(2**31)).normal(0, std * 255, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_downup(img: np.ndarray, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    nh, nw = max(8, int(h * scale)), max(8, int(w * scale))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Recapture chains
# ---------------------------------------------------------------------------

@dataclass
class TraceChain:
    """One sampled acquisition-trace recipe. Parameters are fixed at sample time so the
    SAME chain can be applied to different regions (coherent) or contrasted against a
    different chain (incoherent)."""
    jpeg_qualities: list[int] = field(default_factory=list)
    moire: tuple[float, float, float, float] | None = None   # amp, freq, theta, phase
    blur_sigma: float | None = None
    noise_std: float | None = None
    downup_scale: float | None = None

    def __call__(self, img: np.ndarray, rng: random.Random) -> np.ndarray:
        out = img
        if self.downup_scale is not None:
            out = apply_downup(out, self.downup_scale)
        if self.blur_sigma is not None:
            out = apply_blur(out, self.blur_sigma)
        if self.moire is not None:
            out = apply_moire(out, *self.moire)
        if self.noise_std is not None:
            out = apply_noise(out, self.noise_std, rng)
        if self.jpeg_qualities:
            out = apply_jpeg_chain(out, self.jpeg_qualities)
        return out


def sample_chain(rng: random.Random) -> TraceChain:
    return TraceChain(
        jpeg_qualities=[rng.randint(60, 95) for _ in range(rng.randint(1, 3))],
        moire=((rng.uniform(4, 18), rng.uniform(0.05, 0.5),
                rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi))
               if rng.random() < 0.6 else None),
        blur_sigma=rng.uniform(0.5, 1.6) if rng.random() < 0.5 else None,
        noise_std=rng.uniform(0.02, 0.08) if rng.random() < 0.4 else None,
        downup_scale=rng.uniform(0.3, 0.7) if rng.random() < 0.5 else None,
    )


def sample_contrasting_chain(base: TraceChain, rng: random.Random) -> TraceChain:
    """Sample a chain guaranteed to differ from `base` in at least the JPEG-QF chain and
    the moiré configuration — the two strongest trace fingerprints. Without this
    constraint two independent samples can coincide and poison the consistency labels."""
    base_q = base.jpeg_qualities[-1] if base.jpeg_qualities else 95
    lo, hi = (60, max(61, base_q - 15)) if base_q >= 78 else (min(94, base_q + 15), 95)
    q_last = rng.randint(*sorted((lo, hi)))
    qualities = [rng.randint(60, 95) for _ in range(rng.randint(0, 2))] + [q_last]

    base_f = base.moire[1] if base.moire is not None else None
    while True:
        f = rng.uniform(0.05, 0.5)
        if base_f is None or abs(f - base_f) > 0.1:
            break
    moire = (rng.uniform(6, 18), f, rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi))

    return TraceChain(
        jpeg_qualities=qualities,
        moire=moire,
        blur_sigma=rng.uniform(0.5, 1.6) if rng.random() < 0.5 else None,
        noise_std=rng.uniform(0.02, 0.08) if rng.random() < 0.4 else None,
        downup_scale=rng.uniform(0.3, 0.7) if rng.random() < 0.5 else None,
    )


# ---------------------------------------------------------------------------
# Region masks
# ---------------------------------------------------------------------------

def _region_mask(h: int, w: int, rng: random.Random,
                 area_range=(0.10, 0.40), feather: int = 15) -> np.ndarray:
    """Float [0,1] mask of 1–2 rectangles (occasionally an ellipse, mimicking a photo
    field), feathered so paste seams are soft and trace statistics carry the signal."""
    mask = np.zeros((h, w), np.float32)
    target = rng.uniform(*area_range)
    placed = 0.0
    for _ in range(rng.randint(1, 2)):
        frac = min(target - placed, rng.uniform(0.08, 0.30))
        if frac <= 0.01:
            break
        ar = rng.uniform(0.5, 2.0)
        rh = int(np.sqrt(frac * h * w / ar))
        rw = int(rh * ar)
        rh, rw = min(rh, h - 2), min(rw, w - 2)
        if rh < 8 or rw < 8:
            continue
        y0 = rng.randint(0, h - rh - 1)
        x0 = rng.randint(0, w - rw - 1)
        if rng.random() < 0.3:
            cv2.ellipse(mask, (x0 + rw // 2, y0 + rh // 2), (rw // 2, rh // 2),
                        0, 0, 360, 1.0, -1)
        else:
            mask[y0:y0 + rh, x0:x0 + rw] = 1.0
        placed += frac
    if feather > 0:
        k = feather | 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


# ---------------------------------------------------------------------------
# TraceMix
# ---------------------------------------------------------------------------

class TraceMix:
    """Callable: uint8 RGB image -> (augmented image, consistency label).

    y_cons = 1.0: coherent — clean, or one chain applied uniformly.
    y_cons = 0.0: incoherent — chain A globally, contrasting chain B inside region(s).
    """

    def __init__(self, p_incoherent: float = 0.5, p_chain_if_coherent: float = 0.6,
                 area_range=(0.10, 0.40), seed: int | None = None):
        self.p_incoherent = p_incoherent
        self.p_chain_if_coherent = p_chain_if_coherent
        self.area_range = area_range
        self.rng = random.Random(seed)

    def __call__(self, img: np.ndarray, return_mask: bool = False):
        """Returns (img, y_cons) or, if return_mask, (img, y_cons, mask) where mask is the
        HxW float [0,1] alien-region map (the trace-inconsistent area) — all-zeros for
        coherent samples. Used to supervise the DTC localization head (iteration #D6)."""
        rng = self.rng
        h, w = img.shape[:2]
        if rng.random() >= self.p_incoherent:
            if rng.random() < self.p_chain_if_coherent:
                img = sample_chain(rng)(img, rng)
            if return_mask:
                return img, 1.0, np.zeros((h, w), np.float32)
            return img, 1.0

        chain_a = sample_chain(rng)
        chain_b = sample_contrasting_chain(chain_a, rng)
        base = chain_a(img, rng)
        alien = chain_b(img, rng)
        mask2d = _region_mask(h, w, rng, self.area_range)
        mask = mask2d[..., None]
        mixed = base.astype(np.float32) * (1 - mask) + alien.astype(np.float32) * mask
        out = np.clip(mixed, 0, 255).astype(np.uint8)
        if return_mask:
            return out, 0.0, mask2d
        return out, 0.0


def fda_amplitude_swap(img: np.ndarray, donor: np.ndarray, beta: float = 0.05) -> np.ndarray:
    """FDA (CVPR'20) low-frequency amplitude swap: keep img's PHASE (content + forensic
    traces) but take the donor's low-freq AMPLITUDE (global appearance/style).
    Iteration #16: randomizes doc-type appearance during training so type identity
    cannot colonize the decision space (the EGYPT/BENIN hard-fold wall)."""
    h, w = img.shape[:2]
    if donor.shape[:2] != (h, w):
        donor = cv2.resize(donor, (w, h), interpolation=cv2.INTER_LINEAR)
    out = np.empty_like(img, dtype=np.float32)
    b = max(1, int(min(h, w) * beta))
    cy, cx = h // 2, w // 2
    for c in range(3):
        fs = np.fft.fftshift(np.fft.fft2(img[..., c].astype(np.float32)))
        fd = np.fft.fftshift(np.fft.fft2(donor[..., c].astype(np.float32)))
        amp, pha = np.abs(fs), np.angle(fs)
        amp[cy - b:cy + b, cx - b:cx + b] = np.abs(fd)[cy - b:cy + b, cx - b:cx + b]
        out[..., c] = np.real(np.fft.ifft2(np.fft.ifftshift(amp * np.exp(1j * pha))))
    return np.clip(out, 0, 255).astype(np.uint8)
