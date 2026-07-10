"""Fourier Domain Adaptation (FDA) augmentation for cross-doc-type generalization.

Iteration 5. 5-fold LODO showed the model fails on held-out doc-types whose low-frequency
"style" (color cast, illumination, substrate tint, global layout chroma) differs from the
training types (esp. MAURITIUS/ID). FDA (Yang & Soatto, CVPR'20) swaps the LOW-FREQUENCY
amplitude band of a training image with that of a random reference image from a DIFFERENT
doc-type, keeping phase (content) intact → the fraud cue stays, the doc-type style is
randomized → the model stops keying on doc-type-specific style and generalizes to unseen
types. Corroborated lever (parallel pipeline used FDA-stabilized members).
"""
from __future__ import annotations
import random
import cv2
import numpy as np
import albumentations as A


def _fda_swap(src: np.ndarray, ref: np.ndarray, beta: float) -> np.ndarray:
    """src, ref: HxWx3 uint8 (same size). Replace src low-freq amplitude with ref's."""
    s = src.astype(np.float32)
    r = ref.astype(np.float32)
    out = np.empty_like(s)
    H, W = s.shape[:2]
    b = max(1, int(min(H, W) * beta) // 2)
    cy, cx = H // 2, W // 2
    for c in range(3):
        sf = np.fft.fftshift(np.fft.fft2(s[..., c]))
        rf = np.fft.fftshift(np.fft.fft2(r[..., c]))
        s_amp, s_pha = np.abs(sf), np.angle(sf)
        s_amp[cy - b:cy + b, cx - b:cx + b] = np.abs(rf)[cy - b:cy + b, cx - b:cx + b]
        rec = np.fft.ifft2(np.fft.ifftshift(s_amp * np.exp(1j * s_pha))).real
        out[..., c] = rec
    return np.clip(out, 0, 255).astype(np.uint8)


class FDATransform(A.ImageOnlyTransform):
    """Swap low-freq amplitude with a random reference (pre-loaded, resized to `size`)."""

    def __init__(self, ref_paths, size: int, beta: float = 0.05,
                 n_pool: int = 96, p: float = 0.5):
        super().__init__(p=p)
        self.beta = beta
        self.size = size
        # pre-load a pool of references resized to size x size (uint8 RGB)
        self.pool = []
        for pth in list(ref_paths)[:n_pool]:
            img = cv2.imread(pth, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            self.pool.append(img)

    def get_params(self):
        return {"ridx": random.randrange(len(self.pool)) if self.pool else -1}

    def apply(self, img, ridx=-1, **params):
        if ridx < 0 or not self.pool:
            return img
        ref = self.pool[ridx]
        if ref.shape[:2] != img.shape[:2]:
            ref = cv2.resize(ref, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
        return _fda_swap(img, ref, self.beta)
