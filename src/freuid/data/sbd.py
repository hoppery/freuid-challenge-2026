"""SBD — Self-Blended Documents (iteration #10).

Problem (docs/iterations.md): the EGYPT-fold wall (~0.21) — discriminators cannot catch
ATTACK STYLES absent from training types; the private test's unseen doc types will pose
the same threat. Inspired by Self-Blended Images (SBI, CVPR'22), the strongest OOD win
in face-forgery detection: synthesize pseudo-attacks from bona-fide images only, so the
model learns the generic signature of MANIPULATION ITSELF (splice boundaries, mismatched
processing history) rather than memorizing the attack styles present in training data.

A pseudo-attack = bona-fide base + donor region (from another document) blended in with
a feathered mask, after the donor patch gets a slightly different processing history
(resize/JPEG/color shift) — exactly the forensic situation a real local forgery creates.
Labels: fraud=1 (it IS a manipulated document) and trace-consistency=0 (mixed history),
so SBD supervises BOTH heads of DTCNet.
"""
from __future__ import annotations
import random
import cv2
import numpy as np

from freuid.data.tracemix import _region_mask, apply_jpeg_chain


def _donor_patch_transform(patch: np.ndarray, rng: random.Random) -> np.ndarray:
    """Give the donor patch a slightly different processing history."""
    h, w = patch.shape[:2]
    if rng.random() < 0.7:   # resolution history mismatch
        s = rng.uniform(0.6, 0.95)
        small = cv2.resize(patch, (max(4, int(w * s)), max(4, int(h * s))),
                           interpolation=cv2.INTER_AREA)
        patch = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.7:   # color/illumination mismatch
        shift = np.array([rng.uniform(-12, 12) for _ in range(3)], np.float32)
        gain = rng.uniform(0.92, 1.08)
        patch = np.clip(patch.astype(np.float32) * gain + shift, 0, 255).astype(np.uint8)
    if rng.random() < 0.6:   # compression history mismatch
        patch = apply_jpeg_chain(patch, [rng.randint(60, 92)])
    return patch


class SelfBlendedDoc:
    """Callable (base, donor) -> pseudo-attack image. Both uint8 RGB, same HxW."""

    def __init__(self, area_range=(0.04, 0.25), feather: int = 21,
                 seed: int | None = None):
        self.area_range = area_range
        self.feather = feather
        self.rng = random.Random(seed)

    def __call__(self, base: np.ndarray, donor: np.ndarray) -> np.ndarray:
        rng = self.rng
        h, w = base.shape[:2]
        if donor.shape[:2] != (h, w):
            donor = cv2.resize(donor, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = _region_mask(h, w, rng, self.area_range, self.feather)[..., None]
        donor = _donor_patch_transform(donor, rng)
        out = base.astype(np.float32) * (1 - mask) + donor.astype(np.float32) * mask
        return np.clip(out, 0, 255).astype(np.uint8)
