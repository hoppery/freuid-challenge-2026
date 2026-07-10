from __future__ import annotations
import cv2
import torch
from torch.utils.data import Dataset


class ManifestDataset(Dataset):
    """Reads images listed in a manifest DataFrame. Returns (tensor, label) or
    (tensor, image_id) when with_label=False (inference)."""

    def __init__(self, df, transform, with_label: bool = True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.with_label = with_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.imread(r["path"], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(r["path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = self.transform(image=img)["image"]
        if not self.with_label:
            ident = r["image_id"] if "image_id" in self.df.columns else i
            return x, ident
        return x, torch.tensor(float(r["label"]))


class PatchEvalDataset(Dataset):
    """Eval-time multi-crop: returns (N,3,H,W) of 4 corner + 1 center crops per image
    (each crop = `frac` of the short side, resized to `size`). The evaluator averages
    sigmoid scores over crops — layout/script identity never enters the decision."""

    def __init__(self, df, size: int = 384, frac: float = 0.45):
        from freuid.data.transforms import build_patch_eval_transform
        self.df = df.reset_index(drop=True)
        self.size = size
        self.frac = frac
        self.post = build_patch_eval_transform(size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.imread(r["path"], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(r["path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        c = max(32, int(min(h, w) * self.frac))
        ys = [0, 0, h - c, h - c, (h - c) // 2]
        xs = [0, w - c, 0, w - c, (w - c) // 2]
        crops = []
        for y0, x0 in zip(ys, xs):
            patch = img[y0:y0 + c, x0:x0 + c]
            patch = cv2.resize(patch, (self.size, self.size), interpolation=cv2.INTER_AREA)
            crops.append(self.post(image=patch)["image"])
        return torch.stack(crops), torch.tensor(float(r["label"]))


class DTCTrainDataset(Dataset):
    """Training dataset for the DTC method: geometric pre-aug -> TraceMix (yields the
    self-supervised consistency label) -> normalize. Returns (tensor, y_fraud, y_cons).
    TraceMix is class-agnostic: both fraud and bona-fide get coherent/incoherent traces,
    so the consistency label is independent of (orthogonal to) the fraud label."""

    def __init__(self, df, size: int = 384, p_incoherent: float = 0.5,
                 sbd_p: float = 0.0, patch_mode: bool = False,
                 return_doc: bool = False, fda_p: float = 0.0,
                 heavy_recapture: bool = False, loc_grid: int = 0,
                 appearance_inv: bool = False, appearance_inv_mild: bool = False,
                 patch_scale=(0.08, 0.35), fhag: bool = False, field_tamper_p: float = 0.0):
        from freuid.data.transforms import build_dtc_transforms, build_patch_transforms
        from freuid.data.tracemix import TraceMix
        self.df = df.reset_index(drop=True)
        # localization supervision (iteration #D6): return TraceMix paste mask downsampled
        # to a loc_grid×loc_grid map for the per-patch trace-inconsistency head.
        self.loc_grid = loc_grid
        self.pre, self.post = (build_patch_transforms(size, scale=tuple(patch_scale)) if patch_mode
                               else build_dtc_transforms(size, heavy_recapture, appearance_inv,
                                                         appearance_inv_mild, fhag=fhag))
        # doc-type index for the adversarial GRL head (iteration #13)
        self.return_doc = return_doc
        if return_doc:
            types = sorted(self.df["doc_type"].astype(str).unique())
            self._doc_idx = {t: i for i, t in enumerate(types)}
        # FDA style randomization (iteration #16): swap low-freq amplitude with a donor
        # of a DIFFERENT doc_type -> type appearance randomized, forensic phase kept
        self.fda_p = fda_p
        if fda_p > 0:
            self._by_type = {t: g.index.to_numpy()
                             for t, g in self.df.groupby("doc_type")}
        self.tracemix = TraceMix(p_incoherent=p_incoherent)
        # SBD (iteration #10): convert a fraction of bona-fides into self-blended
        # pseudo-attacks so the model learns generic manipulation traces, not the
        # attack styles of the seen doc types (see data/sbd.py).
        self.sbd_p = sbd_p
        if sbd_p > 0:
            import random as _random
            from freuid.data.sbd import SelfBlendedDoc
            self.sbd = SelfBlendedDoc()
            self._rng = _random.Random()
        # targeted field-manipulation synthesis (research 2026-06-21): genuine → subtle text-field
        # tampering + localized low-QF JPEG (the DeepID-winner compression signal). data/field_tamper.py
        self.field_tamper_p = field_tamper_p
        if field_tamper_p > 0:
            import random as _random2
            self._ft_rng = _random2.Random()

    def __len__(self):
        return len(self.df)

    def _load_pre(self, row):
        img = cv2.imread(row["path"], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(row["path"])
        return self.pre(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))["image"]

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = self._load_pre(r)
        y = float(r["label"])
        if self.fda_p > 0 and len(self._by_type) > 1:
            import random as _r
            if _r.random() < self.fda_p:
                from freuid.data.tracemix import fda_amplitude_swap
                other = [t for t in self._by_type if t != str(r["doc_type"])]
                donor_idx = int(_r.choice(self._by_type[_r.choice(other)]))
                donor = self._load_pre(self.df.iloc[donor_idx])
                img = fda_amplitude_swap(img, donor, beta=_r.uniform(0.01, 0.08))
        forced_incoherent = False
        if self.sbd_p > 0 and y == 0.0 and self._rng.random() < self.sbd_p:
            donor = self._load_pre(self.df.iloc[self._rng.randrange(len(self.df))])
            img = self.sbd(img, donor)
            y = 1.0                    # it is now a manipulated document
            forced_incoherent = True   # mixed processing history by construction
        if (self.field_tamper_p > 0 and y == 0.0 and not forced_incoherent
                and self._ft_rng.random() < self.field_tamper_p):
            from freuid.data.field_tamper import field_tamper
            img = field_tamper(img)    # targeted text-field tampering + localized low-QF JPEG (research)
            y = 1.0
            forced_incoherent = True
        if self.loc_grid > 0:
            img, y_cons, mask = self.tracemix(img, return_mask=True)
            if forced_incoherent:
                y_cons = 0.0
            g = self.loc_grid
            m = cv2.resize(mask, (g, g), interpolation=cv2.INTER_AREA)
            loc = torch.tensor((m > 0.25).astype("float32").reshape(-1))   # (g*g,)
            x = self.post(image=img)["image"]
            return x, torch.tensor(y), torch.tensor(y_cons), loc
        img, y_cons = self.tracemix(img)
        if forced_incoherent:
            y_cons = 0.0
        x = self.post(image=img)["image"]
        if self.return_doc:
            return (x, torch.tensor(y), torch.tensor(y_cons),
                    torch.tensor(self._doc_idx[str(r["doc_type"])]))
        return x, torch.tensor(y), torch.tensor(y_cons)
