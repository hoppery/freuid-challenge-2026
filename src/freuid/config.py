from __future__ import annotations
from dataclasses import dataclass, asdict
import yaml


@dataclass
class Config:
    manifest: str = "manifests/freuid.parquet"
    backbone: str = "convnextv2_tiny.fcmae_ft_in22k_in1k"
    img_size: int = 384
    batch_size: int = 32
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 0.05
    num_workers: int = 8
    val_fraction: float = 0.15
    holdout_by: str = "doc_type"   # domain-holdout column ("" = random split)
    seed: int = 42
    out_dir: str = "checkpoints/baseline"
    amp: bool = True
    pretrained: bool = True

    @staticmethod
    def load(path: str) -> "Config":
        with open(path) as f:
            return Config(**(yaml.safe_load(f) or {}))

    def save(self, path: str):
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f)
