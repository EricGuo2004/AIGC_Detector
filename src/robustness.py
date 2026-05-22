from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from .features import (
    FeatureConfig,
    apply_gaussian_noise,
    apply_jpeg_compression,
    apply_resize_attack,
    extract_feature_vector,
    load_grayscale,
    select_feature_array,
)


@dataclass
class RobustnessResult:
    attack: str
    level: str
    accuracy: float
    macro_f1: float


def evaluate_robustness(
    model: object,
    sample_paths: List[str],
    sample_labels: np.ndarray,
    feature_cfg: FeatureConfig,
    feature_set: str = "all",
    rng_seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    results: List[RobustnessResult] = []

    attack_specs = [
        ("jpeg", [95, 75, 50]),
        ("resize", [0.5, 0.75, 1.5]),
        ("noise", [2, 5, 10]),
    ]

    attack_iter = tqdm(attack_specs, desc="Robustness attacks", leave=False)
    for attack_name, levels in attack_iter:
        for level in tqdm(levels, desc=f"{attack_name} levels", leave=False):
            feats = []
            for p in tqdm(sample_paths, desc=f"{attack_name}={level} samples", leave=False):
                image = load_grayscale(p, feature_cfg.image_size)
                if attack_name == "jpeg":
                    image = apply_jpeg_compression(image, int(level))
                elif attack_name == "resize":
                    image = apply_resize_attack(image, float(level))
                elif attack_name == "noise":
                    image = apply_gaussian_noise(image, float(level), rng)
                feats.append(extract_feature_vector(image, feature_cfg))

            X = np.asarray(feats, dtype=np.float32)
            X = select_feature_array(X, feature_cfg, feature_set)
            pred = model.predict(X)
            results.append(
                RobustnessResult(
                    attack=attack_name,
                    level=str(level),
                    accuracy=float(accuracy_score(sample_labels, pred)),
                    macro_f1=float(f1_score(sample_labels, pred, average="macro")),
                )
            )

    return pd.DataFrame([r.__dict__ for r in results])
