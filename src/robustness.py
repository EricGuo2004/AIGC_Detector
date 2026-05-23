from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from tqdm import tqdm

from .features import (
    FeatureConfig,
    apply_gaussian_noise,
    apply_jpeg_compression,
    apply_resize_attack,
    extract_feature_vector,
    load_feature_image,
    select_feature_array,
)


ROBUSTNESS_ATTACKS: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("clean", ("none",)),
    ("jpeg", (95, 75, 50)),
    ("resize", (0.5, 0.75, 1.5)),
    ("noise", (2, 5, 10)),
)


@dataclass
class RobustnessResult:
    task: str
    attack: str
    level: str
    n_samples: int
    accuracy: float
    macro_f1: float
    auc: float
    positive_rate: float
    mean_ai_probability: float
    skipped_images: int
    model_output: str
    feature_profile: str
    true_distribution: str
    prediction_distribution: str


def _cfg_json(cfg: FeatureConfig) -> str:
    return json.dumps(cfg.__dict__, sort_keys=True)


def _safe_token(value: object) -> str:
    text = str(value).replace(".", "p")
    keep = [ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text]
    return "".join(keep).strip("_") or "default"


def robustness_cache_path(
    cache_dir: Path | None,
    task_name: str,
    cache_tag: str,
    attack_name: str,
    level: object,
) -> Path | None:
    if cache_dir is None:
        return None
    name = (
        f"{_safe_token(task_name)}_{_safe_token(cache_tag)}_"
        f"{_safe_token(attack_name)}_{_safe_token(level)}.npz"
    )
    return cache_dir / name


def _load_attack_cache(
    cache_path: Path,
    sample_paths: Sequence[str],
    sample_labels: np.ndarray,
    feature_cfg: FeatureConfig,
    attack_name: str,
    level: object,
    desc: str,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]] | None:
    if not cache_path.exists():
        return None

    try:
        with np.load(cache_path, allow_pickle=False) as data:
            if str(data["cfg_json"][0]) != _cfg_json(feature_cfg):
                return None
            if str(data["attack_name"][0]) != str(attack_name):
                return None
            if str(data["level"][0]) != str(level):
                return None
            if data["input_paths"].tolist() != list(sample_paths):
                return None
            if data["input_labels"].tolist() != sample_labels.tolist():
                return None

            X = data["X"].astype(np.float32, copy=False)
            y = data["kept_labels"].astype(sample_labels.dtype, copy=False)
            skipped = [
                (str(path), str(err))
                for path, err in zip(data["skipped_paths"].tolist(), data["skipped_errors"].tolist())
            ]
            print(f"[robust cache hit] {desc}: {cache_path}")
            return X, y, skipped
    except Exception as exc:
        print(f"[robust cache ignored] {cache_path}: {exc}")
        return None


def _write_attack_cache(
    cache_path: Path | None,
    sample_paths: Sequence[str],
    sample_labels: np.ndarray,
    feature_cfg: FeatureConfig,
    attack_name: str,
    level: object,
    X: np.ndarray,
    y_kept: np.ndarray,
    skipped: Sequence[Tuple[str, str]],
    desc: str,
) -> None:
    if cache_path is None:
        return

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        X=X.astype(np.float32, copy=False),
        kept_labels=np.asarray(y_kept),
        cfg_json=np.asarray([_cfg_json(feature_cfg)]),
        attack_name=np.asarray([str(attack_name)]),
        level=np.asarray([str(level)]),
        input_paths=np.asarray(list(sample_paths)),
        input_labels=np.asarray(sample_labels),
        skipped_paths=np.asarray([p for p, _ in skipped]),
        skipped_errors=np.asarray([err for _, err in skipped]),
    )
    print(f"[robust cache write] {desc}: {cache_path}")


def _apply_attack(
    image: np.ndarray,
    attack_name: str,
    level: object,
    rng: np.random.Generator,
) -> np.ndarray:
    if attack_name == "clean":
        return image
    if attack_name == "jpeg":
        return apply_jpeg_compression(image, int(float(level)))
    if attack_name == "resize":
        return apply_resize_attack(image, float(level))
    if attack_name == "noise":
        return apply_gaussian_noise(image, float(level), rng)
    raise ValueError(f"Unknown robustness attack: {attack_name}")


def _extract_attacked_feature(
    payload: Tuple[int, str, int, dict, str, object, int],
) -> Tuple[int, np.ndarray | None, int | None, Tuple[str, str] | None]:
    idx, path, label, cfg_dict, attack_name, level, rng_seed = payload
    cfg = FeatureConfig(**cfg_dict)
    try:
        image = load_feature_image(path, cfg)
        level_seed = 0 if str(level) == "none" else int(float(level) * 1000)
        rng = np.random.default_rng((rng_seed + idx * 1_000_003 + level_seed) % (2**32))
        attacked = _apply_attack(image, attack_name, level, rng)
        return idx, extract_feature_vector(attacked, cfg), label, None
    except Exception as exc:
        return idx, None, None, (path, str(exc))


def _extract_attack_matrix(
    sample_paths: Sequence[str],
    sample_labels: np.ndarray,
    feature_cfg: FeatureConfig,
    attack_name: str,
    level: object,
    desc: str,
    num_workers: int = 0,
    chunksize: int = 32,
    cache_path: Path | None = None,
    rng_seed: int = 42,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    if cache_path is not None and not force_recompute:
        cached = _load_attack_cache(cache_path, sample_paths, sample_labels, feature_cfg, attack_name, level, desc)
        if cached is not None:
            return cached

    feats: List[np.ndarray] = []
    y_kept: List[int] = []
    skipped: List[Tuple[str, str]] = []

    if num_workers and num_workers > 1 and len(sample_paths) > 1:
        worker_count = min(num_workers, len(sample_paths))
        cfg_dict = feature_cfg.__dict__.copy()
        payloads = (
            (idx, str(path), int(label), cfg_dict, attack_name, level, rng_seed)
            for idx, (path, label) in enumerate(zip(sample_paths, sample_labels))
        )
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            iterator = executor.map(_extract_attacked_feature, payloads, chunksize=max(1, chunksize))
            for _idx, feat, label, err in tqdm(iterator, total=len(sample_paths), desc=desc, leave=False):
                if err is not None:
                    skipped.append(err)
                elif feat is not None and label is not None:
                    feats.append(feat)
                    y_kept.append(label)
    else:
        cfg_dict = feature_cfg.__dict__.copy()
        for idx, (path, label) in enumerate(tqdm(zip(sample_paths, sample_labels), total=len(sample_paths), desc=desc, leave=False)):
            _idx, feat, kept_label, err = _extract_attacked_feature(
                (idx, str(path), int(label), cfg_dict, attack_name, level, rng_seed)
            )
            if err is not None:
                skipped.append(err)
            elif feat is not None and kept_label is not None:
                feats.append(feat)
                y_kept.append(kept_label)

    X = np.asarray(feats, dtype=np.float32)
    y_arr = np.asarray(y_kept, dtype=sample_labels.dtype)
    _write_attack_cache(
        cache_path,
        sample_paths,
        sample_labels,
        feature_cfg,
        attack_name,
        level,
        X,
        y_arr,
        skipped,
        desc,
    )
    return X, y_arr, skipped


def evaluate_robustness(
    model: object,
    sample_paths: List[str],
    sample_labels: np.ndarray,
    feature_cfg: FeatureConfig,
    feature_set: str = "all",
    rng_seed: int = 42,
    task_name: str = "",
    model_output: str = "",
    num_workers: int = 0,
    chunksize: int = 32,
    cache_dir: Path | None = None,
    cache_tag: str = "default",
    force_recompute: bool = False,
    expected_n_features: int | None = None,
    details_dir: Path | None = None,
    id_to_label: dict[int, str] | None = None,
) -> pd.DataFrame:
    results: List[RobustnessResult] = []
    labels = [id_to_label[i] for i in sorted(id_to_label)] if id_to_label else [str(i) for i in sorted(set(sample_labels.tolist()))]
    ai_idx = None
    if id_to_label:
        for idx, label in id_to_label.items():
            if str(label).lower() == "ai":
                ai_idx = int(idx)
                break

    attack_iter = tqdm(ROBUSTNESS_ATTACKS, desc="Robustness attacks", leave=False)
    for attack_name, levels in attack_iter:
        for level in tqdm(levels, desc=f"{attack_name} levels", leave=False):
            desc = f"{task_name or 'task'} {attack_name}={level}"
            cache_path = robustness_cache_path(cache_dir, task_name or "task", cache_tag, attack_name, level)
            X_full, y_kept, skipped = _extract_attack_matrix(
                sample_paths,
                sample_labels,
                feature_cfg,
                attack_name,
                level,
                desc=desc,
                num_workers=num_workers,
                chunksize=chunksize,
                cache_path=cache_path,
                rng_seed=rng_seed,
                force_recompute=force_recompute,
            )
            if X_full.shape[0] == 0:
                raise ValueError(f"No valid images left after applying {attack_name}={level}.")

            X = select_feature_array(X_full, feature_cfg, feature_set)
            if expected_n_features is not None and X.shape[1] != expected_n_features:
                raise ValueError(
                    f"Feature dimension mismatch for {desc}: got {X.shape[1]}, "
                    f"expected {expected_n_features}."
                )
            pred = model.predict(X)
            proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
            true_counts = {
                labels[int(label_id)] if int(label_id) < len(labels) else str(label_id): int((y_kept == label_id).sum())
                for label_id in sorted(set(y_kept.tolist()))
            }
            pred_counts = {
                labels[int(label_id)] if int(label_id) < len(labels) else str(label_id): int((pred == label_id).sum())
                for label_id in sorted(set(pred.tolist()))
            }
            auc = float("nan")
            mean_ai_probability = float("nan")
            positive_rate = float("nan")
            if ai_idx is not None:
                positive_rate = float(np.mean(pred == ai_idx))
                if proba is not None and proba.shape[1] > ai_idx:
                    ai_proba = proba[:, ai_idx]
                    mean_ai_probability = float(np.mean(ai_proba))
                    if len(np.unique(y_kept == ai_idx)) == 2:
                        auc = float(roc_auc_score((y_kept == ai_idx).astype(int), ai_proba))

            if details_dir is not None:
                details_dir.mkdir(parents=True, exist_ok=True)
                token = f"{_safe_token(attack_name)}_{_safe_token(level)}"
                cm = confusion_matrix(y_kept, pred, labels=list(range(len(labels))))
                pd.DataFrame(cm, index=labels, columns=labels).to_csv(details_dir / f"confusion_matrix_{token}.csv")
                pd.DataFrame(
                    [
                        {"label": label, "true_count": true_counts.get(label, 0), "pred_count": pred_counts.get(label, 0)}
                        for label in labels
                    ]
                ).to_csv(details_dir / f"prediction_distribution_{token}.csv", index=False)

            results.append(
                RobustnessResult(
                    task=task_name,
                    attack=attack_name,
                    level=str(level),
                    n_samples=int(len(y_kept)),
                    accuracy=float(accuracy_score(y_kept, pred)),
                    macro_f1=float(f1_score(y_kept, pred, average="macro")),
                    auc=auc,
                    positive_rate=positive_rate,
                    mean_ai_probability=mean_ai_probability,
                    skipped_images=int(len(skipped)),
                    model_output=model_output,
                    feature_profile=feature_cfg.feature_profile,
                    true_distribution=json.dumps(true_counts, sort_keys=True, ensure_ascii=False),
                    prediction_distribution=json.dumps(pred_counts, sort_keys=True, ensure_ascii=False),
                )
            )

    return pd.DataFrame([r.__dict__ for r in results])
