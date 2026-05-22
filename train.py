from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_utils import (
    Sample,
    discover_ai_subsource_from_roots,
    discover_ai_subsource_split,
    discover_binary_split_multi_root,
    discover_model_roots,
    summarize_labels,
    validate_non_empty,
)
from src.features import (
    FEATURE_SET_CHOICES,
    FEATURE_PROFILE_CHOICES,
    TRAIN_AUGMENTATION_CHOICES,
    FeatureConfig,
    extract_feature_variants_from_path,
    make_feature_names,
    select_feature_columns,
)
from src.robustness import evaluate_robustness
from src.training import (
    LGBM_PROFILE_CHOICES,
    MODEL_ARCHITECTURE_CHOICES,
    MODEL_SET_CHOICES,
    feature_importance_df,
    train_and_select,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AIGC frequency-domain fingerprint training pipeline")
    parser.add_argument("--dataset-root", type=str, default="data", help="Dataset root folder")
    parser.add_argument("--out-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--radial-bins", type=int, default=64)
    parser.add_argument("--angular-bins", type=int, default=18)
    parser.add_argument("--patch-grid", type=int, default=4)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Parallel workers for feature extraction. 0 or 1 runs sequentially.",
    )
    parser.add_argument(
        "--feature-chunksize",
        type=int,
        default=32,
        help="Chunk size for parallel feature extraction.",
    )
    parser.add_argument(
        "--lightgbm-device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Device for LightGBM training. Feature extraction remains CPU-based.",
    )
    parser.add_argument(
        "--lgbm-profile",
        choices=LGBM_PROFILE_CHOICES,
        default="baseline",
        help="LightGBM hyperparameter profile.",
    )
    parser.add_argument(
        "--model-set",
        choices=MODEL_SET_CHOICES,
        default="all",
        help="Candidate models to train. Use lightgbm for tuning-only runs.",
    )
    parser.add_argument(
        "--feature-cache-dir",
        type=str,
        default="",
        help="Optional directory for cached full feature matrices.",
    )
    parser.add_argument(
        "--feature-set",
        choices=FEATURE_SET_CHOICES,
        default="all",
        help="Feature subset used for training. Cache always stores the full feature vector.",
    )
    parser.add_argument(
        "--feature-profile",
        choices=FEATURE_PROFILE_CHOICES,
        default="baseline",
        help="Feature extraction profile. Use enhanced to append extra interpretable frequency features.",
    )
    parser.add_argument(
        "--model-architecture",
        choices=MODEL_ARCHITECTURE_CHOICES,
        default="flat",
        help="Model architecture for LightGBM-based experiments.",
    )
    parser.add_argument(
        "--train-augmentation",
        choices=TRAIN_AUGMENTATION_CHOICES,
        default="none",
        help="Feature-space training augmentation generated from image perturbations. Validation is never augmented.",
    )
    parser.add_argument(
        "--calibrate-threshold",
        action="store_true",
        help="Use a training-only calibration split to tune binary decision threshold.",
    )
    parser.add_argument("--skip-robustness", action="store_true", help="Skip robustness evaluation")
    return parser.parse_args()


def encode_labels(samples: Sequence[Sample], label_to_id: Dict[str, int]) -> np.ndarray:
    return np.asarray([label_to_id[s.label] for s in samples], dtype=np.int64)


def _cfg_json(cfg: FeatureConfig) -> str:
    payload = cfg.__dict__.copy()
    # Keep baseline cache compatibility with feature matrices written before
    # feature_profile existed. Enhanced caches still get a distinct signature.
    if payload.get("feature_profile") == "baseline":
        payload.pop("feature_profile", None)
    return json.dumps(payload, sort_keys=True)


def _safe_cache_token(value: str) -> str:
    keep = []
    for ch in value:
        keep.append(ch if ch.isalnum() or ch in ("-", "_", ".") else "_")
    return "".join(keep).strip("_") or "default"


def sample_group(sample: Sample) -> str:
    parts = list(sample.path.parts)
    for marker in ("train", "val", "test"):
        if marker in parts:
            idx = parts.index(marker)
            if idx > 0:
                return parts[idx - 1]
    return sample.path.parent.name


def feature_cache_path(
    cache_dir: Path | None,
    task_name: str,
    split_name: str,
    cache_tag: str,
) -> Path | None:
    if cache_dir is None:
        return None
    name = f"{task_name}_{split_name}_{_safe_cache_token(cache_tag)}.npz"
    return cache_dir / name


def _load_feature_cache(
    cache_path: Path,
    samples: Sequence[Sample],
    cfg: FeatureConfig,
    desc: str,
) -> Tuple[np.ndarray, List[Sample], List[Tuple[str, str]]] | None:
    if not cache_path.exists():
        return None

    expected_paths = [str(s.path) for s in samples]
    expected_labels = [s.label for s in samples]
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            if str(data["cfg_json"][0]) != _cfg_json(cfg):
                return None
            if data["input_paths"].tolist() != expected_paths:
                return None
            if data["input_labels"].tolist() != expected_labels:
                return None

            X = data["X"].astype(np.float32, copy=False)
            kept_samples = [
                Sample(path=Path(p), label=str(label))
                for p, label in zip(data["kept_paths"].tolist(), data["kept_labels"].tolist())
            ]
            skipped = [
                (str(path), str(err))
                for path, err in zip(data["skipped_paths"].tolist(), data["skipped_errors"].tolist())
            ]
            print(f"[cache hit] {desc}: {cache_path}")
            return X, kept_samples, skipped
    except Exception as exc:
        print(f"[cache ignored] {cache_path}: {exc}")
        return None


def _write_feature_cache(
    cache_path: Path | None,
    samples: Sequence[Sample],
    cfg: FeatureConfig,
    X: np.ndarray,
    kept_samples: Sequence[Sample],
    skipped: Sequence[Tuple[str, str]],
    desc: str,
) -> None:
    if cache_path is None:
        return

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    skipped_paths = [p for p, _ in skipped]
    skipped_errors = [err for _, err in skipped]
    np.savez(
        cache_path,
        X=X.astype(np.float32, copy=False),
        cfg_json=np.asarray([_cfg_json(cfg)]),
        input_paths=np.asarray([str(s.path) for s in samples]),
        input_labels=np.asarray([s.label for s in samples]),
        kept_paths=np.asarray([str(s.path) for s in kept_samples]),
        kept_labels=np.asarray([s.label for s in kept_samples]),
        skipped_paths=np.asarray(skipped_paths),
        skipped_errors=np.asarray(skipped_errors),
    )
    print(f"[cache write] {desc}: {cache_path}")


def extract_matrix(
    samples: Sequence[Sample],
    cfg: FeatureConfig,
    desc: str,
    num_workers: int = 0,
    chunksize: int = 32,
    cache_path: Path | None = None,
    train_augmentation: str = "none",
) -> Tuple[np.ndarray, List[Sample], List[Tuple[str, str]]]:
    if cache_path is not None:
        cached = _load_feature_cache(cache_path, samples, cfg, desc)
        if cached is not None:
            return cached

    feats: List[np.ndarray] = []
    kept_samples: List[Sample] = []
    skipped: List[Tuple[str, str]] = []

    if num_workers and num_workers > 1 and len(samples) > 1:
        cfg_dict = cfg.__dict__.copy()
        worker_count = min(num_workers, len(samples))
        payloads = ((i, s, cfg_dict, train_augmentation) for i, s in enumerate(samples))
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            iterator = executor.map(_extract_one_feature, payloads, chunksize=max(1, chunksize))
            for _idx, sample, sample_feats, err in tqdm(iterator, total=len(samples), desc=desc, leave=False):
                if err is not None:
                    skipped.append(err)
                elif sample_feats:
                    for feat in sample_feats:
                        feats.append(feat)
                        kept_samples.append(sample)
        X = np.asarray(feats, dtype=np.float32)
        _write_feature_cache(cache_path, samples, cfg, X, kept_samples, skipped, desc)
        return X, kept_samples, skipped

    for s in tqdm(samples, desc=desc, leave=False):
        try:
            sample_feats = extract_feature_variants_from_path(
                str(s.path),
                cfg,
                train_augmentation=train_augmentation,
                seed=abs(hash(str(s.path))) % (2**32),
            )
            for feat in sample_feats:
                feats.append(feat)
                kept_samples.append(s)
        except Exception as exc:
            skipped.append((str(s.path), str(exc)))
    X = np.asarray(feats, dtype=np.float32)
    _write_feature_cache(cache_path, samples, cfg, X, kept_samples, skipped, desc)
    return X, kept_samples, skipped


def _extract_one_feature(
    payload: Tuple[int, Sample, Dict[str, object], str]
) -> Tuple[int, Sample, List[np.ndarray] | None, Tuple[str, str] | None]:
    idx, sample, cfg_dict, train_augmentation = payload
    cfg = FeatureConfig(**cfg_dict)
    try:
        feats = extract_feature_variants_from_path(
            str(sample.path),
            cfg,
            train_augmentation=train_augmentation,
            seed=(idx + 17) % (2**32),
        )
        return idx, sample, feats, None
    except Exception as exc:
        return idx, sample, None, (str(sample.path), str(exc))


def save_metrics_block(
    out_path: Path,
    best_name: str,
    all_results: List[dict],
    label_names: List[str],
) -> None:
    content = {
        "best_model": best_name,
        "all_models": all_results,
        "labels": label_names,
    }
    out_path.write_text(json.dumps(content, indent=2), encoding="utf-8")


def run_task(
    task_name: str,
    train_samples: Sequence[Sample],
    val_samples: Sequence[Sample],
    out_dir: Path,
    cfg: FeatureConfig,
    run_robustness: bool,
    lightgbm_device: str = "cpu",
    num_workers: int = 0,
    feature_chunksize: int = 32,
    feature_cache_dir: Path | None = None,
    cache_tag: str = "default",
    feature_set: str = "all",
    lightgbm_profile: str = "baseline",
    model_set: str = "all",
    model_architecture: str = "flat",
    train_augmentation: str = "none",
    calibrate_threshold: bool = False,
) -> None:
    print(f"\n===== Task: {task_name} =====")
    print(f"Train counts: {summarize_labels(train_samples)}")
    print(f"Val counts:   {summarize_labels(val_samples)}")
    print(f"Feature set:  {feature_set}")
    print(f"Feature profile: {cfg.feature_profile}")
    print(f"LGBM profile: {lightgbm_profile}")
    print(f"Model set:    {model_set}")
    print(f"Model architecture: {model_architecture}")
    print(f"Train augmentation: {train_augmentation}")
    print(f"Calibrate threshold: {calibrate_threshold}")

    label_names = sorted({s.label for s in train_samples})
    label_to_id = {name: i for i, name in enumerate(label_names)}

    X_train, train_kept, train_skipped = extract_matrix(
        train_samples,
        cfg,
        desc=f"{task_name} train features",
        num_workers=num_workers,
        chunksize=feature_chunksize,
        cache_path=feature_cache_path(
            feature_cache_dir,
            task_name,
            "train",
            f"{cache_tag}_aug{train_augmentation}",
        ),
        train_augmentation=train_augmentation,
    )
    X_val, val_kept, val_skipped = extract_matrix(
        val_samples,
        cfg,
        desc=f"{task_name} val(test) features",
        num_workers=num_workers,
        chunksize=feature_chunksize,
        cache_path=feature_cache_path(feature_cache_dir, task_name, "val", cache_tag),
        train_augmentation="none",
    )
    y_train = encode_labels(train_kept, label_to_id)
    y_val = encode_labels(val_kept, label_to_id)

    if len(train_skipped) > 0 or len(val_skipped) > 0:
        print(
            f"[Warning] Skipped invalid images - train: {len(train_skipped)}, val: {len(val_skipped)}"
        )
        preview = (train_skipped + val_skipped)[:3]
        for p, err in preview:
            print(f"  skipped: {p}")
            print(f"  reason : {err}")

    if X_train.shape[0] == 0 or X_val.shape[0] == 0:
        raise ValueError("No valid images left after filtering unreadable/corrupted files.")

    full_feat_names = make_feature_names(cfg)
    X_train, feat_names = select_feature_columns(X_train, full_feat_names, cfg, feature_set)
    X_val, _ = select_feature_columns(X_val, full_feat_names, cfg, feature_set)
    print(f"Selected features: {len(feat_names)} / {len(full_feat_names)}")
    train_groups = [sample_group(s) for s in train_kept]
    val_groups = [sample_group(s) for s in val_kept]

    best, all_results = train_and_select(
        X_train,
        y_train,
        X_val,
        y_val,
        label_names,
        lightgbm_device=lightgbm_device,
        lightgbm_profile=lightgbm_profile,
        model_set=model_set,
        model_architecture=model_architecture,
        calibrate_threshold=calibrate_threshold,
        train_groups=train_groups,
        val_groups=val_groups,
    )
    task_out = out_dir / task_name
    task_out.mkdir(parents=True, exist_ok=True)

    fi = feature_importance_df(best.model, feat_names)
    fi.to_csv(task_out / "feature_importance.csv", index=False)

    all_rows = []
    for r in all_results:
        row = {"model": r.model_name, **r.metrics}
        all_rows.append(row)
        (task_out / f"classification_report_{r.model_name}.txt").write_text(r.report_text, encoding="utf-8")
        pd.DataFrame(r.confusion).to_csv(task_out / f"confusion_matrix_{r.model_name}.csv", index=False)
    model_compare_df = pd.DataFrame(all_rows)
    model_compare_df.to_csv(task_out / "model_comparison.csv", index=False)
    save_metrics_block(task_out / "metrics_summary.json", best.model_name, all_rows, label_names)

    model_bundle = {
        "model": best.model,
        "label_to_id": label_to_id,
        "id_to_label": {v: k for k, v in label_to_id.items()},
        "feature_config": cfg.__dict__,
        "feature_names": feat_names,
        "feature_profile": cfg.feature_profile,
        "task_name": task_name,
        "lightgbm_device": lightgbm_device,
        "lightgbm_profile": lightgbm_profile,
        "model_set": model_set,
        "model_architecture": model_architecture,
        "train_augmentation": train_augmentation,
        "calibrate_threshold": calibrate_threshold,
        "num_workers": num_workers,
        "feature_set": feature_set,
        "cache_tag": cache_tag,
    }
    joblib.dump(model_bundle, task_out / "best_model.joblib")

    # Terminal summary for test results (val is used as test split in this project).
    print(f"\n[Terminal Summary] {task_name} (tested on val)")
    print(f"Best model: {best.model_name}")
    print("Best metrics:")
    for k, v in best.metrics.items():
        print(f"  {k}: {v:.6f}")
    print("\nAll model comparison:")
    print(model_compare_df.to_string(index=False))
    print(f"\nClassification report ({best.model_name}):")
    print(best.report_text)
    print(f"Confusion matrix ({best.model_name}):")
    print(pd.DataFrame(best.confusion).to_string(index=False, header=False))

    if run_robustness:
        val_paths = [str(s.path) for s in val_kept]
        robust_df = evaluate_robustness(best.model, val_paths, y_val, cfg, feature_set=feature_set)
        robust_df.to_csv(task_out / "robustness_results.csv", index=False)
        print("\nRobustness summary (val under attacks):")
        print(robust_df.to_string(index=False))
        print("Robustness evaluation saved.")


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    feature_cache_dir = Path(args.feature_cache_dir) if args.feature_cache_dir else None

    cfg = FeatureConfig(
        image_size=args.image_size,
        radial_bins=args.radial_bins,
        angular_bins=args.angular_bins,
        patch_grid=args.patch_grid,
        feature_profile=args.feature_profile,
    )

    # Task 1 in the plan: detect whether image is AI-generated (ai vs nature).
    model_roots = discover_model_roots(dataset_root)
    used_root = dataset_root

    # Friendly fallback: if default "data" is not valid, try current folder / script folder.
    if not model_roots:
        fallback_roots: List[Path] = [Path("."), Path(__file__).resolve().parent]
        for candidate in fallback_roots:
            candidate_roots = discover_model_roots(candidate)
            if candidate_roots:
                model_roots = candidate_roots
                used_root = candidate
                print(
                    f"[Info] dataset root '{dataset_root}' not found/invalid. "
                    f"Auto-switched to '{candidate}'."
                )
                break

    if not model_roots:
        raise ValueError(
            f"No valid dataset structure found under '{dataset_root}'. "
            "Expected either (1) train/val directly under dataset root, or "
            "(2) nested model folders each containing train/val. "
            "Try: python train.py --dataset-root . --out-dir outputs"
        )
    print("Detected dataset roots:")
    for r in model_roots:
        print(f"  - {r}")

    print("\n[Plan Task 1] Running AI-vs-Nature detection (binary classification)")
    train_binary = discover_binary_split_multi_root(used_root, "train")
    val_binary = discover_binary_split_multi_root(used_root, "val")
    validate_non_empty(train_binary, "train")
    validate_non_empty(val_binary, "val")
    run_task(
        task_name="binary_ai_vs_nature",
        train_samples=train_binary,
        val_samples=val_binary,
        out_dir=out_dir,
        cfg=cfg,
        run_robustness=not args.skip_robustness,
        lightgbm_device=args.lightgbm_device,
        num_workers=args.num_workers,
        feature_chunksize=args.feature_chunksize,
        feature_cache_dir=feature_cache_dir,
        cache_tag="full",
        feature_set=args.feature_set,
        lightgbm_profile=args.lgbm_profile,
        model_set=args.model_set,
        model_architecture=args.model_architecture,
        train_augmentation=args.train_augmentation,
        calibrate_threshold=args.calibrate_threshold,
    )

    # Task 2 in the plan: identify which AI model/source generated the image.
    train_sub = discover_ai_subsource_split(used_root, "train")
    val_sub = discover_ai_subsource_split(used_root, "val")
    if not train_sub or not val_sub:
        # Fallback for layout where each model has its own root/train/ai folder.
        train_sub = discover_ai_subsource_from_roots(used_root, "train")
        val_sub = discover_ai_subsource_from_roots(used_root, "val")

    if train_sub and val_sub and len({s.label for s in train_sub}) >= 2:
        print("\n[Plan Task 2] Running AI source attribution (multi-class classification)")
        run_task(
            task_name="ai_subsource_attribution",
            train_samples=train_sub,
            val_samples=val_sub,
            out_dir=out_dir,
            cfg=cfg,
            run_robustness=False,
            lightgbm_device=args.lightgbm_device,
            num_workers=args.num_workers,
            feature_chunksize=args.feature_chunksize,
            feature_cache_dir=feature_cache_dir,
            cache_tag="full",
            feature_set=args.feature_set,
            lightgbm_profile=args.lgbm_profile,
            model_set=args.model_set,
            model_architecture=args.model_architecture,
            train_augmentation=args.train_augmentation,
            calibrate_threshold=args.calibrate_threshold,
        )
    else:
        print(
            "\nSkip ai_subsource_attribution: no ai subfolders found. "
            "If needed, prepare data as train/ai/ADM_SELECTED/... or provide multiple model roots."
        )

    print("\nAll done. Check outputs/ for models and reports.")


if __name__ == "__main__":
    main()
