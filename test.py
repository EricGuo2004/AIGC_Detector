from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Sequence

from src.data_utils import (
    Sample,
    discover_ai_subsource_from_roots,
    discover_ai_subsource_split,
    discover_binary_split_multi_root,
    discover_model_roots,
    validate_non_empty,
)
from src.features import FEATURE_PROFILE_CHOICES, FEATURE_SET_CHOICES, TRAIN_AUGMENTATION_CHOICES, FeatureConfig
from src.training import LGBM_PROFILE_CHOICES, MODEL_ARCHITECTURE_CHOICES, MODEL_SET_CHOICES
from train import run_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick smoke test pipeline (same flow, sampled data).")
    parser.add_argument("--dataset-root", type=str, default="data", help="Dataset root folder")
    parser.add_argument("--out-dir", type=str, default="outputs_smoke", help="Output directory")
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
        "--sample-cache-dir",
        type=str,
        default="",
        help="Optional directory for cached sampled path lists.",
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
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.1,
        help="Per-label sampling fraction for smoke test. Default is 0.1.",
    )
    parser.add_argument("--sample-seed", type=int, default=42, help="Random seed for sample subsetting")
    return parser.parse_args()


def subsample_by_label(samples: Sequence[Sample], fraction: float, seed: int) -> List[Sample]:
    if fraction >= 1.0:
        return list(samples)
    if fraction <= 0.0:
        raise ValueError("--sample-fraction must be > 0")

    import numpy as np

    grouped = defaultdict(list)
    for s in samples:
        grouped[s.label].append(s)

    rng = np.random.default_rng(seed)
    out: List[Sample] = []
    for label, items in grouped.items():
        keep = max(1, int(round(len(items) * fraction)))
        keep = min(keep, len(items))
        idx = rng.choice(len(items), size=keep, replace=False)
        out.extend([items[i] for i in idx])
    return out


def cache_tag_for_sample(fraction: float, seed: int) -> str:
    fraction_token = f"{fraction:g}".replace(".", "p")
    return f"frac{fraction_token}_seed{seed}"


def sample_cache_path(
    sample_cache_dir: Path | None,
    task_name: str,
    split_name: str,
    fraction: float,
    experiment_seed: int,
) -> Path | None:
    if sample_cache_dir is None:
        return None
    tag = cache_tag_for_sample(fraction, experiment_seed)
    return sample_cache_dir / f"{task_name}_{split_name}_{tag}.csv"


def load_sample_cache(path: Path | None) -> List[Sample] | None:
    if path is None or not path.exists():
        return None
    samples: List[Sample] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            samples.append(Sample(path=Path(row["path"]), label=row["label"]))
    print(f"[sample cache hit] {path} ({len(samples)} samples)")
    return samples


def write_sample_cache(path: Path | None, samples: Sequence[Sample]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label"])
        writer.writeheader()
        for sample in samples:
            writer.writerow({"path": str(sample.path), "label": sample.label})
    print(f"[sample cache write] {path} ({len(samples)} samples)")


def cached_subsample(
    sample_cache_dir: Path | None,
    task_name: str,
    split_name: str,
    fraction: float,
    experiment_seed: int,
    subsample_seed: int,
    discover: Callable[[], List[Sample]],
) -> List[Sample]:
    cache_path = sample_cache_path(sample_cache_dir, task_name, split_name, fraction, experiment_seed)
    cached = load_sample_cache(cache_path)
    if cached is not None:
        return cached

    samples = subsample_by_label(discover(), fraction, subsample_seed)
    write_sample_cache(cache_path, samples)
    return samples


def discover_subsource_samples(dataset_root: Path, split: str) -> List[Sample]:
    samples = discover_ai_subsource_split(dataset_root, split)
    if not samples:
        samples = discover_ai_subsource_from_roots(dataset_root, split)
    return samples


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    feature_cache_dir = Path(args.feature_cache_dir) if args.feature_cache_dir else None
    sample_cache_dir = Path(args.sample_cache_dir) if args.sample_cache_dir else None
    cache_tag = cache_tag_for_sample(args.sample_fraction, args.sample_seed)

    cfg = FeatureConfig(
        image_size=args.image_size,
        radial_bins=args.radial_bins,
        angular_bins=args.angular_bins,
        patch_grid=args.patch_grid,
        feature_profile=args.feature_profile,
    )

    model_roots = discover_model_roots(dataset_root)
    used_root = dataset_root
    if not model_roots:
        fallback_roots: List[Path] = [Path("."), Path(__file__).resolve().parent]
        for candidate in fallback_roots:
            candidate_roots = discover_model_roots(candidate)
            if candidate_roots:
                model_roots = candidate_roots
                used_root = candidate
                print(f"[Info] dataset root '{dataset_root}' not found/invalid. Auto-switched to '{candidate}'.")
                break

    if not model_roots:
        raise ValueError(
            f"No valid dataset structure found under '{dataset_root}'. "
            "Try: python test.py --dataset-root . --out-dir outputs_smoke"
        )

    print("[Smoke Test] Using sampled data with same pipeline.")
    print(f"[Smoke Test] sample_fraction={args.sample_fraction}, sample_seed={args.sample_seed}")
    print(f"[Smoke Test] feature_profile={args.feature_profile}")
    print(f"[Smoke Test] model_architecture={args.model_architecture}")
    print(f"[Smoke Test] train_augmentation={args.train_augmentation}")
    print("Detected dataset roots:")
    for r in model_roots:
        print(f"  - {r}")

    train_binary = cached_subsample(
        sample_cache_dir,
        "binary_ai_vs_nature",
        "train",
        args.sample_fraction,
        args.sample_seed,
        args.sample_seed,
        lambda: discover_binary_split_multi_root(used_root, "train"),
    )
    val_binary = cached_subsample(
        sample_cache_dir,
        "binary_ai_vs_nature",
        "val",
        args.sample_fraction,
        args.sample_seed,
        args.sample_seed + 1,
        lambda: discover_binary_split_multi_root(used_root, "val"),
    )
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
        cache_tag=cache_tag,
        feature_set=args.feature_set,
        lightgbm_profile=args.lgbm_profile,
        model_set=args.model_set,
        model_architecture=args.model_architecture,
        train_augmentation=args.train_augmentation,
        calibrate_threshold=args.calibrate_threshold,
    )

    train_sub = cached_subsample(
        sample_cache_dir,
        "ai_subsource_attribution",
        "train",
        args.sample_fraction,
        args.sample_seed,
        args.sample_seed,
        lambda: discover_subsource_samples(used_root, "train"),
    )
    val_sub = cached_subsample(
        sample_cache_dir,
        "ai_subsource_attribution",
        "val",
        args.sample_fraction,
        args.sample_seed,
        args.sample_seed + 1,
        lambda: discover_subsource_samples(used_root, "val"),
    )
    if train_sub and val_sub and len({s.label for s in train_sub}) >= 2:
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
            cache_tag=cache_tag,
            feature_set=args.feature_set,
            lightgbm_profile=args.lgbm_profile,
            model_set=args.model_set,
            model_architecture=args.model_architecture,
            train_augmentation=args.train_augmentation,
            calibrate_threshold=args.calibrate_threshold,
        )
    else:
        print("Skip ai_subsource_attribution in smoke test.")

    print("\nSmoke test done. Check outputs_smoke/ for results.")


if __name__ == "__main__":
    main()
