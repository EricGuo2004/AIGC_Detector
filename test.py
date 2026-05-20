from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import List, Sequence

from src.data_utils import (
    Sample,
    discover_ai_subsource_from_roots,
    discover_ai_subsource_split,
    discover_binary_split_multi_root,
    discover_model_roots,
    validate_non_empty,
)
from src.features import FeatureConfig
from train import run_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick smoke test pipeline (same flow, sampled data).")
    parser.add_argument("--dataset-root", type=str, default="data", help="Dataset root folder")
    parser.add_argument("--out-dir", type=str, default="outputs_smoke", help="Output directory")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--radial-bins", type=int, default=64)
    parser.add_argument("--angular-bins", type=int, default=18)
    parser.add_argument("--patch-grid", type=int, default=4)
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


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = FeatureConfig(
        image_size=args.image_size,
        radial_bins=args.radial_bins,
        angular_bins=args.angular_bins,
        patch_grid=args.patch_grid,
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
    print("Detected dataset roots:")
    for r in model_roots:
        print(f"  - {r}")

    train_binary = discover_binary_split_multi_root(used_root, "train")
    val_binary = discover_binary_split_multi_root(used_root, "val")
    train_binary = subsample_by_label(train_binary, args.sample_fraction, args.sample_seed)
    val_binary = subsample_by_label(val_binary, args.sample_fraction, args.sample_seed + 1)
    validate_non_empty(train_binary, "train")
    validate_non_empty(val_binary, "val")
    run_task(
        task_name="binary_ai_vs_nature",
        train_samples=train_binary,
        val_samples=val_binary,
        out_dir=out_dir,
        cfg=cfg,
        run_robustness=not args.skip_robustness,
    )

    train_sub = discover_ai_subsource_split(used_root, "train")
    val_sub = discover_ai_subsource_split(used_root, "val")
    if not train_sub or not val_sub:
        train_sub = discover_ai_subsource_from_roots(used_root, "train")
        val_sub = discover_ai_subsource_from_roots(used_root, "val")

    train_sub = subsample_by_label(train_sub, args.sample_fraction, args.sample_seed)
    val_sub = subsample_by_label(val_sub, args.sample_fraction, args.sample_seed + 1)
    if train_sub and val_sub and len({s.label for s in train_sub}) >= 2:
        run_task(
            task_name="ai_subsource_attribution",
            train_samples=train_sub,
            val_samples=val_sub,
            out_dir=out_dir,
            cfg=cfg,
            run_robustness=False,
        )
    else:
        print("Skip ai_subsource_attribution in smoke test.")

    print("\nSmoke test done. Check outputs_smoke/ for results.")


if __name__ == "__main__":
    main()
