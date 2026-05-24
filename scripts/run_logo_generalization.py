from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_utils import Sample, discover_binary_split, discover_model_roots, validate_non_empty  # noqa: E402
from src.features import FEATURE_PROFILE_CHOICES, FEATURE_SET_CHOICES, FeatureConfig  # noqa: E402
from src.training import LGBM_PROFILE_CHOICES, MODEL_SET_CHOICES  # noqa: E402
from test import cache_tag_for_sample, subsample_by_label  # noqa: E402
from train import run_task  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leave-one-generator-out binary generalization experiment for the 4-generator GenImage setup."
    )
    parser.add_argument("--dataset-root", default=r"C:\Users\99303\git\GenImage_data")
    parser.add_argument("--out-dir", default="outputs_4gen_logo")
    parser.add_argument(
        "--heldout",
        nargs="*",
        default=["all"],
        help="Generator names to hold out. Use all to run every discovered generator.",
    )
    parser.add_argument("--sample-fraction", type=float, default=0.2, help="Per-label train/val sampling fraction.")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--radial-bins", type=int, default=64)
    parser.add_argument("--angular-bins", type=int, default=18)
    parser.add_argument("--patch-grid", type=int, default=4)
    parser.add_argument("--feature-profile", choices=FEATURE_PROFILE_CHOICES, default="fusion_freq")
    parser.add_argument("--feature-set", choices=FEATURE_SET_CHOICES, default="all")
    parser.add_argument("--feature-cache-dir", default="feature_cache_logo")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--feature-chunksize", type=int, default=32)
    parser.add_argument("--lightgbm-device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--lgbm-profile", choices=LGBM_PROFILE_CHOICES, default="wide")
    parser.add_argument("--model-set", choices=MODEL_SET_CHOICES, default="lightgbm")
    parser.add_argument("--calibrate-threshold", action="store_true")
    parser.add_argument("--resume-completed", action="store_true")
    return parser.parse_args()


def root_label(root: Path, dataset_root: Path) -> str:
    return root.parent.name if root.parent.resolve() != dataset_root.resolve() else root.name


def complete_task(out_dir: Path) -> bool:
    task_dir = out_dir / "binary_ai_vs_nature"
    return (task_dir / "metrics_summary.json").exists() and (task_dir / "model_comparison.csv").exists()


def normalize_requested(names: Sequence[str], available: Sequence[str]) -> List[str]:
    if any(name.lower() == "all" for name in names):
        return list(available)
    by_lower = {name.lower(): name for name in available}
    out = []
    unknown = []
    for name in names:
        match = by_lower.get(name.lower())
        if match is None:
            unknown.append(name)
        else:
            out.append(match)
    if unknown:
        raise SystemExit(f"Unknown heldout generator(s): {unknown}. Available: {list(available)}")
    return out


def sample_many(samples: Iterable[Sample], fraction: float, seed: int) -> List[Sample]:
    return subsample_by_label(list(samples), fraction, seed)


def read_best_metrics(task_dir: Path) -> dict:
    metrics = json.loads((task_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(task_dir / "model_comparison.csv")
    best_model = metrics.get("best_model")
    row = comparison[comparison["model"] == best_model]
    if row.empty:
        row = comparison.iloc[[0]]
    record = row.iloc[0].to_dict()
    return {
        "best_model": best_model,
        "accuracy": float(record.get("accuracy", math.nan)),
        "macro_f1": float(record.get("macro_f1", math.nan)),
        "auc": float(record.get("auc", math.nan)) if "auc" in record else math.nan,
    }


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    roots = discover_model_roots(dataset_root)
    if len(roots) < 2:
        raise SystemExit(f"Need at least two generator roots under {dataset_root}; found {len(roots)}")

    labels = [root_label(root, dataset_root) for root in roots]
    requested = normalize_requested(args.heldout, labels)
    by_label = dict(zip(labels, roots))
    cfg = FeatureConfig(
        image_size=args.image_size,
        radial_bins=args.radial_bins,
        angular_bins=args.angular_bins,
        patch_grid=args.patch_grid,
        feature_profile=args.feature_profile,
    )
    cache_dir = Path(args.feature_cache_dir) if args.feature_cache_dir else None
    sample_tag = cache_tag_for_sample(args.sample_fraction, args.sample_seed)

    rows = []
    for heldout in requested:
        run_name = f"leave_{heldout}_out"
        run_dir = out_dir / run_name
        if args.resume_completed and complete_task(run_dir):
            print(f"[resume] Skip completed LOGO run: {run_dir}")
        else:
            train_samples: List[Sample] = []
            for label, root in by_label.items():
                if label == heldout:
                    continue
                train_samples.extend(discover_binary_split(root, "train"))
            val_samples = discover_binary_split(by_label[heldout], "val")
            train_samples = sample_many(train_samples, args.sample_fraction, args.sample_seed)
            val_samples = sample_many(val_samples, args.sample_fraction, args.sample_seed + 1)
            validate_non_empty(train_samples, f"{run_name} train")
            validate_non_empty(val_samples, f"{run_name} val")

            print(f"[LOGO] heldout={heldout} train={len(train_samples)} val={len(val_samples)}")
            run_task(
                task_name="binary_ai_vs_nature",
                train_samples=train_samples,
                val_samples=val_samples,
                out_dir=run_dir,
                cfg=cfg,
                run_robustness=False,
                lightgbm_device=args.lightgbm_device,
                num_workers=args.num_workers,
                feature_chunksize=args.feature_chunksize,
                feature_cache_dir=cache_dir,
                cache_tag=f"{run_name}_{sample_tag}",
                feature_set=args.feature_set,
                lightgbm_profile=args.lgbm_profile,
                model_set=args.model_set,
                model_architecture="flat",
                train_augmentation="none",
                calibrate_threshold=args.calibrate_threshold,
            )

        task_dir = run_dir / "binary_ai_vs_nature"
        if complete_task(run_dir):
            rows.append(
                {
                    "heldout_generator": heldout,
                    "train_generators": ",".join(label for label in labels if label != heldout),
                    "sample_fraction": args.sample_fraction,
                    "feature_profile": args.feature_profile,
                    "feature_set": args.feature_set,
                    "lgbm_profile": args.lgbm_profile,
                    **read_best_metrics(task_dir),
                    "run_dir": str(run_dir),
                }
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "logo_generalization_summary.csv", index=False)
    print(summary.to_string(index=False) if not summary.empty else "No completed LOGO runs.")
    print(f"[write] {out_dir / 'logo_generalization_summary.csv'}")


if __name__ == "__main__":
    main()
