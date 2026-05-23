from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import (
    Sample,
    discover_ai_subsource_from_roots,
    discover_ai_subsource_split,
    discover_binary_split_multi_root,
    validate_non_empty,
)
from src.features import FeatureConfig
from src.robustness import evaluate_robustness


TASK_ALIASES = {
    "binary": "binary_ai_vs_nature",
    "attribution": "ai_subsource_attribution",
}
TASKS = tuple(TASK_ALIASES.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate robustness for saved best models without retraining.")
    parser.add_argument("--dataset-root", required=True, help="GenImage data root.")
    parser.add_argument("--model-output", default="outputs_v2_full_best", help="Output directory containing saved models.")
    parser.add_argument("--out-dir", required=True, help="Directory for robustness CSV outputs.")
    parser.add_argument("--sample-fraction", type=float, default=0.2, help="Per-label val sampling fraction.")
    parser.add_argument("--sample-seed", type=int, default=42, help="Seed for per-label val sampling.")
    parser.add_argument(
        "--tasks",
        choices=["binary", "attribution", "both"],
        default="both",
        help="Task(s) to evaluate.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="Parallel feature extraction workers.")
    parser.add_argument("--feature-chunksize", type=int, default=32, help="Parallel feature extraction chunksize.")
    parser.add_argument("--robust-cache-dir", default="", help="Optional attack-level feature cache directory.")
    parser.add_argument("--force", action="store_true", help="Recompute even if output CSV or cache files exist.")
    return parser.parse_args()


def subsample_by_label(samples: Sequence[Sample], fraction: float, seed: int) -> List[Sample]:
    if fraction >= 1.0:
        return list(samples)
    if fraction <= 0.0:
        raise ValueError("--sample-fraction must be > 0")

    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.label].append(sample)

    rng = np.random.default_rng(seed)
    selected: List[Sample] = []
    for label in sorted(grouped):
        items = grouped[label]
        keep = max(1, int(round(len(items) * fraction)))
        keep = min(keep, len(items))
        idx = rng.choice(len(items), size=keep, replace=False)
        selected.extend([items[i] for i in idx])
    return selected


def cache_tag_for_sample(fraction: float, seed: int) -> str:
    fraction_token = f"{fraction:g}".replace(".", "p")
    return f"frac{fraction_token}_seed{seed}"


def requested_tasks(value: str) -> List[str]:
    if value == "both":
        return list(TASKS)
    return [TASK_ALIASES[value]]


def discover_task_samples(dataset_root: Path, task_name: str) -> List[Sample]:
    if task_name == "binary_ai_vs_nature":
        return discover_binary_split_multi_root(dataset_root, "val")

    samples = discover_ai_subsource_split(dataset_root, "val")
    if not samples:
        samples = discover_ai_subsource_from_roots(dataset_root, "val")
    return samples


def load_model_bundle(model_output: Path, task_name: str) -> dict:
    model_path = model_output / task_name / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing saved model: {model_path}")
    bundle = joblib.load(model_path)
    required = ["model", "label_to_id", "feature_config", "feature_set", "feature_names"]
    missing = [key for key in required if key not in bundle]
    if missing:
        raise ValueError(f"Saved model is missing keys {missing}: {model_path}")
    return bundle


def encode_samples(samples: Sequence[Sample], label_to_id: dict) -> tuple[list[str], np.ndarray]:
    paths: list[str] = []
    labels: list[int] = []
    unknown = sorted({sample.label for sample in samples if sample.label not in label_to_id})
    if unknown:
        raise ValueError(f"Samples contain labels not present in saved model: {unknown}")
    for sample in samples:
        paths.append(str(sample.path))
        labels.append(int(label_to_id[sample.label]))
    return paths, np.asarray(labels, dtype=np.int64)


def normalize_id_to_label(bundle: dict) -> dict[int, str]:
    raw = bundle.get("id_to_label") or {v: k for k, v in bundle["label_to_id"].items()}
    return {int(k): str(v) for k, v in raw.items()}


def write_run_config(out_dir: Path, args: argparse.Namespace, tasks: Iterable[str]) -> None:
    config = vars(args).copy()
    config["resolved_tasks"] = list(tasks)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    model_output = Path(args.model_output)
    out_dir = Path(args.out_dir)
    cache_dir = Path(args.robust_cache_dir) if args.robust_cache_dir else None
    tasks = requested_tasks(args.tasks)
    write_run_config(out_dir, args, tasks)
    cache_tag = cache_tag_for_sample(args.sample_fraction, args.sample_seed)

    for task_name in tasks:
        task_out = out_dir / task_name
        task_out.mkdir(parents=True, exist_ok=True)
        output_csv = task_out / "robustness_results.csv"
        if output_csv.exists() and not args.force:
            print(f"[skip] {output_csv} already exists. Use --force to recompute.")
            continue

        bundle = load_model_bundle(model_output, task_name)
        cfg = FeatureConfig(**bundle["feature_config"])
        samples = subsample_by_label(
            discover_task_samples(dataset_root, task_name),
            args.sample_fraction,
            args.sample_seed,
        )
        validate_non_empty(samples, f"{task_name} val")
        sample_paths, sample_labels = encode_samples(samples, bundle["label_to_id"])

        feature_names = bundle["feature_names"]
        expected_n_features = len(feature_names) if feature_names else None
        if expected_n_features is None or expected_n_features <= 0 or math.isnan(float(expected_n_features)):
            raise ValueError(f"Invalid feature_names in saved model for task {task_name}")

        print(f"[run] {task_name}")
        print(f"  samples: {len(samples)}")
        print(f"  feature_profile: {cfg.feature_profile}")
        print(f"  feature_set: {bundle['feature_set']}")
        print(f"  expected features: {expected_n_features}")

        df = evaluate_robustness(
            bundle["model"],
            sample_paths,
            sample_labels,
            cfg,
            feature_set=bundle["feature_set"],
            rng_seed=args.sample_seed,
            task_name=task_name,
            model_output=str(model_output),
            num_workers=args.num_workers,
            chunksize=args.feature_chunksize,
            cache_dir=cache_dir,
            cache_tag=cache_tag,
            force_recompute=args.force,
            expected_n_features=expected_n_features,
            details_dir=task_out,
            id_to_label=normalize_id_to_label(bundle),
        )
        df.to_csv(output_csv, index=False)
        print(df.to_string(index=False))
        print(f"[write] {output_csv}")


if __name__ == "__main__":
    main()
