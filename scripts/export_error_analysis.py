from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_utils import (  # noqa: E402
    Sample,
    discover_ai_subsource_from_roots,
    discover_ai_subsource_split,
    discover_binary_split_multi_root,
)
from src.features import FeatureConfig, extract_feature_vector_from_path, select_feature_array  # noqa: E402
from test import cache_tag_for_sample, subsample_by_label  # noqa: E402
from train import _load_feature_cache, feature_cache_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export prediction details and mistakes for a saved experiment.")
    parser.add_argument("--dataset-root", default=os.environ.get("GENIMAGE_DATA_ROOT", "data/GenImage_data"))
    parser.add_argument("--output-dir", required=True, help="Experiment output directory, e.g. outputs_4gen_1pct")
    parser.add_argument("--task", choices=["binary_ai_vs_nature", "ai_subsource_attribution"], required=True)
    parser.add_argument("--sample-fraction", type=float, required=True)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--feature-cache-dir",
        default="",
        help="Optional feature cache directory. Reuses cached val features when available.",
    )
    return parser.parse_args()


def samples_for_task(dataset_root: Path, task: str, fraction: float, seed: int) -> List[Sample]:
    if task == "binary_ai_vs_nature":
        samples = discover_binary_split_multi_root(dataset_root, "val")
    else:
        samples = discover_ai_subsource_split(dataset_root, "val")
        if not samples:
            samples = discover_ai_subsource_from_roots(dataset_root, "val")
    return subsample_by_label(samples, fraction, seed + 1)


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")

    args = parse_args()
    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.output_dir)
    task_dir = out_dir / args.task
    bundle_path = task_dir / "best_model.joblib"
    if not bundle_path.exists():
        raise SystemExit(f"Missing model bundle: {bundle_path}")

    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    id_to_label = {int(k): v for k, v in bundle["id_to_label"].items()}
    label_to_id = {v: k for k, v in id_to_label.items()}
    cfg = FeatureConfig(**bundle["feature_config"])
    feature_set = bundle.get("feature_set", "all")

    samples = [s for s in samples_for_task(dataset_root, args.task, args.sample_fraction, args.sample_seed) if s.label in label_to_id]
    feature_cache_dir = Path(args.feature_cache_dir) if args.feature_cache_dir else None
    cached = None
    if feature_cache_dir is not None:
        cache_path = feature_cache_path(
            feature_cache_dir,
            args.task,
            "val",
            cache_tag_for_sample(args.sample_fraction, args.sample_seed),
        )
        if cache_path is not None:
            cached = _load_feature_cache(cache_path, samples, cfg, f"{args.task} val error-analysis features")

    rows = []
    if cached is not None:
        X, kept_samples, skipped = cached
        if skipped:
            print(f"[cache] skipped invalid images recorded in cache: {len(skipped)}")
        X = select_feature_array(X, cfg, feature_set)
        iterator = zip(kept_samples, X)
        total = len(kept_samples)
    else:
        iterator = ((sample, None) for sample in samples)
        total = len(samples)

    for sample, cached_feat in tqdm(iterator, total=total, desc=f"Predicting {args.task}"):
        if cached_feat is None:
            feat = extract_feature_vector_from_path(str(sample.path), cfg).reshape(1, -1)
            feat = select_feature_array(feat, cfg, feature_set)
        else:
            feat = cached_feat.reshape(1, -1)
        pred_id = int(model.predict(feat)[0])
        pred_label = id_to_label[pred_id]
        confidence = np.nan
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(feat)[0]
            confidence = float(np.max(proba))
        rows.append(
            {
                "path": str(sample.path),
                "true_label": sample.label,
                "pred_label": pred_label,
                "correct": sample.label == pred_label,
                "confidence": confidence,
            }
        )

    details = pd.DataFrame(rows)
    details.to_csv(task_dir / "prediction_details.csv", index=False)
    details[~details["correct"]].to_csv(task_dir / "prediction_errors.csv", index=False)
    print(f"Wrote {len(details)} predictions and {(~details['correct']).sum()} errors under {task_dir}")


if __name__ == "__main__":
    main()
