from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_utils import (  # noqa: E402
    Sample,
    discover_ai_subsource_from_roots,
    discover_ai_subsource_split,
    discover_binary_split_multi_root,
    validate_non_empty,
)
from src.features import FeatureConfig, select_feature_array  # noqa: E402
from src.training import _predict, _predict_proba  # noqa: E402
from test import cache_tag_for_sample, subsample_by_label  # noqa: E402
from train import _load_feature_cache, extract_matrix, feature_cache_path, sample_group  # noqa: E402


TASKS = ("binary_ai_vs_nature", "ai_subsource_attribution")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze confidence, rejection, and high-confidence errors.")
    parser.add_argument("--dataset-root", default=r"C:\Users\99303\git\GenImage_data")
    parser.add_argument("--output-dir", default="outputs_v2_full_best", help="Experiment output containing best_model.joblib.")
    parser.add_argument("--tasks", nargs="*", choices=[*TASKS, "both"], default=["both"])
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--feature-cache-dir", default="", help="Optional feature cache directory.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--feature-chunksize", type=int, default=32)
    parser.add_argument("--analysis-out-dir", default="", help="Optional separate directory for analysis CSV files.")
    return parser.parse_args()


def requested_tasks(values: Sequence[str]) -> List[str]:
    if any(value == "both" for value in values):
        return list(TASKS)
    return list(values)


def discover_samples(dataset_root: Path, task: str, fraction: float, seed: int) -> List[Sample]:
    if task == "binary_ai_vs_nature":
        samples = discover_binary_split_multi_root(dataset_root, "val")
    else:
        samples = discover_ai_subsource_split(dataset_root, "val")
        if not samples:
            samples = discover_ai_subsource_from_roots(dataset_root, "val")
    return subsample_by_label(samples, fraction, seed + 1)


def probability_entropy(proba: np.ndarray) -> float:
    probs = np.asarray(proba, dtype=np.float64)
    probs = probs / (probs.sum() + 1e-12)
    if probs.size <= 1:
        return 0.0
    return float(-np.sum(probs * np.log(probs + 1e-12)) / np.log(probs.size))


def load_or_extract_features(
    samples: Sequence[Sample],
    cfg: FeatureConfig,
    task: str,
    bundle: dict,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[Sample]]:
    cache_dir = Path(args.feature_cache_dir) if args.feature_cache_dir else None
    if cache_dir is not None:
        candidate_tags = []
        bundle_tag = str(bundle.get("cache_tag", "")).strip()
        if bundle_tag:
            candidate_tags.append(bundle_tag)
        candidate_tags.append(cache_tag_for_sample(args.sample_fraction, args.sample_seed))
        candidate_tags.append("full")
        for tag in dict.fromkeys(candidate_tags):
            cache_path = feature_cache_path(cache_dir, task, "val", tag)
            if cache_path is None:
                continue
            cached = _load_feature_cache(cache_path, samples, cfg, f"{task} confidence features")
            if cached is not None:
                X, kept_samples, skipped = cached
                if skipped:
                    print(f"[cache] skipped invalid images recorded in cache: {len(skipped)}")
                return X, kept_samples

    write_path = None
    if cache_dir is not None:
        write_path = feature_cache_path(cache_dir, task, "val", cache_tag_for_sample(args.sample_fraction, args.sample_seed))
    X, kept_samples, skipped = extract_matrix(
        samples,
        cfg,
        desc=f"{task} confidence features",
        num_workers=args.num_workers,
        chunksize=args.feature_chunksize,
        cache_path=write_path,
        train_augmentation="none",
    )
    if skipped:
        print(f"[Warning] skipped invalid images: {len(skipped)}")
    return X, kept_samples


def coverage_table(details: pd.DataFrame) -> pd.DataFrame:
    thresholds = np.linspace(0.0, 0.99, 100)
    rows = []
    y_true = details["true_label"].to_numpy()
    for threshold in thresholds:
        covered = details["confidence"] >= threshold
        if not covered.any():
            rows.append(
                {
                    "threshold": float(threshold),
                    "coverage": 0.0,
                    "n_covered": 0,
                    "accuracy": math.nan,
                    "macro_f1": math.nan,
                }
            )
            continue
        y_cov = y_true[covered.to_numpy()]
        p_cov = details.loc[covered, "pred_label"].to_numpy()
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": float(covered.mean()),
                "n_covered": int(covered.sum()),
                "accuracy": float(accuracy_score(y_cov, p_cov)),
                "macro_f1": float(f1_score(y_cov, p_cov, average="macro")),
            }
        )
    return pd.DataFrame(rows)


def analyze_task(args: argparse.Namespace, task: str) -> None:
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    task_dir = output_dir / task
    bundle_path = task_dir / "best_model.joblib"
    if not bundle_path.exists():
        raise SystemExit(f"Missing model bundle: {bundle_path}")

    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    id_to_label = {int(k): str(v) for k, v in bundle.get("id_to_label", {}).items()}
    if not id_to_label:
        id_to_label = {int(v): str(k) for k, v in bundle["label_to_id"].items()}
    label_to_id = {label: idx for idx, label in id_to_label.items()}
    cfg = FeatureConfig(**bundle["feature_config"])
    feature_set = bundle.get("feature_set", "all")

    samples = [s for s in discover_samples(dataset_root, task, args.sample_fraction, args.sample_seed) if s.label in label_to_id]
    validate_non_empty(samples, f"{task} val")
    X_full, kept_samples = load_or_extract_features(samples, cfg, task, bundle, args)
    X = select_feature_array(X_full, cfg, feature_set)
    y_true = np.asarray([label_to_id[s.label] for s in kept_samples], dtype=np.int64)
    groups = [sample_group(s) for s in kept_samples]
    pred = _predict(model, X, groups)
    proba = _predict_proba(model, X, groups)
    if proba is None:
        raise SystemExit(f"Model for {task} does not expose predict_proba; confidence analysis cannot run.")

    rows = []
    for idx, sample in enumerate(kept_samples):
        probs = np.asarray(proba[idx], dtype=np.float64)
        order = np.sort(probs)
        rows.append(
            {
                "path": str(sample.path),
                "generator": sample_group(sample),
                "true_label": id_to_label[int(y_true[idx])],
                "pred_label": id_to_label[int(pred[idx])],
                "correct": bool(int(y_true[idx]) == int(pred[idx])),
                "confidence": float(order[-1]),
                "margin": float(order[-1] - order[-2]) if order.size >= 2 else float(order[-1]),
                "entropy": probability_entropy(probs),
            }
        )
    details = pd.DataFrame(rows)

    out_base = Path(args.analysis_out_dir) / task if args.analysis_out_dir else task_dir
    out_base.mkdir(parents=True, exist_ok=True)
    details.to_csv(out_base / "confidence_details.csv", index=False)
    coverage = coverage_table(details)
    coverage.to_csv(out_base / "confidence_coverage_accuracy.csv", index=False)
    details.sort_values(["confidence"], ascending=True).head(200).to_csv(
        out_base / "confidence_low_confidence_samples.csv",
        index=False,
    )
    details[~details["correct"]].sort_values("confidence", ascending=False).head(200).to_csv(
        out_base / "confidence_high_confidence_errors.csv",
        index=False,
    )
    by_generator = (
        details.groupby("generator", dropna=False)
        .agg(
            n_samples=("path", "count"),
            accuracy=("correct", "mean"),
            mean_confidence=("confidence", "mean"),
            mean_margin=("margin", "mean"),
            mean_entropy=("entropy", "mean"),
        )
        .reset_index()
    )
    by_generator.to_csv(out_base / "confidence_by_generator.csv", index=False)
    print(f"[write] confidence analysis for {task}: {out_base}")


def main() -> None:
    args = parse_args()
    for task in requested_tasks(args.tasks):
        analyze_task(args, task)


if __name__ == "__main__":
    main()
