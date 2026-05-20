from __future__ import annotations

import argparse
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
from src.features import FeatureConfig, extract_feature_vector, load_grayscale, make_feature_names
from src.robustness import evaluate_robustness
from src.training import feature_importance_df, train_and_select


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AIGC frequency-domain fingerprint training pipeline")
    parser.add_argument("--dataset-root", type=str, default="data", help="Dataset root folder")
    parser.add_argument("--out-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--radial-bins", type=int, default=64)
    parser.add_argument("--angular-bins", type=int, default=18)
    parser.add_argument("--patch-grid", type=int, default=4)
    parser.add_argument("--skip-robustness", action="store_true", help="Skip robustness evaluation")
    return parser.parse_args()


def encode_labels(samples: Sequence[Sample], label_to_id: Dict[str, int]) -> np.ndarray:
    return np.asarray([label_to_id[s.label] for s in samples], dtype=np.int64)


def extract_matrix(
    samples: Sequence[Sample], cfg: FeatureConfig, desc: str
) -> Tuple[np.ndarray, List[Sample], List[Tuple[str, str]]]:
    feats: List[np.ndarray] = []
    kept_samples: List[Sample] = []
    skipped: List[Tuple[str, str]] = []
    for s in tqdm(samples, desc=desc, leave=False):
        try:
            img = load_grayscale(str(s.path), cfg.image_size)
            feats.append(extract_feature_vector(img, cfg))
            kept_samples.append(s)
        except Exception as exc:
            skipped.append((str(s.path), str(exc)))
    return np.asarray(feats, dtype=np.float32), kept_samples, skipped


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
) -> None:
    print(f"\n===== Task: {task_name} =====")
    print(f"Train counts: {summarize_labels(train_samples)}")
    print(f"Val counts:   {summarize_labels(val_samples)}")

    label_names = sorted({s.label for s in train_samples})
    label_to_id = {name: i for i, name in enumerate(label_names)}

    X_train, train_kept, train_skipped = extract_matrix(train_samples, cfg, desc=f"{task_name} train features")
    X_val, val_kept, val_skipped = extract_matrix(val_samples, cfg, desc=f"{task_name} val(test) features")
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

    best, all_results = train_and_select(X_train, y_train, X_val, y_val, label_names)
    task_out = out_dir / task_name
    task_out.mkdir(parents=True, exist_ok=True)

    feat_names = make_feature_names(cfg)
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
        "task_name": task_name,
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
        val_paths = [str(s.path) for s in val_samples]
        robust_df = evaluate_robustness(best.model, val_paths, y_val, cfg)
        robust_df.to_csv(task_out / "robustness_results.csv", index=False)
        print("\nRobustness summary (val under attacks):")
        print(robust_df.to_string(index=False))
        print("Robustness evaluation saved.")


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
        )
    else:
        print(
            "\nSkip ai_subsource_attribution: no ai subfolders found. "
            "If needed, prepare data as train/ai/ADM_SELECTED/... or provide multiple model roots."
        )

    print("\nAll done. Check outputs/ for models and reports.")


if __name__ == "__main__":
    main()
