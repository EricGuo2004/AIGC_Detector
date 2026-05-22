from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize current baseline errors for v2 optimization planning.")
    parser.add_argument("--output-dir", default="outputs_4gen_full_best")
    parser.add_argument("--report-dir", default="report")
    return parser.parse_args()


def generator_from_path(path: str) -> str:
    parts = Path(path).parts
    for marker in ("train", "val", "test"):
        if marker in parts:
            idx = parts.index(marker)
            if idx > 0:
                return parts[idx - 1]
    return ""


def summarize_errors(task_dir: Path, task: str, table_dir: Path) -> None:
    path = task_dir / "prediction_errors.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    df["generator"] = df["path"].map(generator_from_path)
    by_pair = (
        df.groupby(["generator", "true_label", "pred_label"], dropna=False)
        .agg(errors=("path", "count"), mean_confidence=("confidence", "mean"))
        .reset_index()
        .sort_values("errors", ascending=False)
    )
    by_pair.to_csv(table_dir / f"v2_error_breakdown_{task}.csv", index=False)

    high_conf = df[df["confidence"] >= 0.8].copy()
    high_conf.sort_values("confidence", ascending=False).head(200).to_csv(
        table_dir / f"v2_high_confidence_errors_{task}.csv",
        index=False,
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    table_dir = Path(args.report_dir) / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    summarize_errors(output_dir / "binary_ai_vs_nature", "binary_ai_vs_nature", table_dir)
    summarize_errors(output_dir / "ai_subsource_attribution", "ai_subsource_attribution", table_dir)
    print(f"Wrote v2 error bottleneck summaries under {table_dir}")


if __name__ == "__main__":
    main()
