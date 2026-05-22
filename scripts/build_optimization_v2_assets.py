from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


TASKS = ("binary_ai_vs_nature", "ai_subsource_attribution")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize second-round frequency-first optimization outputs.")
    parser.add_argument("--report-dir", default="report")
    parser.add_argument("--outputs", nargs="*", default=[], help="Specific outputs to summarize. Defaults to outputs_v2_*.")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_outputs(paths: list[str]) -> list[Path]:
    if paths:
        return [Path(p) for p in paths if Path(p).exists()]
    return sorted(p for p in Path(".").glob("outputs_v2_*") if p.is_dir())


def collect_rows(outputs: list[Path]) -> pd.DataFrame:
    rows = []
    for out_dir in outputs:
        cfg = {}
        cfg_path = out_dir / "run_config.json"
        if cfg_path.exists():
            cfg = read_json(cfg_path)
        record = {
            "run": out_dir.name,
            "feature_profile": cfg.get("feature_profile", ""),
            "model_architecture": cfg.get("model_architecture", ""),
            "train_augmentation": cfg.get("train_augmentation", "none"),
            "sample_fraction": cfg.get("sample_fraction", 1.0),
            "sample_seed": cfg.get("sample_seed", 42),
            "calibrate_threshold": cfg.get("calibrate_threshold", False),
            "lgbm_profile": cfg.get("lgbm_profile", ""),
        }
        for task in TASKS:
            task_dir = out_dir / task
            metrics_path = task_dir / "metrics_summary.json"
            comparison_path = task_dir / "model_comparison.csv"
            if not metrics_path.exists() or not comparison_path.exists():
                continue
            metrics = read_json(metrics_path)
            comparison = pd.read_csv(comparison_path)
            best_name = metrics.get("best_model")
            best = comparison[comparison["model"] == best_name]
            if best.empty:
                best = comparison.iloc[[0]]
            best_row = best.iloc[0].to_dict()
            record[f"{task}_model"] = best_name
            record[f"{task}_accuracy"] = best_row.get("accuracy", math.nan)
            record[f"{task}_macro_f1"] = best_row.get("macro_f1", math.nan)
            record[f"{task}_auc"] = best_row.get("auc", math.nan)
        if "binary_ai_vs_nature_macro_f1" in record and "ai_subsource_attribution_macro_f1" in record:
            record["combined_macro_f1"] = (
                float(record["binary_ai_vs_nature_macro_f1"]) + float(record["ai_subsource_attribution_macro_f1"])
            ) / 2.0
        rows.append(record)
    return pd.DataFrame(rows)


def add_selection(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "combined_macro_f1" not in df:
        return df
    df = df.copy()
    df["selected"] = False
    df["selection_status"] = "not_selected"
    current_binary = 0.899714775693452
    current_attr = 0.9085996400582268
    current_combined = 0.9041572078758394
    valid = df[
        (df["binary_ai_vs_nature_macro_f1"] >= current_binary - 0.003)
        & (df["ai_subsource_attribution_macro_f1"] >= current_attr - 0.003)
    ].copy()
    if valid.empty:
        return df.sort_values("combined_macro_f1", ascending=False).reset_index(drop=True)
    best = valid.sort_values(["combined_macro_f1", "binary_ai_vs_nature_macro_f1"], ascending=False).iloc[0]
    if float(best["combined_macro_f1"]) - current_combined >= 0.003:
        df.loc[df["run"] == best["run"], "selected"] = True
        df.loc[df["run"] == best["run"], "selection_status"] = "selected_v2_best"
    else:
        df["selection_status"] = "below_replacement_threshold"
    return df.sort_values(["selected", "combined_macro_f1"], ascending=[False, False]).reset_index(drop=True)


def save_plot(df: pd.DataFrame, figure_dir: Path) -> None:
    if df.empty or "combined_macro_f1" not in df:
        return
    plot_df = df.sort_values("combined_macro_f1", ascending=True).tail(20).copy()
    plot_df["label"] = (
        plot_df["feature_profile"].astype(str)
        + "\n"
        + plot_df["model_architecture"].astype(str)
        + "\naug="
        + plot_df["train_augmentation"].astype(str)
    )
    fig, ax = plt.subplots(figsize=(7.2, max(4.0, 0.5 * len(plot_df))))
    ax.barh(plot_df["label"], plot_df["combined_macro_f1"], color="#4f7fba")
    ax.axvline(0.9041572078758394, color="#b34d4d", linestyle="--", linewidth=1.2, label="current full baseline")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Average Macro-F1")
    ax.set_title("Second-round frequency-first optimization")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figure_dir / "optimization_v2_macro_f1.png", dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    report_dir = Path(args.report_dir)
    table_dir = report_dir / "tables"
    figure_dir = report_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    outputs = discover_outputs(args.outputs)
    df = add_selection(collect_rows(outputs))
    df.to_csv(table_dir / "optimization_v2_summary.csv", index=False)
    save_plot(df, figure_dir)
    print(f"Wrote {len(df)} v2 optimization rows to {table_dir / 'optimization_v2_summary.csv'}")


if __name__ == "__main__":
    main()
