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
STYLE = {
    "ink": "#1c2a3b",
    "green": "#66a38a",
    "orange": "#c9823a",
    "plum": "#7f3a58",
    "blue": "#4f77b6",
    "grid": "#d9ddd8",
    "paper": "#fbfbf8",
}
OLD_FULL_BASELINE = 0.9041572078758394


def apply_ppt_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": STYLE["paper"],
            "axes.edgecolor": STYLE["ink"],
            "axes.labelcolor": STYLE["ink"],
            "axes.titlecolor": STYLE["ink"],
            "xtick.color": STYLE["ink"],
            "ytick.color": STYLE["ink"],
            "grid.color": STYLE["grid"],
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    )


def polish_axes(ax, *, grid_axis: str = "x") -> None:
    ax.grid(axis=grid_axis, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b8c0bd")
    ax.spines["bottom"].set_color("#b8c0bd")


def run_label(row: pd.Series) -> str:
    run = str(row.get("run", ""))
    labels = {
        "outputs_v2_full_best": "Fusion full",
        "outputs_v2_50pct_best": "Fusion 50%",
        "outputs_v2_20pct_best": "Fusion 20%",
        "outputs_v2_20pct_fusion_flat_seed123": "Fusion 20% seed123",
        "outputs_v2_5pct_fusion_freq_flat": "Fusion 5%",
        "outputs_v2_5pct_fusion_freq_binary_expert_ensemble": "Expert 5%",
        "outputs_v2_5pct_fusion_freq_pairwise_ovo_attribution": "OVO 5%",
        "outputs_v2_5pct_fusion_freq_hierarchical_attribution": "Hierarchical 5%",
        "outputs_v2_5pct_color_freq_flat": "Color 5%",
        "outputs_v2_5pct_block_dct_flat": "Block DCT 5%",
        "outputs_v2_5pct_multiscale_freq_flat": "Multi-scale 5%",
        "outputs_v2_5pct_residual_freq_flat": "Residual 5%",
    }
    if run in labels:
        return labels[run]
    profile = str(row.get("feature_profile", "")).replace("_freq", "").replace("_", " ").title()
    frac = row.get("sample_fraction", "")
    try:
        frac_label = f"{float(frac) * 100:.0f}%"
    except (TypeError, ValueError):
        frac_label = str(frac)
    return f"{profile} {frac_label}".strip()


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
    plot_df = df.copy()
    plot_df = plot_df[~plot_df["run"].astype(str).str.contains("smoke|robust", case=False, na=False)].copy()
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values("combined_macro_f1", ascending=False).head(10).iloc[::-1].copy()
    plot_df["label"] = plot_df.apply(run_label, axis=1)
    colors = [STYLE["green"] if bool(x) else STYLE["blue"] for x in plot_df["selected"]]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.barh(plot_df["label"], plot_df["combined_macro_f1"], color=colors, height=0.66)
    ax.axvline(OLD_FULL_BASELINE, color=STYLE["plum"], linestyle="--", linewidth=1.3, label="Old gray full baseline")
    for _, row in plot_df.iterrows():
        ax.text(
            float(row["combined_macro_f1"]) + 0.004,
            row["label"],
            f"{float(row['combined_macro_f1']):.3f}",
            va="center",
            fontsize=8.3,
            color=STYLE["ink"],
        )
    ax.set_xlim(0.86, 1.02)
    ax.set_xlabel("Combined Macro-F1 / 双任务平均 F1")
    ax.set_title("Second-round frequency optimization / 第二轮频域优化")
    ax.legend(loc="lower right")
    polish_axes(ax)
    plt.tight_layout()
    plt.savefig(figure_dir / "optimization_v2_macro_f1.png", dpi=240, facecolor="white")
    plt.close()


def main() -> None:
    args = parse_args()
    apply_ppt_style()
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
