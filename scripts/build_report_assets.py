from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
TASKS = ["binary_ai_vs_nature", "ai_subsource_attribution"]
TASK_LABELS = {
    "binary_ai_vs_nature": "AI vs Nature",
    "ai_subsource_attribution": "Source Attribution",
}
STYLE = {
    "ink": "#1c2a3b",
    "muted": "#827f67",
    "teal": "#546f68",
    "green": "#66a38a",
    "orange": "#c9823a",
    "rose": "#a97b74",
    "plum": "#7f3a58",
    "blue": "#4f77b6",
    "grid": "#d9ddd8",
    "paper": "#fbfbf8",
}
SERIES_COLORS = [STYLE["blue"], STYLE["green"], STYLE["orange"], STYLE["plum"], STYLE["teal"], STYLE["rose"]]


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


def polish_axes(ax, *, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b8c0bd")
    ax.spines["bottom"].set_color("#b8c0bd")


def task_label(task: str) -> str:
    return TASK_LABELS.get(task, task.replace("_", " "))


RUN_LABELS = {
    "outputs_4gen_5pct": "Gray 5%",
    "outputs_4gen_10pct_best": "Tuned 10%",
    "outputs_4gen_full_best": "Gray full",
    "outputs_v2_5pct_color_freq_flat": "Color 5%",
    "outputs_v2_5pct_multiscale_freq_flat": "Multi-scale 5%",
    "outputs_v2_5pct_block_dct_flat": "Block DCT 5%",
    "outputs_v2_5pct_residual_freq_flat": "Residual 5%",
    "outputs_v2_5pct_fusion_freq_flat": "Fusion 5%",
    "outputs_v2_full_best": "Fusion full",
    "outputs_4gen_20pct_stable_freq": "Stable 20%",
    "outputs_4gen_20pct_fusion_mild_aug": "Mild aug 20%",
    "outputs_4gen_20pct_fusion_robust_aug": "Robust aug 20%",
    "outputs_4gen_full_best_robust_20pct": "Gray full",
    "outputs_v2_full_best_robust_20pct": "Fusion full",
    "outputs_4gen_20pct_stable_freq_robust_20pct": "Stable 20%",
    "outputs_4gen_20pct_fusion_mild_aug_robust_20pct": "Mild aug 20%",
    "outputs_4gen_20pct_fusion_robust_aug_robust_20pct": "Robust aug 20%",
}


def run_label(run: str) -> str:
    return RUN_LABELS.get(run, run.replace("outputs_", "").replace("_", " "))


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=240, facecolor="white")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build report CSV/PNG assets from AIGC_Detector outputs.")
    parser.add_argument("--dataset-root", default=os.environ.get("GENIMAGE_DATA_ROOT", "data/GenImage_data"))
    parser.add_argument("--report-dir", default="report")
    parser.add_argument(
        "--outputs",
        nargs="*",
        default=[],
        help="Output directories to summarize. Defaults to all local outputs_* directories.",
    )
    parser.add_argument(
        "--primary-output",
        default="",
        help="Primary output directory for confusion matrices and feature-importance plots.",
    )
    parser.add_argument(
        "--robustness-output",
        default="",
        help="Optional output directory containing robustness_results.csv files. Defaults to the largest robustness run.",
    )
    parser.add_argument(
        "--robustness-compare-outputs",
        nargs="*",
        default=[],
        help="Optional robustness output directories to compare in a separate CSV/PNG.",
    )
    return parser.parse_args()


def iter_images(folder: Path):
    if not folder.exists():
        return
    for root, _, files in os.walk(folder):
        for name in files:
            path = Path(root) / name
            if path.suffix.lower() in IMAGE_SUFFIXES:
                yield path


def count_images(folder: Path) -> int:
    return sum(1 for _ in iter_images(folder))


def first_image(folder: Path) -> Path | None:
    for path in iter_images(folder):
        return path
    return None


def count_dataset(dataset_root: Path) -> pd.DataFrame:
    rows = []
    for gen_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir() and not p.name.startswith("_")]):
        row = {"generator": gen_dir.name}
        for rel in ["train/ai", "train/nature", "val/ai", "val/nature"]:
            row[rel.replace("/", "_")] = count_images(gen_dir / rel)
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty and "run" in df:
        non_smoke = ~df["run"].astype(str).str.contains("smoke", case=False, na=False)
        if non_smoke.any():
            df = df[non_smoke].copy()
    return df


def discover_outputs(paths: Iterable[str]) -> List[Path]:
    if paths:
        return [Path(p) for p in paths]
    return sorted([p for p in Path(".").glob("outputs*") if p.is_dir()])


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_experiment_tables(output_dirs: List[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    comparison_rows = []
    for out_dir in output_dirs:
        for task in TASKS:
            task_dir = out_dir / task
            metrics_path = task_dir / "metrics_summary.json"
            comparison_path = task_dir / "model_comparison.csv"
            if not metrics_path.exists() or not comparison_path.exists():
                continue

            metrics = read_json(metrics_path)
            labels = "|".join(metrics.get("labels", []))
            comparison = pd.read_csv(comparison_path)
            for row in comparison.to_dict(orient="records"):
                comparison_rows.append({"run": out_dir.name, "task": task, "labels": labels, **row})

            best_name = metrics.get("best_model")
            best_row = comparison[comparison["model"] == best_name]
            if len(best_row) == 0:
                best_row = comparison.iloc[[0]]
            best = best_row.iloc[0].to_dict()
            summary_rows.append(
                {
                    "run": out_dir.name,
                    "task": task,
                    "best_model": best_name,
                    "labels": labels,
                    "accuracy": best.get("accuracy", math.nan),
                    "macro_f1": best.get("macro_f1", math.nan),
                    "auc": best.get("auc", math.nan),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(comparison_rows)


def seed_info_from_run(run_name: str) -> tuple[str, int] | None:
    if run_name == "outputs_4gen_5pct":
        return "5pct_baseline", 42
    match = re.fullmatch(r"outputs_4gen_5pct_seed(\d+)", run_name)
    if match:
        return "5pct_baseline", int(match.group(1))
    if run_name == "outputs_4gen_10pct_best":
        return "10pct_tuned", 42
    match = re.fullmatch(r"outputs_4gen_10pct_best_seed(\d+)", run_name)
    if match:
        return "10pct_tuned", int(match.group(1))
    return None


def collect_seed_stability(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    df = summary.copy()
    seed_info = df["run"].map(seed_info_from_run)
    df["experiment"] = seed_info.map(lambda x: x[0] if x else None)
    df["seed"] = seed_info.map(lambda x: x[1] if x else None)
    df = df[df["seed"].notna()].copy()
    if df.empty:
        return pd.DataFrame()
    df["seed"] = df["seed"].astype(int)
    df = df.sort_values("run").drop_duplicates(["experiment", "task", "seed"], keep="last")
    rows = []
    for (experiment, task), group in df.groupby(["experiment", "task"]):
        rows.append(
            {
                "experiment": experiment,
                "task": task,
                "num_seeds": int(group["seed"].nunique()),
                "seeds": ",".join(str(s) for s in sorted(group["seed"].unique())),
                "accuracy_mean": float(group["accuracy"].mean()),
                "accuracy_std": float(group["accuracy"].std(ddof=0)),
                "macro_f1_mean": float(group["macro_f1"].mean()),
                "macro_f1_std": float(group["macro_f1"].std(ddof=0)),
                "auc_mean": float(group["auc"].mean()) if group["auc"].notna().any() else math.nan,
                "auc_std": float(group["auc"].std(ddof=0)) if group["auc"].notna().any() else math.nan,
            }
        )
    return pd.DataFrame(rows)


def feature_set_from_run(run_name: str) -> str | None:
    prefix = "outputs_4gen_5pct_ablation_"
    if run_name.startswith(prefix):
        return run_name[len(prefix) :]
    return None


def collect_feature_ablation(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    df = summary.copy()
    df["feature_set"] = df["run"].map(feature_set_from_run)
    df = df[df["feature_set"].notna()].copy()
    if df.empty:
        return pd.DataFrame()
    columns = ["feature_set", "task", "best_model", "accuracy", "macro_f1", "auc", "run"]
    return df[columns].sort_values(["task", "macro_f1"], ascending=[True, False]).reset_index(drop=True)


def tuning_config_from_run(run_name: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"outputs_4gen_5pct_tune_(baseline|regularized|large|wide)_(.+)", run_name)
    if not match:
        return None
    return match.group(1), match.group(2)


def is_tuning_run(run_name: str) -> bool:
    return tuning_config_from_run(run_name) is not None


def collect_lgbm_tuning(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for row in summary.to_dict(orient="records"):
        config = tuning_config_from_run(str(row["run"]))
        if config is None:
            continue
        profile, feature_set = config
        rows.append({"profile": profile, "feature_set": feature_set, **row})
    if not rows:
        return pd.DataFrame()
    columns = ["profile", "feature_set", "task", "best_model", "accuracy", "macro_f1", "auc", "run"]
    return pd.DataFrame(rows)[columns].sort_values(["task", "macro_f1"], ascending=[True, False]).reset_index(drop=True)


def search_metadata_from_run(run_name: str) -> dict | None:
    fixed = {
        "outputs_4gen_10pct_best": ("baseline", "wide", 0.10, "10pct_best"),
        "outputs_4gen_20pct_best": ("baseline", "wide", 0.20, "20pct_best"),
        "outputs_4gen_50pct_best": ("baseline", "wide", 0.50, "50pct_best"),
        "outputs_4gen_full_best": ("baseline", "wide", 1.00, "full_best"),
        "outputs_4gen_20pct_enhanced_best": ("enhanced", "best", 0.20, "20pct_enhanced_best"),
        "outputs_4gen_50pct_enhanced_best": ("enhanced", "best", 0.50, "50pct_enhanced_best"),
        "outputs_4gen_full_enhanced_best": ("enhanced", "best", 1.00, "full_enhanced_best"),
    }
    if run_name in fixed:
        feature_profile, lgbm_profile, fraction, stage = fixed[run_name]
        return {
            "feature_profile": feature_profile,
            "lgbm_profile": lgbm_profile,
            "sample_fraction": fraction,
            "stage": stage,
        }
    match = re.fullmatch(r"outputs_4gen_10pct_enhanced_(baseline|regularized|large|wide)", run_name)
    if match:
        return {
            "feature_profile": "enhanced",
            "lgbm_profile": match.group(1),
            "sample_fraction": 0.10,
            "stage": "10pct_enhanced_tuning",
        }
    return None


def collect_final_selection(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    by_run: dict[str, dict] = {}
    for row in summary.to_dict(orient="records"):
        meta = search_metadata_from_run(str(row["run"]))
        if meta is None:
            continue
        record = by_run.setdefault(str(row["run"]), {"run": str(row["run"]), **meta})
        task = str(row["task"])
        if task == "binary_ai_vs_nature":
            record["binary_accuracy"] = row.get("accuracy", math.nan)
            record["binary_macro_f1"] = row.get("macro_f1", math.nan)
            record["binary_auc"] = row.get("auc", math.nan)
        elif task == "ai_subsource_attribution":
            record["attribution_accuracy"] = row.get("accuracy", math.nan)
            record["attribution_macro_f1"] = row.get("macro_f1", math.nan)

    rows = []
    for record in by_run.values():
        binary = record.get("binary_macro_f1", math.nan)
        attribution = record.get("attribution_macro_f1", math.nan)
        if pd.isna(binary) or pd.isna(attribution):
            continue
        record["combined_macro_f1"] = float((binary + attribution) / 2.0)
        rows.append(record)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    baseline = df[df["run"] == "outputs_4gen_10pct_best"]
    df["valid_against_floor"] = True
    df["selected"] = False
    df["selection_status"] = "not_selected"
    if not baseline.empty:
        base = baseline.iloc[0]
        binary_floor = float(base["binary_macro_f1"]) - 0.005
        attr_floor = float(base["attribution_macro_f1"]) - 0.005
        base_combined = float(base["combined_macro_f1"])
        df["valid_against_floor"] = (df["binary_macro_f1"] >= binary_floor) & (
            df["attribution_macro_f1"] >= attr_floor
        )
        valid = df[df["valid_against_floor"]].copy()
        if valid.empty:
            selected_run = "outputs_4gen_10pct_best"
            df.loc[df["run"] == selected_run, "selection_status"] = "retained_current"
        else:
            best = valid.sort_values(["combined_macro_f1", "binary_macro_f1"], ascending=False).iloc[0]
            selected_run = str(best["run"])
            if float(best["combined_macro_f1"]) - base_combined < 0.003:
                selected_run = "outputs_4gen_10pct_best"
                df.loc[df["run"] == selected_run, "selection_status"] = "retained_current"
            else:
                df.loc[df["run"] == selected_run, "selection_status"] = "selected_best"
        df.loc[~df["valid_against_floor"], "selection_status"] = "invalid_regression"
    else:
        selected_run = str(df.sort_values(["combined_macro_f1", "binary_macro_f1"], ascending=False).iloc[0]["run"])
        df.loc[df["run"] == selected_run, "selection_status"] = "selected_best"

    df.loc[df["run"] == selected_run, "selected"] = True
    return df.sort_values(["selected", "combined_macro_f1"], ascending=[False, False]).reset_index(drop=True)


def save_scaleup_plot(selection: pd.DataFrame, figure_dir: Path) -> None:
    if selection.empty:
        return
    df = selection[selection["stage"].isin(["10pct_best", "20pct_best", "50pct_best", "full_best"])].copy()
    if df.empty:
        return
    df = df.sort_values("sample_fraction")
    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    ax.plot(df["sample_fraction"], df["binary_macro_f1"], marker="o", linewidth=2.2, color=STYLE["blue"], label="真假检测")
    ax.plot(df["sample_fraction"], df["attribution_macro_f1"], marker="o", linewidth=2.2, color=STYLE["green"], label="生成源归因")
    ax.plot(df["sample_fraction"], df["combined_macro_f1"], marker="o", linewidth=2.2, color=STYLE["orange"], label="双任务平均")
    metric_cols = ["binary_macro_f1", "attribution_macro_f1", "combined_macro_f1"]
    y_min = float(df[metric_cols].min().min())
    y_max = float(df[metric_cols].max().max())
    pad = max(0.004, (y_max - y_min) * 0.28)
    ax.set_ylim(max(0.0, y_min - pad), min(1.0, y_max + pad))
    ax.set_xlabel("训练抽样比例")
    ax.set_ylabel("Macro-F1 (zoomed)")
    ax.set_title("Scale-up results, zoomed / 抽样比例扩大")
    ax.legend(loc="lower right")
    polish_axes(ax)
    save_plot(figure_dir / "scaleup_macro_f1.png")


def save_enhanced_search_plot(selection: pd.DataFrame, figure_dir: Path) -> None:
    if selection.empty:
        return
    df = selection[selection["feature_profile"] == "enhanced"].copy()
    if df.empty:
        return
    df = df.sort_values("combined_macro_f1", ascending=True)
    df["config"] = df["stage"] + "\n" + df["lgbm_profile"].astype(str)
    fig, ax = plt.subplots(figsize=(6.8, max(3.8, 0.42 * len(df))))
    ax.barh(df["config"], df["combined_macro_f1"], color=STYLE["plum"])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("双任务平均 Macro-F1")
    ax.set_ylabel("增强特征配置")
    ax.set_title("Enhanced feature search / 增强频域特征搜索")
    polish_axes(ax)
    save_plot(figure_dir / "enhanced_feature_search_macro_f1.png")


def save_model_comparison_plot(comparison: pd.DataFrame, figure_dir: Path) -> None:
    if comparison.empty or "macro_f1" not in comparison:
        return
    df = comparison.copy()
    selected_runs = [
        "outputs_4gen_5pct",
        "outputs_4gen_10pct_best",
        "outputs_4gen_full_best",
        "outputs_v2_5pct_color_freq_flat",
        "outputs_v2_5pct_multiscale_freq_flat",
        "outputs_v2_5pct_block_dct_flat",
        "outputs_v2_5pct_residual_freq_flat",
        "outputs_v2_5pct_fusion_freq_flat",
        "outputs_v2_full_best",
        "outputs_4gen_20pct_stable_freq",
        "outputs_4gen_20pct_fusion_mild_aug",
        "outputs_4gen_20pct_fusion_robust_aug",
    ]
    df = df[df["run"].isin(selected_runs)].copy()
    if df.empty:
        return
    df = df.sort_values("macro_f1", ascending=False).drop_duplicates(["run", "task"], keep="first")
    df["run_label"] = pd.Categorical(df["run"].map(run_label), [run_label(r) for r in selected_runs], ordered=True)
    pivot = df.pivot_table(index="run_label", columns="task", values="macro_f1", aggfunc="first").sort_index()
    pivot = pivot.rename(columns={task: task_label(task) for task in TASKS})
    ax = pivot.plot(kind="bar", figsize=(9.6, 4.8), rot=28, color=[STYLE["blue"], STYLE["green"]])
    ax.set_ylabel("Macro-F1")
    ax.set_xlabel("Experiment output / 实验输出")
    ax.set_ylim(0, 1.05)
    ax.set_title("Key experiment comparison / 关键实验对比")
    ax.legend(title="Task / 任务", loc="center left", bbox_to_anchor=(1.01, 0.5))
    polish_axes(ax)
    save_plot(figure_dir / "model_comparison_macro_f1.png")


def collect_single_vs_fusion(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    run_to_profile = {
        "outputs_v2_5pct_color_freq_flat": ("color_freq", "single-domain"),
        "outputs_v2_5pct_multiscale_freq_flat": ("multiscale_freq", "single-domain"),
        "outputs_v2_5pct_block_dct_flat": ("block_dct", "single-domain"),
        "outputs_v2_5pct_residual_freq_flat": ("residual_freq", "single-domain"),
        "outputs_v2_5pct_fusion_freq_flat": ("fusion_freq", "fusion"),
        "outputs_v2_full_best": ("fusion_freq_full", "fusion"),
    }
    rows = []
    for row in summary.to_dict(orient="records"):
        meta = run_to_profile.get(str(row["run"]))
        if meta is None:
            continue
        profile, family = meta
        rows.append(
            {
                "profile": profile,
                "family": family,
                "task": row["task"],
                "sample_fraction": 1.0 if row["run"] == "outputs_v2_full_best" else 0.05,
                "macro_f1": row["macro_f1"],
                "accuracy": row["accuracy"],
                "auc": row["auc"],
                "run": row["run"],
            }
        )
    if not rows:
        return pd.DataFrame()
    order = {
        "color_freq": 0,
        "multiscale_freq": 1,
        "block_dct": 2,
        "residual_freq": 3,
        "fusion_freq": 4,
        "fusion_freq_full": 5,
    }
    df = pd.DataFrame(rows)
    df["profile_order"] = df["profile"].map(order)
    return df.sort_values(["profile_order", "task"]).drop(columns=["profile_order"]).reset_index(drop=True)


def save_single_vs_fusion_plot(single_vs_fusion: pd.DataFrame, figure_dir: Path) -> None:
    if single_vs_fusion.empty:
        return
    df = single_vs_fusion.copy()
    task_labels = {
        "binary_ai_vs_nature": "AI vs Nature",
        "ai_subsource_attribution": "Attribution",
    }
    df["task_label"] = df["task"].map(task_labels).fillna(df["task"])
    df["profile_label"] = df["profile"].map(
        {
            "color_freq": "Color",
            "multiscale_freq": "Multi-scale",
            "block_dct": "Block DCT",
            "residual_freq": "Residual",
            "fusion_freq": "Fusion 5%",
            "fusion_freq_full": "Fusion full",
        }
    )
    pivot = df.pivot_table(index="profile_label", columns="task_label", values="macro_f1", aggfunc="first")
    order = ["Color", "Multi-scale", "Block DCT", "Residual", "Fusion 5%", "Fusion full"]
    pivot = pivot.reindex([x for x in order if x in pivot.index])
    ax = pivot.plot(kind="bar", figsize=(8.8, 4.4), ylim=(0.75, 1.01), width=0.75, color=[STYLE["blue"], STYLE["green"]])
    ax.set_xlabel("频域特征 profile")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Single-domain vs fused frequency profile / 单一频域与融合频域")
    ax.legend(title="任务", loc="center left", bbox_to_anchor=(1.01, 0.5))
    polish_axes(ax)
    plt.xticks(rotation=20, ha="right")
    save_plot(figure_dir / "single_vs_fusion_feature_profiles.png")


def save_ablation_plot(ablation: pd.DataFrame, figure_dir: Path) -> None:
    if ablation.empty:
        return
    for task, group in ablation.groupby("task"):
        group = group.sort_values("macro_f1", ascending=True)
        fig, ax = plt.subplots(figsize=(6.4, max(3.8, 0.42 * len(group))))
        ax.barh(group["feature_set"], group["macro_f1"], color=STYLE["teal"])
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Macro-F1")
        ax.set_ylabel("特征组")
        ax.set_title(f"Feature ablation / 单组频域消融：{task_label(task)}")
        polish_axes(ax)
        save_plot(figure_dir / f"feature_ablation_{task}.png")


def save_lgbm_tuning_plot(tuning: pd.DataFrame, figure_dir: Path) -> None:
    if tuning.empty:
        return
    for task, group in tuning.groupby("task"):
        group = group.sort_values("macro_f1", ascending=True).copy()
        group["config"] = group["profile"] + "\n" + group["feature_set"]
        fig, ax = plt.subplots(figsize=(6.6, max(3.8, 0.46 * len(group))))
        ax.barh(group["config"], group["macro_f1"], color=STYLE["blue"])
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Macro-F1")
        ax.set_ylabel("参数 profile / 特征集")
        ax.set_title(f"LightGBM tuning / 参数调优：{task_label(task)}")
        polish_axes(ax)
        save_plot(figure_dir / f"lgbm_tuning_{task}.png")


def read_confusion(path: Path) -> pd.DataFrame:
    # train.py writes numeric column headers by default; keep them as labels.
    return pd.read_csv(path)


def save_confusion_plot(task_dir: Path, task: str, figure_dir: Path) -> None:
    metrics_path = task_dir / "metrics_summary.json"
    if not metrics_path.exists():
        return
    metrics = read_json(metrics_path)
    best_model = metrics.get("best_model", "lightgbm")
    path = task_dir / f"confusion_matrix_{best_model}.csv"
    if not path.exists():
        path = task_dir / "confusion_matrix_lightgbm.csv"
    if not path.exists():
        return
    labels = metrics.get("labels", [])
    cm = read_confusion(path).to_numpy()
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="YlGnBu")
    ax.set_title(f"Confusion matrix / 混淆矩阵：{task_label(task)}")
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    if labels and len(labels) == cm.shape[0]:
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        ax.set_yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(figure_dir / f"confusion_{task}_lightgbm.png", dpi=240, facecolor="white")
    if best_model != "lightgbm":
        fig.savefig(figure_dir / f"confusion_{task}_{best_model}.png", dpi=240, facecolor="white")
    plt.close(fig)


def save_feature_importance_plot(task_dir: Path, task: str, figure_dir: Path, table_dir: Path) -> None:
    path = task_dir / "feature_importance.csv"
    if not path.exists():
        return
    df = pd.read_csv(path).head(20)
    if df.empty:
        return
    df.to_csv(table_dir / f"top_features_{task}.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    ax.barh(df["feature"][::-1], df["importance"][::-1], color=STYLE["blue"])
    ax.set_xlabel("LightGBM importance")
    ax.set_title(f"Top-20 feature importance / 特征重要性：{task_label(task)}")
    polish_axes(ax)
    save_plot(figure_dir / f"feature_importance_{task}_top20.png")


def save_robustness_plot(task_dir: Path, task: str, figure_dir: Path, table_dir: Path) -> None:
    path = task_dir / "robustness_results.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    df.to_csv(table_dir / f"robustness_{task}.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    clean = df[df["attack"] == "clean"]
    if not clean.empty:
        clean_f1 = float(clean.iloc[0]["macro_f1"])
        ax.axhline(clean_f1, color="#555555", linestyle="--", linewidth=1.1, label=f"clean ({clean_f1:.3f})")
    plot_df = df[df["attack"] != "clean"].copy()
    for attack, group in plot_df.groupby("attack"):
        color = {"jpeg": STYLE["blue"], "resize": STYLE["green"], "noise": STYLE["orange"]}.get(str(attack), STYLE["teal"])
        ax.plot(group["level"].astype(str), group["macro_f1"], marker="o", linewidth=2.0, color=color, label=attack)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("扰动强度")
    ax.set_ylabel("Macro-F1")
    ax.set_title(f"Robustness / 鲁棒性：{task_label(task)}")
    ax.legend(title="扰动")
    polish_axes(ax)
    save_plot(figure_dir / f"robustness_{task}.png")


def has_robustness_results(output_dir: Path) -> bool:
    return any((output_dir / task / "robustness_results.csv").exists() for task in TASKS)


def robustness_fraction(output_dir: Path) -> float:
    cfg_path = output_dir / "run_config.json"
    if not cfg_path.exists():
        return 0.0
    try:
        return float(read_json(cfg_path).get("sample_fraction", 0.0))
    except Exception:
        return 0.0


def select_robustness_output(outputs: List[Path], override: str) -> Path | None:
    if override:
        path = Path(override)
        if not path.exists():
            raise SystemExit(f"Robustness output does not exist: {path}")
        return path if has_robustness_results(path) else None

    candidates = [p for p in outputs if has_robustness_results(p)]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda p: (
            robustness_fraction(p),
            1 if "full" in p.name else 0,
            0 if "smoke" in p.name else 1,
            p.stat().st_mtime,
        ),
        reverse=True,
    )[0]


def collect_robustness_comparison(output_dirs: List[Path]) -> pd.DataFrame:
    rows = []
    for out_dir in output_dirs:
        if not out_dir.exists():
            continue
        for task in TASKS:
            path = out_dir / task / "robustness_results.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            for row in df.to_dict(orient="records"):
                rows.append({"run": out_dir.name, "task": task, **row})
    return pd.DataFrame(rows)


def save_robustness_comparison(df: pd.DataFrame, figure_dir: Path, table_dir: Path) -> None:
    if df.empty:
        return
    df.to_csv(table_dir / "robustness_comparison.csv", index=False)
    save_robustness_tradeoff(df, figure_dir, table_dir)
    for task, group in df.groupby("task"):
        plot_df = group.copy()
        plot_df["condition"] = plot_df["attack"].astype(str) + "=" + plot_df["level"].astype(str)
        order = (
            ["clean=none"]
            + [f"jpeg={x}" for x in ("95", "75", "50")]
            + [f"resize={x}" for x in ("0.5", "0.75", "1.5")]
            + [f"noise={x}" for x in ("2", "5", "10")]
        )
        plot_df["condition"] = pd.Categorical(plot_df["condition"], categories=order, ordered=True)
        pivot = plot_df.pivot_table(index="condition", columns="run", values="macro_f1", aggfunc="first")
        pivot = pivot.dropna(how="all")
        pivot = pivot.rename(columns={run: run_label(run) for run in pivot.columns})
        display_labels = {
            "clean=none": "Clean",
            "jpeg=95": "JPEG 95",
            "jpeg=75": "JPEG 75",
            "jpeg=50": "JPEG 50",
            "resize=0.5": "Resize .5",
            "resize=0.75": "Resize .75",
            "resize=1.5": "Resize 1.5",
            "noise=2": "Noise 2",
            "noise=5": "Noise 5",
            "noise=10": "Noise 10",
        }
        pivot.index = [display_labels.get(str(idx), str(idx)) for idx in pivot.index]
        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        for color, column in zip(SERIES_COLORS, pivot.columns):
            ax.plot(pivot.index, pivot[column], marker="o", linewidth=2.0, markersize=4.5, color=color, label=column)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Degradation condition / 扰动条件")
        ax.set_ylabel("Macro-F1")
        ax.set_title(f"Robustness comparison / 鲁棒性对比: {task_label(task)}")
        ax.tick_params(axis="x", rotation=28)
        ax.legend(title="Output / 实验", loc="center left", bbox_to_anchor=(1.01, 0.5))
        polish_axes(ax)
        save_plot(figure_dir / f"robustness_comparison_{task}.png")


def save_robustness_tradeoff(df: pd.DataFrame, figure_dir: Path, table_dir: Path) -> None:
    rows = []
    for (run, task), group in df.groupby(["run", "task"]):
        clean = group[group["attack"] == "clean"]
        degraded = group[group["attack"] != "clean"]
        if clean.empty or degraded.empty:
            continue
        rows.append(
            {
                "run": run,
                "task": task,
                "clean_macro_f1": float(clean.iloc[0]["macro_f1"]),
                "degraded_mean_macro_f1": float(degraded["macro_f1"].mean()),
                "degraded_min_macro_f1": float(degraded["macro_f1"].min()),
                "clean_minus_degraded_mean": float(clean.iloc[0]["macro_f1"] - degraded["macro_f1"].mean()),
            }
        )
    tradeoff = pd.DataFrame(rows)
    if tradeoff.empty:
        return
    tradeoff.to_csv(table_dir / "robustness_tradeoff.csv", index=False)
    for task, group in tradeoff.groupby("task"):
        fig, ax = plt.subplots(figsize=(5.6, 4.4))
        for _, row in group.iterrows():
            ax.scatter(row["clean_macro_f1"], row["degraded_mean_macro_f1"], s=72, color=SERIES_COLORS[len(ax.collections) % len(SERIES_COLORS)], edgecolor="white", linewidth=0.8)
            ax.text(row["clean_macro_f1"] + 0.005, row["degraded_mean_macro_f1"], str(row["run"]), fontsize=8)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Clean Macro-F1")
        ax.set_ylabel("Degraded 平均 Macro-F1")
        ax.set_title(f"Clean vs degraded tradeoff / 取舍：{task_label(task)}")
        polish_axes(ax)
        save_plot(figure_dir / f"robustness_tradeoff_{task}.png")


def collect_logo_generalization(output_dirs: List[Path]) -> pd.DataFrame:
    rows = []
    for out_dir in output_dirs:
        summary_path = out_dir / "logo_generalization_summary.csv"
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            for row in df.to_dict(orient="records"):
                rows.append({"run": out_dir.name, **row})
            continue
        for run_dir in sorted(out_dir.glob("leave_*_out")):
            task_dir = run_dir / "binary_ai_vs_nature"
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
            record = best.iloc[0].to_dict()
            heldout = run_dir.name.removeprefix("leave_").removesuffix("_out")
            rows.append(
                {
                    "run": out_dir.name,
                    "heldout_generator": heldout,
                    "best_model": best_name,
                    "accuracy": record.get("accuracy", math.nan),
                    "macro_f1": record.get("macro_f1", math.nan),
                    "auc": record.get("auc", math.nan),
                    "run_dir": str(run_dir),
                }
            )
    return pd.DataFrame(rows)


def save_logo_generalization(df: pd.DataFrame, figure_dir: Path, table_dir: Path) -> None:
    if df.empty:
        return
    df = df.sort_values(["run", "heldout_generator"]).reset_index(drop=True)
    df.to_csv(table_dir / "logo_generalization.csv", index=False)
    plot_df = df.copy()
    plot_df["label"] = plot_df["run"] + "\nheld-out " + plot_df["heldout_generator"].astype(str)
    fig, ax = plt.subplots(figsize=(max(7.0, 0.9 * len(plot_df)), 4.4))
    ax.bar(plot_df["label"], plot_df["macro_f1"], color=STYLE["green"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Held-out Macro-F1")
    ax.set_xlabel("留一生成器实验")
    ax.set_title("Cross-generator generalization / 跨生成器泛化")
    ax.tick_params(axis="x", rotation=35)
    polish_axes(ax)
    save_plot(figure_dir / "logo_generalization_macro_f1.png")


def save_confidence_assets(primary: Path, figure_dir: Path, table_dir: Path) -> None:
    for task in TASKS:
        task_dir = primary / task
        details_path = task_dir / "confidence_details.csv"
        coverage_path = task_dir / "confidence_coverage_accuracy.csv"
        by_generator_path = task_dir / "confidence_by_generator.csv"
        if coverage_path.exists():
            coverage = pd.read_csv(coverage_path)
            coverage.to_csv(table_dir / f"confidence_coverage_{task}.csv", index=False)
            fig, ax1 = plt.subplots(figsize=(6.4, 4.2))
            ax1.plot(coverage["coverage"], coverage["accuracy"], label="Accuracy", color=STYLE["blue"], linewidth=2.0)
            ax1.plot(coverage["coverage"], coverage["macro_f1"], label="Macro-F1", color=STYLE["orange"], linewidth=2.0)
            ax1.set_xlim(1.02, -0.02)
            ax1.set_ylim(0, 1.05)
            ax1.set_xlabel("拒识低置信度样本后的 coverage")
            ax1.set_ylabel("保留样本得分")
            ax1.set_title(f"Coverage-accuracy / 置信度拒识：{task_label(task)}")
            ax1.legend(loc="lower left")
            polish_axes(ax1)
            save_plot(figure_dir / f"confidence_coverage_{task}.png")
        if details_path.exists():
            details = pd.read_csv(details_path)
            details.to_csv(table_dir / f"confidence_details_{task}.csv", index=False)
            fig, ax = plt.subplots(figsize=(6.2, 4.0))
            for correct, group in details.groupby("correct"):
                label = "correct" if bool(correct) else "wrong"
                color = STYLE["green"] if bool(correct) else STYLE["rose"]
                ax.hist(group["confidence"], bins=30, alpha=0.72, label=label, color=color)
            ax.set_xlabel("预测置信度")
            ax.set_ylabel("样本数")
            ax.set_title(f"Confidence distribution / 置信度分布：{task_label(task)}")
            ax.legend()
            polish_axes(ax)
            save_plot(figure_dir / f"confidence_histogram_{task}.png")
        if by_generator_path.exists():
            pd.read_csv(by_generator_path).to_csv(table_dir / f"confidence_by_generator_{task}.csv", index=False)


def fft_power(path: Path, size: int = 256) -> np.ndarray:
    image = Image.open(path).convert("L").resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(image, dtype=np.float32)
    arr = arr - arr.mean()
    arr = arr * np.outer(np.hanning(arr.shape[0]), np.hanning(arr.shape[1]))
    spectrum = np.fft.fftshift(np.fft.fft2(arr))
    return np.log1p(np.abs(spectrum) ** 2)


def save_spectrum_examples(dataset_root: Path, figure_dir: Path) -> None:
    generators = [p for p in sorted(dataset_root.iterdir()) if p.is_dir() and not p.name.startswith("_")]
    generators = generators[:4]
    if not generators:
        return
    fig, axes = plt.subplots(2, len(generators), figsize=(3.0 * len(generators), 5.2))
    if len(generators) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for col, gen in enumerate(generators):
        ai_image = first_image(gen / "val" / "ai")
        nature_image = first_image(gen / "val" / "nature")
        for row, (label, image_path) in enumerate([("AI", ai_image), ("Nature", nature_image)]):
            ax = axes[row, col]
            if image_path is not None:
                power = fft_power(image_path)
                ax.imshow(power, cmap="magma")
            ax.set_title(f"{gen.name} {label}", fontsize=9, color=STYLE["ink"])
            ax.set_axis_off()
    fig.suptitle("FFT power spectrum examples / 频谱示例", fontsize=13, color=STYLE["ink"])
    save_plot(figure_dir / "spectrum_examples.png")


def write_manifest(
    report_dir: Path,
    outputs: List[Path],
    primary: Path,
    dataset_counts: pd.DataFrame,
    robustness_primary: Path | None = None,
) -> None:
    lines = [
        "# Report Asset Manifest",
        "",
        f"- Primary output: `{primary}`",
        f"- Robustness output: `{robustness_primary}`" if robustness_primary else "- Robustness output: none",
        f"- Output directories: {', '.join(f'`{p}`' for p in outputs)}",
        f"- Dataset rows: {len(dataset_counts)}",
        "",
        "Generated files are stored under `report/tables` and `report/figures`.",
    ]
    (report_dir / "asset_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    apply_ppt_style()
    dataset_root = Path(args.dataset_root)
    report_dir = Path(args.report_dir)
    figure_dir = report_dir / "figures"
    table_dir = report_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    outputs = [p for p in discover_outputs(args.outputs) if p.exists()]
    if not outputs:
        raise SystemExit("No output directories found.")
    primary = Path(args.primary_output) if args.primary_output else outputs[-1]
    if not primary.exists():
        raise SystemExit(f"Primary output does not exist: {primary}")
    robustness_primary = select_robustness_output(outputs, args.robustness_output)
    robustness_compare_outputs = [Path(p) for p in args.robustness_compare_outputs]

    dataset_counts = count_dataset(dataset_root)
    dataset_counts.to_csv(table_dir / "dataset_counts.csv", index=False)

    summary, comparison = collect_experiment_tables(outputs)
    summary.to_csv(table_dir / "experiment_summary.csv", index=False)
    comparison.to_csv(table_dir / "model_comparison_long.csv", index=False)
    save_model_comparison_plot(comparison, figure_dir)
    single_vs_fusion = collect_single_vs_fusion(summary)
    if not single_vs_fusion.empty:
        single_vs_fusion.to_csv(table_dir / "single_vs_fusion_profiles.csv", index=False)
        save_single_vs_fusion_plot(single_vs_fusion, figure_dir)
    save_logo_generalization(collect_logo_generalization(outputs), figure_dir, table_dir)
    seed_stability = collect_seed_stability(summary)
    if not seed_stability.empty:
        seed_stability.to_csv(table_dir / "seed_stability.csv", index=False)
    ablation = collect_feature_ablation(summary)
    if not ablation.empty:
        ablation.to_csv(table_dir / "feature_ablation.csv", index=False)
        save_ablation_plot(ablation, figure_dir)
    tuning = collect_lgbm_tuning(summary)
    if not tuning.empty:
        tuning.to_csv(table_dir / "lgbm_tuning.csv", index=False)
        save_lgbm_tuning_plot(tuning, figure_dir)
    selection = collect_final_selection(summary)
    if not selection.empty:
        selection.to_csv(table_dir / "final_result_selection.csv", index=False)
        selection.to_csv(table_dir / "scaleup_results.csv", index=False)
        save_scaleup_plot(selection, figure_dir)
        save_enhanced_search_plot(selection, figure_dir)

    for task in TASKS:
        task_dir = primary / task
        save_confusion_plot(task_dir, task, figure_dir)
        save_feature_importance_plot(task_dir, task, figure_dir, table_dir)
        robust_task_dir = (robustness_primary / task) if robustness_primary else task_dir
        save_robustness_plot(robust_task_dir, task, figure_dir, table_dir)
    save_confidence_assets(primary, figure_dir, table_dir)
    if robustness_compare_outputs:
        save_robustness_comparison(
            collect_robustness_comparison(robustness_compare_outputs),
            figure_dir,
            table_dir,
        )

    save_spectrum_examples(dataset_root, figure_dir)
    write_manifest(report_dir, outputs, primary, dataset_counts, robustness_primary)

    print(f"Wrote report assets to {report_dir.resolve()}")


if __name__ == "__main__":
    main()
