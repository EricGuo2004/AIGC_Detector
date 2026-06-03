from __future__ import annotations

import argparse
import os
import csv
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


TASKS = ["binary_ai_vs_nature", "ai_subsource_attribution"]
REQUIRED_SPLITS = ["train/ai", "train/nature", "val/ai", "val/nature"]
REQUIRED_GENERATORS = ["ADM", "BigGAN", "VQDM", "glide"]


@dataclass
class Check:
    area: str
    name: str
    status: str
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate final AIGC_Detector submission assets.")
    parser.add_argument("--dataset-root", default=os.environ.get("GENIMAGE_DATA_ROOT", "data/GenImage_data"))
    parser.add_argument("--report-dir", default="report")
    parser.add_argument("--primary-output", default="outputs_v2_full_best")
    parser.add_argument("--robustness-output", default="outputs_v2_full_best_robust_20pct")
    parser.add_argument("--baseline-robustness-output", default="outputs_4gen_full_best_robust_20pct")
    parser.add_argument("--max-pages", type=int, default=7)
    parser.add_argument("--write-report", action="store_true")
    return parser.parse_args()


def add(checks: list[Check], area: str, name: str, ok: bool, detail: str, warn: bool = False) -> None:
    status = "PASS" if ok else ("WARN" if warn else "FAIL")
    checks.append(Check(area, name, status, detail))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def dir_exists(path: Path) -> bool:
    return path.exists() and path.is_dir()


def check_dataset(checks: list[Check], dataset_root: Path) -> None:
    add(checks, "dataset", "dataset root", dir_exists(dataset_root), str(dataset_root))
    if not dataset_root.exists():
        return
    for gen in REQUIRED_GENERATORS:
        gen_dir = dataset_root / gen
        add(checks, "dataset", f"{gen} root", dir_exists(gen_dir), str(gen_dir))
        for rel in REQUIRED_SPLITS:
            split_dir = gen_dir / rel
            count = 0
            if split_dir.exists():
                count = sum(1 for p in split_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"})
            add(checks, "dataset", f"{gen}/{rel}", count > 0, f"{count} images")


def check_report_assets(checks: list[Check], report_dir: Path, max_pages: int) -> None:
    required_files = [
        report_dir / "main.tex",
        report_dir / "main.pdf",
        report_dir / "references.bib",
        report_dir / "tables" / "optimization_v2_summary.csv",
        report_dir / "tables" / "robustness_comparison.csv",
        report_dir / "tables" / "dataset_counts.csv",
        report_dir / "figures" / "optimization_v2_macro_f1.png",
        report_dir / "figures" / "robustness_comparison_binary_ai_vs_nature.png",
        report_dir / "figures" / "robustness_comparison_ai_subsource_attribution.png",
        report_dir / "figures" / "feature_importance_binary_ai_vs_nature_top20.png",
    ]
    for path in required_files:
        add(checks, "report", path.as_posix(), file_exists(path), f"{path.stat().st_size if path.exists() else 0} bytes")

    log_path = report_dir / "main.log"
    if log_path.exists():
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"Output written on main\.pdf \((\d+) pages?\)", text)
        pages = int(match.group(1)) if match else -1
        add(checks, "report", "pdf page count", 0 < pages <= max_pages, f"{pages} pages, max {max_pages}")
    else:
        add(checks, "report", "pdf page count", False, "main.log missing", warn=True)


def check_notebook(checks: list[Check], notebook_path: Path) -> None:
    try:
        nb = read_json(notebook_path)
        cells = nb.get("cells", [])
        code_cells = sum(1 for cell in cells if cell.get("cell_type") == "code")
        add(checks, "notebook", "valid JSON", len(cells) > 0, f"{len(cells)} cells, {code_cells} code cells")
    except Exception as exc:  # noqa: BLE001
        add(checks, "notebook", "valid JSON", False, str(exc))


def best_model_metrics(output_dir: Path, task: str) -> dict:
    metrics = read_json(output_dir / task / "metrics_summary.json")
    best_model = metrics["best_model"]
    rows = metrics["all_models"]
    best = next((row for row in rows if row.get("model") == best_model), rows[0])
    return {"best_model": best_model, **best}


def check_primary_results(checks: list[Check], primary_output: Path, report_dir: Path) -> None:
    add(checks, "results", "primary output", dir_exists(primary_output), str(primary_output))
    if not primary_output.exists():
        return

    binary = best_model_metrics(primary_output, "binary_ai_vs_nature")
    attribution = best_model_metrics(primary_output, "ai_subsource_attribution")
    combined = (float(binary["macro_f1"]) + float(attribution["macro_f1"])) / 2.0

    add(checks, "results", "binary macro-F1", float(binary["macro_f1"]) >= 0.99, f"{binary['macro_f1']:.6f}")
    add(checks, "results", "binary AUC", float(binary.get("auc", 0.0)) >= 0.999, f"{binary.get('auc', 0.0):.6f}")
    add(checks, "results", "attribution macro-F1", float(attribution["macro_f1"]) >= 0.999, f"{attribution['macro_f1']:.6f}")
    add(checks, "results", "combined macro-F1", combined >= 0.995, f"{combined:.6f}")

    summary_path = report_dir / "tables" / "optimization_v2_summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        selected = df[df["selected"] == True]  # noqa: E712
        ok = len(selected) == 1 and selected.iloc[0]["run"] == primary_output.name
        detail = selected.iloc[0]["run"] if len(selected) else "no selected row"
        add(checks, "results", "selected result table", ok, detail)


def check_robustness_output(checks: list[Check], output_dir: Path, area_suffix: str) -> None:
    add(checks, "robustness", f"{area_suffix} output", dir_exists(output_dir), str(output_dir))
    for task in TASKS:
        path = output_dir / task / "robustness_results.csv"
        if not path.exists():
            add(checks, "robustness", f"{area_suffix} {task}", False, "missing robustness_results.csv")
            continue
        df = pd.read_csv(path)
        attacks = set(df["attack"].astype(str))
        expected_attacks = {"clean", "jpeg", "resize", "noise"}
        ok = len(df) == 10 and expected_attacks.issubset(attacks)
        clean = df[df["attack"] == "clean"]["macro_f1"]
        clean_detail = f", clean={clean.iloc[0]:.6f}" if len(clean) else ""
        add(checks, "robustness", f"{area_suffix} {task} rows", ok, f"{len(df)} rows{clean_detail}")


def check_git_hygiene(checks: list[Check]) -> None:
    result = subprocess.run(["git", "ls-files"], check=True, capture_output=True, text=True)
    tracked = result.stdout.splitlines()
    forbidden_prefixes = (
        ".venv/",
        "outputs",
        "feature_cache",
        "robustness_cache",
        "sample_cache/",
        "experiment_logs/",
        "GenImage_data/",
    )
    bad = [path for path in tracked if path.startswith(forbidden_prefixes)]
    add(checks, "git", "large/local directories excluded", not bad, ", ".join(bad[:10]) if bad else "ok")

    status = subprocess.run(["git", "status", "--short"], check=True, capture_output=True, text=True).stdout.strip()
    add(checks, "git", "working tree status", status == "", status if status else "clean", warn=status != "")


def write_reports(checks: list[Check], report_dir: Path) -> None:
    table_dir = report_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    csv_path = table_dir / "submission_validation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["area", "name", "status", "detail"])
        writer.writeheader()
        for check in checks:
            writer.writerow(check.__dict__)

    total = len(checks)
    passed = sum(1 for check in checks if check.status == "PASS")
    warned = sum(1 for check in checks if check.status == "WARN")
    failed = sum(1 for check in checks if check.status == "FAIL")
    lines = [
        "# Submission Checklist",
        "",
        f"- Total checks: {total}",
        f"- Passed: {passed}",
        f"- Warnings: {warned}",
        f"- Failed: {failed}",
        "",
        "| Area | Check | Status | Detail |",
        "| --- | --- | --- | --- |",
    ]
    for check in checks:
        detail = check.detail.replace("|", "\\|")
        lines.append(f"| {check.area} | {check.name} | {check.status} | {detail} |")
    (report_dir / "submission_checklist.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(checks: Iterable[Check]) -> int:
    checks = list(checks)
    for check in checks:
        print(f"[{check.status}] {check.area}: {check.name} - {check.detail}")
    failed = sum(1 for check in checks if check.status == "FAIL")
    warned = sum(1 for check in checks if check.status == "WARN")
    print(f"\nSummary: {len(checks) - failed - warned} pass, {warned} warn, {failed} fail")
    return 1 if failed else 0


def main() -> None:
    args = parse_args()
    checks: list[Check] = []
    dataset_root = Path(args.dataset_root)
    report_dir = Path(args.report_dir)
    primary_output = Path(args.primary_output)
    robustness_output = Path(args.robustness_output)
    baseline_robustness_output = Path(args.baseline_robustness_output)

    check_dataset(checks, dataset_root)
    check_report_assets(checks, report_dir, args.max_pages)
    check_notebook(checks, Path("notebooks") / "final_project_aigc_detector.ipynb")
    check_primary_results(checks, primary_output, report_dir)
    check_robustness_output(checks, robustness_output, "v2")
    check_robustness_output(checks, baseline_robustness_output, "baseline")
    check_git_hygiene(checks)

    if args.write_report:
        write_reports(checks, report_dir)

    raise SystemExit(print_summary(checks))


if __name__ == "__main__":
    main()
