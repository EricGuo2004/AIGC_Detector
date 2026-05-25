# Submission Checklist

- Total checks: 47
- Passed: 46
- Warnings: 1
- Failed: 0

| Area | Check | Status | Detail |
| --- | --- | --- | --- |
| dataset | dataset root | PASS | C:\Users\99303\git\GenImage_data |
| dataset | ADM root | PASS | C:\Users\99303\git\GenImage_data\ADM |
| dataset | ADM/train/ai | PASS | 162000 images |
| dataset | ADM/train/nature | PASS | 157453 images |
| dataset | ADM/val/ai | PASS | 6000 images |
| dataset | ADM/val/nature | PASS | 6000 images |
| dataset | BigGAN root | PASS | C:\Users\99303\git\GenImage_data\BigGAN |
| dataset | BigGAN/train/ai | PASS | 162000 images |
| dataset | BigGAN/train/nature | PASS | 162000 images |
| dataset | BigGAN/val/ai | PASS | 6000 images |
| dataset | BigGAN/val/nature | PASS | 6000 images |
| dataset | VQDM root | PASS | C:\Users\99303\git\GenImage_data\VQDM |
| dataset | VQDM/train/ai | PASS | 162000 images |
| dataset | VQDM/train/nature | PASS | 162000 images |
| dataset | VQDM/val/ai | PASS | 6000 images |
| dataset | VQDM/val/nature | PASS | 6000 images |
| dataset | glide root | PASS | C:\Users\99303\git\GenImage_data\glide |
| dataset | glide/train/ai | PASS | 162000 images |
| dataset | glide/train/nature | PASS | 162000 images |
| dataset | glide/val/ai | PASS | 6000 images |
| dataset | glide/val/nature | PASS | 6000 images |
| report | report/main.tex | PASS | 13905 bytes |
| report | report/main.pdf | PASS | 1159204 bytes |
| report | report/references.bib | PASS | 893 bytes |
| report | report/tables/optimization_v2_summary.csv | PASS | 5277 bytes |
| report | report/tables/robustness_comparison.csv | PASS | 28510 bytes |
| report | report/tables/dataset_counts.csv | PASS | 173 bytes |
| report | report/figures/optimization_v2_macro_f1.png | PASS | 130260 bytes |
| report | report/figures/robustness_comparison_binary_ai_vs_nature.png | PASS | 134195 bytes |
| report | report/figures/robustness_comparison_ai_subsource_attribution.png | PASS | 132303 bytes |
| report | report/figures/feature_importance_binary_ai_vs_nature_top20.png | PASS | 137872 bytes |
| report | pdf page count | PASS | 4 pages, max 7 |
| notebook | valid JSON | PASS | 23 cells, 12 code cells |
| results | primary output | PASS | outputs_v2_full_best |
| results | binary macro-F1 | PASS | 0.991250 |
| results | binary AUC | PASS | 0.999618 |
| results | attribution macro-F1 | PASS | 0.999083 |
| results | combined macro-F1 | PASS | 0.995167 |
| results | selected result table | PASS | outputs_v2_full_best |
| robustness | v2 output | PASS | outputs_v2_full_best_robust_20pct |
| robustness | v2 binary_ai_vs_nature rows | PASS | 10 rows, clean=0.990833 |
| robustness | v2 ai_subsource_attribution rows | PASS | 10 rows, clean=0.999167 |
| robustness | baseline output | PASS | outputs_4gen_full_best_robust_20pct |
| robustness | baseline binary_ai_vs_nature rows | PASS | 10 rows, clean=0.894157 |
| robustness | baseline ai_subsource_attribution rows | PASS | 10 rows, clean=0.904850 |
| git | large/local directories excluded | PASS | ok |
| git | working tree status | WARN | M  .gitignore
M  README.md
M  notebooks/final_project_aigc_detector.ipynb
M  report/asset_manifest.md
A  report/figures/confidence_coverage_ai_subsource_attribution.png
A  report/figures/confidence_coverage_binary_ai_vs_nature.png
A  report/figures/confidence_histogram_ai_subsource_attribution.png
A  report/figures/confidence_histogram_binary_ai_vs_nature.png
A  report/figures/logo_generalization_macro_f1.png
M  report/figures/model_comparison_macro_f1.png
M  report/figures/robustness_comparison_ai_subsource_attribution.png
M  report/figures/robustness_comparison_binary_ai_vs_nature.png
A  report/figures/robustness_tradeoff_ai_subsource_attribution.png
A  report/figures/robustness_tradeoff_binary_ai_vs_nature.png
M  report/main.pdf
M  report/main.tex
M  report/submission_checklist.md
A  report/tables/confidence_by_generator_ai_subsource_attribution.csv
A  report/tables/confidence_by_generator_binary_ai_vs_nature.csv
A  report/tables/confidence_coverage_ai_subsource_attribution.csv
A  report/tables/confidence_coverage_binary_ai_vs_nature.csv
M  report/tables/experiment_summary.csv
A  report/tables/logo_generalization.csv
M  report/tables/model_comparison_long.csv
M  report/tables/robustness_comparison.csv
A  report/tables/robustness_tradeoff.csv
M  report/tables/submission_validation.csv |
