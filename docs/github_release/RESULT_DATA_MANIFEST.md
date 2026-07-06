# Result Data Manifest

This file lists the result data that should accompany the GitHub release.

## Primary Manuscript Tables

| Purpose | Path |
|---|---|
| Cold paired tests | `analysis_outputs/statistics_jbhi/jbhi_main_paired_tests.csv` |
| Cold paired-test summary | `analysis_outputs/statistics_jbhi/jbhi_main_paired_tests_summary.csv` |
| Ranking LaTeX table | `analysis_outputs/statistics_jbhi/jbhi_main_ranking_table.tex` |
| Regression LaTeX table | `analysis_outputs/statistics_jbhi/jbhi_main_regression_table.tex` |
| Hot-side-effect removal | `analysis_outputs/hot_side_effect_bias_cafnet_dg/removed_hot_side_effect_metrics_summary.csv` |
| Hot-side-effect top-k fraction | `analysis_outputs/hot_side_effect_bias_cafnet_dg/topk_hot_fraction_summary.csv` |
| Per-drug success/failure | `analysis_outputs/cafnet_dg_per_drug/per_drug_delta_metrics.csv` |
| ATC-stratified diagnostics | `analysis_outputs/cafnet_dg_per_drug/atc_stratified/` |
| Scaffold-stratified diagnostics | `analysis_outputs/cafnet_dg_per_drug/scaffold_stratified/` |

## CAFNet-DG Prediction Files

| Model | Path | Notes |
|---|---|---|
| CAFNet | `result_ICS/10ICS_CAFNet_.../blind_pred.csv` | 10-fold cold-start same-mask prediction matrix |
| A3Net | `result_ICS/10ICS_A3_Net_.../blind_pred.csv` | 10-fold cold-start same-mask prediction matrix |
| CAFNet-D | `result_ICS/10cd3e100f10_CAFNetDecoupled_.../blind_pred.csv` | CAFNet-D branch |
| CAFNet-DG | `result_ICS/10cafnet_dg_ensemble06_cafnetd04_cafnet/blind_pred.csv` | Main fixed residual-fusion result |

## Baseline Result Folders

```text
result_baselines_a3net_rdkit_cold_v1/cold/RF/
result_baselines_a3net_rdkit_cold_v1/cold/XGB/
result_baselines_hstrans_same_masks_epoch100/cold/HSTrans/
result_baselines_popularity/cold/GLOBAL_POPULARITY/
result_baselines_hmmf_smoke/
analysis_outputs/hmmf_cold_5fold/
```

HMMF is descriptive/supplementary because it uses regenerated PubMedBERT embeddings and only 5 cold-start folds.

## External Validation Results

| Purpose | Path |
|---|---|
| OFFSIDES global summary | `analysis_outputs/cafnet_dg_external_validation/offsides_matched_control_global_summary.csv` |
| OFFSIDES per-drug tests | `analysis_outputs/cafnet_dg_external_validation/offsides_matched_control_paired_tests.csv` |
| OFFSIDES extended summary | `analysis_outputs/cafnet_dg_external_validation/offsides_extended_condition_summary.csv` |
| OFFSIDES extended per-drug | `analysis_outputs/cafnet_dg_external_validation/offsides_extended_condition_per_drug.csv` |
| OFFSIDES extended paired tests | `analysis_outputs/cafnet_dg_external_validation/offsides_extended_condition_paired_tests.csv` |
| OFFSIDES drug-level bootstrap CI | `analysis_outputs/cafnet_dg_external_validation/offsides_external_drug_bootstrap_ci.csv` |
| OFFSIDES report | `analysis_outputs/cafnet_dg_external_validation/EXTENDED_OFFSIDES_EXTERNAL_VALIDATION_REPORT_20260701.md` |

Large external validation files to upload as Release/LFS artifacts:

```text
analysis_outputs/cafnet_dg_external_validation/offsides_matched_control_scored_pairs.csv
analysis_outputs/cafnet_dg_external_validation/offsides_matched_control_scored_pairs_with_prevalence_meta.csv
data_external/external_pairs/offsides_contrastive_pairs_all_folds.csv
```

## Warm-Start CAFNet-DG Prioritization Outputs

```text
analysis_outputs/cafnet_dg_completion_20260701/warm_full_fusion/
```

The warm full-matrix predictions under `result_WS/10WSFULL_*` are large and should be uploaded only if reviewers need raw full matrices. The summary CSVs above are enough to reproduce the reported warm supplementary tables.

## Integrity Checks

Known SHA256 values for key files:

```text
result_ICS/10cafnet_dg_ensemble06_cafnetd04_cafnet/blind_pred.csv
FACE33AB24060840FABF5F9A80CBA3E8F3C685F0407793FCD065E5F09E94ED31
```

Recompute hashes before release:

```powershell
Get-FileHash data/raw_frequency_750.mat -Algorithm SHA256
Get-FileHash result_ICS/10cafnet_dg_ensemble06_cafnetd04_cafnet/blind_pred.csv -Algorithm SHA256
```

