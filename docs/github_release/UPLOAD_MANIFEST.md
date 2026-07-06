# CAFNet-DG GitHub Upload Manifest

This manifest separates files for the public GitHub repository from large result artifacts that should be uploaded through Git LFS, GitHub Releases, or an external archive such as Zenodo/OSF.

## Recommended Repository Layout

```text
CAFNet-DG/
  README.md
  LICENSE
  environment.yml
  Net.py
  utils.py
  vector.py
  cold-scence.py
  warm-scence.py
  independent_test_onsides.py
  analysis_scripts/
  analysis_outputs/
  data/
  data_external/
  result_ICS/
  result_baselines_*/
  docs/github_release/
```

## Must Upload: Core Code

Upload these to the normal Git repository:

```text
Net.py
utils.py
vector.py
cold-scence.py
warm-scence.py
independent_test_onsides.py
analysis_scripts/
analysis_outputs/cafnet_dg_completion_20260701/*.py
analysis_outputs/cafnet_dg_external_validation/*.py
analysis_outputs/cafnet_dg_per_drug/*.py
analysis_outputs/hot_side_effect_bias_cafnet_dg/*.py
analysis_outputs/statistics_jbhi/
environment.yml
```

## Must Upload: Original and Processed Benchmark Data

These are required for reproducing the reported same-mask experiments. They should be tracked with Git LFS because several `.mat` masks are larger than normal GitHub comfort limits.

```text
data/raw_frequency_750.mat
data/frequency_data.txt
data/drug_SMILES_750.csv
data/drug_SMILES_750.txt
data/side_effect_label_750.mat
data/mask_mat_750.mat
data/blind_mask_mat_750.mat
data/scaffold_mask_mat_750.mat
side_effect_label_750.mat
frequencyMat.csv
```

Recommended `.gitattributes` entries:

```text
*.mat filter=lfs diff=lfs merge=lfs -text
*.npy filter=lfs diff=lfs merge=lfs -text
*.npz filter=lfs diff=lfs merge=lfs -text
*.csv filter=lfs diff=lfs merge=lfs -text
```

If avoiding Git LFS, upload the `data/` folder as a compressed release artifact and keep a download note in `README.md`.

## Must Upload: Main Result Data

These are the minimum result artifacts needed for reviewers to reproduce the manuscript tables without retraining.

### Main cold-start predictions

```text
result_ICS/10ICS_CAFNet_knn=5_wd=0.001_epoch=100_lamb=0.03_lr0.0004_dim=200_eps=0.5_DF=False_PCA=False_not-FC=False_cosine/blind_pred.csv
result_ICS/10ICS_A3_Net_knn=5_wd=0.001_epoch=100_lamb=0.03_lr0.0004_dim=200_eps=0.5_DF=False_PCA=False_not-FC=False_cosine/blind_pred.csv
result_ICS/10cd3e100f10_CAFNetDecoupled_knn=5_wd=0.001_epoch=100_lamb=0.03_lr0.0004_dim=200_eps=0.5_DF=False_PCA=False_not-FC=False_cross=True_fusion=gate_gate=new_fa=0.5_gatdrop=0.0_mix=0.3_aw=1.0_fw=1.0_rw=0.05_popw=0.1_biasw=1.0_listw=0.1_abw=1.0_arw=1.0_cosine/blind_pred.csv
result_ICS/10cafnet_dg_ensemble06_cafnetd04_cafnet/blind_pred.csv
```

### Main baseline predictions

```text
result_baselines_a3net_rdkit_cold_v1/cold/RF/
result_baselines_a3net_rdkit_cold_v1/cold/XGB/
result_baselines_hstrans_same_masks_epoch100/cold/HSTrans/
result_baselines_popularity/cold/GLOBAL_POPULARITY/
```

### Main analysis outputs

```text
analysis_outputs/statistics_jbhi/
analysis_outputs/hot_side_effect_bias_cafnet_dg/
analysis_outputs/cafnet_dg_per_drug/
analysis_outputs/cafnet_dg_completion_20260701/rho_sensitivity/
analysis_outputs/cafnet_dg_completion_20260701/warm_full_fusion/
analysis_outputs/cafnet_dg_external_validation/offsides_extended_condition_summary.csv
analysis_outputs/cafnet_dg_external_validation/offsides_extended_condition_per_drug.csv
analysis_outputs/cafnet_dg_external_validation/offsides_extended_condition_paired_tests.csv
analysis_outputs/cafnet_dg_external_validation/offsides_external_drug_bootstrap_ci.csv
analysis_outputs/cafnet_dg_external_validation/offsides_matched_control_global_summary.csv
analysis_outputs/cafnet_dg_external_validation/offsides_matched_control_paired_tests.csv
analysis_outputs/cafnet_dg_external_validation/EXTENDED_OFFSIDES_EXTERNAL_VALIDATION_REPORT_20260701.md
```

## Large Result Artifacts: Upload via Release/LFS, Not Normal Git

These are useful for full reproducibility but too large/noisy for ordinary Git tracking.

```text
analysis_outputs/cafnet_dg_external_validation/offsides_matched_control_scored_pairs.csv                  # ~118 MB
analysis_outputs/cafnet_dg_external_validation/offsides_matched_control_scored_pairs_with_prevalence_meta.csv # ~145 MB
data_external/external_pairs/offsides_contrastive_pairs_all_folds.csv                                      # ~107 MB
```

Suggested release artifact:

```text
cafnet-dg-large-results-20260701.zip
```

## Do Not Upload

Do not upload these to the public GitHub repository:

```text
.local_pkgs/
.conda_envs/
.idea/
__pycache__/
result_WS/                         # very large; includes many checkpoints/full prediction intermediates
new_data/
code_snapshots/
submission_flat/                   # manuscripts, old templates, duplicate PDFs
*.pt
*.pth
*.model
save_model/
texput.log
```

## Baseline Code

Upload only baseline code that is necessary to reproduce the reported tables and include upstream citation/license notes. Do not upload environment folders or downloaded pretrained checkpoints. For HMMF, clearly state that PubMedBERT regenerated embeddings were used rather than the original unreleased KV-PLM embeddings.

## Minimal Reviewer Reproduction Path

1. Install environment:

```bash
conda env create -f environment.yml
conda activate cafnet-dg
```

2. Verify data files in `data/`.
3. Use saved `blind_pred.csv` files and `analysis_outputs/` scripts to regenerate manuscript tables.
4. Optional: rerun `cold-scence.py` or `warm-scence.py` for training, noting that training is slower and GPU-dependent.

