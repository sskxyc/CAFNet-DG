# CAFNet-DG Reproducibility Package

This repository contains code, benchmark data, and result artifacts for CAFNet-DG, a frequency-aware and prevalence-aware framework for drug side-effect prioritization.

## Contents

- `Net.py`, `utils.py`, `vector.py`: model and utility code.
- `cold-scence.py`: cold-start training/evaluation entry point.
- `warm-scence.py`: warm-start training/evaluation entry point.
- `analysis_scripts/`: auxiliary analysis scripts.
- `analysis_outputs/`: curated result summaries and scripts used to generate manuscript tables.
- `data/`: SIDER-derived benchmark matrices and fixed train/test masks.
- `data_external/`: external evidence mappings used for supplementary analyses.
- `result_ICS/`: saved cold-start prediction matrices for CAFNet, A3Net, CAFNet-D, and CAFNet-DG.
- `result_baselines_*`: saved baseline predictions used in the manuscript.

## Installation

```bash
conda env create -f environment.yml
conda activate cafnet-dg
```

The original experiments used Python 3.9, PyTorch 1.12.1 with CUDA 11.3, PyTorch Geometric 1.7.2, RDKit 2022.9.4, NumPy 1.22.4, Pandas 1.5.1, SciPy 1.10.1, and scikit-learn 1.2.1.

## Data

The main benchmark files are:

```text
data/raw_frequency_750.mat
data/frequency_data.txt
data/drug_SMILES_750.csv
data/side_effect_label_750.mat
data/mask_mat_750.mat
data/blind_mask_mat_750.mat
data/scaffold_mask_mat_750.mat
```

The warm-start split is `mask_mat_750.mat`; the drug-disjoint cold-start split is `blind_mask_mat_750.mat`.

## Reproducing Main Tables from Saved Predictions

The manuscript tables can be reproduced from saved predictions and analysis outputs without retraining:

```text
analysis_outputs/statistics_jbhi/
analysis_outputs/hot_side_effect_bias_cafnet_dg/
analysis_outputs/cafnet_dg_external_validation/
analysis_outputs/cafnet_dg_per_drug/
analysis_outputs/cafnet_dg_completion_20260701/
```

The main CAFNet-DG cold-start prediction file is:

```text
result_ICS/10cafnet_dg_ensemble06_cafnetd04_cafnet/blind_pred.csv
```

## External Validation

OFFSIDES matched-control external validation outputs are in:

```text
analysis_outputs/cafnet_dg_external_validation/
```

The external validation is an ADR prioritization/ranking analysis, not an external frequency-regression or causal clinical validation. Matched controls are unobserved pairs matched by side-effect prevalence, not confirmed negative adverse reactions.

## Large Files

Large scored-pair CSVs are distributed through Git LFS or the release archive, not normal Git tracking:

```text
analysis_outputs/cafnet_dg_external_validation/offsides_matched_control_scored_pairs.csv
analysis_outputs/cafnet_dg_external_validation/offsides_matched_control_scored_pairs_with_prevalence_meta.csv
data_external/external_pairs/offsides_contrastive_pairs_all_folds.csv
```

## Citation

Please cite the associated manuscript when using this code or result package.

