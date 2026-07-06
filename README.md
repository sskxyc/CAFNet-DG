# CAFNet-DG: Drug Side-Effect Frequency-Aware Prioritization

This repository contains code, benchmark data, and curated result artifacts for CAFNet-DG, a frequency-aware and prevalence-aware framework for drug side-effect prioritization.

CAFNet-DG combines a task-decoupled CAFNet-D branch with the original CAFNet score as a residual ranking signal. The released materials support reproduction of the same-mask warm-start and drug-disjoint cold-start analyses, popularity-controlled diagnostics, and OFFSIDES matched-control external validation reported in the manuscript.

## Repository Contents

```text
Net.py                         Model definitions.
utils.py                       Training and evaluation utilities.
vector.py                      SMILES-to-graph preprocessing helpers.
cold-scence.py                 Cold-start training/evaluation entry point.
warm-scence.py                 Warm-start training/evaluation entry point.
analysis_scripts/              Additional analysis scripts.
analysis_outputs/              Curated tables, reports, and analysis scripts.
data/                          Benchmark matrices and fixed train/test masks.
data_external/                 External evidence mappings used in supplementary analyses.
result_ICS/                    Saved cold-start prediction matrices.
result_baselines_*/            Saved baseline predictions used in the manuscript.
docs/github_release/           Upload manifest and result-data release notes.
```

## Installation

The experiments were run with Python 3.9, PyTorch 1.12.1 + CUDA 11.3, PyTorch Geometric 1.7.2, RDKit 2022.9.4, NumPy 1.22.4, Pandas 1.5.1, SciPy 1.10.1, and scikit-learn 1.2.1.

```bash
conda env create -f environment.yml
conda activate cafnet-dg
```

## Data

The main benchmark files are:

```text
data/raw_frequency_750.mat
data/frequency_data.txt
data/drug_SMILES_750.csv
data/drug_SMILES_750.txt
data/side_effect_label_750.mat
data/mask_mat_750.mat
data/blind_mask_mat_750.mat
data/scaffold_mask_mat_750.mat
```

`mask_mat_750.mat` defines warm-start folds. `blind_mask_mat_750.mat` defines drug-disjoint cold-start folds. `scaffold_mask_mat_750.mat` is used for scaffold-split feasibility analyses.

## Reproducing Manuscript Tables from Saved Results

The fastest path is to use the saved prediction matrices and curated analysis outputs:

```text
analysis_outputs/statistics_jbhi/
analysis_outputs/hot_side_effect_bias_cafnet_dg/
analysis_outputs/cafnet_dg_per_drug/
analysis_outputs/cafnet_dg_external_validation/
analysis_outputs/cafnet_dg_completion_20260701/
```

The main CAFNet-DG cold-start prediction matrix is:

```text
result_ICS/10cafnet_dg_ensemble06_cafnetd04_cafnet/blind_pred.csv
```

Important baseline result folders include:

```text
result_baselines_a3net_rdkit_cold_v1/cold/RF/
result_baselines_a3net_rdkit_cold_v1/cold/XGB/
result_baselines_hstrans_same_masks_epoch100/cold/HSTrans/
result_baselines_popularity/cold/GLOBAL_POPULARITY/
```

## Training

Cold-start training example:

```bash
python cold-scence.py --tenfold --epoch 100 --lr 0.0004
```

Warm-start training example:

```bash
python warm-scence.py --tenfold --epoch 100 --lr 0.0004
```

Exact result-folder names encode the hyperparameters used for the reported runs.

## External Validation

OFFSIDES matched-control external validation outputs are in:

```text
analysis_outputs/cafnet_dg_external_validation/
```

This validation is an external ADR prioritization/ranking analysis. It is not an external frequency-regression validation and should not be interpreted as clinical causal validation. Matched controls are prevalence-matched unobserved pairs rather than confirmed negative adverse reactions.

## Large Files and Git LFS

Large benchmark files and large result tables should be tracked with Git LFS or uploaded as GitHub Release assets. See:

```text
docs/github_release/UPLOAD_MANIFEST.md
docs/github_release/RESULT_DATA_MANIFEST.md
```

Do not upload local environment caches, old manuscript snapshots, or intermediate checkpoints unless they are explicitly needed for a release.

## Citation

Please cite the associated manuscript when using this code, data, or result package.
