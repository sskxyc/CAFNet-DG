# CAFNet-DG: Drug Side-Effect Frequency-Aware Prioritization

This repository contains code, benchmark data, and curated result artifacts for CAFNet-DG, a frequency-aware and prevalence-aware framework for drug side-effect prioritization.

CAFNet-DG combines a task-decoupled CAFNet-D branch with the original CAFNet structural score as a fixed residual ranking signal. The released materials support same-mask warm-start and drug-disjoint analyses, a formal scaffold-disjoint evaluation, ordinal-frequency diagnostics, popularity controls, and independent validation against CT-ADE, OnSIDES, FDA safety-signal, and OFFSIDES evidence.

The repository contains code, processed benchmark data, fixed splits, and derived result artifacts only. Manuscript source files, submission PDFs, cover letters, and author correspondence are intentionally excluded.

## Repository Contents

```text
Net.py                         Model definitions.
utils.py                       Training and evaluation utilities.
vector.py                      SMILES-to-graph preprocessing helpers.
cold-scence.py                 Cold-start training/evaluation entry point.
warm-scence.py                 Warm-start training/evaluation entry point.
analysis_scripts/              Reproducible evaluation and audit scripts.
analysis_outputs/              Curated tables, reports, and protocol metadata.
experiments/                   Frozen experimental variants and launch settings.
data/                          Benchmark matrices and fixed train/test masks.
result_ICS/                    Saved cold-start prediction matrices.
result_baselines_*/            Saved baseline predictions used in the reported comparisons.
docs/github_release/           Upload manifest and result-data release notes.
```

## Installation

The experiments were run with Python 3.9, PyTorch 1.12.1 + CUDA 11.3, PyTorch Geometric 1.7.2, RDKit 2022.9.4, NumPy 1.22.4, Pandas 1.5.1, SciPy 1.10.1, and scikit-learn 1.2.1.

```bash
conda env create -f environment.yml
conda activate cafnet-dg
```

### Windows checkout

Clone the repository into a short local path (for example, `D:\\cfdg`). The release uses portable short names for result directories; enabling `git config --global core.longpaths true` is also recommended on Windows.

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

`mask_mat_750.mat` defines warm-start folds. `blind_mask_mat_750.mat` defines drug-disjoint cold-start folds. `scaffold_mask_mat_750.mat` defines the frozen 10-fold Bemis--Murcko group split used for formal scaffold-disjoint evaluation.

## Reproducing Reported Tables from Saved Results

The authoritative post-audit metrics, provenance hashes, paired tests, and manuscript-ready table sources are in:

```text
analysis_outputs/unified_evaluation_20260811/
```

`POST_LEAKAGE_CORRECTION_COMPARISON_SUMMARY.md` records which earlier comparison artifacts were invalidated and which corrected results may be used. The score-scale audit for the fixed residual fusion is provided in `analysis_outputs/fusion_scale_audit_20260814/`.

The main CAFNet-DG cold-start prediction matrix is:

```text
result_ICS/10cafnet_dg_ensemble06_cafnetd04_cafnet/blind_pred.csv
```

Important baseline result folders used in the reported analyses include:

```text
result_baselines_a3net_rdkit_cold_v1/cold/RF/
result_baselines_a3net_rdkit_cold_v1/cold/XGB/
result_baselines_hstrans_foldlocal_v3_compute_matched/
result_baselines_popularity/cold/GLOBAL_POPULARITY/
```

The HSTrans folder contains the fold-local, compute-matched rerun used in the final comparison. The earlier full-matrix preprocessing result is intentionally excluded because its feature construction did not respect fold boundaries.

## Training

Cold-start training example:

```bash
python cold-scence.py --tenfold --epoch 100 --lr 0.0004
```

Warm-start training example:

```bash
python warm-scence.py --tenfold --epoch 100 --lr 0.0004
```

Exact run settings and provenance are stored with the corresponding result folders and in `analysis_outputs/unified_evaluation_20260811/prediction_provenance_sha256.csv`.

## Independent and External Validation

The scientific-gap resolution package contains:

```text
analysis_outputs/scientific_gap_resolution_20260807/ct_ade_pt_external_validation/
analysis_outputs/scientific_gap_resolution_20260807/ct_ade_frequency_validation/
analysis_outputs/scientific_gap_resolution_20260807/onsides_v311_high_confidence/
analysis_outputs/scientific_gap_resolution_20260807/fda_aems_signal_validation/
analysis_outputs/scientific_gap_resolution_20260807/scaffold_cafnet_dg/
analysis_outputs/scientific_gap_resolution_20260807/ordinal_diagnostics/
analysis_outputs/scientific_gap_resolution_20260807/determinism_audit/
analysis_outputs/scientific_gap_resolution_20260807/rare_e2e_screen/
```

These resources support controlled ADR prioritization analyses. They do not establish patient-specific incidence or causal drug--event effects. CT-ADE external ordinal-frequency calibration is reported as a negative result, and the OnSIDES/FDA positive sets are too small to support a claim of uniform model superiority.

The predeclared 10-fold rare-aware/trainable-fusion screen is also retained as a negative result. Group-balanced ranking and global, drug-conditioned, and drug-by-prevalence-stratum gates did not pass the frozen rare-ADR and aggregate-ranking promotion criteria. The reported CAFNet-DG therefore remains the fixed `0.6 * CAFNet-D + 0.4 * CAFNet` residual score fusion.

Raw external source archives are not redistributed. Source URLs, checksums, exact mapping audits, derived non-identifying pairs, and summary statistics are provided where licensing and size permit.

## Reproducibility Boundary

The release fixes data splits, seeds, scripts, and environment versions. The legacy CUDA `torch_scatter` kernel used by the GAT stack is not bitwise deterministic. The determinism audit includes three same-seed repeats and reports numerical variation; reproducibility claims are statistical rather than bitwise.

## Large Files and Git LFS

Large benchmark files and large result tables should be tracked with Git LFS or uploaded as GitHub Release assets. See:

```text
docs/github_release/UPLOAD_MANIFEST.md
docs/github_release/RESULT_DATA_MANIFEST.md
```

Do not upload local environment caches, old manuscript snapshots, or intermediate checkpoints unless they are explicitly needed for a release.

## Citation

Citation metadata will be added after publication.
