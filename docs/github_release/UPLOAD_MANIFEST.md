# CAFNet-DG Public Repository Manifest

This manifest defines the clean public release. The repository contains code, benchmark data, fixed splits, saved predictions, and derived result tables. It excludes manuscript and submission materials.

## Core code

```text
Net.py
utils.py
vector.py
cold-scence.py
warm-scence.py
independent_test_onsides.py
analysis_scripts/
experiments/scientific_gap_resolution_20260807_vnext/
environment.yml
```

## Benchmark data and fixed splits

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

Binary arrays are tracked through Git LFS according to `.gitattributes`.

## Saved model and baseline results

```text
result_ICS/
result_baselines_a3net_rdkit_cold_v1/
result_baselines_hstrans_foldlocal_v3_compute_matched/
result_baselines_popularity/
```

The principal fixed-fusion cold-start predictions are in:

```text
result_ICS/10cafnet_dg_ensemble06_cafnetd04_cafnet/blind_pred.csv
```

## Curated current analyses

The authoritative comparison package and fixed-fusion scale audit are:

```text
analysis_outputs/unified_evaluation_20260811/
analysis_outputs/fusion_scale_audit_20260814/
```

The following directories retain supporting scientific-gap analyses:

```text
analysis_outputs/scientific_gap_resolution_20260807/ordinal_diagnostics/
analysis_outputs/scientific_gap_resolution_20260807/scaffold_cafnet_dg/
analysis_outputs/scientific_gap_resolution_20260807/ct_ade_pt_external_validation/
analysis_outputs/scientific_gap_resolution_20260807/ct_ade_frequency_validation/
analysis_outputs/scientific_gap_resolution_20260807/onsides_v311_high_confidence/
analysis_outputs/scientific_gap_resolution_20260807/fda_aems_signal_validation/
analysis_outputs/scientific_gap_resolution_20260807/determinism_audit/
analysis_outputs/scientific_gap_resolution_20260807/rare_e2e_screen/
```

These directories include exact mapping audits, per-drug/fold results, paired tests, uncertainty summaries, protocol metadata, and negative findings. Raw external archives are not redistributed; source locations and checksums are recorded where permitted.

## Reproducibility metadata

```text
docs/reproducibility/
analysis_outputs/scientific_gap_resolution_20260807/completion_audit.json
```

The legacy CUDA scatter-add path is not bitwise deterministic. The release fixes splits, seeds, code snapshots, and environment versions and reports same-seed numerical variation.

## Excluded materials

The following must never be committed:

```text
submission_flat/
manuscript/
main.tex
supplement*.tex
manuscript*.pdf
submission*.pdf
*.docx
cover_letter*
author correspondence
local environments and caches
trained checkpoints not required for inference
```

Before each push, verify the candidate file list and scan it for local absolute paths and publication files.
