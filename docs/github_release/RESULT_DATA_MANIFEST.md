# Result Data Manifest

## Primary prediction artifacts

| Model or diagnostic | Public path |
|---|---|
| CAFNet cold-start | `result_ICS/10ICS_CAFNet_*/blind_pred.csv` |
| A3Net cold-start | `result_ICS/10ICS_A3_Net_*/blind_pred.csv` |
| CAFNet-D cold-start | `result_ICS/10cd3e100f10_CAFNetDecoupled_*/blind_pred.csv` |
| CAFNet-DG fixed residual fusion | `result_ICS/10cafnet_dg_ensemble06_cafnetd04_cafnet/blind_pred.csv` |
| RF/XGBoost baselines | `result_baselines_a3net_rdkit_cold_v1/` |
| HSTrans baseline | `result_baselines_hstrans_same_masks_epoch100/` |
| Global-popularity diagnostic | `result_baselines_popularity/` |

## Current derived result package

| Scientific question | Public path |
|---|---|
| Cold/warm ordinal diagnostics and confusion matrices | `analysis_outputs/scientific_gap_resolution_20260807/ordinal_diagnostics/` |
| Formal 10-fold scaffold-disjoint evaluation | `analysis_outputs/scientific_gap_resolution_20260807/scaffold_cafnet_dg/` |
| CT-ADE prioritization validation | `analysis_outputs/scientific_gap_resolution_20260807/ct_ade_pt_external_validation/` |
| CT-ADE ordinal-frequency validation | `analysis_outputs/scientific_gap_resolution_20260807/ct_ade_frequency_validation/` |
| OnSIDES high-confidence validation | `analysis_outputs/scientific_gap_resolution_20260807/onsides_v311_high_confidence/` |
| FDA temporal signal validation | `analysis_outputs/scientific_gap_resolution_20260807/fda_aems_signal_validation/` |
| Same-seed determinism audit | `analysis_outputs/scientific_gap_resolution_20260807/determinism_audit/` |
| Rare-aware/end-to-end screen | `analysis_outputs/scientific_gap_resolution_20260807/rare_e2e_screen/` |

Each result directory contains the summary used for interpretation together with lower-level fold/drug rows when available. External results retain mapping attrition and negative controls; they do not imply causal ADR confirmation or patient-specific incidence.

## Integrity

The reproducibility directory contains code, data, and runtime manifests. Key public artifacts should be hash-checked before release. The known SHA256 for the fixed CAFNet-DG cold prediction matrix is:

```text
FACE33AB24060840FABF5F9A80CBA3E8F3C685F0407793FCD065E5F09E94ED31
```

## Publication-material exclusion

No manuscript source, submission PDF, cover letter, response letter, or author correspondence belongs in this result release.
