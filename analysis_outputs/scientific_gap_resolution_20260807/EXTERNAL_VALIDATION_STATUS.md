# Independent External-Validation Status

## CT-ADE-PT controlled monopharmacy clinical-trial analysis

- Source: the public CT-ADE-PT test split derived from ClinicalTrials.gov
  monopharmacy results (Scientific Data, 2025).
- Exact canonical-structure mapping covers 31 benchmark drugs and 254 trial
  groups; exact normalized MedDRA-PT mapping covers 982/994 side effects.
- Trial-group labels are aggregated by drug. After removing every pair present
  in the SIDER-derived matrix, 1,336 positives remain across 29 drugs. The
  evaluation contains 28,874 non-SIDER pairs.
- The CT-ADE positive label means there is 95% confidence that at least 1% of
  the trial group experienced the event. This is an independent clinical-trial
  prioritization endpoint, not the original 1--5 ordinal-frequency target.

CAFNet-DG obtains pooled AUROC 0.754 and AUPR 0.199. Drug-level paired tests
show significant improvements over the original CAFNet for AUROC and AUPR
after Holm correction, but not over CAFNet-D. Against a 1:5 same-drug
nearest-prevalence control set, CAFNet-DG obtains pooled AUROC 0.679 and AUPR
0.321 and significantly exceeds global popularity in drug-level paired tests.

The independent rare-prevalence subset contains 174 positives. CAFNet-DG is
significantly better than global popularity there, but it is not significantly
better than CAFNet or CAFNet-D. The clinical-trial analysis therefore closes
the absence of independent clinical labels while confirming that rare-ADR
retrieval remains unresolved relative to the learned references.

### External ordinal-frequency diagnostic

The CT-ADE frequency file yields 2,781 non-SIDER drug--ADR pairs with finite,
nonzero at-risk-weighted trial proportions over 30 mapped drugs. Converting
proportions to the standard five frequency bands gives a deliberately strict
external calibration test. The CAFNet-D frequency head obtains QWK 0.288,
Spearman 0.347, within-one accuracy 0.755, RMSE 1.391, and MAE 1.126. A
training-fold side-effect-mean baseline has similar rank correlation but
significantly better within-one accuracy, RMSE, and MAE (0.894, 0.901, and
0.727). Thus, the model retains moderate external ordinal association but its
numeric scale is not well calibrated across resources. This result must remain
visible and rules out claims of cross-resource calibrated incidence
prediction.

## OnSIDES v3.1.1 high-confidence analysis

- Official release: OnSIDES v3.1.1, 2026-04-22.
- Downloaded release asset: `onsides-v3.1.1.zip`, MD5
  `ab934e47108f2f53b3d7107405fa93d3`.
- The authors define `high_confidence.csv` as ingredient-effect pairs observed
  across all four label sources: US, UK, EU, and Japan.
- Benchmark mapping is restricted to exact normalized RxNorm ingredient names
  and exact normalized MedDRA preferred terms.
- Pairs already present in the SIDER-derived frequency matrix are excluded.
- The final external set contains 91 positives over 27 drugs and 455
  same-drug, training-fold-prevalence-matched controls.

CAFNet-DG obtains AUROC 0.540 and AUPR 0.552. The drug-bootstrap 95% interval
for AUROC is 0.510--0.578. Its tie-aware positive-control concordance is 0.602.
The result supports above-chance transfer to independent multi-regulatory-label
pairs, but CAFNet-DG is not significantly better than CAFNet or CAFNet-D.
Global popularity remains strong (AUPR 0.564; concordance 0.625), showing that
the external high-confidence subset is also prevalence-dominated.

The rare subgroup has only four positive pairs and cannot support an
inferential rare-ADR claim. In the non-hot-100 subset (28 positives), CAFNet-DG
improves over CAFNet-D but remains below CAFNet.

## FDA AEMS/FAERS regulatory-signal analysis

- Thirteen official quarterly pages from 2023--2026 were cached and parsed.
- From 255 regulatory entries, strict monotherapy/identical-combination drug
  mapping and maximal-specific ADR mapping produced 62 pairs.
- Five semantically invalid single-drug or overly broad mappings were excluded
  before scoring.
- After SIDER exclusion, 53 temporal external signals over 39 drugs remained,
  with 265 prevalence-matched controls.

CAFNet-DG obtains AUROC 0.547, AUPR 0.538, and tie-aware concordance 0.555.
None of the paired per-drug comparisons against CAFNet-D, CAFNet, or global
popularity is significant after Holm correction. This is a negative-to-modest
temporal regulatory robustness result, not evidence of causality.

## Interpretation boundary

CAFNet-DG was quantitatively evaluated on a
controlled monopharmacy clinical-trial benchmark, non-SIDER OnSIDES
high-confidence label pairs, and post-2022 FDA regulatory safety signals, with
exact mapping audits, prevalence-matched controls, and drug-level uncertainty.
The stronger CT-ADE discrimination result must be distinguished from the modest
OnSIDES/FDA effects; none of these analyses establishes
patient-specific prediction, calibrated incidence, EHR effectiveness, or
causality. It must also report that the independent CT-ADE ordinal-frequency
diagnostic did not transfer successfully.

Primary data and results:

- `analysis_outputs/scientific_gap_resolution_20260807/onsides_v311_high_confidence/`
- `analysis_outputs/scientific_gap_resolution_20260807/fda_aems_signal_validation/`
- `analysis_outputs/scientific_gap_resolution_20260807/ct_ade_pt_external_validation/`
- `analysis_outputs/scientific_gap_resolution_20260807/ct_ade_frequency_validation/`
- `data_external/onsides_v3.1.1/`
- `data_external/ct_ade_pt_2025/test.csv`
