# Scientific-Gap Completion Audit

All work packages verified: **True**

## frozen_reproducibility_inputs

- Verified: `True`
- Evidence: `docs/reproducibility/code_SHA256.csv`
- Detail: Code hashes and runtime manifest must both exist.

## ordinal_diagnostics

- Verified: `True`
- Evidence: `analysis_outputs/scientific_gap_resolution_20260807/ordinal_diagnostics`
- Detail: cold_10fold=True, warm_10fold=True; QWK/within-one/confusion files required.

## scaffold_disjoint

- Verified: `True`
- Evidence: `analysis_outputs/scientific_gap_resolution_20260807/scaffold_cafnet_dg`
- Detail: Requires 3 models x 10 folds, similarity audit, and paired tests.

## determinism_audit

- Verified: `True`
- Evidence: `analysis_outputs/scientific_gap_resolution_20260807/determinism_audit`
- Detail: strict_supported=False; bitwise_equal=False; max_abs_difference=6.81255841255188

## rare_adr_and_end_to_end_fusion

- Verified: `True`
- Evidence: `analysis_outputs/scientific_gap_resolution_20260807/rare_e2e_screen`
- Detail: all_variants_reported=True; promotable_variants=[]

## independent_external_validation

- Verified: `True`
- Evidence: `analysis_outputs/scientific_gap_resolution_20260807/ct_ade_pt_external_validation/coverage_and_protocol.json; analysis_outputs/scientific_gap_resolution_20260807/ct_ade_frequency_validation/coverage_and_protocol.json; analysis_outputs/scientific_gap_resolution_20260807/onsides_v311_high_confidence/coverage_and_protocol.json; analysis_outputs/scientific_gap_resolution_20260807/fda_aems_signal_validation/coverage_and_protocol.json`
- Detail: CT-ADE positives=1336 over 31 drugs, OnSIDES positives=91, FDA signals=53; CT-ADE frequency pairs=2781; controlled-trial prioritization but failed external ordinal calibration and no EHR/incidence/causal claim.
