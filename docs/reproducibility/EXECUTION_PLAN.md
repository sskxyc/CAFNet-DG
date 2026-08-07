# CAFNet-DG Scientific-Gap Resolution Plan

Execution date: 2026-08-07 onward.

## Invariants

1. The submitted CAFNet-DG artifacts and root training scripts are frozen.
2. New runs use unique prefixes and must refuse to overwrite existing output.
3. Side-effect prevalence and co-occurrence statistics are computed from the
   current training fold only.
4. Test folds are not used to select losses, fusion gates, or coefficients.
5. Primary comparisons use paired 10-fold tests. Two-sided Wilcoxon signed-rank
   tests and family-wise Holm correction remain the primary convention.
6. A new model is promoted only if it improves the targeted weakness without a
   material loss on the established overall metrics.

## Work packages and gates

### WP1: ordinal diagnostics

- Export fold-specific CAFNet-D frequency matrices for warm and cold start.
- Exclude label 0 as unknown.
- Clip continuous predictions to [1, 5] only for ordinal classification
  diagnostics; retain unclipped predictions for RMSE, MAE, and Spearman.
- Report quadratic weighted kappa, within-one-class accuracy, per-class recall,
  and row-normalized confusion matrices.

### WP2: scaffold-disjoint evaluation

- Reuse the fixed scaffold masks and the verified CAFNet scaffold predictions.
- Train CAFNet-D for 10 folds with the final 100-epoch configuration.
- Construct CAFNet-DG with fixed rho=0.6, without scaffold-test tuning.
- Verify zero Bemis-Murcko scaffold overlap and report train-test molecular
  similarity diagnostics.

### WP3: determinism audit

- Test strict deterministic mode and identify unsupported operations.
- Repeat one fixed fold three times in the same environment.
- Separate same-seed numerical nondeterminism from different-seed uncertainty.
- Do not replace a nondeterministic GAT operator while reusing historical model
  claims; any such replacement is a new model variant.

### WP4: rare-aware model screen

- Compare the frozen final CAFNet-D against prevalence-stratified sampling and a
  low-capacity head/tail dual-expert variant.
- Run every predeclared low-capacity variant on all 10 outer folds and retain all
  outcomes, avoiding selection on a subset of outer test folds. A variant is
  considered promotable only if rare AP improves by at least 5% relative to both
  CAFNet and CAFNet-DG, middle AP does not decline, and overall MAP and nDCG
  decline by no more than 1%. The rare-AP gain must also be positive and
  significant against both references in the paired Wilcoxon tests after Holm
  correction.
- Reject variants with NaN training or greater than 90% single-expert routing.

### WP5: end-to-end fusion screen

- Compare global learnable rho, drug-only gate, and drug-plus-prevalence-stratum
  gate. A full pairwise gate is attempted only if a lower-capacity gate passes.
- Evaluate all predeclared low-capacity gates across the same 10 folds rather
  than choosing a gate from outer-test performance. A gate can replace fixed
  fusion only if overall MAP/nDCG and rare/middle AP jointly meet or exceed the
  fixed-fusion reference and routing remains non-degenerate.

### WP6: independent validation

- Use the public CT-ADE-PT test split as a controlled monopharmacy
  clinical-trial endpoint. Restrict mapping to exact canonical structures and
  exact normalized MedDRA preferred terms, aggregate trial groups by drug, and
  remove all SIDER-observed pairs before scoring frozen cold-start OOF outputs.
- Expand non-SIDER OnSIDES mapping before considering a FAERS/AEMS analysis.
- Treat unmapped or unlabelled pairs as unknown, not confirmed negatives.
- For FAERS/AEMS, deduplicate cases, separate primary-suspect drugs, map to
  ingredient and MedDRA PT levels, and use reporting signals rather than claims
  of incidence or causality.
- External labels are never used for model or coefficient selection.

## Promotion rule

The current fixed CAFNet-DG remains the submitted model unless a replacement
passes every predeclared gate. Negative experiments are retained as evidence and
must not be hidden or reframed as successful improvements.
