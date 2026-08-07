# Rare-ADR and end-to-end fusion experiment protocol

## Frozen comparison

- Dataset: 750-drug by 994-side-effect benchmark.
- Split: the same 10 drug-disjoint folds used by the reported CAFNet family.
- Training budget: 100 epochs per fold.
- Seed: 42.
- Optimizer settings: lr 0.0004, weight decay 0.001, and the frozen CAFNet-D loss weights.
- No test-fold tuning or output replacement is allowed.

## Variants

1. **Group-only**: CAFNet-D with an additional prevalence-group-balanced ranking term; no learned score gate.
2. **Global gate**: one jointly optimized fusion coefficient shared by all pairs.
3. **Drug gate**: a jointly optimized gate conditioned on the drug embedding.
4. **Drug-stratum gate**: a jointly optimized gate conditioned on the drug embedding and the rare/middle/frequent prevalence stratum.

All learned gates are initialized around the fixed CAFNet-DG coefficient (0.6 for CAFNet-D) and regularized toward that prior.

## Predeclared promotion gates

A variant can replace fixed CAFNet-DG only if all conditions hold:

- rare AP is at least 5% higher than both CAFNet and fixed CAFNet-DG;
- rare AP is positive and Holm-significant versus both references;
- middle AP does not decrease versus either reference;
- macro AP and nDCG@10 decrease by no more than 1% versus fixed CAFNet-DG;
- learned routing does not place more than 90% of predictions at a single-expert extreme.

Failure is retained as a negative result. It does not justify post-hoc threshold relaxation or additional test-guided tuning.
