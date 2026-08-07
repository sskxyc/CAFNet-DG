# Portable rare-aware and trainable-fusion commands

Run from the repository root in the `cafnet-dg` environment. Each command uses the same frozen 10 drug-disjoint folds, 100 epochs, and seed 42. Use a fresh output prefix; the runner refuses to overwrite an existing result directory.

Common arguments:

```text
--model 3 --tenfold --max_folds 10 --epoch 100
--lr 0.0004 --wd 0.001 --lamb 0.03 --knn 5 --dim 200 --eps 0.5
--short_result_name --rank_score_mix 0.3
--assoc_weight 1.0 --freq_weight 1.0 --rank_weight 0.05
--pop_weight 0.1 --bias_weight 1.0 --list_weight 0.1
--assoc_base_weight 1.0 --assoc_residual_weight 1.0
--group_rank_weight 0.05 --gate_prior 0.6 --gate_reg_weight 0.01
--seed 42
```

Variant-specific arguments:

```text
Group-only:         --result_prefix R8GROUP  --rank_fusion_mode none
Global gate:        --result_prefix R8GLOBAL --rank_fusion_mode global
Drug gate:          --result_prefix R8DRUG   --rank_fusion_mode drug
Drug-stratum gate:  --result_prefix R8STRAT  --rank_fusion_mode drug_stratum
```

Example:

```bash
python experiments/scientific_gap_resolution_20260807_vnext/cold-scence.py --model 3 --tenfold --max_folds 10 --epoch 100 --lr 0.0004 --wd 0.001 --lamb 0.03 --knn 5 --dim 200 --eps 0.5 --short_result_name --rank_score_mix 0.3 --assoc_weight 1.0 --freq_weight 1.0 --rank_weight 0.05 --pop_weight 0.1 --bias_weight 1.0 --list_weight 0.1 --assoc_base_weight 1.0 --assoc_residual_weight 1.0 --group_rank_weight 0.05 --rank_fusion_mode drug_stratum --gate_prior 0.6 --gate_reg_weight 0.01 --result_prefix R8STRAT --seed 42
```

After all four variants finish, run:

```bash
python analysis_scripts/evaluate_rare_e2e_screen_20260807.py
python analysis_scripts/scientific_gap_completion_audit_20260807.py
```

The evaluator applies the frozen promotion rules in `analysis_outputs/scientific_gap_resolution_20260807/RARE_E2E_EXPERIMENT_PROTOCOL.md`.
