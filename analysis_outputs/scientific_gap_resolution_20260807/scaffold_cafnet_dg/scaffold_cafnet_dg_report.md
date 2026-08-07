# CAFNet-DG Scaffold-Disjoint Evaluation

All 10 folds have zero overlap in the frozen split_group field. CAFNet-DG uses fixed rho=0.6 without scaffold-test tuning.

```text
                 AP           global_AUROC           global_AUPR             nDCG@10             rare_AP           middle_AP           frequent_AP           nonhot100_AP
               mean       std         mean       std        mean       std      mean       std      mean       std      mean       std        mean       std         mean       std
model
CAFNet     0.334424  0.013086     0.787622  0.018889    0.246127  0.016073  0.482913  0.031105  0.066847  0.013137  0.098522  0.012295    0.382795  0.016847     0.105286  0.014624
CAFNet-D   0.403818  0.020775     0.809224  0.019659    0.291639  0.019486  0.625507  0.025022  0.059128  0.019443  0.093184  0.014589    0.453036  0.018809     0.103726  0.016458
CAFNet-DG  0.408671  0.020439     0.817075  0.018818    0.298179  0.015642  0.621239  0.026871  0.065840  0.020299  0.103452  0.018865    0.456281  0.019087     0.117050  0.014289
```

The train-test similarity audit uses 2048-bit radius-2 Morgan fingerprints with chirality and is reported separately from the Bemis-Murcko split definition. Exact canonical-structure matches are also audited.
