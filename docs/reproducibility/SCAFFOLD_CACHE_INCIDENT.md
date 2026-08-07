# Scaffold cache incident and corrective action

## Detection

The first CAFNet-D scaffold run completed ten training folds, but the evaluation
guard rejected it because its exported `blind_raw.csv` did not match the
previously verified CAFNet scaffold labels.

## Root cause

The command correctly supplied `data/scaffold_mask_mat_750.mat`, but the PyG
`InMemoryDataset` names were identical to those used by the ordinary
drug-disjoint cold-start run. Existing processed datasets were therefore loaded
from `data_ICS/processed` instead of rebuilding examples for the scaffold mask.
The exported row order matched `data/blind_mask_mat_750.mat`, confirming cache
reuse rather than a scaffold-label mismatch.

## Disposition

The invalid output was retained for audit at
`result_ICS/10SCFD8COLD_CAFNetDecoupled_invalid_blind_cache_20260808` and is
excluded from every analysis. It was not overwritten or used for any result.

## Corrective action

The experimental runner now passes `--dataset_cache_tag SCFD8`, which gives the
scaffold run a separate PyG cache namespace. The alignment guard remains in
place and the full ten-fold run is repeated before any scaffold conclusion is
reported.
