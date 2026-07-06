$ErrorActionPreference = "Stop"

$Python = "C:\ProgramData\Miniconda3\envs\A3\python.exe"
$Root = "D:\CAFNet-master-master"
Set-Location $Root

# Warm-start full-matrix rerun for CAFNet.
# The added --save_full_pred flag preserves the historical masked pred_result.csv
# and additionally writes full_predictions/full_pred_fold{0..9}.csv.
& $Python .\warm-scence.py `
  --model 0 `
  --tenfold `
  --save_full_pred `
  --result_prefix "WSFULL_CAFNet" `
  --knn 10 `
  --wd 0.001 `
  --epoch 100 `
  --lamb 0.03 `
  --lr 0.0004 `
  --dim 200 `
  --eps 0.5 `
  --fusion_mode gate `
  --gate_mode new `
  --loss_type focal `
  --max_folds 10

# Warm-start full-matrix rerun for CAFNet-D / CAFNetDecoupled.
& $Python .\warm-scence.py `
  --model 4 `
  --tenfold `
  --save_full_pred `
  --result_prefix "WSFULL_CAFNetD" `
  --knn 10 `
  --wd 0.001 `
  --epoch 100 `
  --lamb 0.03 `
  --lr 0.0004 `
  --dim 200 `
  --eps 0.5 `
  --fusion_mode gate `
  --gate_mode new `
  --loss_type focal `
  --rank_score_mix 0.3 `
  --assoc_weight 1.0 `
  --freq_weight 1.0 `
  --rank_weight 0.05 `
  --pop_weight 0.1 `
  --bias_weight 1.0 `
  --list_weight 0.1 `
  --assoc_base_weight 1.0 `
  --assoc_residual_weight 1.0 `
  --max_folds 10

& $Python .\analysis_outputs\cafnet_dg_completion_20260701\warm_fusion_from_full_predictions.py
