$ErrorActionPreference = "Continue"
$PSNativeCommandUseErrorActionPreference = $false

$python = "C:\ProgramData\Miniconda3\envs\A3\python.exe"
$root = "D:\CAFNet-master-master"
$logDir = Join-Path $root "analysis_outputs\external_evidence_experiments"

Set-Location $root

& $python cold-scence.py `
  --model 3 `
  --epoch 50 `
  --max_folds 3 `
  --tenfold `
  --result_prefix cd3ext_atctarget_f3e50 `
  --short_result_name `
  --lr 0.0004 `
  --wd 0.001 `
  --lamb 0.03 `
  --knn 5 `
  --dim 200 `
  --eps 0.5 `
  --train_batch 10 `
  --rank_score_mix 0.3 `
  --assoc_weight 1.0 `
  --freq_weight 1.0 `
  --rank_weight 0.05 `
  --pop_weight 0.1 `
  --bias_weight 1.0 `
  --list_weight 0.1 `
  --assoc_base_weight 1.0 `
  --assoc_residual_weight 1.0 `
  --drug_feature_file data_external\chembl_atc_target_features.npy `
  --drug_feature_tag atctarget `
  --evidence_dropout 0.1 `
  > (Join-Path $logDir "cd3ext_atctarget_f3e50_stdout.log") `
  2> (Join-Path $logDir "cd3ext_atctarget_f3e50_stderr.log")
