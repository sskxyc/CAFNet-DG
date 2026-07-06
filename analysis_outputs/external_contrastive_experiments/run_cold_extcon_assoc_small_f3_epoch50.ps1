$ErrorActionPreference = "Continue"
$PSNativeCommandUseErrorActionPreference = $false

$python = "C:\ProgramData\Miniconda3\envs\A3\python.exe"
$root = "D:\CAFNet-master-master"
$logDir = Join-Path $root "analysis_outputs\external_contrastive_experiments"

Set-Location $root

$configs = @(
  @{prefix="extcon_assoc_lam0005_f3e50"; weight="0.005"},
  @{prefix="extcon_assoc_lam001_f3e50"; weight="0.01"},
  @{prefix="extcon_assoc_lam002_f3e50"; weight="0.02"}
)

foreach ($cfg in $configs) {
  & $python cold-scence.py `
    --model 3 `
    --epoch 50 `
    --max_folds 3 `
    --tenfold `
    --result_prefix $cfg.prefix `
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
    --external_pairs_dir data_external\external_pairs `
    --external_weight $cfg.weight `
    --external_samples_per_drug 16 `
    --external_target assoc `
    > (Join-Path $logDir ($cfg.prefix + "_stdout.log")) `
    2> (Join-Path $logDir ($cfg.prefix + "_stderr.log"))
}
