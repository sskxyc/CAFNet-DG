$ErrorActionPreference = "Continue"
$py = "C:\ProgramData\Miniconda3\envs\A3\python.exe"
$wd = "D:\CAFNet-master-master"
$logDir = Join-Path $wd "analysis_outputs\decoupled_v3_experiments\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
Set-Location $wd

$prefix = "cd3e100f10"
$log = Join-Path $logDir "$prefix.log"
& $py .\cold-scence.py `
    --model 3 --tenfold --max_folds 10 --epoch 100 `
    --lr 0.0004 --wd 0.001 --lamb 0.03 --knn 5 --dim 200 --eps 0.5 `
    --result_prefix $prefix `
    --rank_score_mix 0.3 --assoc_weight 1.0 --freq_weight 1.0 --rank_weight 0.05 `
    --pop_weight 0.1 --bias_weight 1.0 --list_weight 0.1 `
    --assoc_base_weight 1.0 --assoc_residual_weight 1.0 `
    --seed 42 --cuda_name cuda:0 *> $log
