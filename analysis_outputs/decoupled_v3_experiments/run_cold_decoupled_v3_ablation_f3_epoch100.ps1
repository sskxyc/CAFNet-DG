$ErrorActionPreference = "Continue"
$py = "C:\ProgramData\Miniconda3\envs\A3\python.exe"
$wd = "D:\CAFNet-master-master"
$logDir = Join-Path $wd "analysis_outputs\decoupled_v3_experiments\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
Set-Location $wd

$configs = @(
    @{prefix = "cd3abl_nores_f3"; pop = "0.1"; bias = "1.0"; arw = "0.0"},
    @{prefix = "cd3abl_nobias_f3"; pop = "0.0"; bias = "0.0"; arw = "1.0"}
)

foreach ($c in $configs) {
    $log = Join-Path $logDir "$($c.prefix).log"
    & $py .\cold-scence.py `
        --model 3 --tenfold --max_folds 3 --epoch 100 `
        --lr 0.0004 --wd 0.001 --lamb 0.03 --knn 5 --dim 200 --eps 0.5 `
        --result_prefix $c.prefix --short_result_name `
        --rank_score_mix 0.3 --assoc_weight 1.0 --freq_weight 1.0 --rank_weight 0.05 `
        --pop_weight $c.pop --bias_weight $c.bias --list_weight 0.1 `
        --assoc_base_weight 1.0 --assoc_residual_weight $c.arw `
        --seed 42 --cuda_name cuda:0 *> $log
}
