$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$python = "C:\ProgramData\Miniconda3\envs\A3\python.exe"
$root = "D:\CAFNet-master-master"
$logDir = Join-Path $root "analysis_outputs\debias_experiments"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$configs = @(
    @{ prefix = "cd3deb_wg1_rb15_pop002_nores_f3e50"; gamma = "1.0"; boost = "1.5"; pop = "0.02"; arw = "0.0" },
    @{ prefix = "cd3deb_wg1_rb20_pop002_nores_f3e50"; gamma = "1.0"; boost = "2.0"; pop = "0.02"; arw = "0.0" }
)

foreach ($c in $configs) {
    $stdout = Join-Path $logDir ($c.prefix + "_stdout.log")
    $stderr = Join-Path $logDir ($c.prefix + "_stderr.log")
    "Running $($c.prefix)" | Tee-Object -FilePath $stdout
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $python "$root\cold-scence.py" `
        --model 3 --tenfold --max_folds 3 --epoch 50 `
        --lr 0.0004 --wd 0.001 --lamb 0.03 --knn 5 --dim 200 --eps 0.5 `
        --result_prefix $c.prefix --short_result_name `
        --rank_score_mix 0.3 --assoc_weight 1.0 --freq_weight 1.0 --rank_weight 0.05 `
        --pop_weight $c.pop --bias_weight 1.0 --list_weight 0.1 `
        --assoc_base_weight 1.0 --assoc_residual_weight $c.arw `
        --prevalence_debias --debias_gamma $c.gamma --rare_pos_boost $c.boost `
        --fusion_mode gate --gate_mode new --fusion_alpha 0.5 --gat_dropout 0.0 `
        > $stdout 2> $stderr
    $ErrorActionPreference = $oldErrorActionPreference
    if ($LASTEXITCODE -ne 0) {
        throw "Run failed: $($c.prefix). See $stdout and $stderr"
    }
}
