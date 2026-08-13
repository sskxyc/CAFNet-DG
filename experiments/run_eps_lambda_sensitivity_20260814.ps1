param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$entry = Join-Path $root "experiments\scientific_gap_resolution_20260807_vnext\cold-scence.py"
$logs = Join-Path $root "analysis_outputs\eps_lambda_sensitivity_20260814\logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null

$settings = @(
    @{ Name = "BASE";     Lambda = "0.03"; Epsilon = "0.5" },
    @{ Name = "EPS0";     Lambda = "0.03"; Epsilon = "0.0" },
    @{ Name = "EPS025";   Lambda = "0.03"; Epsilon = "0.25" },
    @{ Name = "EPS075";   Lambda = "0.03"; Epsilon = "0.75" },
    @{ Name = "LAM0";     Lambda = "0.0";  Epsilon = "0.5" },
    @{ Name = "LAM001";   Lambda = "0.01"; Epsilon = "0.5" },
    @{ Name = "LAM01";    Lambda = "0.1";  Epsilon = "0.5" }
)

foreach ($setting in $settings) {
    $prefix = "SENS_$($setting.Name)"
    $result = Join-Path $root "result_ICS\10${prefix}_CAFNetDecoupled"
    $complete = Join-Path $result "blind_freq_pred.csv"
    if ((Test-Path $complete) -and ((Get-Content -LiteralPath $complete).Count -eq 750)) {
        Write-Host "Skipping completed setting $($setting.Name)"
        continue
    }

    $log = Join-Path $logs "$($setting.Name).log"
    $ErrorActionPreference = "Continue"
    & $Python $entry --model 3 --tenfold --epoch 100 --lr 0.0004 --wd 0.001 `
        --lamb $setting.Lambda --knn 5 --dim 200 --eps $setting.Epsilon `
        --rank_score_mix 0.3 --assoc_weight 1 --freq_weight 1 --rank_weight 0.05 `
        --pop_weight 0.1 --bias_weight 1 --list_weight 0.1 `
        --assoc_base_weight 1 --assoc_residual_weight 1 --seed 3 `
        --short_result_name --result_prefix $prefix 2>&1 | Tee-Object -FilePath $log
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    if ($exitCode -ne 0) { throw "Setting $($setting.Name) failed with exit code $exitCode" }
}

& $Python (Join-Path $root "analysis\summarize_eps_lambda_sensitivity_20260814.py")
if ($LASTEXITCODE -ne 0) { throw "Sensitivity summarization failed" }
