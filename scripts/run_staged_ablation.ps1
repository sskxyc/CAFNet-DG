[CmdletBinding()]
param(
    [int]$Seed = 42,
    [ValidateSet("Cold", "Warm", "Both")]
    [string]$Scenario = "Cold",
    [string[]]$Variants = @("decoupled_head", "lambda0_zero"),
    [switch]$Overwrite,
    [int]$Epochs = 100,
    [string]$PythonExecutable = "python"
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$allowedVariants = @(
    "decoupled_head",
    "decoupled_ranking",
    "no_assoc_residual",
    "no_prior",
    "lambda0_zero"
)
foreach ($variant in $Variants) {
    if ($variant -notin $allowedVariants) {
        throw "Unknown variant '$variant'. Allowed: $($allowedVariants -join ', ')"
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
if (-not (Test-Path -LiteralPath $PythonExecutable)) {
    throw "A3 Python was not found: $PythonExecutable"
}

$logDir = Join-Path $repoRoot "logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $logDir "cafnetdg_ablations_$timestamp.log"

function Invoke-LoggedPython {
    param([string]$Label, [string[]]$Arguments)
    "`n== $Label ==`n$PythonExecutable $($Arguments -join ' ')" | Tee-Object -FilePath $logFile -Append
    $oldErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $PythonExecutable @Arguments 2>&1 | Tee-Object -FilePath $logFile -Append
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $oldErrorAction
    if ($exitCode -ne 0) {
        throw "$Label failed with exit code $exitCode. See $logFile"
    }
}

function New-VariantArguments {
    param([string]$EntryScript, [int]$Model, [int]$Knn, [string]$Variant)

    $rankWeight = "0.05"
    $popWeight = "0.1"
    $biasWeight = "1.0"
    $listWeight = "0.1"
    $assocResidualWeight = "1.0"
    $lambda0 = "0.03"

    switch ($Variant) {
        "decoupled_head" {
            $rankWeight = "0.0"; $popWeight = "0.0"; $biasWeight = "0.0"; $listWeight = "0.0"
        }
        "decoupled_ranking" {
            $popWeight = "0.0"; $biasWeight = "0.0"
        }
        "no_assoc_residual" {
            $assocResidualWeight = "0.0"
        }
        "no_prior" {
            $popWeight = "0.0"; $biasWeight = "0.0"
        }
        "lambda0_zero" {
            $lambda0 = "0.0"
        }
    }

    $scenarioTag = if ($EntryScript -eq "cold-scence.py") { "cold" } else { "warm" }
    $arguments = @(
        $EntryScript,
        "--model", $Model.ToString(),
        "--tenfold",
        "--seed", $Seed.ToString(),
        "--epoch", $Epochs.ToString(),
        "--lr", "0.0004",
        "--lamb", $lambda0,
        "--eps", "0.5",
        "--train_batch", "10",
        "--knn", $Knn.ToString(),
        "--rank_score_mix", "0.3",
        "--assoc_weight", "1.0",
        "--freq_weight", "1.0",
        "--rank_weight", $rankWeight,
        "--pop_weight", $popWeight,
        "--bias_weight", $biasWeight,
        "--list_weight", $listWeight,
        "--assoc_base_weight", "1.0",
        "--assoc_residual_weight", $assocResidualWeight,
        "--short_result_name",
        "--result_prefix", "cafnetdg_${variant}_${scenarioTag}_seed_$Seed"
    )
    if ($EntryScript -eq "warm-scence.py") {
        $arguments += @("--listnet_target", "binary", "--save_full_pred")
    }
    if ($Overwrite) {
        $arguments += "--overwrite"
    }
    return $arguments
}

@"
CAFNet-DG CAFNet-D staged ablation run
timestamp: $timestamp
seed: $Seed
scenario: $Scenario
variants: $($Variants -join ', ')
epochs: $Epochs
python: $PythonExecutable
"@ | Set-Content -LiteralPath $logFile -Encoding utf8

foreach ($variant in $Variants) {
    if ($Scenario -in @("Cold", "Both")) {
        $coldArgs = New-VariantArguments "cold-scence.py" 3 5 $variant
        Invoke-LoggedPython "$variant cold seed $Seed" $coldArgs
    }
    if ($Scenario -in @("Warm", "Both")) {
        $warmArgs = New-VariantArguments "warm-scence.py" 4 10 $variant
        Invoke-LoggedPython "$variant warm seed $Seed" $warmArgs
    }
}

"`n== CAFNet-DG ablations completed ==`nlog: $logFile" | Tee-Object -FilePath $logFile -Append
