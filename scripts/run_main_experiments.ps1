[CmdletBinding()]
param(
    [int[]]$Seeds = @(42),
    [ValidateSet("Cold", "Warm", "Both")]
    [string]$Scenario = "Both",
    [switch]$RunCAFNet,
    [switch]$Overwrite,
    [ValidateRange(0, 9)]
    [int]$WarmStartFold = 0,
    [int]$Epochs = 100,
    [string]$PythonExecutable = "python"
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

if (-not (Test-Path -LiteralPath $PythonExecutable)) {
    throw "A3 Python was not found: $PythonExecutable"
}

$logDir = Join-Path $repoRoot "logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $logDir "cafnetdg_main_$timestamp.log"

function Invoke-LoggedPython {
    param(
        [string]$Label,
        [string[]]$Arguments,
        [string]$ExpectedArtifact = ""
    )

    $commandLine = "$PythonExecutable $($Arguments -join ' ')"
    "`n== $Label ==`n$commandLine" | Tee-Object -FilePath $logFile -Append
    $startedAt = Get-Date
    $oldErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $PythonExecutable @Arguments 2>&1 | Tee-Object -FilePath $logFile -Append
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $oldErrorAction
    if ($exitCode -ne 0) {
        if ($ExpectedArtifact -and (Test-Path -LiteralPath $ExpectedArtifact)) {
            $artifact = Get-Item -LiteralPath $ExpectedArtifact
            if ($artifact.LastWriteTime -ge $startedAt.AddSeconds(-2)) {
                $message = "${Label}: Python exited with code $exitCode during fold cleanup, but the required artifact was written; continuing."
                $message | Tee-Object -FilePath $logFile -Append
                return
            }
        }
        throw "$Label failed with exit code $exitCode. See $logFile"
    }
}

function Add-OverwriteFlag {
    param([string[]]$Arguments, [switch]$Resume)
    if ($Overwrite -and -not $Resume) {
        return @($Arguments) + "--overwrite"
    }
    return @($Arguments)
}

function New-CAFNetDArguments {
    param(
        [string]$EntryScript,
        [int]$Model,
        [int]$Seed,
        [int]$Knn,
        [string]$Prefix,
        [int]$StartFold = -1
    )

    $arguments = @(
        $EntryScript,
        "--model", $Model.ToString(),
        "--tenfold",
        "--seed", $Seed.ToString(),
        "--epoch", $Epochs.ToString(),
        "--lr", "0.0004",
        "--lamb", "0.03",
        "--eps", "0.5",
        "--train_batch", "10",
        "--knn", $Knn.ToString(),
        "--rank_score_mix", "0.3",
        "--assoc_weight", "1.0",
        "--freq_weight", "1.0",
        "--rank_weight", "0.05",
        "--pop_weight", "0.1",
        "--bias_weight", "1.0",
        "--list_weight", "0.1",
        "--assoc_base_weight", "1.0",
        "--assoc_residual_weight", "1.0",
        "--short_result_name",
        "--result_prefix", $Prefix
    )
    if ($EntryScript -eq "warm-scence.py") {
        $arguments += @("--listnet_target", "binary", "--save_full_pred")
        $effectiveStartFold = if ($StartFold -ge 0) { $StartFold } else { $WarmStartFold }
        if ($effectiveStartFold -gt 0) {
            $arguments += @("--start_fold", $effectiveStartFold.ToString())
        }
        if ($StartFold -ge 0) {
            $arguments += @("--max_folds", ($StartFold + 1).ToString())
        }
    }
    return Add-OverwriteFlag $arguments -Resume:($EntryScript -eq "warm-scence.py" -and $effectiveStartFold -gt 0)
}

function New-CAFNetArguments {
    param([string]$EntryScript, [int]$Seed, [int]$Knn, [string]$Prefix)

    $arguments = @(
        $EntryScript,
        "--model", "0",
        "--tenfold",
        "--seed", $Seed.ToString(),
        "--epoch", $Epochs.ToString(),
        "--lr", "0.0004",
        "--lamb", "0.03",
        "--eps", "0.5",
        "--train_batch", "10",
        "--knn", $Knn.ToString(),
        "--short_result_name",
        "--result_prefix", $Prefix
    )
    if ($EntryScript -eq "warm-scence.py") {
        $arguments += "--save_full_pred"
    }
    return Add-OverwriteFlag $arguments
}

@"
CAFNet-DG CAFNet-DG main experiment run
timestamp: $timestamp
seeds: $($Seeds -join ', ')
scenario: $Scenario
run_cafnet_component: $RunCAFNet
epochs: $Epochs
warm_start_fold: $WarmStartFold
python: $PythonExecutable
CAFNet-D parameters: eta=0.3, pop=0.1, assoc=1.0, freq=1.0, rank=0.05, list=0.1
"@ | Set-Content -LiteralPath $logFile -Encoding utf8

foreach ($seed in $Seeds) {
    if ($Scenario -in @("Cold", "Both")) {
        if ($RunCAFNet) {
            $cafnetCold = New-CAFNetArguments "cold-scence.py" $seed 5 "cafnetdg_cafnet_cold_seed_$seed"
            Invoke-LoggedPython "CAFNet cold seed $seed" $cafnetCold
        }
        $cafnetDCold = New-CAFNetDArguments "cold-scence.py" 3 $seed 5 "cafnetdg_cafnet_d_cold_seed_$seed"
        Invoke-LoggedPython "CAFNet-D cold seed $seed" $cafnetDCold
    }

    if ($Scenario -in @("Warm", "Both")) {
        if ($RunCAFNet) {
            $cafnetWarm = New-CAFNetArguments "warm-scence.py" $seed 10 "cafnetdg_cafnet_warm_seed_$seed"
            Invoke-LoggedPython "CAFNet warm seed $seed" $cafnetWarm
        }
        for ($fold = $WarmStartFold; $fold -lt 10; $fold++) {
            $cafnetDWarm = New-CAFNetDArguments "warm-scence.py" 4 $seed 10 "cafnetdg_cafnet_d_warm_seed_$seed" $fold
            $artifact = Join-Path $repoRoot "result_WS\cafnetdg_cafnet_d_warm_seed_$seed\full_predictions\full_pred_fold$fold.csv"
            Invoke-LoggedPython "CAFNet-D warm seed $seed fold $fold" $cafnetDWarm $artifact
        }
    }
}

"`n== CAFNet-DG main experiments completed ==`nlog: $logFile" | Tee-Object -FilePath $logFile -Append
