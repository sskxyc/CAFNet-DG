param(
    [string]$Destination = "D:\CAFNet-DG-github-release-staging",
    [switch]$IncludeLargeArtifacts
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Dest = $Destination

function Copy-ItemSafe {
    param(
        [string]$RelativePath
    )
    $src = Join-Path $Root $RelativePath
    if (-not (Test-Path $src)) {
        Write-Warning "Missing: $RelativePath"
        return
    }
    $dst = Join-Path $Dest $RelativePath
    $parent = Split-Path $dst -Parent
    New-Item -ItemType Directory -Force $parent | Out-Null
    if ((Get-Item $src).PSIsContainer) {
        Copy-Item -Path $src -Destination $parent -Recurse -Force
    } else {
        Copy-Item -Path $src -Destination $dst -Force
    }
}

if (Test-Path $Dest) {
    Write-Host "Destination already exists: $Dest"
    Write-Host "Remove it manually if you want a clean rebuild."
} else {
    New-Item -ItemType Directory -Force $Dest | Out-Null
}

$core = @(
    "README.md",
    ".gitignore",
    ".gitattributes",
    "environment.yml",
    "Net.py",
    "utils.py",
    "vector.py",
    "cold-scence.py",
    "warm-scence.py",
    "independent_test_onsides.py",
    "analysis_scripts",
    "docs\github_release",
    "data\raw_frequency_750.mat",
    "data\frequency_data.txt",
    "data\drug_SMILES_750.csv",
    "data\drug_SMILES_750.txt",
    "data\side_effect_label_750.mat",
    "data\mask_mat_750.mat",
    "data\blind_mask_mat_750.mat",
    "data\scaffold_mask_mat_750.mat",
    "analysis_outputs\statistics_jbhi",
    "analysis_outputs\hot_side_effect_bias_cafnet_dg",
    "analysis_outputs\cafnet_dg_per_drug",
    "analysis_outputs\cafnet_dg_completion_20260701",
    "analysis_outputs\cafnet_dg_external_validation",
    "result_ICS\10ICS_CAFNet_knn=5_wd=0.001_epoch=100_lamb=0.03_lr0.0004_dim=200_eps=0.5_DF=False_PCA=False_not-FC=False_cosine",
    "result_ICS\10ICS_A3_Net_knn=5_wd=0.001_epoch=100_lamb=0.03_lr0.0004_dim=200_eps=0.5_DF=False_PCA=False_not-FC=False_cosine",
    "result_ICS\10cd3e100f10_CAFNetDecoupled_knn=5_wd=0.001_epoch=100_lamb=0.03_lr0.0004_dim=200_eps=0.5_DF=False_PCA=False_not-FC=False_cross=True_fusion=gate_gate=new_fa=0.5_gatdrop=0.0_mix=0.3_aw=1.0_fw=1.0_rw=0.05_popw=0.1_biasw=1.0_listw=0.1_abw=1.0_arw=1.0_cosine",
    "result_ICS\10cafnet_dg_ensemble06_cafnetd04_cafnet",
    "result_baselines_a3net_rdkit_cold_v1\cold\RF",
    "result_baselines_a3net_rdkit_cold_v1\cold\XGB",
    "result_baselines_hstrans_same_masks_epoch100\cold\HSTrans",
    "result_baselines_popularity\cold\GLOBAL_POPULARITY"
)

foreach ($item in $core) {
    Copy-ItemSafe $item
}

if (-not $IncludeLargeArtifacts) {
    $largeInDest = @(
        "analysis_outputs\cafnet_dg_external_validation\offsides_matched_control_scored_pairs.csv",
        "analysis_outputs\cafnet_dg_external_validation\offsides_matched_control_scored_pairs_with_prevalence_meta.csv"
    )
    foreach ($item in $largeInDest) {
        $p = Join-Path $Dest $item
        if (Test-Path $p) {
            Remove-Item -LiteralPath $p -Force
            Write-Host "Removed large artifact from staging: $item"
        }
    }
} else {
    Copy-ItemSafe "data_external\external_pairs\offsides_contrastive_pairs_all_folds.csv"
}

Write-Host "Staging directory prepared: $Dest"
Write-Host "Review it before pushing to GitHub."
