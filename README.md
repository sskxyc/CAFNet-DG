# CAFNet-DG

Official code release for CAFNet-DG, a graph-based framework for drug-side-effect prioritization and frequency estimation.

## Repository scope

This repository contains source code, experiment configurations, fixed split masks, and analysis utilities required to reproduce the computational results. Manuscripts, journal submission packages, cover letters, compiled PDFs, and internal revision documents are intentionally excluded.

## Environment

The experiments were developed with Python, PyTorch, PyTorch Geometric, RDKit, NumPy, SciPy, pandas, and scikit-learn. Activate the project environment before running commands.

    conda activate A3

## Main experiments

The frozen main protocol uses seed 42, 100 epochs, learning rate 0.0004, lambda 0.03, epsilon 0.5, and batch size 10. CAFNet-D uses rank_score_mix=0.3, association/frequency weights of 1.0, ranking weight 0.05, prevalence weight 0.1, bias weight 1.0, and ListNet weight 0.1. The warm-start protocol uses binary ListNet targets.

Run the complete cold- and warm-start workflow:

    powershell -ExecutionPolicy Bypass -File scripts/run_main_experiments.ps1 -PythonExecutable python

Individual configuration-driven runs are available through:

    python scripts/train_cold.py --config configs/cafnet_d_cold_main.json
    python scripts/train_warm.py --config configs/cafnet_d_warm_main.json

Fixed split masks are required. The training programs do not silently regenerate missing masks.

## Staged task-decoupling ablation

    powershell -ExecutionPolicy Bypass -File scripts/run_staged_ablation.ps1 -PythonExecutable python

The staged protocol covers the decoupled prediction head, ranking objectives, association residual, prevalence prior, and zero-weight control settings.

## Inference fusion

CAFNet and CAFNet-D are optimized separately. Their frozen prediction scores are combined only at inference:

    s_DG = rho * s_D + (1 - rho) * s_C, rho = 0.6.

This is a fixed residual score fusion rule, not an end-to-end trainable module. The frequency branch outputs a continuous ordinal-frequency estimate rather than a separate ordinal-classification head.

## Outputs

Training scripts write fold-level metrics and prediction files under result directories. Analysis utilities are located in analysis_scripts/. Generated logs, figures, tables, and submission artifacts should remain outside version control.
