from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "analysis_outputs" / "cafnet_dg_completion_20260701" / "warm_full_fusion"
OUT_DIR = IN_DIR

RANKING = ["MAP", "AUROC", "AUPR", "nDCG@10", "P@1", "P@15", "R@15"]
REGRESSION = ["Spearman", "RMSE", "MAE"]


def holm(p_values: list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def wilcoxon(diff: np.ndarray) -> float:
    if np.allclose(diff, 0):
        return 1.0
    try:
        return float(stats.wilcoxon(diff, zero_method="wilcox").pvalue)
    except ValueError:
        return 1.0


def fmt(mean: float, std: float, bold: bool = False) -> str:
    text = f"${mean:.4f} \\pm {std:.4f}$"
    return f"\\textbf{{{text}}}" if bold else text


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = RANKING + REGRESSION + ["nonhot100_AP", "rare_AP", "middle_AP"]
    rows = []
    for model, sub in df.groupby("model"):
        row = {"model": model}
        for metric in metrics:
            row[f"{metric}_mean"] = float(sub[metric].mean())
            row[f"{metric}_std"] = float(sub[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def make_paired(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    target = df[df.model.eq("CAFNet-DG")].sort_values("fold")
    lower_better = {"RMSE", "MAE"}
    paired_metrics = RANKING + ["nonhot100_AP", "rare_AP", "middle_AP"]
    for baseline in ["CAFNet-D", "CAFNet"]:
        base = df[df.model.eq(baseline)].sort_values("fold")
        for metric in paired_metrics:
            a = target[metric].to_numpy(float)
            b = base[metric].to_numpy(float)
            diff = b - a if metric in lower_better else a - b
            rows.append(
                {
                    "comparison": f"CAFNet-DG vs {baseline}",
                    "metric": metric,
                    "target_mean": float(a.mean()),
                    "baseline_mean": float(b.mean()),
                    "improvement_mean": float(diff.mean()),
                    "wilcoxon_p": wilcoxon(diff),
                }
            )
    paired = pd.DataFrame(rows)
    parts = []
    for _, sub in paired.groupby("comparison", sort=False):
        sub = sub.copy()
        sub["wilcoxon_p_holm"] = holm(sub["wilcoxon_p"].tolist())
        sub["holm_sig"] = sub["wilcoxon_p_holm"] < 0.05
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def write_ranking_tex(summary: pd.DataFrame) -> None:
    models = ["CAFNet", "CAFNet-D", "CAFNet-DG"]
    max_by_metric = {m: summary[f"{m}_mean"].max() for m in RANKING}
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Warm-start full-matrix prioritization results for CAFNet-DG. Values are mean $\\pm$ standard deviation over 10 folds.}",
        "\\label{tab:warm_full_cafnet_dg_ranking}",
        "\\scriptsize",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Model & MAP & AUROC & AUPR & nDCG@10 & P@1 & P@15 & R@15 \\\\",
        "\\midrule",
    ]
    for model in models:
        row = summary[summary.model.eq(model)].iloc[0]
        vals = []
        for metric in RANKING:
            vals.append(fmt(row[f"{metric}_mean"], row[f"{metric}_std"], abs(row[f"{metric}_mean"] - max_by_metric[metric]) < 1e-12))
        lines.append(f"{model} & " + " & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    (OUT_DIR / "warm_full_cafnet_dg_ranking_table.tex").write_text("\n".join(lines), encoding="utf-8")


def write_rho_tex(rho: pd.DataFrame) -> None:
    metrics = ["MAP", "AUROC", "AUPR", "nDCG@10", "nonhot100_AP", "rare_AP", "middle_AP"]
    max_by_metric = {m: rho[f"{m}_mean"].max() for m in metrics}
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Warm-start sensitivity of CAFNet-DG fixed residual fusion. Values are 10-fold means.}",
        "\\label{tab:warm_cafnet_dg_rho_sensitivity}",
        "\\scriptsize",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "$\\rho$ & MAP & AUROC & AUPR & nDCG@10 & Non-hot100 AP & Rare AP & Middle AP \\\\",
        "\\midrule",
    ]
    for _, row in rho.iterrows():
        vals = [f"{row['rho']:.1f}"]
        for metric in metrics:
            text = f"{row[f'{metric}_mean']:.3f}"
            if abs(row[f"{metric}_mean"] - max_by_metric[metric]) < 1e-12:
                text = f"\\textbf{{{text}}}"
            vals.append(text)
        lines.append(" & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    (OUT_DIR / "warm_rho_sensitivity_table.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = pd.read_csv(IN_DIR / "warm_full_cafnet_dg_by_fold.csv")
    summary = make_summary(df)
    summary.to_csv(OUT_DIR / "warm_full_cafnet_dg_summary_flat.csv", index=False)
    paired = make_paired(df)
    paired.to_csv(OUT_DIR / "warm_full_cafnet_dg_ranking_paired_tests.csv", index=False)
    write_ranking_tex(summary)
    rho = pd.read_csv(IN_DIR / "warm_rho_sensitivity_summary.csv")
    write_rho_tex(rho)
    print(summary.to_string(index=False))
    print()
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
