from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_outputs" / "decoupled_v3_experiments"

MODELS = {
    "CAFNet": ROOT / "result_ICS" / "10ICS_CAFNet_knn=5_wd=0.001_epoch=100_lamb=0.03_lr0.0004_dim=200_eps=0.5_DF=False_PCA=False_not-FC=False_cosine" / "CAFNet_result.csv",
    "CAFNet-D": ROOT / "result_ICS" / "10cd3e100f10_CAFNetDecoupled_knn=5_wd=0.001_epoch=100_lamb=0.03_lr0.0004_dim=200_eps=0.5_DF=False_PCA=False_not-FC=False_cross=True_fusion=gate_gate=new_fa=0.5_gatdrop=0.0_mix=0.3_aw=1.0_fw=1.0_rw=0.05_popw=0.1_biasw=1.0_listw=0.1_abw=1.0_arw=1.0_cosine" / "CAFNetDecoupled_result.csv",
    "w/o association residual": ROOT / "result_ICS" / "10cd3noresf10_CAFNetDecoupled" / "CAFNetDecoupled_result.csv",
    "w/o bias/popularity prior": ROOT / "result_ICS" / "10cd3abl_nobias_f10_CAFNetDecoupled" / "CAFNetDecoupled_result.csv",
}

METRICS = [
    ("MAP", "MAP", "higher", "ranking"),
    ("auc_all", "AUROC", "higher", "ranking"),
    ("aupr_all", "AUPR", "higher", "ranking"),
    ("nDCG", "nDCG@10", "higher", "ranking"),
    ("P15", "P@15", "higher", "ranking"),
    ("R15", "R@15", "higher", "ranking"),
    ("spearman", "Spearman", "higher", "regression"),
    ("rMSE", "RMSE", "lower", "regression"),
    ("MAE", "MAE", "lower", "regression"),
]


def read_results(path):
    df = pd.read_csv(path)
    numeric_guard = ["pearson"] + [metric for metric, *_ in METRICS]
    for metric in numeric_guard:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=numeric_guard).copy()
    if "fold" not in df.columns:
        df = df.head(10).copy()
        df.insert(0, "fold", range(len(df)))
    else:
        df["fold"] = pd.to_numeric(df["fold"], errors="coerce")
        df = df.dropna(subset=["fold"]).copy()
    return df.sort_values("fold").reset_index(drop=True)


def holm(values):
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    out = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        out[idx] = min(running, 1.0)
    return out


def paired_test(target, baseline, target_name, baseline_name):
    merged = target.merge(baseline, on="fold", suffixes=("_target", "_baseline"))
    rows = []
    for metric, label, direction, group in METRICS:
        a = merged[f"{metric}_target"].to_numpy(float)
        b = merged[f"{metric}_baseline"].to_numpy(float)
        improvement = a - b if direction == "higher" else b - a
        t = stats.ttest_rel(a, b)
        try:
            w = stats.wilcoxon(improvement, zero_method="wilcox", alternative="two-sided")
            w_stat, w_p = float(w.statistic), float(w.pvalue)
        except ValueError:
            w_stat, w_p = np.nan, 1.0
        rows.append({
            "comparison": f"{target_name} vs {baseline_name}",
            "target": target_name,
            "baseline": baseline_name,
            "metric": metric,
            "metric_label": label,
            "group": group,
            "direction": direction,
            "target_mean": float(np.mean(a)),
            "target_std": float(np.std(a, ddof=1)),
            "baseline_mean": float(np.mean(b)),
            "baseline_std": float(np.std(b, ddof=1)),
            "improvement_mean": float(np.mean(improvement)),
            "paired_t_p": float(t.pvalue),
            "wilcoxon_p": w_p,
            "wilcoxon_stat": w_stat,
        })
    return rows


def fmt(mean, std):
    return f"${mean:.4f} \\pm {std:.4f}$"


def main():
    data = {name: read_results(path) for name, path in MODELS.items()}

    summary_rows = []
    for name, df in data.items():
        row = {"Model": name}
        for metric, label, *_ in METRICS:
            row[f"{metric}_mean"] = df[metric].mean()
            row[f"{metric}_std"] = df[metric].std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "cold_decoupled_v3_ablation_f10_summary.csv", index=False)

    rows = []
    for baseline in ["CAFNet", "w/o association residual", "w/o bias/popularity prior"]:
        rows.extend(paired_test(data["CAFNet-D"], data[baseline], "CAFNet-D", baseline))
    paired = pd.DataFrame(rows)
    parts = []
    for _, sub in paired.groupby(["comparison", "group"], sort=False):
        sub = sub.copy()
        sub["paired_t_p_holm"] = holm(sub["paired_t_p"])
        sub["wilcoxon_p_holm"] = holm(sub["wilcoxon_p"])
        sub["paired_t_holm_sig"] = sub["paired_t_p_holm"] < 0.05
        sub["wilcoxon_holm_sig"] = sub["wilcoxon_p_holm"] < 0.05
        parts.append(sub)
    paired = pd.concat(parts, ignore_index=True)
    paired.to_csv(OUT / "cold_decoupled_v3_ablation_f10_paired_tests.csv", index=False)

    best = {}
    for metric, label, direction, _ in METRICS:
        values = summary.set_index("Model")[f"{metric}_mean"]
        best[metric] = values.idxmax() if direction == "higher" else values.idxmin()

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Cold-start 10-fold ablation study for CAFNet-D. Values are mean $\\pm$ standard deviation. Best values are bolded.}",
        "\\label{tab:ablation_full}",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{lccccccccc}",
        "\\toprule",
        "Variant & MAP & AUROC & AUPR & nDCG@10 & P@15 & R@15 & Spearman & RMSE & MAE \\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        cells = [row["Model"]]
        for metric, *_ in METRICS:
            text = fmt(row[f"{metric}_mean"], row[f"{metric}_std"])
            if best[metric] == row["Model"]:
                text = "\\textbf{" + text + "}"
            cells.append(text)
        lines.append(" & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{table*}",
        "",
    ]
    (OUT / "cold_decoupled_v3_ablation_f10_table.tex").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
