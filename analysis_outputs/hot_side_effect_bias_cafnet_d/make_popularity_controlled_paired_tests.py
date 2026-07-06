from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_outputs" / "hot_side_effect_bias_cafnet_d"
INFILE = OUT / "removed_hot_side_effect_metrics_by_fold.csv"

TARGET = "Decoupled-v3"
BASELINES = ["CAFNet", "A3Net", "Global popularity", "HSTrans"]
REMOVED = [10, 20, 50, 100]
METRICS = ["MAP", "nDCG", "P15", "R15"]


def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def wilcoxon_safe(diff):
    diff = np.asarray(diff, dtype=float)
    if np.allclose(diff, 0):
        return np.nan, 1.0
    try:
        result = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        return float(result.statistic), float(result.pvalue)
    except ValueError:
        return np.nan, 1.0


def fmt(mean, std):
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def fmt_p(p):
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def main():
    df = pd.read_csv(INFILE)
    df = df[df["scenario"].eq("cold")].copy()
    rows = []

    for removed in REMOVED:
        for baseline in BASELINES:
            target = df[
                df["model"].eq(TARGET)
                & df["removed_top_n_hot_side_effects"].eq(removed)
            ][["fold"] + METRICS].sort_values("fold")
            base = df[
                df["model"].eq(baseline)
                & df["removed_top_n_hot_side_effects"].eq(removed)
            ][["fold"] + METRICS].sort_values("fold")
            merged = target.merge(base, on="fold", suffixes=("_target", "_baseline"))
            if len(merged) != 10:
                raise ValueError(f"{baseline}, top{removed}: expected 10 folds, got {len(merged)}")
            for metric in METRICS:
                a = merged[f"{metric}_target"].astype(float).to_numpy()
                b = merged[f"{metric}_baseline"].astype(float).to_numpy()
                diff = a - b
                t = stats.ttest_rel(a, b)
                w_stat, w_p = wilcoxon_safe(diff)
                rows.append(
                    {
                        "removed_top_n": removed,
                        "comparison": f"{TARGET} vs {baseline}",
                        "baseline": baseline,
                        "metric": metric,
                        "target_mean": float(np.mean(a)),
                        "target_std": float(np.std(a, ddof=1)),
                        "baseline_mean": float(np.mean(b)),
                        "baseline_std": float(np.std(b, ddof=1)),
                        "delta_mean": float(np.mean(diff)),
                        "paired_t_p": float(t.pvalue),
                        "wilcoxon_p": w_p,
                        "wilcoxon_stat": w_stat,
                    }
                )

    paired = pd.DataFrame(rows)
    parts = []
    for _, sub in paired.groupby(["removed_top_n", "comparison"], sort=False):
        sub = sub.copy()
        sub["paired_t_p_holm"] = holm(sub["paired_t_p"])
        sub["wilcoxon_p_holm"] = holm(sub["wilcoxon_p"])
        sub["paired_t_holm_sig"] = sub["paired_t_p_holm"] < 0.05
        sub["wilcoxon_holm_sig"] = sub["wilcoxon_p_holm"] < 0.05
        parts.append(sub)
    paired = pd.concat(parts, ignore_index=True)
    paired.to_csv(OUT / "popularity_controlled_paired_tests.csv", index=False)

    summary_rows = []
    for (removed, baseline), sub in paired.groupby(["removed_top_n", "baseline"], sort=False):
        wins = sub[(sub["delta_mean"] > 0) & sub["paired_t_holm_sig"]]["metric"].tolist()
        numeric = sub[(sub["delta_mean"] > 0) & (~sub["paired_t_holm_sig"])]["metric"].tolist()
        losses = sub[(sub["delta_mean"] < 0) & sub["paired_t_holm_sig"]]["metric"].tolist()
        summary_rows.append(
            {
                "removed_top_n": removed,
                "baseline": baseline,
                "holm_sig_improvements": "; ".join(wins) if wins else "None",
                "numeric_only_improvements": "; ".join(numeric) if numeric else "None",
                "holm_sig_losses": "; ".join(losses) if losses else "None",
            }
        )
    pd.DataFrame(summary_rows).to_csv(OUT / "popularity_controlled_paired_tests_summary.csv", index=False)

    lines = [
        "\\begin{longtable}{lllrrrrr}",
        "\\caption{Popularity-controlled cold-start paired tests after removing the top-$N$ most frequent side effects. Values are mean $\\pm$ standard deviation over 10 folds. Positive $\\Delta$ indicates better CAFNet-D performance.}\\label{tab:popularity_controlled_paired}\\\\",
        "\\toprule",
        "Removed & Baseline & Metric & Baseline & CAFNet-D & $\\Delta$ & $p_t^{Holm}$ & $p_w^{Holm}$ \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        "Removed & Baseline & Metric & Baseline & CAFNet-D & $\\Delta$ & $p_t^{Holm}$ & $p_w^{Holm}$ \\\\",
        "\\midrule",
        "\\endhead",
    ]
    for _, row in paired.iterrows():
        marker = "$^{\\dagger}$" if row["paired_t_holm_sig"] and row["delta_mean"] > 0 else ""
        loss = "$^{\\ddagger}$" if row["paired_t_holm_sig"] and row["delta_mean"] < 0 else ""
        lines.append(
            f"Top-{int(row['removed_top_n'])} & {row['baseline']} & {row['metric']} & "
            f"{fmt(row['baseline_mean'], row['baseline_std'])} & "
            f"{fmt(row['target_mean'], row['target_std'])} & "
            f"{row['delta_mean']:.3f}{marker}{loss} & "
            f"{fmt_p(row['paired_t_p_holm'])} & {fmt_p(row['wilcoxon_p_holm'])} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{longtable}",
        "",
        "\\noindent $\\dagger$ indicates a Holm-corrected paired $t$-test improvement for CAFNet-D; $\\ddagger$ indicates a significant loss.",
        "",
    ]
    (OUT / "popularity_controlled_paired_tests.tex").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
