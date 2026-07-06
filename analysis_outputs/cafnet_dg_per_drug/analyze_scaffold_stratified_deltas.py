from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_outputs" / "cafnet_dg_per_drug" / "scaffold_stratified"
OUT.mkdir(parents=True, exist_ok=True)

DELTA_FILE = ROOT / "analysis_outputs" / "cafnet_dg_per_drug" / "per_drug_delta_metrics.csv"
SCAFFOLD_FILE = ROOT / "analysis_outputs" / "scaffold_split_audit" / "drug_scaffold_assignments.csv"

FOCUS_METRICS = [
    "delta_DG_minus_D_AP",
    "delta_DG_minus_D_rare_AP",
    "delta_DG_minus_D_middle_AP",
    "delta_DG_minus_D_nonhot100_AP",
]


def summarize(values: pd.Series) -> dict:
    arr = values.dropna().to_numpy(dtype=float)
    out = {
        "n_drugs": int(arr.size),
        "mean_delta": np.nan,
        "median_delta": np.nan,
        "std_delta": np.nan,
        "positive_fraction": np.nan,
        "wilcoxon_p_two_sided": np.nan,
    }
    if arr.size == 0:
        return out
    out["mean_delta"] = float(np.mean(arr))
    out["median_delta"] = float(np.median(arr))
    out["std_delta"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    out["positive_fraction"] = float(np.mean(arr > 0))
    nonzero = arr[arr != 0]
    if nonzero.size >= 5:
        try:
            out["wilcoxon_p_two_sided"] = float(wilcoxon(nonzero, alternative="two-sided").pvalue)
        except ValueError:
            pass
    return out


def holm_adjust(pvals: pd.Series) -> pd.Series:
    valid = pvals.dropna().sort_values()
    adjusted = pd.Series(np.nan, index=pvals.index, dtype=float)
    running = 0.0
    m = len(valid)
    for rank, (idx, p) in enumerate(valid.items(), start=1):
        running = max(running, min(1.0, (m - rank + 1) * p))
        adjusted.loc[idx] = running
    return adjusted


def size_bin(n: int) -> str:
    if n <= 1:
        return "singleton"
    if n <= 2:
        return "shared_2"
    if n <= 4:
        return "shared_3_4"
    return "shared_5plus"


def make_summary(df: pd.DataFrame, group_col: str, min_n_for_holm: int = 10) -> pd.DataFrame:
    rows = []
    for group_value, group in df.groupby(group_col, dropna=False):
        for metric in FOCUS_METRICS:
            row = summarize(group[metric])
            row.update({"grouping": group_col, "group": str(group_value), "metric": metric})
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary = summary[["grouping", "group", "metric", "n_drugs", "mean_delta", "median_delta", "std_delta", "positive_fraction", "wilcoxon_p_two_sided"]]
    summary["holm_p_within_metric"] = np.nan
    for metric, idx in summary.groupby("metric").groups.items():
        mask = summary.index.isin(idx) & (summary["n_drugs"] >= min_n_for_holm)
        summary.loc[mask, "holm_p_within_metric"] = holm_adjust(summary.loc[mask, "wilcoxon_p_two_sided"])
    return summary


def main() -> None:
    deltas = pd.read_csv(DELTA_FILE)
    scaffolds = pd.read_csv(SCAFFOLD_FILE)
    counts = scaffolds.groupby("split_group").size().rename("scaffold_group_size").reset_index()
    scaffolds = scaffolds.merge(counts, on="split_group", how="left")
    scaffolds["scaffold_size_bin"] = scaffolds["scaffold_group_size"].apply(size_bin)
    scaffolds["scaffold_type"] = np.where(scaffolds["has_murcko_scaffold"].astype(bool), "murcko", "acyclic_or_empty")
    scaffolds["repeat_group"] = np.where(scaffolds["scaffold_group_size"] >= 2, "shared_scaffold_group", "singleton_scaffold_group")

    merged = deltas.merge(
        scaffolds[[
            "drug_index",
            "drug",
            "murcko_scaffold",
            "split_group",
            "has_murcko_scaffold",
            "scaffold_group_size",
            "scaffold_size_bin",
            "scaffold_type",
            "repeat_group",
            "scaffold_fold",
        ]],
        on="drug_index",
        how="left",
    )
    merged.to_csv(OUT / "per_drug_delta_with_scaffold.csv", index=False)

    summaries = []
    for group_col in ["scaffold_type", "repeat_group", "scaffold_size_bin", "scaffold_fold"]:
        summaries.append(make_summary(merged, group_col))
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(OUT / "scaffold_stratified_delta_summary.csv", index=False)

    repeated = merged[merged["scaffold_group_size"] >= 2].copy()
    exact_rows = []
    for split_group, group in repeated.groupby("split_group"):
        if len(group) < 2:
            continue
        for metric in FOCUS_METRICS:
            row = summarize(group[metric])
            row.update({
                "split_group": split_group,
                "scaffold_group_size": int(len(group)),
                "metric": metric,
                "example_drugs": "; ".join(group["drug_name"].fillna(group["drug"]).astype(str).head(5)),
            })
            exact_rows.append(row)
    exact = pd.DataFrame(exact_rows)
    if not exact.empty:
        exact = exact.sort_values(["metric", "scaffold_group_size", "mean_delta"], ascending=[True, False, False])
    exact.to_csv(OUT / "repeated_scaffold_group_delta_summary.csv", index=False)

    broad_rows = []
    for grouping in ["scaffold_type", "repeat_group", "scaffold_size_bin"]:
        sub = summary[summary["grouping"] == grouping]
        for metric in FOCUS_METRICS:
            m = sub[sub["metric"] == metric]
            broad_rows.append({
                "grouping": grouping,
                "metric": metric,
                "groups_with_positive_mean": int((m["mean_delta"] > 0).sum()),
                "groups_total": int(m.shape[0]),
                "median_group_mean_delta": float(m["mean_delta"].median()),
                "min_group_mean_delta": float(m["mean_delta"].min()),
                "max_group_mean_delta": float(m["mean_delta"].max()),
            })
    broad = pd.DataFrame(broad_rows)
    broad.to_csv(OUT / "scaffold_broad_support_summary.csv", index=False)

    report = [
        "# CAFNet-DG Scaffold-Stratified Drug-Specific Delta Analysis",
        "",
        "This diagnostic merges per-drug CAFNet-DG minus CAFNet-D deltas with RDKit Bemis-Murcko scaffold assignments.",
        "It tests whether residual-fusion improvements persist across structural scaffold categories. It is not a direct scaffold-disjoint retraining experiment.",
        "",
        "## Coverage",
        "",
        f"- Per-drug rows: {deltas.shape[0]}",
        f"- Scaffold-assigned drugs: {scaffolds['drug_index'].nunique()}",
        f"- Unique split groups: {scaffolds['split_group'].nunique()}",
        f"- Drugs with non-empty Murcko scaffold: {int(scaffolds['has_murcko_scaffold'].astype(bool).sum())}",
        f"- Drugs with acyclic/empty scaffold fallback: {int((~scaffolds['has_murcko_scaffold'].astype(bool)).sum())}",
        "",
        "## Broad Support Summary",
        "",
    ]
    for _, row in broad.iterrows():
        report.append(
            f"- {row['grouping']} / {row['metric']}: positive mean in "
            f"{int(row['groups_with_positive_mean'])}/{int(row['groups_total'])} groups; "
            f"median group mean delta={row['median_group_mean_delta']:.4f}; "
            f"range={row['min_group_mean_delta']:.4f} to {row['max_group_mean_delta']:.4f}."
        )
    report.extend(["", "## Key Size-Bin Results", ""])
    size_sub = summary[summary["grouping"] == "scaffold_size_bin"].copy()
    for metric in FOCUS_METRICS:
        report.append(f"### {metric}")
        report.append("")
        for _, row in size_sub[size_sub["metric"] == metric].sort_values("group").iterrows():
            report.append(
                f"- {row['group']}, n={int(row['n_drugs'])}: mean={row['mean_delta']:.4f}, "
                f"median={row['median_delta']:.4f}, positive_fraction={row['positive_fraction']:.3f}, "
                f"Holm p={row['holm_p_within_metric']:.4g}"
            )
        report.append("")
    report.extend([
        "## Interpretation",
        "",
        "Use this as structural diagnostic evidence only. Positive deltas across singleton and shared scaffold bins support that CAFNet-DG changes drug-specific rankings beyond one repeated chemical series, but a formal scaffold-disjoint CAFNet-DG experiment would still be stronger.",
    ])
    (OUT / "SCAFFOLD_STRATIFIED_DELTA_REPORT_20260701.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
