from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_outputs" / "cafnet_dg_per_drug" / "atc_stratified"
OUT.mkdir(parents=True, exist_ok=True)

DELTA_FILE = ROOT / "analysis_outputs" / "cafnet_dg_per_drug" / "per_drug_delta_metrics.csv"
ATC_FILE = ROOT / "data_external" / "chembl_atc_drug_evidence_mapping.csv"

ATC_L1_NAMES = {
    "A": "Alimentary tract and metabolism",
    "B": "Blood and blood forming organs",
    "C": "Cardiovascular system",
    "D": "Dermatologicals",
    "G": "Genito-urinary system and sex hormones",
    "H": "Systemic hormonal preparations",
    "J": "Antiinfectives for systemic use",
    "L": "Antineoplastic and immunomodulating agents",
    "M": "Musculo-skeletal system",
    "N": "Nervous system",
    "P": "Antiparasitic products",
    "R": "Respiratory system",
    "S": "Sensory organs",
    "V": "Various",
}

METRICS = [
    "delta_DG_minus_D_AP",
    "delta_DG_minus_D_rare_AP",
    "delta_DG_minus_D_middle_AP",
    "delta_DG_minus_D_nonhot100_AP",
    "delta_DG_minus_D_AUROC",
    "delta_DG_minus_D_nDCG@10",
    "delta_DG_minus_D_P@15",
    "delta_DG_minus_D_R@15",
]


def explode_atc_l1(atc: object) -> list[str]:
    if pd.isna(atc) or str(atc).strip() == "":
        return ["Unmapped"]
    values = []
    for item in str(atc).split("|"):
        item = item.strip()
        if item:
            values.append(item[0])
    return sorted(set(values)) or ["Unmapped"]


def summarize_group(group: pd.DataFrame, metric: str) -> dict:
    values = group[metric].dropna().to_numpy(dtype=float)
    out = {
        "metric": metric,
        "n_drugs": int(values.size),
        "mean_delta": np.nan,
        "median_delta": np.nan,
        "std_delta": np.nan,
        "positive_fraction": np.nan,
        "wilcoxon_p_two_sided": np.nan,
    }
    if values.size == 0:
        return out
    out["mean_delta"] = float(np.mean(values))
    out["median_delta"] = float(np.median(values))
    out["std_delta"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    out["positive_fraction"] = float(np.mean(values > 0))
    nonzero = values[values != 0]
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


def main() -> None:
    deltas = pd.read_csv(DELTA_FILE)
    atc = pd.read_csv(ATC_FILE)
    atc = atc[["drug_index", "sider_name", "atc_l1", "atc_codes"]].copy()
    atc["atc_l1_list"] = atc["atc_l1"].apply(explode_atc_l1)
    exploded = atc.explode("atc_l1_list").rename(columns={"atc_l1_list": "atc_l1_group"})
    exploded["atc_l1_name"] = exploded["atc_l1_group"].map(ATC_L1_NAMES).fillna(exploded["atc_l1_group"])

    merged = deltas.merge(exploded, on="drug_index", how="left")
    merged["atc_l1_group"] = merged["atc_l1_group"].fillna("Unmapped")
    merged["atc_l1_name"] = merged["atc_l1_name"].fillna("Unmapped")
    merged.to_csv(OUT / "per_drug_delta_with_atc_l1.csv", index=False)

    rows = []
    for (code, name), group in merged.groupby(["atc_l1_group", "atc_l1_name"], dropna=False):
        for metric in METRICS:
            row = summarize_group(group, metric)
            row.update({"atc_l1": code, "atc_l1_name": name})
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary = summary[["atc_l1", "atc_l1_name", "metric", "n_drugs", "mean_delta", "median_delta", "std_delta", "positive_fraction", "wilcoxon_p_two_sided"]]
    summary["holm_p_within_metric"] = np.nan
    for metric, idx in summary.groupby("metric").groups.items():
        mask = summary.index.isin(idx) & (summary["n_drugs"] >= 10)
        summary.loc[mask, "holm_p_within_metric"] = holm_adjust(summary.loc[mask, "wilcoxon_p_two_sided"])
    summary.to_csv(OUT / "atc_l1_delta_summary.csv", index=False)

    focus_metrics = [
        "delta_DG_minus_D_AP",
        "delta_DG_minus_D_rare_AP",
        "delta_DG_minus_D_middle_AP",
        "delta_DG_minus_D_nonhot100_AP",
    ]
    compact = summary[
        (summary["metric"].isin(focus_metrics))
        & (summary["n_drugs"] >= 20)
    ].copy()
    compact = compact.sort_values(["metric", "mean_delta"], ascending=[True, False])
    compact.to_csv(OUT / "atc_l1_delta_compact_focus.csv", index=False)

    report_lines = [
        "# CAFNet-DG ATC-Stratified Drug-Specific Delta Analysis",
        "",
        "This diagnostic maps per-drug CAFNet-DG minus CAFNet-D improvements to ATC level-1 therapeutic classes.",
        "It is intended to test whether residual-fusion gains appear across drug classes rather than only through a global side-effect popularity shift.",
        "",
        "## Coverage",
        "",
        f"- Per-drug rows: {deltas.shape[0]}",
        f"- Rows after ATC level-1 explosion: {merged.shape[0]}",
        f"- Unique drugs with at least one ATC level-1 code: {atc.loc[atc['atc_l1'].notna() & (atc['atc_l1'].astype(str) != ''), 'drug_index'].nunique()}",
        "",
        "## Main Focus Metrics by ATC Level-1 Class",
        "",
    ]
    for metric in focus_metrics:
        sub = compact[compact["metric"] == metric].sort_values("mean_delta", ascending=False)
        report_lines.append(f"### {metric}")
        report_lines.append("")
        for _, row in sub.iterrows():
            report_lines.append(
                f"- {row['atc_l1']} ({row['atc_l1_name']}), n={int(row['n_drugs'])}: "
                f"mean={row['mean_delta']:.4f}, median={row['median_delta']:.4f}, "
                f"positive_fraction={row['positive_fraction']:.3f}, "
                f"Holm p={row['holm_p_within_metric']:.4g}"
            )
        report_lines.append("")

    broad = compact.groupby("metric").agg(
        classes_with_positive_mean=("mean_delta", lambda s: int((s > 0).sum())),
        classes_total=("mean_delta", "size"),
        median_class_mean_delta=("mean_delta", "median"),
        min_class_mean_delta=("mean_delta", "min"),
        max_class_mean_delta=("mean_delta", "max"),
    ).reset_index()
    broad.to_csv(OUT / "atc_l1_broad_support_summary.csv", index=False)
    report_lines.extend([
        "## Broad Support Summary",
        "",
    ])
    for _, row in broad.iterrows():
        report_lines.append(
            f"- {row['metric']}: positive mean in {int(row['classes_with_positive_mean'])}/"
            f"{int(row['classes_total'])} ATC classes; median class mean delta="
            f"{row['median_class_mean_delta']:.4f}; range="
            f"{row['min_class_mean_delta']:.4f} to {row['max_class_mean_delta']:.4f}."
        )
    report_lines.extend([
        "",
        "## Interpretation",
        "",
        "Use this as diagnostic evidence only. A positive ATC-stratified pattern supports that CAFNet-DG changes drug-specific rankings across therapeutic groups, but it does not establish clinical causality or independent external validation.",
    ])
    (OUT / "ATC_STRATIFIED_DELTA_REPORT_20260701.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
