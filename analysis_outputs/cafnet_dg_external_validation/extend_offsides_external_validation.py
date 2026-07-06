from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_outputs" / "cafnet_dg_external_validation"
SCORED_FILE = OUT / "offsides_matched_control_scored_pairs.csv"
RAW_FILE = ROOT / "data" / "raw_frequency_750.mat"
MASK_FILE = ROOT / "data" / "blind_mask_mat_750.mat"

MODELS = ["CAFNet", "CAFNet-D", "CAFNet-DG", "Global popularity"]
TARGET = "CAFNet-DG"
METRICS = ["external_AUROC", "external_AUPR", "pos_gt_neg_rate", "mean_pos_minus_neg"]


def safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def safe_aupr(y: np.ndarray, s: np.ndarray) -> float:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, s))


def metric_row(df: pd.DataFrame) -> dict[str, float | int]:
    if df.empty:
        return {
            "n_matched_rows": 0,
            "n_drugs": 0,
            "n_positive_pairs": 0,
            "external_AUROC": np.nan,
            "external_AUPR": np.nan,
            "pos_gt_neg_rate": np.nan,
            "mean_pos_minus_neg": np.nan,
        }
    labels = np.r_[np.ones(len(df)), np.zeros(len(df))]
    scores = np.r_[df["score"].to_numpy(float), df["paired_score"].to_numpy(float)]
    return {
        "n_matched_rows": int(len(df)),
        "n_drugs": int(df["drug_index"].nunique()),
        "n_positive_pairs": int(df[["drug_index", "side_index"]].drop_duplicates().shape[0]),
        "external_AUROC": safe_auroc(labels, scores),
        "external_AUPR": safe_aupr(labels, scores),
        "pos_gt_neg_rate": float(df["pos_gt_neg"].mean()),
        "mean_pos_minus_neg": float(df["pos_minus_neg"].mean()),
    }


def holm(p_values: list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def cold_train_rows(masks: dict[str, np.ndarray]) -> dict[int, np.ndarray]:
    out = {}
    for fold in range(10):
        mask = masks[f"mask{fold}"].astype(float)
        out[fold] = mask[:, 0] != 0
    return out


def fold_prevalence_metadata(raw: np.ndarray, masks: dict[str, np.ndarray]) -> tuple[dict[int, set[int]], dict[int, set[int]], dict[int, set[int]], pd.DataFrame]:
    train_rows = cold_train_rows(masks)
    hot20, hot50, hot100 = {}, {}, {}
    rows = []
    for fold in range(10):
        prevalence = (raw[train_rows[fold]] != 0).mean(axis=0)
        q1, q2 = np.quantile(prevalence, [1 / 3, 2 / 3])
        order = np.argsort(prevalence)[::-1]
        hot20[fold] = set(order[:20].astype(int).tolist())
        hot50[fold] = set(order[:50].astype(int).tolist())
        hot100[fold] = set(order[:100].astype(int).tolist())
        for side_idx, prev in enumerate(prevalence):
            if prev <= q1:
                group = "rare"
            elif prev <= q2:
                group = "middle"
            else:
                group = "frequent"
            rows.append(
                {
                    "fold": fold,
                    "side_index": int(side_idx),
                    "train_prevalence": float(prev),
                    "train_positive_count": float(prev * train_rows[fold].sum()),
                    "prevalence_group": group,
                    "is_hot20": int(side_idx in hot20[fold]),
                    "is_hot50": int(side_idx in hot50[fold]),
                    "is_hot100": int(side_idx in hot100[fold]),
                }
            )
    return hot20, hot50, hot100, pd.DataFrame(rows)


def add_metadata(scored: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    pos_meta = meta.rename(
        columns={
            "side_index": "side_index",
            "train_prevalence": "pos_train_prevalence",
            "train_positive_count": "pos_train_positive_count",
            "prevalence_group": "pos_prevalence_group",
            "is_hot20": "pos_is_hot20",
            "is_hot50": "pos_is_hot50",
            "is_hot100": "pos_is_hot100",
        }
    )
    keep = [
        "fold",
        "side_index",
        "pos_train_prevalence",
        "pos_train_positive_count",
        "pos_prevalence_group",
        "pos_is_hot20",
        "pos_is_hot50",
        "pos_is_hot100",
    ]
    scored = scored.merge(pos_meta[keep], on=["fold", "side_index"], how="left")
    ctrl_meta = meta.rename(
        columns={
            "side_index": "control_side_index",
            "is_hot20": "control_is_hot20",
            "is_hot50": "control_is_hot50",
            "is_hot100": "control_is_hot100",
        }
    )
    scored = scored.merge(
        ctrl_meta[["fold", "control_side_index", "control_is_hot20", "control_is_hot50", "control_is_hot100"]],
        on=["fold", "control_side_index"],
        how="left",
    )
    return scored


def summarize_conditions(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    condition_frames: list[tuple[str, str, pd.DataFrame]] = [("all", "all", scored)]
    for n in [20, 50, 100]:
        subset = scored[(scored[f"pos_is_hot{n}"] == 0) & (scored[f"control_is_hot{n}"] == 0)]
        condition_frames.append(("hot_removal", f"remove_top{n}", subset))
    for group in ["rare", "middle", "frequent"]:
        condition_frames.append(("prevalence_group", group, scored[scored["pos_prevalence_group"] == group]))

    summary_rows = []
    per_drug_rows = []
    for family, condition, df_condition in condition_frames:
        for model in MODELS:
            df_model = df_condition[df_condition["model"] == model]
            row = {"condition_family": family, "condition": condition, "model": model}
            row.update(metric_row(df_model))
            summary_rows.append(row)
            for drug_idx, df_drug in df_model.groupby("drug_index"):
                drow = {
                    "condition_family": family,
                    "condition": condition,
                    "model": model,
                    "drug_index": int(drug_idx),
                }
                drow.update(metric_row(df_drug))
                per_drug_rows.append(drow)
    return pd.DataFrame(summary_rows), pd.DataFrame(per_drug_rows)


def paired_tests(per_drug: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, condition), sub in per_drug.groupby(["condition_family", "condition"]):
        p_indexes = []
        p_values = []
        for baseline in [m for m in MODELS if m != TARGET]:
            target = sub[sub["model"] == TARGET]
            base = sub[sub["model"] == baseline]
            merged = target.merge(base, on="drug_index", suffixes=("_target", "_baseline"))
            for metric in METRICS:
                a = merged[f"{metric}_target"].to_numpy(float)
                b = merged[f"{metric}_baseline"].to_numpy(float)
                mask = np.isfinite(a) & np.isfinite(b)
                if mask.sum() < 3 or np.allclose(a[mask], b[mask]):
                    p = 1.0
                    stat = np.nan
                else:
                    stat, p = stats.wilcoxon(a[mask], b[mask], zero_method="wilcox", alternative="two-sided")
                rows.append(
                    {
                        "condition_family": family,
                        "condition": condition,
                        "comparison": f"{TARGET} vs {baseline}",
                        "baseline": baseline,
                        "metric": metric,
                        "n_drugs": int(mask.sum()),
                        "target_mean": float(np.nanmean(a[mask])) if mask.any() else np.nan,
                        "baseline_mean": float(np.nanmean(b[mask])) if mask.any() else np.nan,
                        "improvement_mean": float(np.nanmean(a[mask] - b[mask])) if mask.any() else np.nan,
                        "wilcoxon_stat": float(stat) if np.isfinite(stat) else np.nan,
                        "wilcoxon_p": float(p),
                    }
                )
                p_indexes.append(len(rows) - 1)
                p_values.append(float(p))
        adjusted = holm(p_values)
        for idx, adj in zip(p_indexes, adjusted):
            rows[idx]["wilcoxon_p_holm"] = float(adj)
            rows[idx]["holm_sig"] = bool(adj < 0.05)
    return pd.DataFrame(rows)


def bootstrap_ci_from_per_drug(per_drug: pd.DataFrame, n_boot: int = 2000, seed: int = 20260701) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    sub = per_drug[per_drug["condition"].eq("all")].copy()
    for model in MODELS:
        df_model = sub[sub["model"] == model].sort_values("drug_index")
        values_by_metric = {
            metric: df_model[metric].to_numpy(float)
            for metric in METRICS
        }
        n = len(df_model)
        boot = {metric: [] for metric in METRICS}
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            for metric in METRICS:
                vals = values_by_metric[metric][idx]
                boot[metric].append(float(np.nanmean(vals)))
        for metric in METRICS:
            values = np.asarray(boot[metric], dtype=float)
            values = values[np.isfinite(values)]
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "point": float(np.nanmean(values_by_metric[metric])),
                    "ci_low": float(np.quantile(values, 0.025)),
                    "ci_high": float(np.quantile(values, 0.975)),
                    "n_boot": int(len(values)),
                    "bootstrap_unit": "drug_per_metric_mean",
                }
            )
    return pd.DataFrame(rows)


def fmt3(x: float) -> str:
    return "--" if not np.isfinite(x) else f"{x:.3f}"


def write_latex(summary: pd.DataFrame, ci: pd.DataFrame) -> None:
    ci_wide = ci.pivot(index="model", columns="metric", values=["ci_low", "ci_high"])
    ci_point = ci.pivot(index="model", columns="metric", values="point")
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{OFFSIDES external matched-control analysis with drug-level bootstrap confidence intervals for per-drug metric means. External positives already present in the SIDER-derived matrix were removed.}",
        "\\label{tab:offsides_external_ci_supp}",
        "\\small",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Model & AUROC (95\\% CI) & AUPR (95\\% CI) & Pos$>$Ctrl (95\\% CI) \\\\",
        "\\midrule",
    ]
    for model in MODELS:
        vals = []
        for metric in ["external_AUROC", "external_AUPR", "pos_gt_neg_rate"]:
            point = float(ci_point.loc[model, metric])
            lo = float(ci_wide.loc[model, ("ci_low", metric)])
            hi = float(ci_wide.loc[model, ("ci_high", metric)])
            cell = f"{fmt3(point)} ({fmt3(lo)}--{fmt3(hi)})"
            if model == "CAFNet-DG":
                cell = f"\\textbf{{{cell}}}"
            vals.append(cell)
        lines.append(f"{model} & {vals[0]} & {vals[1]} & {vals[2]} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    (OUT / "offsides_external_main_ci_table.tex").write_text("\n".join(lines), encoding="utf-8")

    subset = summary[
        summary["condition_family"].isin(["hot_removal", "prevalence_group"])
        & summary["model"].isin(["CAFNet", "CAFNet-D", "CAFNet-DG", "Global popularity"])
    ].copy()
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Popularity-controlled and prevalence-stratified OFFSIDES external matched-control analysis. Hot-removal rows exclude matched rows where either the positive or control side effect is among the fold-specific top-$N$ frequent side effects.}",
        "\\label{tab:offsides_external_controlled_supp}",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Condition & Model & Positives & AUROC & AUPR & Pos$>$Ctrl \\\\",
        "\\midrule",
    ]
    order = [
        ("hot_removal", "remove_top20"),
        ("hot_removal", "remove_top50"),
        ("hot_removal", "remove_top100"),
        ("prevalence_group", "rare"),
        ("prevalence_group", "middle"),
        ("prevalence_group", "frequent"),
    ]
    for family, condition in order:
        block = subset[(subset["condition_family"] == family) & (subset["condition"] == condition)]
        for model in MODELS:
            row = block[block["model"] == model].iloc[0]
            best_auc = block["external_AUROC"].max()
            best_aupr = block["external_AUPR"].max()
            best_pos = block["pos_gt_neg_rate"].max()
            auc = fmt3(row["external_AUROC"])
            aupr = fmt3(row["external_AUPR"])
            pos = fmt3(row["pos_gt_neg_rate"])
            if np.isclose(row["external_AUROC"], best_auc, equal_nan=False):
                auc = f"\\textbf{{{auc}}}"
            if np.isclose(row["external_AUPR"], best_aupr, equal_nan=False):
                aupr = f"\\textbf{{{aupr}}}"
            if np.isclose(row["pos_gt_neg_rate"], best_pos, equal_nan=False):
                pos = f"\\textbf{{{pos}}}"
            condition_label = condition.replace("_", "-") if model == MODELS[0] else ""
            lines.append(
                f"{condition_label} & {model} & {int(row['n_positive_pairs'])} & {auc} & {aupr} & {pos} \\\\"
            )
        lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    (OUT / "offsides_external_controlled_table.tex").write_text("\n".join(lines), encoding="utf-8")


def markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(format(float(val), floatfmt) if np.isfinite(val) else "nan")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(summary: pd.DataFrame, tests: pd.DataFrame, ci: pd.DataFrame) -> None:
    main = summary[summary["condition"].eq("all")].copy()
    hot = summary[summary["condition_family"].eq("hot_removal")].copy()
    group = summary[summary["condition_family"].eq("prevalence_group")].copy()
    lines = [
        "# Extended OFFSIDES External Validation for CAFNet-DG",
        "",
        "This analysis extends the non-SIDER OFFSIDES matched-control validation with drug-level bootstrap confidence intervals, hot-side-effect removal, and prevalence-stratified external positives.",
        "",
        "## Global matched-control validation",
        "",
        markdown_table(main[["model", "n_positive_pairs", "n_drugs", "external_AUROC", "external_AUPR", "pos_gt_neg_rate"]]),
        "",
        "## Drug-level bootstrap 95% confidence intervals",
        "",
        markdown_table(ci[ci["metric"].isin(["external_AUROC", "external_AUPR", "pos_gt_neg_rate"])]),
        "",
        "## Hot-side-effect removal",
        "",
        markdown_table(hot[["condition", "model", "n_positive_pairs", "n_drugs", "external_AUROC", "external_AUPR", "pos_gt_neg_rate"]]),
        "",
        "## Prevalence-stratified OFFSIDES positives",
        "",
        markdown_table(group[["condition", "model", "n_positive_pairs", "n_drugs", "external_AUROC", "external_AUPR", "pos_gt_neg_rate"]]),
        "",
        "## Key paired tests for CAFNet-DG",
        "",
        markdown_table(tests[
            tests["metric"].isin(["external_AUROC", "external_AUPR", "pos_gt_neg_rate"])
            & tests["comparison"].isin(["CAFNet-DG vs CAFNet-D", "CAFNet-DG vs CAFNet", "CAFNet-DG vs Global popularity"])
        ], floatfmt=".4g"),
        "",
        "## Interpretation",
        "",
        "- This remains an external association/prioritization analysis, not external frequency-regression validation.",
        "- Controls are prevalence-matched unobserved pairs, not clinically confirmed negatives.",
        "- Hot-removal and prevalence-stratified rows should be used to judge whether the OFFSIDES signal is dominated by common side effects.",
    ]
    (OUT / "EXTENDED_OFFSIDES_EXTERNAL_VALIDATION_REPORT_20260701.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    raw = sio.loadmat(RAW_FILE)["R"].astype(float)
    masks = sio.loadmat(MASK_FILE)
    scored = pd.read_csv(SCORED_FILE)
    _, _, _, meta = fold_prevalence_metadata(raw, masks)
    scored = add_metadata(scored, meta)
    meta_file = OUT / "offsides_matched_control_scored_pairs_with_prevalence_meta.csv"
    if not meta_file.exists():
        scored.to_csv(meta_file, index=False)

    summary, per_drug = summarize_conditions(scored)
    tests = paired_tests(per_drug)
    ci = bootstrap_ci_from_per_drug(per_drug)

    summary.to_csv(OUT / "offsides_extended_condition_summary.csv", index=False)
    per_drug.to_csv(OUT / "offsides_extended_condition_per_drug.csv", index=False)
    tests.to_csv(OUT / "offsides_extended_condition_paired_tests.csv", index=False)
    ci.to_csv(OUT / "offsides_external_drug_bootstrap_ci.csv", index=False)
    write_latex(summary, ci)
    write_report(summary, tests, ci)

    print("Wrote extended OFFSIDES external validation outputs to", OUT)
    print(summary[summary["model"].eq("CAFNet-DG")][["condition_family", "condition", "n_positive_pairs", "external_AUROC", "external_AUPR", "pos_gt_neg_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
