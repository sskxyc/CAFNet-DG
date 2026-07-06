from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_outputs" / "cafnet_dg_final_evidence"
OUT.mkdir(parents=True, exist_ok=True)

RAW_FILE = ROOT / "data" / "raw_frequency_750.mat"
MASK_FILE = ROOT / "data" / "blind_mask_mat_750.mat"

MODEL_PATHS = {
    "CAFNet": ROOT / "result_ICS" / (
        "10ICS_CAFNet_knn=5_wd=0.001_epoch=100_lamb=0.03_lr0.0004_dim=200_"
        "eps=0.5_DF=False_PCA=False_not-FC=False_cosine"
    ) / "blind_pred.csv",
    "CAFNet-D": ROOT / "result_ICS" / (
        "10cd3e100f10_CAFNetDecoupled_knn=5_wd=0.001_epoch=100_lamb=0.03_"
        "lr0.0004_dim=200_eps=0.5_DF=False_PCA=False_not-FC=False_cross=True_"
        "fusion=gate_gate=new_fa=0.5_gatdrop=0.0_mix=0.3_aw=1.0_fw=1.0_"
        "rw=0.05_popw=0.1_biasw=1.0_listw=0.1_abw=1.0_arw=1.0_cosine"
    ) / "blind_pred.csv",
    "CAFNet-DG": ROOT / "result_ICS" / "10cafnet_dg_ensemble06_cafnetd04_cafnet" / "blind_pred.csv",
}

METRICS = [
    ("MAP", "mAP", "higher", "ranking"),
    ("AUROC", "AUROC", "higher", "ranking"),
    ("AUPR", "AUPR", "higher", "ranking"),
    ("nDCG@10", "nDCG@10", "higher", "ranking"),
    ("P@1", "P@1", "higher", "ranking"),
    ("P@15", "P@15", "higher", "ranking"),
    ("R@15", "R@15", "higher", "ranking"),
    ("Spearman", "Spearman", "higher", "regression"),
    ("RMSE", "RMSE", "lower", "regression"),
    ("MAE", "MAE", "lower", "regression"),
]


def read_matrix(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, header=None).values.astype(float)


def split_cold_concat(full: np.ndarray, masks: dict[str, np.ndarray]) -> list[np.ndarray]:
    parts, start = [], 0
    for fold in range(10):
        mask = masks[f"mask{fold}"].astype(float)
        n = int(np.sum(mask[:, 0] == 0))
        parts.append(full[start : start + n])
        start += n
    if start != len(full):
        raise ValueError(f"Expected {start} rows, got {len(full)}")
    return parts


def precision(actual, predicted, n):
    return float(len(set(actual) & set(predicted[:n])) / float(n))


def recall(actual, predicted, n):
    actual = set(actual)
    return 0.0 if not actual else float(len(actual & set(predicted[:n])) / float(len(actual)))


def ndcg(actual, predicted, n=10):
    actual = set(actual)
    dcg = 0.0
    hits = []
    for i, item in enumerate(predicted[:n]):
        hit = 1 if item in actual else 0
        hits.append(hit)
        if hit:
            dcg += 1.0 / np.log2(i + 2)
    hits.sort(reverse=True)
    idcg = sum(v / np.log2(i + 2) for i, v in enumerate(hits[:n]))
    return float(dcg / idcg) if idcg > 0 else 0.0


def map_auc(pos_idx, neg_idx, scores):
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return np.nan, np.nan
    pos_scores = scores[pos_idx]
    neg_scores = scores[neg_idx]
    order = np.argsort(pos_scores)[::-1]
    sorted_pos = pos_scores[order]
    sorted_neg = np.sort(neg_scores)[::-1]
    ap = 0.0
    auc_num = 0.0
    for i, pos_score in enumerate(sorted_pos):
        num_higher_neg = 0.0
        for neg_score in sorted_neg:
            if pos_score <= neg_score:
                num_higher_neg += 1
            else:
                auc_num += 1
        ap += (i + 1) / (i + num_higher_neg + 1)
    return float(ap / len(pos_idx)), float(auc_num / (len(pos_idx) * len(neg_idx)))


def rank_metrics_cold(pred: np.ndarray, label: np.ndarray) -> dict[str, float]:
    positions = [1, 5, 10, 15]
    prec = np.zeros(len(positions), dtype=float)
    rec = np.zeros(len(positions), dtype=float)
    map_values, auc_values, ndcg_values, aupr_values, drug_auc_values = [], [], [], [], []
    candidate = np.arange(label.shape[1])
    used = 0
    for drug_idx in range(label.shape[0]):
        binary = (label[drug_idx] != 0).astype(int)
        pos = np.where(binary == 1)[0]
        if pos.size == 0:
            continue
        scores = pred[drug_idx]
        top = candidate[np.argsort(scores[candidate])[::-1][: max(positions)]]
        for k, n in enumerate(positions):
            prec[k] += precision(pos, top, n)
            rec[k] += recall(pos, top, n)
        neg = np.setdiff1d(candidate, pos, assume_unique=False)
        ap_v, auc_v = map_auc(pos, neg, scores)
        map_values.append(ap_v)
        auc_values.append(auc_v)
        ndcg_values.append(ndcg(pos, top, 10))
        if len(np.unique(binary)) == 2:
            drug_auc_values.append(roc_auc_score(binary, scores))
            aupr_values.append(average_precision_score(binary, scores))
        used += 1
    prec /= max(used, 1)
    rec /= max(used, 1)
    y_bin = (label.reshape(-1) != 0).astype(int)
    s = pred.reshape(-1)
    return {
        "MAP": float(np.nanmean(map_values)),
        "AUROC": float(roc_auc_score(y_bin, s)),
        "AUPR": float(average_precision_score(y_bin, s)),
        "drugAUROC": float(np.nanmean(drug_auc_values)),
        "drugAUPR": float(np.nanmean(aupr_values)),
        "nDCG@10": float(np.nanmean(ndcg_values)),
        "P@1": float(prec[0]),
        "P@5": float(prec[1]),
        "P@10": float(prec[2]),
        "P@15": float(prec[3]),
        "R@1": float(rec[0]),
        "R@5": float(rec[1]),
        "R@10": float(rec[2]),
        "R@15": float(rec[3]),
    }


def regression_metrics(pred: np.ndarray, label: np.ndarray) -> dict[str, float]:
    y = label.reshape(-1)
    s = pred.reshape(-1)
    ok = np.isfinite(y) & np.isfinite(s)
    y, s = y[ok], s[ok]
    return {
        "Pearson": float(np.corrcoef(y, s)[0, 1]) if len(y) > 1 else np.nan,
        "Spearman": float(stats.spearmanr(y, s)[0]) if len(y) > 1 else np.nan,
        "RMSE": float(np.sqrt(np.mean((y - s) ** 2))),
        "MAE": float(mean_absolute_error(y, s)),
    }


def global_popularity_preds(raw: np.ndarray, masks: dict[str, np.ndarray]) -> list[np.ndarray]:
    preds = []
    for fold in range(10):
        mask = masks[f"mask{fold}"].astype(float)
        train = mask[:, 0] != 0
        test = mask[:, 0] == 0
        pop = (raw[train] != 0).mean(axis=0)
        preds.append(np.tile(pop, (int(np.sum(test)), 1)))
    return preds


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
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0 or np.allclose(diff, 0):
        return np.nan, 1.0
    try:
        w = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        return float(w.statistic), float(w.pvalue)
    except ValueError:
        return np.nan, 1.0


def paired_tests(df: pd.DataFrame, target="CAFNet-DG") -> pd.DataFrame:
    rows = []
    for baseline in sorted(set(df["model"]) - {target}):
        merged = df[df["model"].eq(target)].merge(
            df[df["model"].eq(baseline)], on="fold", suffixes=("_target", "_baseline")
        )
        for metric, label, direction, group in METRICS:
            a = merged[f"{metric}_target"].astype(float).to_numpy()
            b = merged[f"{metric}_baseline"].astype(float).to_numpy()
            diff = a - b if direction == "higher" else b - a
            t = stats.ttest_rel(a, b, nan_policy="omit")
            w_stat, w_p = wilcoxon_safe(diff)
            rows.append(
                {
                    "comparison": f"{target} vs {baseline}",
                    "baseline": baseline,
                    "metric": metric,
                    "metric_label": label,
                    "group": group,
                    "direction": direction,
                    "n_folds": len(merged),
                    "target_mean": float(np.mean(a)),
                    "target_std": float(np.std(a, ddof=1)),
                    "baseline_mean": float(np.mean(b)),
                    "baseline_std": float(np.std(b, ddof=1)),
                    "improvement_mean": float(np.mean(diff)),
                    "paired_t_p": float(t.pvalue),
                    "wilcoxon_stat": w_stat,
                    "wilcoxon_p": w_p,
                }
            )
    out = pd.DataFrame(rows)
    parts = []
    for _, sub in out.groupby(["comparison", "group"], sort=False):
        sub = sub.copy()
        sub["paired_t_p_holm"] = holm(sub["paired_t_p"])
        sub["wilcoxon_p_holm"] = holm(sub["wilcoxon_p"])
        sub["paired_t_holm_sig"] = sub["paired_t_p_holm"] < 0.05
        sub["wilcoxon_holm_sig"] = sub["wilcoxon_p_holm"] < 0.05
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def fmt(mean, std):
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def bold_best(summary: pd.DataFrame, model: str, metric: str, value: str) -> str:
    vals = summary.set_index("model")
    direction = next(d for m, _, d, _ in METRICS if m == metric)
    best = vals[metric]["mean"].max() if direction == "higher" else vals[metric]["mean"].min()
    current = vals.loc[model, (metric, "mean")]
    if np.isclose(current, best):
        return f"\\textbf{{{value}}}"
    return value


def write_tables(summary: pd.DataFrame, paired: pd.DataFrame):
    summary_flat = summary.copy()
    summary_flat.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in summary_flat.columns]
    summary_flat.to_csv(OUT / "cafnet_dg_cold_summary.csv", index=False)

    rank_metrics = ["MAP", "AUROC", "AUPR", "nDCG@10", "P@1", "P@15", "R@15"]
    reg_metrics = ["Spearman", "RMSE", "MAE"]
    for filename, metrics, caption in [
        ("cafnet_dg_ranking_table.tex", rank_metrics, "Cold-start ranking comparison for CAFNet-DG."),
        ("cafnet_dg_regression_table.tex", reg_metrics, "Cold-start frequency/regression comparison for CAFNet-DG."),
    ]:
        lines = [
            "\\begin{table*}[t]",
            "\\centering",
            f"\\caption{{{caption}}}",
            "\\begin{tabular}{l" + "c" * len(metrics) + "}",
            "\\toprule",
            "Model & " + " & ".join([next(label for m, label, _, _ in METRICS if m == metric) for metric in metrics]) + r" \\",
            "\\midrule",
        ]
        for model in summary["model"]:
            cells = []
            for metric in metrics:
                mean = summary.loc[summary["model"].eq(model), (metric, "mean")].iloc[0]
                std = summary.loc[summary["model"].eq(model), (metric, "std")].iloc[0]
                cells.append(bold_best(summary, model, metric, fmt(mean, std)))
            lines.append(f"{model} & " + " & ".join(cells) + r" \\")
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
        (OUT / filename).write_text("\n".join(lines), encoding="utf-8")

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Paired tests for CAFNet-DG against attribution-critical baselines. Positive $\\Delta$ indicates better CAFNet-DG performance after metric direction is considered.}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Baseline & Metric & Baseline & CAFNet-DG & $\\Delta$ & $p_w^{Holm}$ \\\\",
        "\\midrule",
    ]
    focus = paired[
        paired["baseline"].isin(["CAFNet-D", "CAFNet", "Global popularity"])
        & paired["metric"].isin(["MAP", "nDCG@10", "P@15", "R@15", "Spearman", "RMSE", "MAE"])
    ].copy()
    for _, row in focus.iterrows():
        marker = "$^{\\dagger}$" if row["wilcoxon_holm_sig"] and row["improvement_mean"] > 0 else ""
        loss = "$^{\\ddagger}$" if row["wilcoxon_holm_sig"] and row["improvement_mean"] < 0 else ""
        lines.append(
            f"{row['baseline']} & {row['metric_label']} & "
            f"{fmt(row['baseline_mean'], row['baseline_std'])} & "
            f"{fmt(row['target_mean'], row['target_std'])} & "
            f"{row['improvement_mean']:.3f}{marker}{loss} & "
            f"{row['wilcoxon_p_holm']:.3g} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
        "",
    ]
    (OUT / "cafnet_dg_paired_focus_table.tex").write_text("\n".join(lines), encoding="utf-8")


def main():
    raw = sio.loadmat(RAW_FILE)["R"].astype(float)
    masks = sio.loadmat(MASK_FILE)
    labels_by_fold = []
    for fold in range(10):
        mask = masks[f"mask{fold}"].astype(float)
        labels_by_fold.append(raw[mask[:, 0] == 0])

    predictions = {name: split_cold_concat(read_matrix(path), masks) for name, path in MODEL_PATHS.items()}
    predictions["Global popularity"] = global_popularity_preds(raw, masks)

    rows = []
    for model, fold_preds in predictions.items():
        for fold, pred in enumerate(fold_preds):
            label = labels_by_fold[fold]
            row = {"model": model, "fold": fold}
            row.update(rank_metrics_cold(pred, label))
            row.update(regression_metrics(pred, label))
            rows.append(row)
    folds = pd.DataFrame(rows)
    folds.to_csv(OUT / "cafnet_dg_cold_fold_metrics.csv", index=False)
    summary = folds.groupby("model")[[m for m, _, _, _ in METRICS]].agg(["mean", "std"]).reset_index()
    paired = paired_tests(folds, target="CAFNet-DG")
    paired.to_csv(OUT / "cafnet_dg_paired_tests.csv", index=False)
    write_tables(summary, paired)
    print(summary.to_string(index=False))
    print("\nPaired summary vs CAFNet-D:")
    print(paired[paired["baseline"].eq("CAFNet-D")][["metric", "improvement_mean", "paired_t_p_holm", "wilcoxon_p_holm", "wilcoxon_holm_sig"]].to_string(index=False))


if __name__ == "__main__":
    main()
