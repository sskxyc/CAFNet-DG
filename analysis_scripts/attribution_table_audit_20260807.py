from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs" / "submission_rigor_audit_20260807" / "attribution_recomputed"
RAW_FILE = ROOT / "data" / "raw_frequency_750.mat"
MASK_FILE = ROOT / "data" / "blind_mask_mat_750.mat"

MODEL_FILES = {
    "A3Net": ROOT
    / "result_ICS"
    / (
        "10ICS_A3_Net_knn=5_wd=0.001_epoch=100_lamb=0.03_lr0.0004_dim=200_"
        "eps=0.5_DF=False_PCA=False_not-FC=False_cosine"
    )
    / "blind_pred.csv",
    "CAFNet": ROOT
    / "result_ICS"
    / (
        "10ICS_CAFNet_knn=5_wd=0.001_epoch=100_lamb=0.03_lr0.0004_dim=200_"
        "eps=0.5_DF=False_PCA=False_not-FC=False_cosine"
    )
    / "blind_pred.csv",
    "CAFNet-D": ROOT
    / "result_ICS"
    / (
        "10cd3e100f10_CAFNetDecoupled_knn=5_wd=0.001_epoch=100_lamb=0.03_"
        "lr0.0004_dim=200_eps=0.5_DF=False_PCA=False_not-FC=False_cross=True_"
        "fusion=gate_gate=new_fa=0.5_gatdrop=0.0_mix=0.3_aw=1.0_fw=1.0_"
        "rw=0.05_popw=0.1_biasw=1.0_listw=0.1_abw=1.0_arw=1.0_cosine"
    )
    / "blind_pred.csv",
    "CAFNet-DG": ROOT
    / "result_ICS"
    / "10cafnet_dg_ensemble06_cafnetd04_cafnet"
    / "blind_pred.csv",
    "w/o association residual": ROOT
    / "result_ICS"
    / "10cd3noresf10_CAFNetDecoupled"
    / "blind_pred.csv",
    "w/o bias/popularity prior": ROOT
    / "result_ICS"
    / "10cd3abl_nobias_f10_CAFNetDecoupled"
    / "blind_pred.csv",
}


def read_matrix(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    matrix = pd.read_csv(path, header=None).to_numpy(dtype=np.float64)
    if matrix.shape[1] != 994:
        raise ValueError(f"Unexpected prediction shape for {path}: {matrix.shape}")
    return matrix


def split_predictions(
    matrix: np.ndarray, masks: dict[str, np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    parts: list[np.ndarray] = []
    drug_ids: list[np.ndarray] = []
    start = 0
    for fold in range(10):
        mask = masks[f"mask{fold}"].astype(float)
        ids = np.flatnonzero(mask[:, 0] == 0)
        stop = start + len(ids)
        parts.append(matrix[start:stop])
        drug_ids.append(ids)
        start = stop
    if start != len(matrix):
        raise ValueError(f"Consumed {start} prediction rows, but matrix has {len(matrix)}")
    return parts, drug_ids


def rank_bins(values: np.ndarray, n_bins: int) -> list[np.ndarray]:
    """Return deterministic equal-size bins from lowest to highest prevalence."""
    order = np.argsort(values, kind="stable")
    return [np.asarray(part, dtype=int) for part in np.array_split(order, n_bins)]


def ap_subset(scores: np.ndarray, labels: np.ndarray, cols: np.ndarray) -> float:
    y = labels[cols]
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return float(average_precision_score(y, scores[cols]))


def ndcg10(scores: np.ndarray, labels: np.ndarray) -> float:
    if labels.sum() == 0:
        return np.nan
    return float(ndcg_score(labels[None, :], scores[None, :], k=10))


def matched_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    prevalence: np.ndarray,
    prevalence_bins: list[np.ndarray],
    seed: int,
    controls_per_positive: int = 5,
) -> tuple[float, float, int]:
    bin_id = np.empty(len(labels), dtype=int)
    for index, cols in enumerate(prevalence_bins):
        bin_id[cols] = index

    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(seed)
    positive_scores: list[float] = []
    control_scores: list[float] = []

    for positive in positives:
        candidates = negatives[bin_id[negatives] == bin_id[positive]]
        if len(candidates) == 0:
            continue
        distance = np.abs(prevalence[candidates] - prevalence[positive])
        nearest = candidates[np.argsort(distance, kind="stable")]
        pool = nearest[: min(len(nearest), max(controls_per_positive * 4, controls_per_positive))]
        count = min(controls_per_positive, len(pool))
        selected = rng.choice(pool, size=count, replace=False)
        positive_scores.extend([float(scores[positive])] * count)
        control_scores.extend(scores[selected].astype(float).tolist())

    if not positive_scores or not control_scores:
        return np.nan, np.nan, 0
    y = np.concatenate([np.ones(len(positive_scores)), np.zeros(len(control_scores))])
    s = np.concatenate([np.asarray(positive_scores), np.asarray(control_scores)])
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s)), len(positive_scores)


def holm(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values, dtype=float)
    running = 0.0
    n = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (n - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = sio.loadmat(RAW_FILE)["R"].astype(float)
    labels_full = (raw != 0).astype(int)
    masks = sio.loadmat(MASK_FILE)

    model_parts: dict[str, list[np.ndarray]] = {}
    fold_drug_ids: list[np.ndarray] | None = None
    for model, path in MODEL_FILES.items():
        parts, ids = split_predictions(read_matrix(path), masks)
        model_parts[model] = parts
        if fold_drug_ids is None:
            fold_drug_ids = ids
        elif any(not np.array_equal(a, b) for a, b in zip(fold_drug_ids, ids)):
            raise ValueError(f"Fold alignment differs for {model}")
    assert fold_drug_ids is not None

    rows: list[dict[str, float | int | str]] = []
    for fold in range(10):
        mask = masks[f"mask{fold}"].astype(float)
        train_rows = mask[:, 0] != 0
        prevalence = labels_full[train_rows].sum(axis=0).astype(float)
        rare, middle, frequent = rank_bins(prevalence, 3)
        matched_bins = rank_bins(prevalence, 5)

        for local_index, drug_index in enumerate(fold_drug_ids[fold]):
            labels = labels_full[int(drug_index)]
            if labels.sum() == 0:
                continue
            scores_by_model = {
                model: parts[fold][local_index] for model, parts in model_parts.items()
            }
            scores_by_model["Global popularity"] = prevalence

            for model, scores in scores_by_model.items():
                matched_auc, matched_aupr, matched_pairs = matched_metrics(
                    scores,
                    labels,
                    prevalence,
                    matched_bins,
                    seed=42 + fold * 100_000 + int(drug_index),
                )
                rows.append(
                    {
                        "fold": fold,
                        "drug_index": int(drug_index),
                        "model": model,
                        "macro_AP": ap_subset(scores, labels, np.arange(raw.shape[1])),
                        "macro_nDCG@10": ndcg10(scores, labels),
                        "frequent_AP": ap_subset(scores, labels, frequent),
                        "middle_AP": ap_subset(scores, labels, middle),
                        "rare_AP": ap_subset(scores, labels, rare),
                        "matched_AUROC": matched_auc,
                        "matched_AUPR": matched_aupr,
                        "matched_positive_control_pairs": matched_pairs,
                    }
                )

    per_drug = pd.DataFrame(rows)
    per_drug.to_csv(OUT / "attribution_metrics_by_drug.csv", index=False)

    metrics = [
        "macro_AP",
        "macro_nDCG@10",
        "frequent_AP",
        "middle_AP",
        "rare_AP",
        "matched_AUROC",
        "matched_AUPR",
    ]
    summary_rows = []
    for model, group in per_drug.groupby("model", sort=False):
        row: dict[str, float | int | str] = {"model": model}
        for metric in metrics:
            values = group[metric].dropna()
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_n"] = int(len(values))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "attribution_summary.csv", index=False)

    paired_rows = []
    target = per_drug[per_drug["model"] == "CAFNet-DG"]
    for comparator in ["CAFNet-D", "CAFNet", "Global popularity"]:
        baseline = per_drug[per_drug["model"] == comparator]
        merged = target.merge(
            baseline,
            on=["fold", "drug_index"],
            suffixes=("_target", "_baseline"),
            validate="one_to_one",
        )
        start = len(paired_rows)
        for metric in metrics:
            paired = merged[[f"{metric}_target", f"{metric}_baseline"]].dropna()
            delta = paired.iloc[:, 0].to_numpy() - paired.iloc[:, 1].to_numpy()
            if np.allclose(delta, 0):
                statistic, p_value = 0.0, 1.0
            else:
                test = stats.wilcoxon(delta, alternative="two-sided", zero_method="wilcox")
                statistic, p_value = float(test.statistic), float(test.pvalue)
            paired_rows.append(
                {
                    "comparison": f"CAFNet-DG vs {comparator}",
                    "metric": metric,
                    "n_drugs": len(delta),
                    "mean_delta": float(np.mean(delta)),
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p": p_value,
                }
            )
        stop = len(paired_rows)
        adjusted = holm(np.asarray([row["wilcoxon_p"] for row in paired_rows[start:stop]]))
        for row, value in zip(paired_rows[start:stop], adjusted):
            row["wilcoxon_p_holm"] = float(value)
            row["holm_significant"] = bool(value < 0.05)
    pd.DataFrame(paired_rows).to_csv(OUT / "attribution_paired_tests.csv", index=False)

    protocol = (
        "All rows were recomputed from the same saved cold-start prediction matrices and the same "
        "drug-disjoint folds. Within each training fold, side effects were ranked by positive count "
        "and split into deterministic equal-size prevalence tertiles. Matched controls used five "
        "same-drug negatives per positive from the same prevalence quintile, preferring the nearest "
        "training prevalence and using seed 42 for deterministic tie sampling. Metrics are macro-"
        "averaged across evaluable held-out drugs."
    )
    (OUT / "PROTOCOL.md").write_text(protocol + "\n", encoding="utf-8")
    print(summary[["model"] + [f"{metric}_mean" for metric in metrics]].to_string(index=False))


if __name__ == "__main__":
    main()
