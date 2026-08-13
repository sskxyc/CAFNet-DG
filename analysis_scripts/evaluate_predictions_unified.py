from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    ndcg_score,
    roc_auc_score,
)


RANKING_METRICS = ["mAP", "AUROC", "AUPR", "nDCG@10", "P@1", "P@15", "R@15"]
ORDINAL_METRICS = ["Spearman", "RMSE", "MAE", "QWK", "Within1"]


def _safe_binary_metric(function, truth: np.ndarray, score: np.ndarray) -> float:
    if truth.size == 0 or np.unique(truth).size < 2:
        return float("nan")
    return float(function(truth, score))


def warm_candidate_mask(raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw)
    mask = np.asarray(mask)
    if raw.shape != mask.shape:
        raise ValueError("raw and mask must have identical shapes")
    heldout = mask == 0
    if np.any(heldout & (raw <= 0)):
        raise ValueError("warm mask contains a held-out original-zero entry")
    return heldout | (raw == 0)


def evaluate_association(
    prediction: np.ndarray,
    raw: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    ndcg_k: int = 10,
    precision_ks: tuple[int, ...] = (1, 15),
    recall_ks: tuple[int, ...] = (15,),
) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    raw = np.asarray(raw)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    if prediction.shape != raw.shape or raw.shape != candidate_mask.shape:
        raise ValueError("prediction, raw, and candidate_mask must align")
    if not np.all(np.isfinite(prediction[candidate_mask])):
        raise ValueError("association prediction contains non-finite candidate scores")

    truth = (raw > 0).astype(np.int8)
    flat_truth = truth[candidate_mask]
    flat_score = prediction[candidate_mask]
    metrics: dict[str, float] = {
        "AUROC": _safe_binary_metric(roc_auc_score, flat_truth, flat_score),
        "AUPR": _safe_binary_metric(average_precision_score, flat_truth, flat_score),
        "candidate_pairs": int(candidate_mask.sum()),
        "positive_pairs": int(flat_truth.sum()),
    }

    per_drug_ap = []
    per_drug_ndcg = []
    precision_values = {k: [] for k in precision_ks}
    recall_values = {k: [] for k in recall_ks}
    evaluable_drugs = 0
    for drug_idx in range(len(raw)):
        candidates = candidate_mask[drug_idx]
        y = truth[drug_idx, candidates]
        score = prediction[drug_idx, candidates]
        positives = int(y.sum())
        if positives == 0:
            continue
        evaluable_drugs += 1
        per_drug_ap.append(float(average_precision_score(y, score)))
        per_drug_ndcg.append(float(ndcg_score(y[None, :], score[None, :], k=min(ndcg_k, len(y)))))
        order = np.argsort(score, kind="stable")[::-1]
        for k in precision_ks:
            top = order[: min(k, len(order))]
            precision_values[k].append(float(y[top].sum() / max(1, len(top))))
        for k in recall_ks:
            top = order[: min(k, len(order))]
            recall_values[k].append(float(y[top].sum() / positives))

    metrics["evaluable_drugs"] = evaluable_drugs
    metrics["mAP"] = float(np.mean(per_drug_ap)) if per_drug_ap else float("nan")
    metrics[f"nDCG@{ndcg_k}"] = float(np.mean(per_drug_ndcg)) if per_drug_ndcg else float("nan")
    for k, values in precision_values.items():
        metrics[f"P@{k}"] = float(np.mean(values)) if values else float("nan")
    for k, values in recall_values.items():
        metrics[f"R@{k}"] = float(np.mean(values)) if values else float("nan")
    return metrics


def evaluate_warm_association(prediction: np.ndarray, raw: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    return evaluate_association(prediction, raw, warm_candidate_mask(raw, mask))


def evaluate_cold_association(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    truth = np.asarray(truth)
    return evaluate_association(prediction, truth, np.ones_like(truth, dtype=bool))


def evaluate_ordinal(
    prediction: np.ndarray,
    truth: np.ndarray,
    eligible_mask: np.ndarray | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if prediction.shape != truth.shape:
        raise ValueError("prediction and truth must align")
    if eligible_mask is None:
        eligible = truth > 0
    else:
        eligible = np.asarray(eligible_mask, dtype=bool) & (truth > 0)
    y_true = truth[eligible]
    y_pred = prediction[eligible]
    if y_true.size == 0:
        raise ValueError("no eligible nonzero ordinal labels")
    if not np.all(np.isfinite(y_pred)):
        raise ValueError("ordinal prediction contains non-finite eligible scores")

    rounded = np.clip(np.rint(y_pred), 1, 5).astype(int)
    integer_truth = y_true.astype(int)
    if not np.allclose(y_true, integer_truth):
        raise ValueError("ordinal truth contains non-integer labels")
    rho = float("nan") if len(y_true) < 2 else spearmanr(y_true, y_pred).statistic
    if np.unique(np.column_stack([integer_truth, rounded])).size < 2:
        qwk = float("nan")
    else:
        qwk = cohen_kappa_score(integer_truth, rounded, labels=[1, 2, 3, 4, 5], weights="quadratic")
    matrix = confusion_matrix(integer_truth, rounded, labels=[1, 2, 3, 4, 5])
    metrics = {
        "Spearman": float(rho),
        "RMSE": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "MAE": float(np.mean(np.abs(y_pred - y_true))),
        "QWK": float(qwk),
        "Within1": float(np.mean(np.abs(rounded - integer_truth) <= 1)),
        "ordinal_pairs": int(len(y_true)),
    }
    return metrics, matrix


def evaluate_warm_ordinal(prediction: np.ndarray, raw: np.ndarray, mask: np.ndarray):
    return evaluate_ordinal(prediction, raw, eligible_mask=np.asarray(mask) == 0)


def evaluate_cold_ordinal(prediction: np.ndarray, truth: np.ndarray):
    return evaluate_ordinal(prediction, truth, eligible_mask=np.asarray(truth) > 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified CAFNet-DG fold evaluator")
    parser.add_argument("--scenario", choices=["warm", "cold"], required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--truth", required=True, help="Raw .mat for warm or truth .npy for cold")
    parser.add_argument("--mask", help="Warm mask .mat")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--frequency_prediction")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    association_prediction = np.load(args.prediction)
    frequency_prediction = np.load(args.frequency_prediction) if args.frequency_prediction else association_prediction
    if args.scenario == "warm":
        raw = scipy.io.loadmat(args.truth)["R"]
        masks = scipy.io.loadmat(args.mask)
        mask = masks[f"mask{args.fold}"]
        ranking = evaluate_warm_association(association_prediction, raw, mask)
        ordinal, confusion = evaluate_warm_ordinal(frequency_prediction, raw, mask)
    else:
        truth = np.load(args.truth)
        ranking = evaluate_cold_association(association_prediction, truth)
        ordinal, confusion = evaluate_cold_ordinal(frequency_prediction, truth)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    row = {"scenario": args.scenario, "fold": args.fold, **ranking, **ordinal}
    pd.DataFrame([row]).to_csv(output, index=False)
    np.save(output.with_name(output.stem + "_confusion.npy"), confusion)
    output.with_name(output.stem + "_contract.json").write_text(
        json.dumps(
            {
                "ranking_candidate_contract": "warm: heldout positives plus original zeros; cold: all ADRs",
                "ordinal_contract": "true nonzero heldout frequency labels only",
                "ndcg": "sklearn.metrics.ndcg_score with k=10",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
