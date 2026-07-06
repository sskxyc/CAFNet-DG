from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_outputs" / "cafnet_dg_per_drug"
OUT.mkdir(parents=True, exist_ok=True)

RAW_FILE = ROOT / "data" / "raw_frequency_750.mat"
MASK_FILE = ROOT / "data" / "blind_mask_mat_750.mat"

MODEL_FILES = {
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


def mat_names(arr: np.ndarray) -> list[str]:
    out = []
    for x in arr.reshape(-1):
        while isinstance(x, np.ndarray):
            x = x.reshape(-1)[0]
        out.append(str(x))
    return out


def read_matrix(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, header=None).values.astype(np.float32)


def split_cold(full: np.ndarray, masks: dict[str, np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    parts, drug_ids, start = [], [], 0
    for fold in range(10):
        mask = masks[f"mask{fold}"].astype(float)
        ids = np.flatnonzero(mask[:, 0] == 0)
        n = len(ids)
        parts.append(full[start : start + n])
        drug_ids.append(ids)
        start += n
    if start != len(full):
        raise ValueError(f"Consumed {start} rows but matrix has {len(full)} rows")
    return parts, drug_ids


def ap_for_cols(scores: np.ndarray, labels: np.ndarray, cols: np.ndarray) -> float:
    if len(cols) == 0:
        return np.nan
    y = (labels[cols] != 0).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return float(average_precision_score(y, scores[cols]))


def auc_for_cols(scores: np.ndarray, labels: np.ndarray, cols: np.ndarray) -> float:
    if len(cols) == 0:
        return np.nan
    y = (labels[cols] != 0).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return float(roc_auc_score(y, scores[cols]))


def ndcg_for_cols(scores: np.ndarray, labels: np.ndarray, cols: np.ndarray, k: int = 10) -> float:
    y = (labels[cols] != 0).astype(int)
    if y.sum() == 0:
        return np.nan
    return float(ndcg_score(y[None, :], scores[cols][None, :], k=min(k, len(cols))))


def topk_hit_rate(scores: np.ndarray, labels: np.ndarray, cols: np.ndarray, k: int = 15) -> float:
    if len(cols) == 0:
        return np.nan
    kk = min(k, len(cols))
    top = cols[np.argsort(scores[cols])[::-1][:kk]]
    return float(np.mean(labels[top] != 0))


def recall_at(scores: np.ndarray, labels: np.ndarray, cols: np.ndarray, k: int = 15) -> float:
    pos = cols[labels[cols] != 0]
    if len(pos) == 0:
        return np.nan
    kk = min(k, len(cols))
    top = cols[np.argsort(scores[cols])[::-1][:kk]]
    return float(len(set(pos) & set(top)) / len(pos))


def top_positive_names(scores: np.ndarray, labels: np.ndarray, names: list[str], cols: np.ndarray, k: int = 5) -> str:
    pos = cols[labels[cols] != 0]
    if len(pos) == 0:
        return ""
    top = pos[np.argsort(scores[pos])[::-1][:k]]
    return "; ".join(names[i] for i in top)


def main() -> None:
    mat = sio.loadmat(RAW_FILE)
    raw = mat["R"].astype(float)
    drug_names = mat_names(mat["drugs"])
    side_names = mat_names(mat["sideeffects"])
    masks = sio.loadmat(MASK_FILE)

    model_parts = {}
    fold_drug_ids = None
    for model, path in MODEL_FILES.items():
        parts, ids = split_cold(read_matrix(path), masks)
        model_parts[model] = parts
        if fold_drug_ids is None:
            fold_drug_ids = ids

    rows = []
    for fold in range(10):
        mask = masks[f"mask{fold}"].astype(float)
        train_rows = mask[:, 0] != 0
        prevalence = (raw[train_rows] != 0).mean(axis=0)
        q1, q2 = np.quantile(prevalence, [1 / 3, 2 / 3])
        rare = np.flatnonzero(prevalence <= q1)
        middle = np.flatnonzero((prevalence > q1) & (prevalence <= q2))
        frequent = np.flatnonzero(prevalence > q2)
        top100_hot = set(np.argsort(prevalence)[::-1][:100].astype(int).tolist())
        non_hot = np.array([i for i in range(raw.shape[1]) if i not in top100_hot], dtype=int)

        for local, drug_idx in enumerate(fold_drug_ids[fold]):
            labels = raw[int(drug_idx)]
            pos_count = int(np.count_nonzero(labels))
            if pos_count == 0:
                continue
            for model, parts in model_parts.items():
                scores = parts[fold][local]
                top15 = np.argsort(scores)[::-1][:15]
                rows.append(
                    {
                        "model": model,
                        "fold": fold,
                        "drug_index": int(drug_idx),
                        "drug_name": drug_names[int(drug_idx)],
                        "positive_count": pos_count,
                        "AP": ap_for_cols(scores, labels, np.arange(raw.shape[1])),
                        "AUROC": auc_for_cols(scores, labels, np.arange(raw.shape[1])),
                        "nDCG@10": ndcg_for_cols(scores, labels, np.arange(raw.shape[1]), 10),
                        "P@15": topk_hit_rate(scores, labels, np.arange(raw.shape[1]), 15),
                        "R@15": recall_at(scores, labels, np.arange(raw.shape[1]), 15),
                        "rare_AP": ap_for_cols(scores, labels, rare),
                        "middle_AP": ap_for_cols(scores, labels, middle),
                        "frequent_AP": ap_for_cols(scores, labels, frequent),
                        "nonhot100_AP": ap_for_cols(scores, labels, non_hot),
                        "top15_hot_fraction": float(np.mean([int(x) in top100_hot for x in top15])),
                        "top_ranked_true_positives": top_positive_names(scores, labels, side_names, np.arange(raw.shape[1]), 5),
                        "top_ranked_nonhot_true_positives": top_positive_names(scores, labels, side_names, non_hot, 5),
                    }
                )

    per_model = pd.DataFrame(rows)
    per_model.to_csv(OUT / "per_drug_model_metrics.csv", index=False)

    wide = per_model.pivot_table(
        index=["fold", "drug_index", "drug_name", "positive_count"],
        columns="model",
        values=["AP", "AUROC", "nDCG@10", "P@15", "R@15", "rare_AP", "middle_AP", "frequent_AP", "nonhot100_AP", "top15_hot_fraction"],
    )
    wide.columns = [f"{metric}_{model}" for metric, model in wide.columns]
    wide = wide.reset_index()
    for metric in ["AP", "AUROC", "nDCG@10", "P@15", "R@15", "rare_AP", "middle_AP", "frequent_AP", "nonhot100_AP"]:
        wide[f"delta_DG_minus_D_{metric}"] = wide[f"{metric}_CAFNet-DG"] - wide[f"{metric}_CAFNet-D"]
        wide[f"delta_DG_minus_CAFNet_{metric}"] = wide[f"{metric}_CAFNet-DG"] - wide[f"{metric}_CAFNet"]
    wide["delta_DG_minus_D_hot_fraction"] = wide["top15_hot_fraction_CAFNet-DG"] - wide["top15_hot_fraction_CAFNet-D"]
    wide.to_csv(OUT / "per_drug_delta_metrics.csv", index=False)

    success = wide.sort_values(
        ["delta_DG_minus_D_rare_AP", "delta_DG_minus_D_middle_AP", "delta_DG_minus_D_nonhot100_AP"],
        ascending=False,
    ).head(25)
    failure = wide.sort_values(
        ["delta_DG_minus_D_rare_AP", "delta_DG_minus_D_middle_AP", "delta_DG_minus_D_nonhot100_AP"],
        ascending=True,
    ).head(25)
    success.to_csv(OUT / "top25_drug_specific_successes_vs_cafnet_d.csv", index=False)
    failure.to_csv(OUT / "top25_drug_specific_failures_vs_cafnet_d.csv", index=False)

    summary_rows = []
    for metric in ["AP", "rare_AP", "middle_AP", "nonhot100_AP", "top15_hot_fraction"]:
        for comparison in ["D", "CAFNet"]:
            col = f"delta_DG_minus_{comparison}_{metric}" if comparison == "CAFNet" else f"delta_DG_minus_D_{metric}"
            if col not in wide.columns:
                continue
            vals = wide[col].dropna().to_numpy(float)
            summary_rows.append(
                {
                    "delta_metric": col,
                    "n_drugs": int(len(vals)),
                    "mean_delta": float(np.mean(vals)),
                    "median_delta": float(np.median(vals)),
                    "fraction_positive": float(np.mean(vals > 0)),
                    "q25": float(np.quantile(vals, 0.25)),
                    "q75": float(np.quantile(vals, 0.75)),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "per_drug_delta_summary.csv", index=False)

    report = [
        "# CAFNet-DG Per-Drug Success/Failure Analysis",
        "",
        "This analysis identifies which cold-start test drugs benefit from CAFNet-DG residual fusion relative to CAFNet-D.",
        "",
        "## Delta Summary",
        "",
    ]
    for _, row in summary.iterrows():
        report.append(
            f"- `{row['delta_metric']}`: mean `{row['mean_delta']:.4f}`, median `{row['median_delta']:.4f}`, "
            f"fraction positive `{row['fraction_positive']:.3f}`, n `{int(row['n_drugs'])}`"
        )
    report.extend(
        [
            "",
            "## Strongest Success Examples vs CAFNet-D",
            "",
        ]
    )
    for _, row in success.head(10).iterrows():
        report.append(
            f"- {row['drug_name']} (fold {int(row['fold'])}): rare AP delta `{row['delta_DG_minus_D_rare_AP']:.4f}`, "
            f"middle AP delta `{row['delta_DG_minus_D_middle_AP']:.4f}`, non-hot100 AP delta `{row['delta_DG_minus_D_nonhot100_AP']:.4f}`"
        )
    report.extend(["", "## Strongest Failure Examples vs CAFNet-D", ""])
    for _, row in failure.head(10).iterrows():
        report.append(
            f"- {row['drug_name']} (fold {int(row['fold'])}): rare AP delta `{row['delta_DG_minus_D_rare_AP']:.4f}`, "
            f"middle AP delta `{row['delta_DG_minus_D_middle_AP']:.4f}`, non-hot100 AP delta `{row['delta_DG_minus_D_nonhot100_AP']:.4f}`"
        )
    report.extend(
        [
            "",
            "## Interpretation",
            "",
            "Use this table to discuss where residual fusion helps and where it fails. It should be reported as a diagnostic analysis, not as a causal mechanism claim.",
        ]
    )
    (OUT / "PER_DRUG_SUCCESS_FAILURE_REPORT_20260701.md").write_text("\n".join(report), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote outputs to {OUT}")


if __name__ == "__main__":
    main()

