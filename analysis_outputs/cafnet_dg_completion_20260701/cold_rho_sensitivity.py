from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_outputs" / "cafnet_dg_completion_20260701" / "rho_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

RAW_FILE = ROOT / "data" / "raw_frequency_750.mat"
MASK_FILE = ROOT / "data" / "blind_mask_mat_750.mat"
CAFNET_FILE = ROOT / "result_ICS" / (
    "10ICS_CAFNet_knn=5_wd=0.001_epoch=100_lamb=0.03_lr0.0004_dim=200_"
    "eps=0.5_DF=False_PCA=False_not-FC=False_cosine"
) / "blind_pred.csv"
CAFNET_D_FILE = ROOT / "result_ICS" / (
    "10cd3e100f10_CAFNetDecoupled_knn=5_wd=0.001_epoch=100_lamb=0.03_"
    "lr0.0004_dim=200_eps=0.5_DF=False_PCA=False_not-FC=False_cross=True_"
    "fusion=gate_gate=new_fa=0.5_gatdrop=0.0_mix=0.3_aw=1.0_fw=1.0_"
    "rw=0.05_popw=0.1_biasw=1.0_listw=0.1_abw=1.0_arw=1.0_cosine"
) / "blind_pred.csv"


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


def safe_ap(y: np.ndarray, s: np.ndarray) -> float:
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return float(average_precision_score(y, s))


def safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return float(roc_auc_score(y, s))


def row_metrics(scores: np.ndarray, labels: np.ndarray, cols: np.ndarray) -> dict[str, float]:
    y = (labels[cols] != 0).astype(int)
    s = scores[cols]
    if y.sum() == 0:
        return {"AP": np.nan, "AUROC": np.nan, "nDCG@10": np.nan, "P@15": np.nan, "R@15": np.nan}
    top15 = np.argsort(s)[::-1][: min(15, len(cols))]
    hits = y[top15].sum()
    return {
        "AP": safe_ap(y, s),
        "AUROC": safe_auc(y, s),
        "nDCG@10": float(ndcg_score(y[None, :], s[None, :], k=min(10, len(cols)))),
        "P@15": float(hits / min(15, len(cols))),
        "R@15": float(hits / y.sum()),
    }


def summarize_fold(pred_rows: np.ndarray, drug_ids: np.ndarray, raw: np.ndarray, train_rows: np.ndarray) -> dict[str, float]:
    prevalence = (raw[train_rows] != 0).mean(axis=0)
    top100 = set(np.argsort(prevalence)[::-1][:100].astype(int).tolist())
    all_cols = np.arange(raw.shape[1])
    nonhot_cols = np.array([i for i in all_cols if int(i) not in top100], dtype=int)
    q1, q2 = np.quantile(prevalence, [1 / 3, 2 / 3])
    rare_cols = np.flatnonzero(prevalence <= q1)
    middle_cols = np.flatnonzero((prevalence > q1) & (prevalence <= q2))

    groups = {
        "overall": all_cols,
        "nonhot100": nonhot_cols,
        "rare": rare_cols,
        "middle": middle_cols,
    }
    bucket: dict[str, list[float]] = {}
    global_y, global_s = [], []
    for local_idx, drug_idx in enumerate(drug_ids):
        labels = raw[int(drug_idx)]
        scores = pred_rows[local_idx]
        global_y.append((labels != 0).astype(int))
        global_s.append(scores)
        for group, cols in groups.items():
            metrics = row_metrics(scores, labels, cols)
            for metric, value in metrics.items():
                bucket.setdefault(f"{group}_{metric}", []).append(value)
        top15 = np.argsort(scores)[::-1][:15]
        bucket.setdefault("top15_hot_fraction", []).append(float(np.mean([int(x) in top100 for x in top15])))

    y_all = np.concatenate(global_y)
    s_all = np.concatenate(global_s)
    out = {
        "global_AUROC": safe_auc(y_all, s_all),
        "global_AUPR": safe_ap(y_all, s_all),
    }
    for key, values in bucket.items():
        out[key] = float(np.nanmean(np.asarray(values, dtype=float)))
    return out


def main() -> None:
    raw = sio.loadmat(RAW_FILE)["R"].astype(float)
    masks = sio.loadmat(MASK_FILE)
    cafnet_parts, drug_ids = split_cold(read_matrix(CAFNET_FILE), masks)
    cafnet_d_parts, _ = split_cold(read_matrix(CAFNET_D_FILE), masks)

    rows = []
    for rho in [round(x / 10, 1) for x in range(11)]:
        for fold in range(10):
            mask = masks[f"mask{fold}"].astype(float)
            train_rows = mask[:, 0] != 0
            pred = rho * cafnet_d_parts[fold] + (1.0 - rho) * cafnet_parts[fold]
            metrics = summarize_fold(pred, drug_ids[fold], raw, train_rows)
            metrics.update({"rho": rho, "fold": fold})
            rows.append(metrics)

    by_fold = pd.DataFrame(rows)
    by_fold.to_csv(OUT / "cold_rho_sensitivity_by_fold.csv", index=False)
    metric_cols = [c for c in by_fold.columns if c not in {"rho", "fold"}]
    summary = by_fold.groupby("rho")[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join([str(x) for x in col if str(x) != ""]) for col in summary.columns]
    summary.to_csv(OUT / "cold_rho_sensitivity_summary.csv", index=False)

    key_cols = [
        "rho",
        "overall_AP_mean",
        "global_AUROC_mean",
        "global_AUPR_mean",
        "overall_nDCG@10_mean",
        "nonhot100_AP_mean",
        "rare_AP_mean",
        "middle_AP_mean",
        "top15_hot_fraction_mean",
    ]
    print(summary[key_cols].to_string(index=False))

    best_rows = []
    for metric in key_cols[1:]:
        best = summary.sort_values(metric, ascending=False).iloc[0]
        best_rows.append({"metric": metric, "best_rho": best["rho"], "best_value": best[metric]})
    pd.DataFrame(best_rows).to_csv(OUT / "cold_rho_sensitivity_best_rho.csv", index=False)


if __name__ == "__main__":
    main()
