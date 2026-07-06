from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
from sklearn.metrics import average_precision_score, mean_absolute_error, ndcg_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_outputs" / "cafnet_dg_completion_20260701" / "warm_full_fusion"
OUT.mkdir(parents=True, exist_ok=True)

RAW_FILE = ROOT / "data" / "raw_frequency_750.mat"
MASK_FILE = ROOT / "data" / "mask_mat_750.mat"

CAFNET_DIR_HINT = "10WSFULL_CAFNet_CAFNet"
CAFNET_D_DIR_HINT = "10WSFULL_CAFNetD_CAFNetDecoupled"


def find_full_prediction_dir(hint: str) -> Path:
    matches = sorted(
        [p for p in (ROOT / "result_WS").glob(f"{hint}*") if (p / "full_predictions").is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            f"No warm full prediction directory matching result_WS/{hint}*/full_predictions. "
            "Run analysis_outputs/cafnet_dg_completion_20260701/run_warm_full_prediction_jobs.ps1 first."
        )
    return matches[0]


def read_fold_matrix(run_dir: Path, fold: int) -> np.ndarray:
    path = run_dir / "full_predictions" / f"full_pred_fold{fold}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    mat = pd.read_csv(path, header=None).values.astype(np.float32)
    if mat.shape != (750, 994):
        raise ValueError(f"{path} has shape {mat.shape}, expected (750, 994)")
    if np.allclose(mat, 0):
        raise ValueError(f"{path} is all-zero; this is not a valid full prediction matrix")
    return mat


def precision_at(pos: np.ndarray, ranked: np.ndarray, k: int) -> float:
    return len(set(pos) & set(ranked[:k])) / float(k)


def recall_at(pos: np.ndarray, ranked: np.ndarray, k: int) -> float:
    return 0.0 if len(pos) == 0 else len(set(pos) & set(ranked[:k])) / float(len(pos))


def row_ap(pos: np.ndarray, candidates: np.ndarray, scores: np.ndarray) -> float:
    y = np.isin(candidates, pos).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return float(average_precision_score(y, scores[candidates]))


def fold_metrics(pred: np.ndarray, raw: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    test = mask == 0
    y_reg = raw[test].reshape(-1)
    s_reg = pred[test].reshape(-1)

    y_bin, s_bin = [], []
    aps, ndcgs, p1s, p15s, r15s = [], [], [], [], []
    for drug_idx in range(raw.shape[0]):
        pos = np.flatnonzero(test[drug_idx])
        if len(pos) == 0:
            continue
        train_row = raw[drug_idx] * mask[drug_idx]
        candidates = np.flatnonzero(train_row == 0)
        labels = np.isin(candidates, pos).astype(int)
        if labels.sum() == 0 or labels.sum() == len(labels):
            continue
        scores = pred[drug_idx]
        ranked = candidates[np.argsort(scores[candidates])[::-1]]
        y_bin.append(labels)
        s_bin.append(scores[candidates])
        aps.append(row_ap(pos, candidates, scores))
        ndcgs.append(float(ndcg_score(labels[None, :], scores[candidates][None, :], k=10)))
        p1s.append(precision_at(pos, ranked, 1))
        p15s.append(precision_at(pos, ranked, 15))
        r15s.append(recall_at(pos, ranked, 15))

    yy = np.concatenate(y_bin)
    ss = np.concatenate(s_bin)
    return {
        "MAP": float(np.nanmean(aps)),
        "AUROC": float(roc_auc_score(yy, ss)),
        "AUPR": float(average_precision_score(yy, ss)),
        "nDCG@10": float(np.nanmean(ndcgs)),
        "P@1": float(np.nanmean(p1s)),
        "P@15": float(np.nanmean(p15s)),
        "R@15": float(np.nanmean(r15s)),
        "Spearman": float(stats.spearmanr(y_reg, s_reg)[0]),
        "RMSE": float(np.sqrt(np.mean((y_reg - s_reg) ** 2))),
        "MAE": float(mean_absolute_error(y_reg, s_reg)),
    }


def subgroup_metrics(pred: np.ndarray, raw: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    train_rows = np.ones(raw.shape[0], dtype=bool)
    prevalence = (raw * mask != 0).sum(axis=0) / max(1, train_rows.sum())
    top100 = set(np.argsort(prevalence)[::-1][:100].astype(int).tolist())
    q1, q2 = np.quantile(prevalence, [1 / 3, 2 / 3])
    groups = {
        "nonhot100": np.array([i for i in range(raw.shape[1]) if int(i) not in top100], dtype=int),
        "rare": np.flatnonzero(prevalence <= q1),
        "middle": np.flatnonzero((prevalence > q1) & (prevalence <= q2)),
    }
    out = {}
    test = mask == 0
    for group, cols in groups.items():
        vals = []
        for drug_idx in range(raw.shape[0]):
            pos = np.intersect1d(np.flatnonzero(test[drug_idx]), cols, assume_unique=False)
            if len(pos) == 0:
                continue
            train_row = raw[drug_idx] * mask[drug_idx]
            candidates = np.array([c for c in cols if train_row[c] == 0], dtype=int)
            vals.append(row_ap(pos, candidates, pred[drug_idx]))
        out[f"{group}_AP"] = float(np.nanmean(vals)) if vals else np.nan
    return out


def main() -> None:
    cafnet_dir = find_full_prediction_dir(CAFNET_DIR_HINT)
    cafnet_d_dir = find_full_prediction_dir(CAFNET_D_DIR_HINT)
    raw = sio.loadmat(RAW_FILE)["R"].astype(float)
    masks = sio.loadmat(MASK_FILE)

    rows = []
    rho_rows = []
    for fold in range(10):
        mask = masks[f"mask{fold}"].astype(float)
        cafnet = read_fold_matrix(cafnet_dir, fold)
        cafnet_d = read_fold_matrix(cafnet_d_dir, fold)
        for model_name, pred in [("CAFNet", cafnet), ("CAFNet-D", cafnet_d), ("CAFNet-DG", 0.6 * cafnet_d + 0.4 * cafnet)]:
            metrics = fold_metrics(pred, raw, mask)
            metrics.update(subgroup_metrics(pred, raw, mask))
            metrics.update({"fold": fold, "model": model_name})
            rows.append(metrics)
        for rho in [round(x / 10, 1) for x in range(11)]:
            pred = rho * cafnet_d + (1.0 - rho) * cafnet
            metrics = fold_metrics(pred, raw, mask)
            metrics.update(subgroup_metrics(pred, raw, mask))
            metrics.update({"fold": fold, "rho": rho})
            rho_rows.append(metrics)

    by_fold = pd.DataFrame(rows)
    by_fold.to_csv(OUT / "warm_full_cafnet_dg_by_fold.csv", index=False)
    metric_cols = [c for c in by_fold.columns if c not in {"model", "fold"}]
    summary = by_fold.groupby("model")[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join([str(x) for x in col if str(x) != ""]) for col in summary.columns]
    summary.to_csv(OUT / "warm_full_cafnet_dg_summary.csv", index=False)

    rho_by_fold = pd.DataFrame(rho_rows)
    rho_by_fold.to_csv(OUT / "warm_rho_sensitivity_by_fold.csv", index=False)
    rho_metric_cols = [c for c in rho_by_fold.columns if c not in {"rho", "fold"}]
    rho_summary = rho_by_fold.groupby("rho")[rho_metric_cols].agg(["mean", "std"]).reset_index()
    rho_summary.columns = ["_".join([str(x) for x in col if str(x) != ""]) for col in rho_summary.columns]
    rho_summary.to_csv(OUT / "warm_rho_sensitivity_summary.csv", index=False)

    print(summary.to_string(index=False))
    print(rho_summary[["rho", "MAP_mean", "AUROC_mean", "AUPR_mean", "nDCG@10_mean", "nonhot100_AP_mean", "rare_AP_mean", "middle_AP_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
