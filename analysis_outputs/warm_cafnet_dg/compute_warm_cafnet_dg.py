from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
from sklearn.metrics import average_precision_score, mean_absolute_error, ndcg_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_outputs" / "warm_cafnet_dg"
OUT.mkdir(parents=True, exist_ok=True)

RAW_FILE = ROOT / "data" / "raw_frequency_750.mat"
MASK_FILE = ROOT / "data" / "mask_mat_750.mat"

CAFNET_PRED = ROOT / "result_WS" / "lr0.0004-2" / (
    "10WS_CAFNet_knn=10_wd=0.001_epoch=100_lamb=0.03_lr0.0004_dim=200_"
    "eps=0.5_DF=False_PCA=False_not-FC=False_cosine_abl=CA_gate_new_loss=focal"
) / "pred_result.csv"
CAFNET_D_PRED = ROOT / "result_WS" / (
    "10d3v3e100_CAFNetDecoupled_knn=10_wd=0.001_epoch=100_lamb=0.03_"
    "lr0.0004_dim=200_eps=0.5_DF=False_PCA=False_not-FC=False_cosine_"
    "abl=CA_gate_new_loss=focal_mix=0.3_aw=1.0_fw=1.0_rw=0.05_popw=0.1_"
    "biasw=1.0_listw=0.1_abw=1.0_arw=1.0"
) / "pred_result.csv"
OUT_PRED_DIR = ROOT / "result_WS" / "10cafnet_dg_warm_ensemble06_cafnetd04_cafnet"


def read_matrix(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, header=None).values.astype(np.float32)


def precision_at(pos: np.ndarray, ranked: np.ndarray, k: int) -> float:
    return len(set(pos) & set(ranked[:k])) / float(k)


def recall_at(pos: np.ndarray, ranked: np.ndarray, k: int) -> float:
    return 0.0 if len(pos) == 0 else len(set(pos) & set(ranked[:k])) / float(len(pos))


def average_precision_for_scores(pos: np.ndarray, candidates: np.ndarray, scores: np.ndarray) -> float:
    y = np.isin(candidates, pos).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return float(average_precision_score(y, scores[candidates]))


def fold_metrics(pred: np.ndarray, raw: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    test = mask == 0
    pos_scores = pred[test].reshape(-1)
    neg_scores = pred[raw == 0].reshape(-1)
    y_bin = np.r_[np.ones(len(pos_scores), dtype=int), np.zeros(len(neg_scores), dtype=int)]
    s_bin = np.r_[pos_scores, neg_scores]
    y = raw[test].reshape(-1)
    s = pos_scores
    out: dict[str, float] = {
        "auc_all": float(roc_auc_score(y_bin, s_bin)),
        "aupr_all": float(average_precision_score(y_bin, s_bin)),
        "spearman": float(stats.spearmanr(y, s)[0]),
        "rMSE": float(np.sqrt(np.mean((y - s) ** 2))),
        "MAE": float(mean_absolute_error(y, s)),
    }

    aps, ndcgs, p1s, p5s, p10s, p15s, r1s, r5s, r10s, r15s = ([] for _ in range(10))
    drug_aucs, drug_auprs = [], []
    for drug_idx in range(raw.shape[0]):
        pos = np.flatnonzero(test[drug_idx])
        if len(pos) == 0:
            continue
        train_data_row = raw[drug_idx] * mask[drug_idx]
        candidates = np.flatnonzero(train_data_row == 0)
        neg = np.setdiff1d(candidates, pos, assume_unique=False)
        if len(neg) == 0:
            continue
        scores = pred[drug_idx]
        ranked = candidates[np.argsort(scores[candidates])[::-1]]
        aps.append(average_precision_for_scores(pos, candidates, scores))
        y_row = np.isin(candidates, pos).astype(int)
        drug_aucs.append(float(roc_auc_score(y_row, scores[candidates])))
        drug_auprs.append(float(average_precision_score(y_row, scores[candidates])))
        ndcgs.append(float(ndcg_score(y_row[None, :], scores[candidates][None, :], k=10)))
        for k, store_p, store_r in [(1, p1s, r1s), (5, p5s, r5s), (10, p10s, r10s), (15, p15s, r15s)]:
            store_p.append(precision_at(pos, ranked, k))
            store_r.append(recall_at(pos, ranked, k))

    out.update(
        {
            "drugAUC": float(np.nanmean(drug_aucs)),
            "drugAUPR": float(np.nanmean(drug_auprs)),
            "MAP": float(np.nanmean(aps)),
            "nDCG": float(np.nanmean(ndcgs)),
            "P1": float(np.nanmean(p1s)),
            "P5": float(np.nanmean(p5s)),
            "P10": float(np.nanmean(p10s)),
            "P15": float(np.nanmean(p15s)),
            "R1": float(np.nanmean(r1s)),
            "R5": float(np.nanmean(r5s)),
            "R10": float(np.nanmean(r10s)),
            "R15": float(np.nanmean(r15s)),
        }
    )
    return out


def holm(p_values: list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["MAP", "auc_all", "aupr_all", "nDCG", "P1", "P15", "R15", "spearman", "rMSE", "MAE"]
    lower = {"rMSE", "MAE"}
    rows = []
    target = df[df.model == "CAFNet-DG"].sort_values("fold")
    for baseline in ["CAFNet", "CAFNet-D"]:
        base = df[df.model == baseline].sort_values("fold")
        for metric in metrics:
            a = target[metric].to_numpy(float)
            b = base[metric].to_numpy(float)
            diff = (b - a) if metric in lower else (a - b)
            try:
                p = float(stats.wilcoxon(diff, zero_method="wilcox").pvalue) if not np.allclose(diff, 0) else 1.0
            except ValueError:
                p = 1.0
            rows.append(
                {
                    "comparison": f"CAFNet-DG vs {baseline}",
                    "metric": metric,
                    "target_mean": float(np.mean(a)),
                    "baseline_mean": float(np.mean(b)),
                    "improvement_mean": float(np.mean(diff)),
                    "wilcoxon_p": p,
                }
            )
    out = pd.DataFrame(rows)
    parts = []
    for _, sub in out.groupby("comparison", sort=False):
        sub = sub.copy()
        sub["wilcoxon_p_holm"] = holm(sub["wilcoxon_p"].tolist())
        sub["holm_sig"] = sub["wilcoxon_p_holm"] < 0.05
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    raw = sio.loadmat(RAW_FILE)["R"].astype(float)
    masks = sio.loadmat(MASK_FILE)
    cafnet = read_matrix(CAFNET_PRED)
    cafnet_d = read_matrix(CAFNET_D_PRED)
    if cafnet.shape != cafnet_d.shape:
        raise ValueError((cafnet.shape, cafnet_d.shape))
    cafnet_dg = 0.6 * cafnet_d + 0.4 * cafnet

    OUT_PRED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cafnet_dg).to_csv(OUT_PRED_DIR / "pred_result.csv", header=False, index=False)
    (OUT_PRED_DIR / "README.txt").write_text(
        "Warm-start CAFNet-DG fixed residual fusion.\n"
        "score = 0.6 * warm CAFNet-D + 0.4 * warm CAFNet.\n"
        f"CAFNet source: {CAFNET_PRED}\n"
        f"CAFNet-D source: {CAFNET_D_PRED}\n",
        encoding="utf-8",
    )

    rows = []
    for fold in range(10):
        mask = masks[f"mask{fold}"].astype(float)
        for name, mat in [("CAFNet", cafnet), ("CAFNet-D", cafnet_d), ("CAFNet-DG", cafnet_dg)]:
            m = fold_metrics(mat, raw, mask)
            m["model"] = name
            m["fold"] = fold
            rows.append(m)
    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(OUT / "warm_cafnet_dg_fold_metrics.csv", index=False)

    metric_cols = [c for c in fold_df.columns if c not in {"model", "fold"}]
    summary = fold_df.groupby("model")[metric_cols].agg(["mean", "std"]).reset_index()
    summary.to_csv(OUT / "warm_cafnet_dg_summary.csv", index=False)
    paired = paired_tests(fold_df)
    paired.to_csv(OUT / "warm_cafnet_dg_paired_tests.csv", index=False)

    report = [
        "# Warm-Start CAFNet-DG Fixed Fusion Check",
        "",
        "CAFNet-DG is computed as `0.6 * CAFNet-D + 0.4 * CAFNet` using available warm-start 10-fold prediction matrices.",
        "",
        "## Summary",
        "",
    ]
    flat = summary.copy()
    flat.columns = ["_".join([str(x) for x in col if str(x) != ""]) for col in flat.columns]
    for _, row in flat.iterrows():
        report.append(
            f"- {row['model']}: MAP `{row['MAP_mean']:.4f}`, AUROC `{row['auc_all_mean']:.4f}`, "
            f"AUPR `{row['aupr_all_mean']:.4f}`, Spearman `{row['spearman_mean']:.4f}`, "
            f"RMSE `{row['rMSE_mean']:.4f}`, MAE `{row['MAE_mean']:.4f}`"
        )
    report.extend(["", "## Interpretation", "", "Use this as a branch-level warm-start check. The main CAFNet-DG evidence remains cold-start unless the manuscript explicitly adds this warm-start fusion result."])
    (OUT / "WARM_CAFNET_DG_REPORT_20260701.md").write_text("\n".join(report), encoding="utf-8")
    print(flat.to_string(index=False))
    print()
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
