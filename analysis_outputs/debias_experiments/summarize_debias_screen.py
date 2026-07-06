from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_outputs" / "debias_experiments"
MASK_FILE = ROOT / "data" / "blind_mask_mat_750.mat"
RAW_FILE = ROOT / "data" / "raw_frequency_750.mat"

CAFNET_D_DIR = ROOT / "result_ICS" / (
    "10cd3e100f10_CAFNetDecoupled_knn=5_wd=0.001_epoch=100_lamb=0.03_"
    "lr0.0004_dim=200_eps=0.5_DF=False_PCA=False_not-FC=False_cross=True_"
    "fusion=gate_gate=new_fa=0.5_gatdrop=0.0_mix=0.3_aw=1.0_fw=1.0_"
    "rw=0.05_popw=0.1_biasw=1.0_listw=0.1_abw=1.0_arw=1.0_cosine"
)
CAFNET_DIR = ROOT / "result_ICS" / (
    "10ICS_CAFNet_knn=5_wd=0.001_epoch=100_lamb=0.03_lr0.0004_dim=200_"
    "eps=0.5_DF=False_PCA=False_not-FC=False_cosine"
)

SCREEN_DIRS = {
    "debias pop0.02 no-res": ROOT / "result_ICS" / "10cd3deb_pop002_nores_f3e50_CAFNetDecoupled",
    "debias pop0.05 no-res": ROOT / "result_ICS" / "10cd3deb_pop005_nores_f3e50_CAFNetDecoupled",
    "debias pop0.02 residual": ROOT / "result_ICS" / "10cd3deb_pop002_res_f3e50_CAFNetDecoupled",
    "debias weighted rb1.5": ROOT / "result_ICS" / "10cd3deb_wg1_rb15_pop002_nores_f3e50_CAFNetDecoupled",
    "debias weighted rb2.0": ROOT / "result_ICS" / "10cd3deb_wg1_rb20_pop002_nores_f3e50_CAFNetDecoupled",
    "matched-BPR pop0.05 no-res": ROOT / "result_ICS" / "10cd3mbpr_pop005_nores_f3e50_CAFNetDecoupled",
    "matched-BPR pop0.10 no-res": ROOT / "result_ICS" / "10cd3mbpr_pop010_nores_f3e50_CAFNetDecoupled",
    "matched-BPR pop0.05 residual": ROOT / "result_ICS" / "10cd3mbpr_pop005_res_f3e50_CAFNetDecoupled",
}


def read_csv_matrix(path: Path) -> np.ndarray:
    return pd.read_csv(path, header=None).values.astype(float)


def test_rows_by_fold(max_folds: int = 3):
    masks = sio.loadmat(MASK_FILE)
    rows = []
    train_rows = {}
    fold_of_local = []
    for fold in range(max_folds):
        mask = masks[f"mask{fold}"].astype(float)
        train = mask[:, 0] > 0
        test = ~train
        idx = np.where(test)[0]
        rows.extend(idx.tolist())
        train_rows[fold] = train
        fold_of_local.extend([fold] * len(idx))
    return np.array(rows, dtype=int), train_rows, np.array(fold_of_local, dtype=int)


def ndcg_at_k(y, score, k=10):
    order = np.argsort(score)[::-1][:k]
    gains = y[order].astype(float)
    denom = np.log2(np.arange(2, len(gains) + 2))
    dcg = np.sum(gains / denom)
    ideal = np.sort(y)[::-1][:k].astype(float)
    idcg = np.sum(ideal / denom[: len(ideal)])
    return np.nan if idcg == 0 else float(dcg / idcg)


def rank_metrics(y_raw, score, candidates=None):
    if candidates is None:
        candidates = np.ones_like(y_raw, dtype=bool)
    y = (y_raw[candidates] > 0).astype(int)
    s = score[candidates].astype(float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if y.sum() == 0 or len(y) == 0:
        return {"AP": np.nan, "nDCG@10": np.nan, "P@15": np.nan, "R@15": np.nan}
    order = np.argsort(s)[::-1]
    top15 = order[:15]
    return {
        "AP": float(average_precision_score(y, s)),
        "nDCG@10": ndcg_at_k(y, s, 10),
        "P@15": float(y[top15].sum() / min(15, len(y))),
        "R@15": float(y[top15].sum() / y.sum()),
    }


def prevalence_groups(prevalence):
    q50, q80 = np.quantile(prevalence, [0.5, 0.8])
    return {
        "rare": prevalence <= q50,
        "middle": (prevalence > q50) & (prevalence <= q80),
        "frequent": prevalence > q80,
    }


def global_popularity_scores(R, train_rows, fold_of_local):
    out = np.zeros((len(fold_of_local), R.shape[1]))
    for local_idx, fold in enumerate(fold_of_local):
        out[local_idx] = (R[train_rows[int(fold)]] > 0).mean(axis=0)
    return out


def load_screen_scores(selected_rows, train_rows, fold_of_local):
    scores = {
        "CAFNet": read_csv_matrix(CAFNET_DIR / "blind_pred.csv")[selected_rows],
        "CAFNet-D full": read_csv_matrix(CAFNET_D_DIR / "blind_pred.csv")[selected_rows],
    }
    y_true = read_csv_matrix(CAFNET_D_DIR / "blind_raw.csv")[selected_rows]
    for name, d in SCREEN_DIRS.items():
        scores[name] = read_csv_matrix(d / "blind_pred.csv")
    R = sio.loadmat(RAW_FILE)["R"].astype(float)
    scores["Global popularity"] = global_popularity_scores(R, train_rows, fold_of_local)
    return y_true, scores, R


def summarize_result_csv():
    rows = []
    for name, d in SCREEN_DIRS.items():
        df = pd.read_csv(d / "CAFNetDecoupled_result.csv")
        df = df[pd.to_numeric(df["MAP"], errors="coerce").notna()].copy()
        for col in ["MAP", "nDCG", "P15", "R15", "auc_all", "aupr_all", "spearman", "rMSE", "MAE"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        row = {"model": name, "folds": len(df)}
        for col in ["MAP", "nDCG", "P15", "R15", "auc_all", "aupr_all", "spearman", "rMSE", "MAE"]:
            row[f"{col}_mean"] = df[col].mean()
            row[f"{col}_std"] = df[col].std(ddof=1)
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "debias_screen_result_csv_summary.csv", index=False)


def macro_and_subgroups(y_true, scores, R, train_rows, fold_of_local):
    macro_rows = []
    subgroup_rows = []
    for model, pred in scores.items():
        for local_idx in range(y_true.shape[0]):
            metrics = rank_metrics(y_true[local_idx], pred[local_idx])
            macro_rows.append({"drug_local_idx": local_idx, "model": model, **metrics})
            fold = int(fold_of_local[local_idx])
            prevalence = (R[train_rows[fold]] > 0).mean(axis=0)
            for group, mask in prevalence_groups(prevalence).items():
                metrics = rank_metrics(y_true[local_idx], pred[local_idx], mask)
                subgroup_rows.append({"drug_local_idx": local_idx, "model": model, "group": group, **metrics})
    macro = pd.DataFrame(macro_rows)
    subgroup = pd.DataFrame(subgroup_rows)
    return macro, subgroup


def matched_eval(y_true, scores, R, train_rows, fold_of_local):
    rng = np.random.default_rng(20260629)
    rows = []
    for local_idx in range(y_true.shape[0]):
        fold = int(fold_of_local[local_idx])
        prevalence = (R[train_rows[fold]] > 0).mean(axis=0)
        bins = pd.qcut(pd.Series(prevalence).rank(method="first"), q=10, labels=False, duplicates="drop").to_numpy()
        positives = np.where(y_true[local_idx] > 0)[0]
        pairs = []
        for pos in positives:
            pool = np.where((y_true[local_idx] == 0) & (bins == bins[pos]))[0]
            if len(pool) == 0:
                pool = np.where(y_true[local_idx] == 0)[0]
            if len(pool) == 0:
                continue
            controls = rng.choice(pool, size=min(5, len(pool)), replace=False)
            pairs.extend((int(pos), int(c)) for c in controls)
        if not pairs:
            continue
        for model, pred in scores.items():
            pos_scores = np.array([pred[local_idx, p] for p, _ in pairs], dtype=float)
            ctrl_scores = np.array([pred[local_idx, c] for _, c in pairs], dtype=float)
            labels = np.r_[np.ones(len(pos_scores)), np.zeros(len(ctrl_scores))]
            vals = np.r_[pos_scores, ctrl_scores]
            rows.append(
                {
                    "drug_local_idx": local_idx,
                    "model": model,
                    "matched_AUROC": roc_auc_score(labels, vals),
                    "matched_AUPR": average_precision_score(labels, vals),
                    "pos_gt_ctrl_rate": float(np.mean(pos_scores > ctrl_scores)),
                    "pos_minus_ctrl_score": float(np.mean(pos_scores - ctrl_scores)),
                }
            )
    return pd.DataFrame(rows)


def summarize(df, groups, metrics):
    out = []
    for key, chunk in df.groupby(groups):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(groups, key))
        for m in metrics:
            x = chunk[m].dropna()
            row[f"{m}_mean"] = x.mean()
            row[f"{m}_std"] = x.std(ddof=1)
            row[f"{m}_n"] = len(x)
        out.append(row)
    return pd.DataFrame(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summarize_result_csv()
    selected_rows, train_rows, fold_of_local = test_rows_by_fold(3)
    y_true, scores, R = load_screen_scores(selected_rows, train_rows, fold_of_local)
    macro, subgroup = macro_and_subgroups(y_true, scores, R, train_rows, fold_of_local)
    matched = matched_eval(y_true, scores, R, train_rows, fold_of_local)
    macro.to_csv(OUT / "debias_screen_macro_by_drug.csv", index=False)
    subgroup.to_csv(OUT / "debias_screen_subgroup_by_drug.csv", index=False)
    matched.to_csv(OUT / "debias_screen_matched_by_drug.csv", index=False)
    macro_s = summarize(macro, ["model"], ["AP", "nDCG@10", "P@15", "R@15"])
    subgroup_s = summarize(subgroup, ["group", "model"], ["AP", "nDCG@10", "P@15", "R@15"])
    matched_s = summarize(
        matched, ["model"], ["matched_AUROC", "matched_AUPR", "pos_gt_ctrl_rate", "pos_minus_ctrl_score"]
    )
    macro_s.to_csv(OUT / "debias_screen_macro_summary.csv", index=False)
    subgroup_s.to_csv(OUT / "debias_screen_subgroup_summary.csv", index=False)
    matched_s.to_csv(OUT / "debias_screen_matched_summary.csv", index=False)
    report = macro_s.merge(matched_s, on="model")
    report = report[
        [
            "model",
            "AP_mean",
            "nDCG@10_mean",
            "P@15_mean",
            "R@15_mean",
            "matched_AUROC_mean",
            "matched_AUPR_mean",
            "pos_gt_ctrl_rate_mean",
        ]
    ]
    report.to_csv(OUT / "debias_screen_report.csv", index=False)
    print(report.sort_values("matched_AUPR_mean", ascending=False).to_string(index=False))
    print("\nRare/middle/frequent AP:")
    print(subgroup_s.pivot(index="model", columns="group", values="AP_mean").to_string())


if __name__ == "__main__":
    sys.exit(main())
