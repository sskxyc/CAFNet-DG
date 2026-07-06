from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_outputs" / "external_contrastive_experiments"
RAW_FILE = ROOT / "data" / "raw_frequency_750.mat"
MASK_FILE = ROOT / "data" / "blind_mask_mat_750.mat"
OFFSIDES_PAIRS = ROOT / "data_external" / "external_pairs" / "offsides_positive_pairs_all.csv"

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
EXTCON_DIRS = {
    "CAFNet-D-Ext lambda=0.03": ROOT / "result_ICS" / "10extcon_lam003_f3e50_CAFNetDecoupled",
    "CAFNet-D-Ext lambda=0.05": ROOT / "result_ICS" / "10extcon_lam005_f3e50_CAFNetDecoupled",
    "CAFNet-D-Ext lambda=0.10": ROOT / "result_ICS" / "10extcon_lam010_f3e50_CAFNetDecoupled",
    "CAFNet-D-Ext assoc lambda=0.005": ROOT / "result_ICS" / "10extcon_assoc_lam0005_f3e50_CAFNetDecoupled",
    "CAFNet-D-Ext assoc lambda=0.01": ROOT / "result_ICS" / "10extcon_assoc_lam001_f3e50_CAFNetDecoupled",
    "CAFNet-D-Ext assoc lambda=0.02": ROOT / "result_ICS" / "10extcon_assoc_lam002_f3e50_CAFNetDecoupled",
}


def read_csv_matrix(path: Path) -> np.ndarray:
    return pd.read_csv(path, header=None).values.astype(float)


def test_rows_by_fold(max_folds=3):
    masks = sio.loadmat(MASK_FILE)
    rows = []
    train_rows = {}
    fold_of_local = []
    global_of_local = []
    for fold in range(max_folds):
        mask = masks[f"mask{fold}"].astype(float)
        train = mask[:, 0] > 0
        test = np.where(~train)[0]
        train_rows[fold] = train
        rows.extend(test.tolist())
        global_of_local.extend(test.tolist())
        fold_of_local.extend([fold] * len(test))
    return np.array(rows, dtype=int), train_rows, np.array(fold_of_local, dtype=int), np.array(global_of_local, dtype=int)


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


def prevalence_bins(prevalence, n_bins=10):
    rank = pd.Series(prevalence).rank(method="first")
    return pd.qcut(rank, q=n_bins, labels=False, duplicates="drop").to_numpy()


def global_popularity_scores(R, train_rows, fold_of_local):
    out = np.zeros((len(fold_of_local), R.shape[1]))
    for local_idx, fold in enumerate(fold_of_local):
        out[local_idx] = (R[train_rows[int(fold)]] > 0).mean(axis=0)
    return out


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


def internal_attribution(y_true, scores, R, train_rows, fold_of_local):
    rng = np.random.default_rng(20260630)
    macro_rows, subgroup_rows, matched_rows = [], [], []
    for local_idx in range(y_true.shape[0]):
        fold = int(fold_of_local[local_idx])
        prevalence = (R[train_rows[fold]] > 0).mean(axis=0)
        groups = prevalence_groups(prevalence)
        bins = prevalence_bins(prevalence)
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
        for model, pred in scores.items():
            macro_rows.append({"drug_local_idx": local_idx, "model": model, **rank_metrics(y_true[local_idx], pred[local_idx])})
            for group, mask in groups.items():
                subgroup_rows.append(
                    {"drug_local_idx": local_idx, "model": model, "group": group, **rank_metrics(y_true[local_idx], pred[local_idx], mask)}
                )
            if pairs:
                pos_scores = np.array([pred[local_idx, p] for p, _ in pairs], dtype=float)
                ctrl_scores = np.array([pred[local_idx, c] for _, c in pairs], dtype=float)
                labels = np.r_[np.ones(len(pos_scores)), np.zeros(len(ctrl_scores))]
                vals = np.r_[pos_scores, ctrl_scores]
                matched_rows.append(
                    {
                        "drug_local_idx": local_idx,
                        "model": model,
                        "matched_AUROC": roc_auc_score(labels, vals),
                        "matched_AUPR": average_precision_score(labels, vals),
                        "pos_gt_ctrl_rate": float(np.mean(pos_scores > ctrl_scores)),
                    }
                )
    return pd.DataFrame(macro_rows), pd.DataFrame(subgroup_rows), pd.DataFrame(matched_rows)


def offsides_external_validation(scores, R, train_rows, fold_of_local, global_of_local):
    pairs = pd.read_csv(OFFSIDES_PAIRS)
    local_by_global = {int(g): i for i, g in enumerate(global_of_local)}
    ext_by_drug = {int(di): set(g["side_index"].astype(int)) for di, g in pairs.groupby("drug_index")}
    rng = np.random.default_rng(20260630)
    rows = []
    for _, pair in pairs.iterrows():
        di = int(pair["drug_index"])
        si = int(pair["side_index"])
        if di not in local_by_global:
            continue
        local = local_by_global[di]
        fold = int(fold_of_local[local])
        prevalence = (R[train_rows[fold]] > 0).mean(axis=0)
        bins = prevalence_bins(prevalence)
        excluded = set(np.where(R[di] != 0)[0].tolist())
        excluded.update(ext_by_drug.get(di, set()))
        pool = np.where((R[di] == 0) & (bins == bins[si]))[0]
        pool = np.array([x for x in pool if int(x) not in excluded], dtype=int)
        if len(pool) == 0:
            pool = np.where(R[di] == 0)[0]
            pool = np.array([x for x in pool if int(x) not in excluded], dtype=int)
        if len(pool) == 0:
            continue
        controls = rng.choice(pool, size=min(5, len(pool)), replace=False)
        for ctrl in controls:
            for model, pred in scores.items():
                rows.append(
                    {
                        "model": model,
                        "drug_index": di,
                        "pos_side_index": si,
                        "neg_side_index": int(ctrl),
                        "pos_score": float(pred[local, si]),
                        "neg_score": float(pred[local, int(ctrl)]),
                        "pos_gt_neg": int(pred[local, si] > pred[local, int(ctrl)]),
                    }
                )
    scored = pd.DataFrame(rows)
    summary = []
    for model, df in scored.groupby("model"):
        labels = np.r_[np.ones(len(df)), np.zeros(len(df))]
        vals = np.r_[df["pos_score"].to_numpy(float), df["neg_score"].to_numpy(float)]
        summary.append(
            {
                "model": model,
                "n_matched_rows": len(df),
                "n_external_positive_pairs": len(df[["drug_index", "pos_side_index"]].drop_duplicates()),
                "external_AUROC": roc_auc_score(labels, vals),
                "external_AUPR": average_precision_score(labels, vals),
                "external_pos_gt_neg_rate": df["pos_gt_neg"].mean(),
                "external_mean_pos_minus_neg": (df["pos_score"] - df["neg_score"]).mean(),
            }
        )
    return scored, pd.DataFrame(summary)


def ordinary_result_summary():
    rows = []
    for model, d in EXTCON_DIRS.items():
        df = pd.read_csv(d / "CAFNetDecoupled_result.csv")
        avg_idx = np.where(df["pearson"].astype(str).str.lower() == "avg")[0]
        if len(avg_idx):
            df = df.iloc[: int(avg_idx[0])].copy()
        for col in ["MAP", "nDCG", "P15", "R15", "auc_all", "aupr_all", "spearman", "rMSE", "MAE"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        row = {"model": model, "folds": len(df)}
        for col in ["MAP", "nDCG", "P15", "R15", "auc_all", "aupr_all", "spearman", "rMSE", "MAE"]:
            row[f"{col}_mean"] = df[col].mean()
            row[f"{col}_std"] = df[col].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = sio.loadmat(RAW_FILE)["R"].astype(float)
    selected_rows, train_rows, fold_of_local, global_of_local = test_rows_by_fold(3)
    y_true = read_csv_matrix(CAFNET_D_DIR / "blind_raw.csv")[selected_rows]
    scores = {
        "CAFNet": read_csv_matrix(CAFNET_DIR / "blind_pred.csv")[selected_rows],
        "CAFNet-D full": read_csv_matrix(CAFNET_D_DIR / "blind_pred.csv")[selected_rows],
        "Global popularity": global_popularity_scores(R, train_rows, fold_of_local),
    }
    for model, d in EXTCON_DIRS.items():
        scores[model] = read_csv_matrix(d / "blind_pred.csv")
    ordinary = ordinary_result_summary()
    macro, subgroup, matched = internal_attribution(y_true, scores, R, train_rows, fold_of_local)
    ext_scored, ext_summary = offsides_external_validation(scores, R, train_rows, fold_of_local, global_of_local)
    macro_s = summarize(macro, ["model"], ["AP", "nDCG@10", "P@15", "R@15"])
    subgroup_s = summarize(subgroup, ["group", "model"], ["AP", "nDCG@10", "P@15", "R@15"])
    matched_s = summarize(matched, ["model"], ["matched_AUROC", "matched_AUPR", "pos_gt_ctrl_rate"])
    report = macro_s.merge(matched_s, on="model").merge(ext_summary, on="model")
    report.to_csv(OUT / "extcon_screen_report.csv", index=False)
    ordinary.to_csv(OUT / "extcon_ordinary_result_summary.csv", index=False)
    macro_s.to_csv(OUT / "extcon_macro_summary.csv", index=False)
    subgroup_s.to_csv(OUT / "extcon_subgroup_summary.csv", index=False)
    matched_s.to_csv(OUT / "extcon_internal_matched_summary.csv", index=False)
    ext_summary.to_csv(OUT / "extcon_offsides_external_summary.csv", index=False)
    ext_scored.to_csv(OUT / "extcon_offsides_external_scored_pairs.csv", index=False)
    print(report.sort_values("external_AUPR", ascending=False).to_string(index=False))
    print("\nRare/middle/frequent AP:")
    print(subgroup_s.pivot(index="model", columns="group", values="AP_mean").to_string())
    print("\nOrdinary:")
    print(ordinary.to_string(index=False))


if __name__ == "__main__":
    main()
