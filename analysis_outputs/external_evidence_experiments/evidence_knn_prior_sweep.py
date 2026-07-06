from __future__ import annotations

from pathlib import Path

import json
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_outputs" / "external_evidence_experiments"
RAW_FILE = ROOT / "data" / "raw_frequency_750.mat"
MASK_FILE = ROOT / "data" / "blind_mask_mat_750.mat"

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

FEATURES = {
    "ATC": ROOT / "data_external" / "chembl_atc_features.npy",
    "ATC+target": ROOT / "data_external" / "chembl_atc_target_features.npy",
}


def read_csv_matrix(path: Path) -> np.ndarray:
    return pd.read_csv(path, header=None).values.astype(float)


def fold_layout(max_folds=10):
    masks = sio.loadmat(MASK_FILE)
    rows = []
    train_rows = {}
    fold_of_local = []
    original_of_local = []
    for fold in range(max_folds):
        mask = masks[f"mask{fold}"].astype(float)
        train = mask[:, 0] > 0
        test_idx = np.where(~train)[0]
        rows.extend(test_idx.tolist())
        train_rows[fold] = train
        fold_of_local.extend([fold] * len(test_idx))
        original_of_local.extend(test_idx.tolist())
    return (
        np.array(rows, dtype=int),
        train_rows,
        np.array(fold_of_local, dtype=int),
        np.array(original_of_local, dtype=int),
    )


def zscore_rows(x):
    x = x.astype(float)
    mu = np.nanmean(x, axis=1, keepdims=True)
    sd = np.nanstd(x, axis=1, keepdims=True)
    return (x - mu) / np.maximum(sd, 1e-8)


def evidence_knn_scores(R, features, train_rows, fold_of_local, original_of_local, k=20, use_frequency=False):
    out = np.zeros((len(original_of_local), R.shape[1]), dtype=float)
    observed = (R != 0).astype(float)
    y_train_all = R.astype(float) if use_frequency else observed
    for local_idx, (fold, drug_idx) in enumerate(zip(fold_of_local, original_of_local)):
        train_mask = train_rows[int(fold)]
        train_idx = np.where(train_mask)[0]
        x_train = features[train_idx]
        x_test = features[int(drug_idx)].reshape(1, -1)
        sim = cosine_similarity(x_test, x_train).ravel()
        sim = np.where(np.isfinite(sim), sim, 0.0)
        order = np.argsort(sim)[::-1][: min(k, len(sim))]
        weights = np.maximum(sim[order], 0.0)
        if weights.sum() <= 1e-12:
            weights = np.ones_like(weights, dtype=float)
        weights = weights / weights.sum()
        out[local_idx] = weights @ y_train_all[train_idx[order]]
    return out


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


def summarize(df, groups, metrics):
    rows = []
    for key, chunk in df.groupby(groups):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(groups, key))
        for metric in metrics:
            x = chunk[metric].dropna()
            row[f"{metric}_mean"] = x.mean()
            row[f"{metric}_std"] = x.std(ddof=1)
            row[f"{metric}_n"] = len(x)
        rows.append(row)
    return pd.DataFrame(rows)


def macro_subgroup_matched(y_true, scores, R, train_rows, fold_of_local):
    macro_rows = []
    subgroup_rows = []
    matched_rows = []
    rng = np.random.default_rng(20260630)
    for local_idx in range(y_true.shape[0]):
        fold = int(fold_of_local[local_idx])
        prevalence = (R[train_rows[fold]] > 0).mean(axis=0)
        groups = prevalence_groups(prevalence)
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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = sio.loadmat(RAW_FILE)["R"].astype(float)
    selected_rows, train_rows, fold_of_local, original_of_local = fold_layout(10)
    y_true = read_csv_matrix(CAFNET_D_DIR / "blind_raw.csv")[selected_rows]
    cafnet_d = read_csv_matrix(CAFNET_D_DIR / "blind_pred.csv")[selected_rows]
    cafnet = read_csv_matrix(CAFNET_DIR / "blind_pred.csv")[selected_rows]

    scores = {
        "CAFNet": cafnet,
        "CAFNet-D full": cafnet_d,
        "Global popularity": global_popularity_scores(R, train_rows, fold_of_local),
    }
    sweep_rows = []
    for feat_name, feat_path in FEATURES.items():
        features = np.load(feat_path).astype(float)
        for k in [5, 10, 20, 50]:
            prior = evidence_knn_scores(R, features, train_rows, fold_of_local, original_of_local, k=k)
            scores[f"Evidence-kNN {feat_name} k={k}"] = prior
            for beta in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
                combo = zscore_rows(cafnet_d) + beta * zscore_rows(prior)
                name = f"CAFNet-D + {feat_name} kNN k={k} beta={beta}"
                scores[name] = combo
                sweep_rows.append({"model": name, "feature": feat_name, "k": k, "beta": beta})

    macro, subgroup, matched = macro_subgroup_matched(y_true, scores, R, train_rows, fold_of_local)
    macro_s = summarize(macro, ["model"], ["AP", "nDCG@10", "P@15", "R@15"])
    subgroup_s = summarize(subgroup, ["group", "model"], ["AP", "nDCG@10", "P@15", "R@15"])
    matched_s = summarize(matched, ["model"], ["matched_AUROC", "matched_AUPR", "pos_gt_ctrl_rate"])
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
    macro.to_csv(OUT / "evidence_knn_macro_by_drug.csv", index=False)
    subgroup.to_csv(OUT / "evidence_knn_subgroup_by_drug.csv", index=False)
    matched.to_csv(OUT / "evidence_knn_matched_by_drug.csv", index=False)
    macro_s.to_csv(OUT / "evidence_knn_macro_summary.csv", index=False)
    subgroup_s.to_csv(OUT / "evidence_knn_subgroup_summary.csv", index=False)
    matched_s.to_csv(OUT / "evidence_knn_matched_summary.csv", index=False)
    report.to_csv(OUT / "evidence_knn_screen_report.csv", index=False)

    best = report.sort_values(["matched_AUPR_mean", "AP_mean"], ascending=False).head(20)
    best.to_csv(OUT / "evidence_knn_top20_by_matched_aupr.csv", index=False)
    rare_middle = subgroup_s[subgroup_s["group"].isin(["rare", "middle"])].pivot(index="model", columns="group", values="AP_mean")
    rare_middle["rare_middle_mean"] = rare_middle[["rare", "middle"]].mean(axis=1)
    rare_middle.sort_values("rare_middle_mean", ascending=False).head(20).to_csv(
        OUT / "evidence_knn_top20_by_rare_middle_ap.csv"
    )
    metadata = {
        "n_test_rows": int(y_true.shape[0]),
        "n_side_effects": int(y_true.shape[1]),
        "features": {k: str(v) for k, v in FEATURES.items()},
    }
    (OUT / "evidence_knn_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Top by matched AUPR")
    print(best.to_string(index=False))
    print("\nTop by rare/middle AP")
    print(rare_middle.sort_values("rare_middle_mean", ascending=False).head(20).to_string())


if __name__ == "__main__":
    main()
