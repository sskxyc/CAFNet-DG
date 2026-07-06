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
PAIR_FILE = ROOT / "data_external" / "external_pairs" / "offsides_positive_pairs_all.csv"

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


def read_csv_matrix(path: Path) -> np.ndarray:
    return pd.read_csv(path, header=None).values.astype(float)


def prevalence_bins(prevalence, n_bins=10):
    rank = pd.Series(prevalence).rank(method="first")
    return pd.qcut(rank, q=n_bins, labels=False, duplicates="drop").to_numpy()


def test_row_layout():
    masks = sio.loadmat(MASK_FILE)
    rows = []
    fold_by_global = {}
    local_by_global = {}
    train_rows = {}
    for fold in range(10):
        mask = masks[f"mask{fold}"].astype(float)
        train = mask[:, 0] > 0
        test = np.where(~train)[0]
        train_rows[fold] = train
        for idx in test:
            fold_by_global[int(idx)] = fold
            local_by_global[int(idx)] = len(rows)
            rows.append(int(idx))
    return np.array(rows, dtype=int), fold_by_global, local_by_global, train_rows


def global_popularity_scores(R, train_rows, fold_of_local):
    out = np.zeros((len(fold_of_local), R.shape[1]))
    for i, fold in enumerate(fold_of_local):
        out[i] = (R[train_rows[int(fold)]] > 0).mean(axis=0)
    return out


def build_scored_pairs():
    OUT.mkdir(parents=True, exist_ok=True)
    R = sio.loadmat(RAW_FILE)["R"].astype(float)
    pairs = pd.read_csv(PAIR_FILE)
    selected_rows, fold_by_global, local_by_global, train_rows = test_row_layout()
    fold_of_local = np.array([fold_by_global[int(x)] for x in selected_rows], dtype=int)
    scores = {
        "CAFNet-D": read_csv_matrix(CAFNET_D_DIR / "blind_pred.csv")[selected_rows],
        "CAFNet": read_csv_matrix(CAFNET_DIR / "blind_pred.csv")[selected_rows],
        "Global popularity": global_popularity_scores(R, train_rows, fold_of_local),
    }
    ext_by_drug = {int(di): set(g["side_index"].astype(int)) for di, g in pairs.groupby("drug_index")}
    rng = np.random.default_rng(20260630)
    rows = []
    for _, pair in pairs.iterrows():
        di = int(pair["drug_index"])
        si = int(pair["side_index"])
        if di not in local_by_global:
            continue
        fold = fold_by_global[di]
        local = local_by_global[di]
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
            for model, mat in scores.items():
                rows.append(
                    {
                        "model": model,
                        "fold": fold,
                        "drug_index": di,
                        "pos_side_index": si,
                        "neg_side_index": int(ctrl),
                        "pos_score": float(mat[local, si]),
                        "neg_score": float(mat[local, int(ctrl)]),
                        "pos_gt_neg": int(mat[local, si] > mat[local, int(ctrl)]),
                        "pos_prevalence": float(prevalence[si]),
                        "neg_prevalence": float(prevalence[int(ctrl)]),
                    }
                )
    return pd.DataFrame(rows)


def summarize(scored):
    rows = []
    for model, df in scored.groupby("model"):
        labels = np.r_[np.ones(len(df)), np.zeros(len(df))]
        vals = np.r_[df["pos_score"].to_numpy(float), df["neg_score"].to_numpy(float)]
        rows.append(
            {
                "model": model,
                "n_matched_rows": len(df),
                "n_drugs": df["drug_index"].nunique(),
                "n_positive_pairs": len(df[["drug_index", "pos_side_index"]].drop_duplicates()),
                "AUROC": roc_auc_score(labels, vals),
                "AUPR": average_precision_score(labels, vals),
                "pos_gt_neg_rate": df["pos_gt_neg"].mean(),
                "mean_pos_minus_neg": (df["pos_score"] - df["neg_score"]).mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("AUPR", ascending=False)


def main():
    scored = build_scored_pairs()
    scored.to_csv(OUT / "offsides_external_sanity_scored_pairs.csv", index=False)
    summary = summarize(scored)
    summary.to_csv(OUT / "offsides_external_sanity_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
