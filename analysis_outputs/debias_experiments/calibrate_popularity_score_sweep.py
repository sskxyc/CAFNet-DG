from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_outputs" / "debias_experiments"
RAW = sio.loadmat(ROOT / "data" / "raw_frequency_750.mat")["R"].astype(float)
MASKS = sio.loadmat(ROOT / "data" / "blind_mask_mat_750.mat")
CAFNET_D = ROOT / "result_ICS" / (
    "10cd3e100f10_CAFNetDecoupled_knn=5_wd=0.001_epoch=100_lamb=0.03_"
    "lr0.0004_dim=200_eps=0.5_DF=False_PCA=False_not-FC=False_cross=True_"
    "fusion=gate_gate=new_fa=0.5_gatdrop=0.0_mix=0.3_aw=1.0_fw=1.0_"
    "rw=0.05_popw=0.1_biasw=1.0_listw=0.1_abw=1.0_arw=1.0_cosine"
)
CAFNET = ROOT / "result_ICS" / (
    "10ICS_CAFNet_knn=5_wd=0.001_epoch=100_lamb=0.03_lr0.0004_dim=200_"
    "eps=0.5_DF=False_PCA=False_not-FC=False_cosine"
)


def read(path):
    return pd.read_csv(path, header=None).values.astype(float)


def fold_arrays():
    fold_of_row = np.full(750, -1, dtype=int)
    pop = np.zeros((750, 994), dtype=float)
    groups = {}
    for fold in range(10):
        mask = MASKS[f"mask{fold}"].astype(float)
        train = mask[:, 0] > 0
        test = ~train
        fold_of_row[test] = fold
        prevalence = (RAW[train] > 0).mean(axis=0)
        zpop = (prevalence - prevalence.mean()) / max(prevalence.std(), 1e-6)
        pop[test] = zpop
        q50, q80 = np.quantile(prevalence, [0.5, 0.8])
        groups[fold] = {
            "rare": prevalence <= q50,
            "middle": (prevalence > q50) & (prevalence <= q80),
            "frequent": prevalence > q80,
        }
    return fold_of_row, pop, groups


def ndcg(y, s, k=10):
    order = np.argsort(s)[::-1][:k]
    gains = y[order].astype(float)
    denom = np.log2(np.arange(2, len(gains) + 2))
    dcg = np.sum(gains / denom)
    ideal = np.sort(y)[::-1][:k].astype(float)
    idcg = np.sum(ideal / denom[: len(ideal)])
    return np.nan if idcg == 0 else float(dcg / idcg)


def metrics(y_raw, score, cand=None):
    if cand is None:
        cand = np.ones_like(y_raw, dtype=bool)
    y = (y_raw[cand] > 0).astype(int)
    s = score[cand]
    if y.sum() == 0:
        return None
    order = np.argsort(s)[::-1][:15]
    return {
        "AP": average_precision_score(y, s),
        "nDCG@10": ndcg(y, s),
        "P@15": y[order].sum() / min(15, len(y)),
        "R@15": y[order].sum() / y.sum(),
    }


def matched(y_true, score, fold_of_row):
    rng = np.random.default_rng(20260629)
    vals = []
    for i in range(y_true.shape[0]):
        fold = int(fold_of_row[i])
        train = MASKS[f"mask{fold}"][:, 0] > 0
        prevalence = (RAW[train] > 0).mean(axis=0)
        bins = pd.qcut(pd.Series(prevalence).rank(method="first"), q=10, labels=False, duplicates="drop").to_numpy()
        pairs = []
        for pos in np.where(y_true[i] > 0)[0]:
            pool = np.where((y_true[i] == 0) & (bins == bins[pos]))[0]
            if len(pool) == 0:
                pool = np.where(y_true[i] == 0)[0]
            if len(pool) == 0:
                continue
            ctrls = rng.choice(pool, size=min(5, len(pool)), replace=False)
            pairs.extend((int(pos), int(c)) for c in ctrls)
        if not pairs:
            continue
        ps = np.array([score[i, p] for p, _ in pairs])
        cs = np.array([score[i, c] for _, c in pairs])
        labels = np.r_[np.ones(len(ps)), np.zeros(len(cs))]
        pred = np.r_[ps, cs]
        vals.append(
            {
                "matched_AUROC": roc_auc_score(labels, pred),
                "matched_AUPR": average_precision_score(labels, pred),
                "pos_gt_ctrl": np.mean(ps > cs),
            }
        )
    return pd.DataFrame(vals).mean().to_dict()


def evaluate(name, score, y_true, fold_of_row, groups):
    rows = []
    for i in range(y_true.shape[0]):
        m = metrics(y_true[i], score[i])
        if m:
            rows.append({"model": name, "group": "macro", **m})
        fold = int(fold_of_row[i])
        for g, cand in groups[fold].items():
            m = metrics(y_true[i], score[i], cand)
            if m:
                rows.append({"model": name, "group": g, **m})
    df = pd.DataFrame(rows)
    out = {"model": name}
    for group, sub in df.groupby("group"):
        for metric in ["AP", "nDCG@10", "P@15", "R@15"]:
            out[f"{group}_{metric}"] = sub[metric].mean()
    out.update(matched(y_true, score, fold_of_row))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    y_true = read(CAFNET_D / "blind_raw.csv")
    base = read(CAFNET_D / "blind_pred.csv")
    cafnet = read(CAFNET / "blind_pred.csv")
    fold_of_row, zpop, groups = fold_arrays()
    rows = [evaluate("CAFNet", cafnet, y_true, fold_of_row, groups)]
    for alpha in [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]:
        rows.append(evaluate(f"CAFNet-D minus {alpha:g}*zpop", base - alpha * zpop, y_true, fold_of_row, groups))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "cafnet_d_popularity_calibration_sweep.csv", index=False)
    cols = ["model", "macro_AP", "macro_nDCG@10", "middle_AP", "rare_AP", "matched_AUROC", "matched_AUPR"]
    print(out[cols].to_string(index=False))


if __name__ == "__main__":
    main()
