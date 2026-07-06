from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_outputs" / "external_contrastive_experiments"
PAIR_DIR = ROOT / "data_external" / "external_pairs"
RAW_FILE = ROOT / "data" / "raw_frequency_750.mat"
MASK_FILE = ROOT / "data" / "blind_mask_mat_750.mat"
DRUG_FEATURE_FILE = ROOT / "data_external" / "chembl_atc_target_features.npy"
OFFSIDES_PAIRS = PAIR_DIR / "offsides_positive_pairs_all.csv"

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


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def read_csv_matrix(path: Path) -> np.ndarray:
    return pd.read_csv(path, header=None).values.astype(float)


def fold_layout(max_folds=10):
    masks = sio.loadmat(MASK_FILE)
    selected_rows, fold_of_local, global_of_local, train_rows = [], [], [], {}
    for fold in range(max_folds):
        mask = masks[f"mask{fold}"].astype(float)
        train = mask[:, 0] > 0
        test = np.where(~train)[0]
        train_rows[fold] = train
        selected_rows.extend(test.tolist())
        global_of_local.extend(test.tolist())
        fold_of_local.extend([fold] * len(test))
    return np.array(selected_rows), np.array(fold_of_local), np.array(global_of_local), train_rows


def prevalence_bins(prevalence, n_bins=10):
    rank = pd.Series(prevalence).rank(method="first")
    return pd.qcut(rank, q=n_bins, labels=False, duplicates="drop").to_numpy()


def make_pair_features(drug_idx, side_idx, base_scores, prevalence, drug_features, side_embed, scaler=None, fit=False):
    dense = np.column_stack(
        [
            base_scores.astype(float),
            prevalence[side_idx].astype(float),
            np.log1p(prevalence[side_idx] * 750.0),
        ]
    )
    if fit:
        scaler = StandardScaler().fit(dense)
    dense = scaler.transform(dense).astype(np.float32)
    x = np.hstack([dense, drug_features[drug_idx].astype(np.float32), side_embed[side_idx].astype(np.float32)])
    return x, scaler


def load_train_pairs(fold, R, base_full, drug_features, side_embed, max_pairs=80000):
    masks = sio.loadmat(MASK_FILE)
    train_idx = np.where(masks[f"mask{fold}"][:, 0].astype(float) > 0)[0]
    prevalence = (R[train_idx] > 0).mean(axis=0)
    npz = np.load(PAIR_DIR / f"offsides_contrastive_pairs_fold{fold}.npz")
    local_drugs = npz["train_local_index"].astype(np.int64)
    pos_side = npz["pos_side_index"].astype(np.int64)
    neg_side = npz["neg_side_index"].astype(np.int64)
    rng = np.random.default_rng(20260630 + fold)
    if len(local_drugs) > max_pairs:
        keep = rng.choice(np.arange(len(local_drugs)), size=max_pairs, replace=False)
        local_drugs, pos_side, neg_side = local_drugs[keep], pos_side[keep], neg_side[keep]
    global_drugs = train_idx[local_drugs]
    drug_idx = np.r_[global_drugs, global_drugs]
    side_idx = np.r_[pos_side, neg_side]
    labels = np.r_[np.ones(len(pos_side)), np.zeros(len(neg_side))].astype(np.float32)
    base_scores = base_full[drug_idx, side_idx]
    x, scaler = make_pair_features(drug_idx, side_idx, base_scores, prevalence, drug_features, side_embed, fit=True)
    return x, labels, scaler, prevalence


def train_residual_mlp(x, y, reg_weight=0.01, epochs=8, batch_size=4096, seed=42):
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResidualMLP(x.shape[1]).to(device)
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(y)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = x_t[idx].to(device)
            yb = y_t[idx].to(device)
            logits = model(xb)
            bce = F.binary_cross_entropy_with_logits(logits, yb)
            penalty = logits.pow(2).mean()
            loss = bce + float(reg_weight) * penalty
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.cpu()


def predict_residual(model, scaler, prevalence, global_test_idx, base_full, drug_features, side_embed):
    drug_idx = np.repeat(global_test_idx, 994)
    side_idx = np.tile(np.arange(994), len(global_test_idx))
    base_scores = base_full[drug_idx, side_idx]
    x, _ = make_pair_features(drug_idx, side_idx, base_scores, prevalence, drug_features, side_embed, scaler=scaler)
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 8192):
            logits = model(torch.tensor(x[start:start + 8192], dtype=torch.float32))
            out.append(torch.sigmoid(logits).numpy())
    return np.concatenate(out).reshape(len(global_test_idx), 994)


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
            vals = chunk[metric].dropna()
            row[f"{metric}_mean"] = vals.mean()
            row[f"{metric}_std"] = vals.std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_internal(y_true, scores, R, train_rows, fold_of_local):
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
        for name, pred in scores.items():
            macro_rows.append({"drug_local_idx": local_idx, "model": name, **rank_metrics(y_true[local_idx], pred[local_idx])})
            for group, mask in groups.items():
                subgroup_rows.append({"drug_local_idx": local_idx, "model": name, "group": group, **rank_metrics(y_true[local_idx], pred[local_idx], mask)})
            if pairs:
                pos_scores = np.array([pred[local_idx, p] for p, _ in pairs])
                ctrl_scores = np.array([pred[local_idx, c] for _, c in pairs])
                labels = np.r_[np.ones(len(pos_scores)), np.zeros(len(ctrl_scores))]
                vals = np.r_[pos_scores, ctrl_scores]
                matched_rows.append(
                    {
                        "drug_local_idx": local_idx,
                        "model": name,
                        "matched_AUROC": roc_auc_score(labels, vals),
                        "matched_AUPR": average_precision_score(labels, vals),
                    }
                )
    return pd.DataFrame(macro_rows), pd.DataFrame(subgroup_rows), pd.DataFrame(matched_rows)


def evaluate_offsides(scores, R, train_rows, fold_of_local, global_of_local):
    pairs = pd.read_csv(OFFSIDES_PAIRS)
    local_by_global = {int(g): i for i, g in enumerate(global_of_local)}
    ext_by_drug = {int(di): set(g["side_index"].astype(int)) for di, g in pairs.groupby("drug_index")}
    rng = np.random.default_rng(20260630)
    rows = []
    for _, pair in pairs.iterrows():
        di, si = int(pair["drug_index"]), int(pair["side_index"])
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
            for name, pred in scores.items():
                rows.append({"model": name, "pos_score": float(pred[local, si]), "neg_score": float(pred[local, int(ctrl)])})
    scored = pd.DataFrame(rows)
    summary = []
    for name, df in scored.groupby("model"):
        labels = np.r_[np.ones(len(df)), np.zeros(len(df))]
        vals = np.r_[df["pos_score"].to_numpy(float), df["neg_score"].to_numpy(float)]
        summary.append(
            {
                "model": name,
                "external_AUROC": roc_auc_score(labels, vals),
                "external_AUPR": average_precision_score(labels, vals),
                "external_pos_gt_neg_rate": float(np.mean(df["pos_score"] > df["neg_score"])),
            }
        )
    return pd.DataFrame(summary)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = sio.loadmat(RAW_FILE)["R"].astype(float)
    drug_features = np.load(DRUG_FEATURE_FILE).astype(np.float32)
    rng = np.random.default_rng(20260630)
    side_embed = rng.normal(0, 0.1, size=(994, 32)).astype(np.float32)
    selected_rows, fold_of_local, global_of_local, train_rows = fold_layout(10)
    base_full = read_csv_matrix(CAFNET_D_DIR / "blind_pred.csv")
    cafnet_full = read_csv_matrix(CAFNET_DIR / "blind_pred.csv")
    y_true = read_csv_matrix(CAFNET_D_DIR / "blind_raw.csv")[selected_rows]
    base = base_full[selected_rows]
    residual_by_reg = {0.01: np.zeros_like(base), 0.05: np.zeros_like(base)}
    for reg in residual_by_reg:
        for fold in range(10):
            test_mask = fold_of_local == fold
            x, y, scaler, prevalence = load_train_pairs(fold, R, base_full, drug_features, side_embed)
            model = train_residual_mlp(x, y, reg_weight=reg, seed=20260630 + fold)
            residual_by_reg[reg][test_mask] = predict_residual(
                model, scaler, prevalence, global_of_local[test_mask], base_full, drug_features, side_embed
            )
            print("trained neural residual", "reg", reg, "fold", fold)
    scores = {
        "CAFNet": cafnet_full[selected_rows],
        "CAFNet-D full": base,
        "Global popularity": global_popularity_scores(R, train_rows, fold_of_local),
    }
    z_base = (base - base.mean(axis=1, keepdims=True)) / np.maximum(base.std(axis=1, keepdims=True), 1e-8)
    for reg, residual in residual_by_reg.items():
        z_res = (residual - residual.mean(axis=1, keepdims=True)) / np.maximum(residual.std(axis=1, keepdims=True), 1e-8)
        scores[f"Neural residual only reg={reg}"] = residual
        for gamma in [0.005, 0.01, 0.03]:
            scores[f"CAFNet-D + neural residual reg={reg} gamma={gamma}"] = z_base + gamma * z_res
    macro, subgroup, matched = evaluate_internal(y_true, scores, R, train_rows, fold_of_local)
    external = evaluate_offsides(scores, R, train_rows, fold_of_local, global_of_local)
    macro_s = summarize(macro, ["model"], ["AP", "nDCG@10", "P@15", "R@15"])
    subgroup_s = summarize(subgroup, ["group", "model"], ["AP"])
    matched_s = summarize(matched, ["model"], ["matched_AUROC", "matched_AUPR"])
    report = macro_s.merge(matched_s, on="model").merge(external, on="model")
    report.to_csv(OUT / "neural_residual_screen_report.csv", index=False)
    subgroup_s.to_csv(OUT / "neural_residual_subgroup_summary.csv", index=False)
    matched_s.to_csv(OUT / "neural_residual_internal_matched_summary.csv", index=False)
    external.to_csv(OUT / "neural_residual_offsides_summary.csv", index=False)
    print(report.sort_values("external_AUPR", ascending=False).to_string(index=False))
    print("\nRare/middle AP:")
    print(subgroup_s[subgroup_s["group"].isin(["rare", "middle"])].pivot(index="model", columns="group", values="AP_mean").to_string())


if __name__ == "__main__":
    main()
