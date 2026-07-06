from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_outputs" / "learned_gate_experiments"
OUT.mkdir(parents=True, exist_ok=True)

RAW_FILE = ROOT / "data" / "raw_frequency_750.mat"
MASK_FILE = ROOT / "data" / "blind_mask_mat_750.mat"

CAFNET_D_FILE = ROOT / "result_ICS" / (
    "10cd3e100f10_CAFNetDecoupled_knn=5_wd=0.001_epoch=100_lamb=0.03_"
    "lr0.0004_dim=200_eps=0.5_DF=False_PCA=False_not-FC=False_cross=True_"
    "fusion=gate_gate=new_fa=0.5_gatdrop=0.0_mix=0.3_aw=1.0_fw=1.0_"
    "rw=0.05_popw=0.1_biasw=1.0_listw=0.1_abw=1.0_arw=1.0_cosine"
) / "blind_pred.csv"
CAFNET_FILE = ROOT / "result_ICS" / (
    "10ICS_CAFNet_knn=5_wd=0.001_epoch=100_lamb=0.03_lr0.0004_dim=200_"
    "eps=0.5_DF=False_PCA=False_not-FC=False_cosine"
) / "blind_pred.csv"
FIXED_DG_FILE = ROOT / "result_ICS" / "10cafnet_dg_ensemble06_cafnetd04_cafnet" / "blind_pred.csv"


def read_pred(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, header=None).values.astype(np.float32)


def load_mat_array(path: Path, preferred: str | None = None) -> np.ndarray:
    mat = sio.loadmat(path)
    if preferred and preferred in mat:
        return np.asarray(mat[preferred])
    keys = [k for k in mat if not k.startswith("__")]
    if len(keys) != 1:
        raise KeyError(f"Cannot infer array key from {path}: {keys}")
    return np.asarray(mat[keys[0]])


def split_cold(full: np.ndarray, masks: dict[str, np.ndarray]) -> list[np.ndarray]:
    parts, start = [], 0
    for fold in range(10):
        mask = masks[f"mask{fold}"].astype(float)
        n = int(np.sum(mask[:, 0] == 0))
        parts.append(full[start : start + n])
        start += n
    if start != len(full):
        raise ValueError(f"Expected {start} rows consumed, got prediction rows={len(full)}")
    return parts


def get_fold_labels(raw: np.ndarray, masks: dict[str, np.ndarray]) -> list[np.ndarray]:
    labels = []
    for fold in range(10):
        mask = masks[f"mask{fold}"].astype(float)
        labels.append(raw[mask[:, 0] == 0])
    return labels


def get_fold_prevalence(raw: np.ndarray, masks: dict[str, np.ndarray]) -> list[np.ndarray]:
    prevs = []
    for fold in range(10):
        mask = masks[f"mask{fold}"].astype(float)
        train = raw[mask[:, 0] != 0]
        prevs.append((train != 0).mean(axis=0).astype(np.float32))
    return prevs


def zscore_like_train(train_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    mu = float(np.nanmean(train_values))
    sd = float(np.nanstd(train_values))
    if not np.isfinite(sd) or sd < 1e-8:
        sd = 1.0
    return (values - mu) / sd


def make_pair_features(
    d_scores: np.ndarray,
    c_scores: np.ndarray,
    prevalence: np.ndarray,
    d_train_flat: np.ndarray,
    c_train_flat: np.ndarray,
) -> np.ndarray:
    d_z = zscore_like_train(d_train_flat, d_scores.reshape(-1))
    c_z = zscore_like_train(c_train_flat, c_scores.reshape(-1))
    if prevalence.size == d_scores.shape[1]:
        prev = np.tile(prevalence, d_scores.shape[0])
    elif prevalence.size == d_scores.size:
        prev = prevalence.reshape(-1)
    else:
        raise ValueError(
            f"Prevalence shape is incompatible with scores: prevalence={prevalence.shape}, scores={d_scores.shape}"
        )
    return np.column_stack([d_z, c_z, d_z - c_z, prev, np.log1p(prev * d_scores.shape[1])]).astype(np.float32)


def sample_train_pairs(labels: np.ndarray, rng: np.random.Generator, max_neg_per_pos: int = 4) -> np.ndarray:
    y = (labels.reshape(-1) != 0)
    pos = np.flatnonzero(y)
    neg = np.flatnonzero(~y)
    if len(pos) == 0:
        raise ValueError("No positives in meta-train labels")
    n_neg = min(len(neg), len(pos) * max_neg_per_pos)
    neg_sample = rng.choice(neg, size=n_neg, replace=False)
    idx = np.concatenate([pos, neg_sample])
    rng.shuffle(idx)
    return idx


def precision_at(pos: np.ndarray, ranked: np.ndarray, k: int) -> float:
    return len(set(pos) & set(ranked[:k])) / float(k)


def recall_at(pos: np.ndarray, ranked: np.ndarray, k: int) -> float:
    return 0.0 if len(pos) == 0 else len(set(pos) & set(ranked[:k])) / float(len(pos))


def ap_manual(pos: np.ndarray, scores: np.ndarray) -> float:
    if len(pos) == 0:
        return np.nan
    y = np.zeros(scores.shape[0], dtype=int)
    y[pos] = 1
    return float(average_precision_score(y, scores))


def fold_metrics(scores: np.ndarray, labels: np.ndarray, prevalence: np.ndarray) -> dict[str, float]:
    y = (labels != 0).astype(int)
    flat_y = y.reshape(-1)
    flat_s = scores.reshape(-1)
    rows = []
    ndcgs, aps, p15s, r15s = [], [], [], []
    rare_aps, middle_aps, frequent_aps = [], [], []
    q1, q2 = np.quantile(prevalence, [1 / 3, 2 / 3])
    rare_cols = np.where(prevalence <= q1)[0]
    middle_cols = np.where((prevalence > q1) & (prevalence <= q2))[0]
    frequent_cols = np.where(prevalence > q2)[0]
    for i in range(labels.shape[0]):
        pos = np.flatnonzero(y[i])
        if len(pos) == 0:
            continue
        ranked = np.argsort(scores[i])[::-1]
        aps.append(ap_manual(pos, scores[i]))
        ndcgs.append(float(ndcg_score(y[i][None, :], scores[i][None, :], k=10)))
        p15s.append(precision_at(pos, ranked, 15))
        r15s.append(recall_at(pos, ranked, 15))
        for cols, store in [(rare_cols, rare_aps), (middle_cols, middle_aps), (frequent_cols, frequent_aps)]:
            yy = y[i, cols]
            if yy.sum() > 0 and yy.sum() < len(yy):
                store.append(float(average_precision_score(yy, scores[i, cols])))
    rows.append(
        {
            "MAP": float(np.nanmean(aps)),
            "AUROC": float(roc_auc_score(flat_y, flat_s)),
            "AUPR": float(average_precision_score(flat_y, flat_s)),
            "nDCG@10": float(np.nanmean(ndcgs)),
            "P@15": float(np.nanmean(p15s)),
            "R@15": float(np.nanmean(r15s)),
            "rare_AP": float(np.nanmean(rare_aps)),
            "middle_AP": float(np.nanmean(middle_aps)),
            "frequent_AP": float(np.nanmean(frequent_aps)),
        }
    )
    return rows[0]


def matched_control_aupr(scores: np.ndarray, labels: np.ndarray, prevalence: np.ndarray, rng: np.random.Generator) -> float:
    bins = np.digitize(prevalence, np.quantile(prevalence, [0.2, 0.4, 0.6, 0.8]), right=True)
    ys, ss = [], []
    for i in range(labels.shape[0]):
        positive_cols = np.flatnonzero(labels[i] != 0)
        negative_cols = np.flatnonzero(labels[i] == 0)
        if len(positive_cols) == 0 or len(negative_cols) == 0:
            continue
        for pos_col in positive_cols:
            same_bin = negative_cols[bins[negative_cols] == bins[pos_col]]
            pool = same_bin if len(same_bin) else negative_cols
            neg_col = int(rng.choice(pool))
            ys.extend([1, 0])
            ss.extend([float(scores[i, pos_col]), float(scores[i, neg_col])])
    if len(set(ys)) < 2:
        return np.nan
    return float(average_precision_score(ys, ss))


def hot_removed_map(scores: np.ndarray, labels: np.ndarray, prevalence: np.ndarray, n_hot: int = 100) -> float:
    hot = np.argsort(prevalence)[::-1][:n_hot]
    keep = np.ones(labels.shape[1], dtype=bool)
    keep[hot] = False
    return fold_metrics(scores[:, keep], labels[:, keep], prevalence[keep])["MAP"]


def holm(p_values: list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def wilcoxon_p(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0 or np.allclose(diff, 0):
        return 1.0
    try:
        return float(stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def main() -> None:
    rng = np.random.default_rng(20260701)
    raw = load_mat_array(RAW_FILE, "R").astype(float)
    masks = sio.loadmat(MASK_FILE)
    labels = get_fold_labels(raw, masks)
    prevalence = get_fold_prevalence(raw, masks)
    d_parts = split_cold(read_pred(CAFNET_D_FILE), masks)
    c_parts = split_cold(read_pred(CAFNET_FILE), masks)
    fixed_parts = split_cold(read_pred(FIXED_DG_FILE), masks)

    learned_parts: list[np.ndarray] = []
    gate_parts: list[np.ndarray] = []
    fold_rows = []
    gate_diag_rows = []

    for test_fold in range(10):
        train_folds = [f for f in range(10) if f != test_fold]
        train_d = np.vstack([d_parts[f] for f in train_folds])
        train_c = np.vstack([c_parts[f] for f in train_folds])
        train_y = np.vstack([labels[f] for f in train_folds])
        train_prev_by_row = np.vstack([np.tile(prevalence[f], (labels[f].shape[0], 1)) for f in train_folds])

        train_features = make_pair_features(
            train_d,
            train_c,
            train_prev_by_row.reshape(-1),
            train_d.reshape(-1),
            train_c.reshape(-1),
        )
        train_labels = (train_y.reshape(-1) != 0).astype(int)
        sampled = sample_train_pairs(train_y, rng)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(train_features[sampled])
        y_train = train_labels[sampled]
        clf = LogisticRegression(
            solver="saga",
            penalty="l2",
            C=0.5,
            class_weight="balanced",
            max_iter=500,
            n_jobs=1,
            random_state=20260701 + test_fold,
        )
        clf.fit(x_train, y_train)

        test_features = make_pair_features(
            d_parts[test_fold],
            c_parts[test_fold],
            prevalence[test_fold],
            train_d.reshape(-1),
            train_c.reshape(-1),
        )
        prob = clf.predict_proba(scaler.transform(test_features))[:, 1]

        # Convert the learned association probability into a local gate. If the learned
        # model sees stronger evidence in the CAFNet-D score than in the CAFNet residual,
        # g moves upward; if not, it falls back toward the residual branch.
        coef = clf.coef_.reshape(-1)
        d_contrib = coef[0] * scaler.transform(test_features)[:, 0]
        c_contrib = coef[1] * scaler.transform(test_features)[:, 1]
        raw_gate = 1.0 / (1.0 + np.exp(-(d_contrib - c_contrib)))
        gate = np.clip(0.20 + 0.60 * raw_gate, 0.20, 0.80).reshape(d_parts[test_fold].shape)
        learned = gate * d_parts[test_fold] + (1.0 - gate) * c_parts[test_fold]
        learned_parts.append(learned)
        gate_parts.append(gate)

        for model_name, score_mat in [
            ("CAFNet", c_parts[test_fold]),
            ("CAFNet-D", d_parts[test_fold]),
            ("CAFNet-DG-fixed", fixed_parts[test_fold]),
            ("CAFNet-DG-learned", learned),
        ]:
            m = fold_metrics(score_mat, labels[test_fold], prevalence[test_fold])
            m["matched_AUPR"] = matched_control_aupr(score_mat, labels[test_fold], prevalence[test_fold], rng)
            m["top100_removed_MAP"] = hot_removed_map(score_mat, labels[test_fold], prevalence[test_fold], 100)
            m.update({"model": model_name, "fold": test_fold})
            fold_rows.append(m)

        q1, q2 = np.quantile(prevalence[test_fold], [1 / 3, 2 / 3])
        for group, mask in [
            ("rare", prevalence[test_fold] <= q1),
            ("middle", (prevalence[test_fold] > q1) & (prevalence[test_fold] <= q2)),
            ("frequent", prevalence[test_fold] > q2),
        ]:
            gate_diag_rows.append(
                {
                    "fold": test_fold,
                    "prevalence_group": group,
                    "mean_gate_to_CAFNet_D": float(np.mean(gate[:, mask])),
                    "std_gate_to_CAFNet_D": float(np.std(gate[:, mask])),
                }
            )

    pd.DataFrame(np.vstack(learned_parts)).to_csv(OUT / "learned_gate_predictions.csv", header=False, index=False)
    pd.DataFrame(np.vstack(gate_parts)).to_csv(OUT / "learned_gate_weights.csv", header=False, index=False)
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(OUT / "learned_gate_fold_metrics.csv", index=False)
    gate_diag = pd.DataFrame(gate_diag_rows)
    gate_diag.to_csv(OUT / "learned_gate_diagnostics.csv", index=False)

    metric_cols = [
        "MAP",
        "AUROC",
        "AUPR",
        "nDCG@10",
        "P@15",
        "R@15",
        "rare_AP",
        "middle_AP",
        "frequent_AP",
        "matched_AUPR",
        "top100_removed_MAP",
    ]
    summary = (
        fold_df.groupby("model")[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.to_csv(OUT / "learned_gate_summary.csv", index=False)

    rows = []
    target = fold_df[fold_df["model"] == "CAFNet-DG-learned"].sort_values("fold")
    for baseline in ["CAFNet", "CAFNet-D", "CAFNet-DG-fixed"]:
        base = fold_df[fold_df["model"] == baseline].sort_values("fold")
        for metric in metric_cols:
            diff = target[metric].to_numpy() - base[metric].to_numpy()
            rows.append(
                {
                    "comparison": f"CAFNet-DG-learned vs {baseline}",
                    "metric": metric,
                    "target_mean": float(target[metric].mean()),
                    "baseline_mean": float(base[metric].mean()),
                    "improvement_mean": float(np.mean(diff)),
                    "wilcoxon_p": wilcoxon_p(diff),
                }
            )
    paired = pd.DataFrame(rows)
    parts = []
    for _, sub in paired.groupby("comparison", sort=False):
        sub = sub.copy()
        sub["wilcoxon_p_holm"] = holm(sub["wilcoxon_p"].tolist())
        sub["holm_sig"] = sub["wilcoxon_p_holm"] < 0.05
        parts.append(sub)
    paired = pd.concat(parts, ignore_index=True)
    paired.to_csv(OUT / "learned_gate_paired_tests.csv", index=False)

    gate_summary = gate_diag.groupby("prevalence_group")["mean_gate_to_CAFNet_D"].agg(["mean", "std"]).reset_index()
    gate_summary.to_csv(OUT / "learned_gate_diagnostics_summary.csv", index=False)

    learned_mean = fold_df[fold_df["model"] == "CAFNet-DG-learned"].set_index("fold")
    fixed_mean = fold_df[fold_df["model"] == "CAFNet-DG-fixed"].set_index("fold")
    d_mean = fold_df[fold_df["model"] == "CAFNet-D"].set_index("fold")
    accept = {
        "mAP_loss_vs_fixed": float(fixed_mean["MAP"].mean() - learned_mean["MAP"].mean()),
        "rare_AP_gain_vs_CAFNet_D": float(learned_mean["rare_AP"].mean() - d_mean["rare_AP"].mean()),
        "middle_AP_gain_vs_CAFNet_D": float(learned_mean["middle_AP"].mean() - d_mean["middle_AP"].mean()),
        "matched_AUPR_gain_vs_CAFNet_D": float(learned_mean["matched_AUPR"].mean() - d_mean["matched_AUPR"].mean()),
        "top100_removed_MAP_gain_vs_CAFNet_D": float(learned_mean["top100_removed_MAP"].mean() - d_mean["top100_removed_MAP"].mean()),
    }
    pd.DataFrame([accept]).to_csv(OUT / "learned_gate_acceptance_check.csv", index=False)

    report = [
        "# Frozen-Branch Learned Gate Report",
        "",
        "This experiment trains a fold-held-out stacking gate using cached CAFNet-D and CAFNet predictions.",
        "For each test fold, the gate is trained only on the other nine out-of-fold prediction blocks.",
        "",
        "## Acceptance Check",
        "",
    ]
    for key, value in accept.items():
        report.append(f"- `{key}`: `{value:.6f}`")
    report.extend(
        [
            "",
            "## Interpretation Template",
            "",
            "Accept the learned gate only if it keeps overall mAP within 0.005 of fixed CAFNet-DG and improves",
            "rare/middle AP, matched-control AUPR, and top-100 hot-removal mAP over CAFNet-D.",
            "",
            "Generated files:",
            "",
            "- `learned_gate_predictions.csv`",
            "- `learned_gate_weights.csv`",
            "- `learned_gate_fold_metrics.csv`",
            "- `learned_gate_summary.csv`",
            "- `learned_gate_paired_tests.csv`",
            "- `learned_gate_diagnostics_summary.csv`",
            "- `learned_gate_acceptance_check.csv`",
        ]
    )
    (OUT / "LEARNED_GATE_REPORT_20260701.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote learned gate outputs to {OUT}")
    print(pd.read_csv(OUT / "learned_gate_acceptance_check.csv").to_string(index=False))


if __name__ == "__main__":
    main()
