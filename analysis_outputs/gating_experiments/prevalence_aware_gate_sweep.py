from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_outputs" / "gating_experiments"
OUT.mkdir(parents=True, exist_ok=True)

RAW_FILE = ROOT / "data" / "raw_frequency_750.mat"
MASK_FILE = ROOT / "data" / "blind_mask_mat_750.mat"
PAIR_DIR = ROOT / "data_external" / "external_pairs"
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


def read_csv_matrix(path: Path) -> np.ndarray:
    return pd.read_csv(path, header=None).values.astype(float)


def load_raw_matrix() -> np.ndarray:
    raw = sio.loadmat(RAW_FILE)
    for key in ("R", "raw_frequency", "frequency", "raw_frequency_750"):
        if key in raw:
            return np.asarray(raw[key], dtype=float)
    for key, val in raw.items():
        if not key.startswith("__") and getattr(val, "shape", None) == (750, 994):
            return np.asarray(val, dtype=float)
    raise KeyError(f"Could not find 750x994 matrix in {RAW_FILE}")


def fold_layout(max_folds: int = 10):
    masks = sio.loadmat(MASK_FILE)
    fold_of_local, global_of_local, train_rows = [], [], {}
    for fold in range(max_folds):
        mask = masks[f"mask{fold}"].astype(float)
        train = mask[:, 0] > 0
        test = np.where(~train)[0]
        train_rows[fold] = train
        global_of_local.extend(test.tolist())
        fold_of_local.extend([fold] * len(test))
    return np.array(fold_of_local), np.array(global_of_local), train_rows


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


def prevalence_bins(prevalence, n_bins=10):
    rank = pd.Series(prevalence).rank(method="first")
    return pd.qcut(rank, q=n_bins, labels=False, duplicates="drop").to_numpy()


def prevalence_groups(prevalence):
    q50, q80 = np.quantile(prevalence, [0.5, 0.8])
    return {
        "rare": prevalence <= q50,
        "middle": (prevalence > q50) & (prevalence <= q80),
        "frequent": prevalence > q80,
    }


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
        for model_name, pred in scores.items():
            macro_rows.append({"drug_local_idx": local_idx, "model": model_name, **rank_metrics(y_true[local_idx], pred[local_idx])})
            for group, mask in groups.items():
                subgroup_rows.append({"drug_local_idx": local_idx, "model": model_name, "group": group, **rank_metrics(y_true[local_idx], pred[local_idx], mask)})
            if pairs:
                pos_scores = np.array([pred[local_idx, p] for p, _ in pairs])
                ctrl_scores = np.array([pred[local_idx, c] for _, c in pairs])
                labels = np.r_[np.ones(len(pos_scores)), np.zeros(len(ctrl_scores))]
                vals = np.r_[pos_scores, ctrl_scores]
                matched_rows.append(
                    {
                        "drug_local_idx": local_idx,
                        "model": model_name,
                        "matched_AUROC": roc_auc_score(labels, vals),
                        "matched_AUPR": average_precision_score(labels, vals),
                        "pos_gt_ctrl_rate": float(np.mean(pos_scores > ctrl_scores)),
                    }
                )
    return pd.DataFrame(macro_rows), pd.DataFrame(subgroup_rows), pd.DataFrame(matched_rows)


def evaluate_offsides(scores, R, train_rows, fold_of_local, global_of_local):
    if not OFFSIDES_PAIRS.exists():
        return pd.DataFrame()
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
            for model_name, pred in scores.items():
                rows.append(
                    {
                        "model": model_name,
                        "pos_score": float(pred[local, si]),
                        "neg_score": float(pred[local, int(ctrl)]),
                        "pos_gt_neg": int(pred[local, si] > pred[local, int(ctrl)]),
                    }
                )
    scored = pd.DataFrame(rows)
    summary = []
    for model_name, df in scored.groupby("model"):
        labels = np.r_[np.ones(len(df)), np.zeros(len(df))]
        vals = np.r_[df["pos_score"].to_numpy(float), df["neg_score"].to_numpy(float)]
        summary.append(
            {
                "model": model_name,
                "external_AUROC": roc_auc_score(labels, vals),
                "external_AUPR": average_precision_score(labels, vals),
                "external_pos_gt_neg_rate": df["pos_gt_neg"].mean(),
            }
        )
    return pd.DataFrame(summary)


def make_gate_scores(cafnet, cafnet_d, R, train_rows, fold_of_local, mode, low, mid, high, temperature):
    out = np.zeros_like(cafnet_d)
    for local_idx, fold in enumerate(fold_of_local):
        prevalence = (R[train_rows[int(fold)]] > 0).mean(axis=0)
        q50, q80 = np.quantile(prevalence, [0.5, 0.8])
        if mode == "piecewise":
            w = np.where(prevalence <= q50, low, np.where(prevalence <= q80, mid, high))
        elif mode == "sigmoid":
            center = q80
            scale = max(float(np.std(prevalence)), 1e-6) * temperature
            w = 1.0 / (1.0 + np.exp(-(prevalence - center) / scale))
            w = low + (high - low) * w
        else:
            raise ValueError(mode)
        out[local_idx] = w * cafnet_d[local_idx] + (1.0 - w) * cafnet[local_idx]
    return out


def main():
    R = load_raw_matrix()
    fold_of_local, global_of_local, train_rows = fold_layout(10)
    y_true = R[global_of_local]
    cafnet = read_csv_matrix(CAFNET_DIR / "blind_pred.csv")
    cafnet_d = read_csv_matrix(CAFNET_D_DIR / "blind_pred.csv")
    scores = {
        "CAFNet": cafnet,
        "CAFNet-D full": cafnet_d,
    }

    configs = []
    for low in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        for mid in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]:
            for high in [0.6, 0.8, 0.9, 1.0]:
                if low <= mid <= high:
                    configs.append(("piecewise", low, mid, high, 1.0))
    for low in [0.0, 0.1, 0.2, 0.4, 0.6]:
        for high in [0.8, 0.9, 1.0]:
            for temperature in [0.5, 1.0, 2.0]:
                configs.append(("sigmoid", low, 0.0, high, temperature))

    for mode, low, mid, high, temp in configs:
        name = f"Gate {mode} low={low:g} mid={mid:g} high={high:g} temp={temp:g}"
        scores[name] = make_gate_scores(cafnet, cafnet_d, R, train_rows, fold_of_local, mode, low, mid, high, temp)

    macro, subgroup, matched = evaluate_internal(y_true, scores, R, train_rows, fold_of_local)
    external = evaluate_offsides(scores, R, train_rows, fold_of_local, global_of_local)
    macro.to_csv(OUT / "prevalence_gate_macro_by_drug.csv", index=False)
    subgroup.to_csv(OUT / "prevalence_gate_subgroup_by_drug.csv", index=False)
    matched.to_csv(OUT / "prevalence_gate_internal_matched_by_drug.csv", index=False)

    macro_s = summarize(macro, ["model"], ["AP", "nDCG@10", "P@15", "R@15"])
    subgroup_s = summarize(subgroup, ["group", "model"], ["AP", "nDCG@10", "P@15", "R@15"])
    matched_s = summarize(matched, ["model"], ["matched_AUROC", "matched_AUPR", "pos_gt_ctrl_rate"])
    report = macro_s.merge(matched_s, on="model", how="left").merge(external, on="model", how="left")
    report.to_csv(OUT / "prevalence_gate_screen_report.csv", index=False)
    macro_s.to_csv(OUT / "prevalence_gate_macro_summary.csv", index=False)
    subgroup_s.to_csv(OUT / "prevalence_gate_subgroup_summary.csv", index=False)
    matched_s.to_csv(OUT / "prevalence_gate_internal_matched_summary.csv", index=False)
    if len(external):
        external.to_csv(OUT / "prevalence_gate_offsides_summary.csv", index=False)

    rare_middle = subgroup_s[subgroup_s["group"].isin(["rare", "middle"])].pivot(index="model", columns="group", values="AP_mean")
    rare_middle["rare_middle_mean"] = rare_middle[["rare", "middle"]].mean(axis=1)
    decision = report.merge(rare_middle.reset_index(), on="model", how="left")
    decision["passes_internal_matched"] = decision["matched_AUPR_mean"] >= float(
        report.loc[report["model"].eq("CAFNet-D full"), "matched_AUPR_mean"].iloc[0]
    )
    decision["passes_macro"] = decision["AP_mean"] >= float(report.loc[report["model"].eq("CAFNet-D full"), "AP_mean"].iloc[0])
    decision["beats_cafnet_rare_middle"] = decision["rare_middle_mean"] >= float(
        rare_middle.loc["CAFNet", "rare_middle_mean"]
    )
    decision.to_csv(OUT / "prevalence_gate_decision_table.csv", index=False)
    selected = [
        "Gate piecewise low=0.4 mid=0.6 high=0.6 temp=1",
        "Gate piecewise low=0.5 mid=0.6 high=0.6 temp=1",
        "Gate piecewise low=0.6 mid=0.6 high=0.6 temp=1",
    ]
    paired_rows = []
    for candidate in selected:
        for metric in ["AP", "nDCG@10", "P@15", "R@15"]:
            wide = macro[macro["model"].isin(["CAFNet-D full", candidate])].pivot(
                index="drug_local_idx", columns="model", values=metric
            ).dropna()
            if len(wide) < 3:
                continue
            diff = wide[candidate] - wide["CAFNet-D full"]
            try:
                w_p = wilcoxon(diff).pvalue
            except ValueError:
                w_p = np.nan
            paired_rows.append(
                {
                    "candidate": candidate,
                    "metric": metric,
                    "n": len(wide),
                    "mean_diff": diff.mean(),
                    "ttest_p": ttest_rel(wide[candidate], wide["CAFNet-D full"]).pvalue,
                    "wilcoxon_p": w_p,
                }
            )
        for group in ["rare", "middle", "frequent"]:
            for metric in ["AP", "nDCG@10"]:
                wide = subgroup[
                    subgroup["model"].isin(["CAFNet-D full", candidate]) & subgroup["group"].eq(group)
                ].pivot(index="drug_local_idx", columns="model", values=metric).dropna()
                if len(wide) < 3:
                    continue
                diff = wide[candidate] - wide["CAFNet-D full"]
                try:
                    w_p = wilcoxon(diff).pvalue
                except ValueError:
                    w_p = np.nan
                paired_rows.append(
                    {
                        "candidate": candidate,
                        "metric": f"{group}_{metric}",
                        "n": len(wide),
                        "mean_diff": diff.mean(),
                        "ttest_p": ttest_rel(wide[candidate], wide["CAFNet-D full"]).pvalue,
                        "wilcoxon_p": w_p,
                    }
                )
        for metric in ["matched_AUROC", "matched_AUPR", "pos_gt_ctrl_rate"]:
            wide = matched[matched["model"].isin(["CAFNet-D full", candidate])].pivot(
                index="drug_local_idx", columns="model", values=metric
            ).dropna()
            if len(wide) < 3:
                continue
            diff = wide[candidate] - wide["CAFNet-D full"]
            try:
                w_p = wilcoxon(diff).pvalue
            except ValueError:
                w_p = np.nan
            paired_rows.append(
                {
                    "candidate": candidate,
                    "metric": metric,
                    "n": len(wide),
                    "mean_diff": diff.mean(),
                    "ttest_p": ttest_rel(wide[candidate], wide["CAFNet-D full"]).pvalue,
                    "wilcoxon_p": w_p,
                }
            )
    pd.DataFrame(paired_rows).to_csv(OUT / "prevalence_gate_paired_tests_vs_cafnet_d.csv", index=False)

    cols = [
        "model",
        "AP_mean",
        "nDCG@10_mean",
        "matched_AUPR_mean",
        "rare",
        "middle",
        "rare_middle_mean",
        "external_AUPR",
        "passes_macro",
        "passes_internal_matched",
        "beats_cafnet_rare_middle",
    ]
    print(decision.sort_values(["passes_macro", "passes_internal_matched", "rare_middle_mean"], ascending=False)[cols].head(25).to_string(index=False))


if __name__ == "__main__":
    main()
