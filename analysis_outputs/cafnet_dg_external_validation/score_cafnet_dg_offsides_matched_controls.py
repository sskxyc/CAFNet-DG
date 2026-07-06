from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_outputs" / "cafnet_dg_external_validation"
OUT.mkdir(parents=True, exist_ok=True)

RAW_FILE = ROOT / "data" / "raw_frequency_750.mat"
MASK_FILE = ROOT / "data" / "blind_mask_mat_750.mat"
PAIR_FILE = ROOT / "data_external" / "external_pairs" / "offsides_positive_pairs_non_sider.csv"

MODEL_FILES = {
    "CAFNet": ROOT
    / "result_ICS"
    / (
        "10ICS_CAFNet_knn=5_wd=0.001_epoch=100_lamb=0.03_lr0.0004_dim=200_"
        "eps=0.5_DF=False_PCA=False_not-FC=False_cosine"
    )
    / "blind_pred.csv",
    "CAFNet-D": ROOT
    / "result_ICS"
    / (
        "10cd3e100f10_CAFNetDecoupled_knn=5_wd=0.001_epoch=100_lamb=0.03_"
        "lr0.0004_dim=200_eps=0.5_DF=False_PCA=False_not-FC=False_cross=True_"
        "fusion=gate_gate=new_fa=0.5_gatdrop=0.0_mix=0.3_aw=1.0_fw=1.0_"
        "rw=0.05_popw=0.1_biasw=1.0_listw=0.1_abw=1.0_arw=1.0_cosine"
    )
    / "blind_pred.csv",
    "CAFNet-DG": ROOT / "result_ICS" / "10cafnet_dg_ensemble06_cafnetd04_cafnet" / "blind_pred.csv",
}


def read_matrix(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, header=None).values.astype(np.float32)


def prevalence_bins(prevalence: np.ndarray, n_bins: int = 10) -> np.ndarray:
    rank = pd.Series(prevalence).rank(method="first")
    return pd.qcut(rank, q=n_bins, labels=False, duplicates="drop").to_numpy()


def cold_layout(masks: dict[str, np.ndarray]):
    rows = []
    fold_by_global: dict[int, int] = {}
    local_by_global: dict[int, int] = {}
    train_rows: dict[int, np.ndarray] = {}
    test_rows: dict[int, np.ndarray] = {}
    for fold in range(10):
        mask = masks[f"mask{fold}"].astype(float)
        train = mask[:, 0] != 0
        test = np.where(~train)[0]
        train_rows[fold] = train
        test_rows[fold] = test
        for idx in test:
            fold_by_global[int(idx)] = fold
            local_by_global[int(idx)] = len(rows)
            rows.append(int(idx))
    return np.array(rows, dtype=int), fold_by_global, local_by_global, train_rows, test_rows


def global_popularity_score(raw: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    return (raw[train_mask] != 0).mean(axis=0).astype(np.float32)


def sample_controls(
    raw: np.ndarray,
    drug_idx: int,
    side_idx: int,
    prevalence: np.ndarray,
    bins: np.ndarray,
    external_by_drug: dict[int, set[int]],
    rng: np.random.Generator,
    n_controls: int = 5,
) -> np.ndarray:
    excluded = set(np.flatnonzero(raw[drug_idx] != 0).astype(int).tolist())
    excluded.update(external_by_drug.get(drug_idx, set()))
    neg = np.flatnonzero(raw[drug_idx] == 0)
    same_bin = np.array([x for x in neg if bins[x] == bins[side_idx] and int(x) not in excluded], dtype=int)
    if len(same_bin) >= n_controls:
        return rng.choice(same_bin, size=n_controls, replace=False)
    wider = np.array([x for x in neg if abs(int(bins[x]) - int(bins[side_idx])) <= 1 and int(x) not in excluded], dtype=int)
    if len(wider) >= n_controls:
        return rng.choice(wider, size=n_controls, replace=False)
    pool = np.array([x for x in neg if int(x) not in excluded], dtype=int)
    if len(pool) == 0:
        return np.array([], dtype=int)
    return rng.choice(pool, size=min(n_controls, len(pool)), replace=False)


def safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def safe_aupr(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, s))


def build_scored_pairs() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = sio.loadmat(RAW_FILE)["R"].astype(float)
    masks = sio.loadmat(MASK_FILE)
    pairs = pd.read_csv(PAIR_FILE)
    _, fold_by_global, local_by_global, train_rows, _ = cold_layout(masks)

    model_mats = {name: read_matrix(path) for name, path in MODEL_FILES.items()}
    if len({m.shape for m in model_mats.values()}) != 1:
        raise ValueError({name: mat.shape for name, mat in model_mats.items()})

    external_by_drug = {int(di): set(g["side_index"].astype(int)) for di, g in pairs.groupby("drug_index")}
    rng = np.random.default_rng(20260701)
    rows = []
    coverage_rows = []

    for fold in range(10):
        fold_pairs = pairs[pairs["drug_index"].map(lambda x: int(x) in fold_by_global and fold_by_global[int(x)] == fold)]
        prevalence = global_popularity_score(raw, train_rows[fold])
        bins = prevalence_bins(prevalence)
        n_with_controls = 0
        n_without_controls = 0
        for _, pair in fold_pairs.iterrows():
            drug_idx = int(pair["drug_index"])
            side_idx = int(pair["side_index"])
            local_idx = local_by_global[drug_idx]
            controls = sample_controls(raw, drug_idx, side_idx, prevalence, bins, external_by_drug, rng)
            if len(controls) == 0:
                n_without_controls += 1
                continue
            n_with_controls += 1
            for ctrl in controls:
                for model_name, mat in model_mats.items():
                    pos_score = float(mat[local_idx, side_idx])
                    neg_score = float(mat[local_idx, int(ctrl)])
                    rows.append(
                        {
                            "model": model_name,
                            "fold": fold,
                            "drug_index": drug_idx,
                            "side_index": side_idx,
                            "control_side_index": int(ctrl),
                            "label": 1,
                            "score": pos_score,
                            "paired_score": neg_score,
                            "pos_gt_neg": int(pos_score > neg_score),
                            "pos_minus_neg": pos_score - neg_score,
                            "pos_prevalence": float(prevalence[side_idx]),
                            "control_prevalence": float(prevalence[int(ctrl)]),
                            "prevalence_abs_diff": float(abs(prevalence[side_idx] - prevalence[int(ctrl)])),
                            "drug_name_external": pair.get("drug_name_external", ""),
                            "side_effect_external": pair.get("side_effect_external", ""),
                        }
                    )
                pop_pos = float(prevalence[side_idx])
                pop_neg = float(prevalence[int(ctrl)])
                rows.append(
                    {
                        "model": "Global popularity",
                        "fold": fold,
                        "drug_index": drug_idx,
                        "side_index": side_idx,
                        "control_side_index": int(ctrl),
                        "label": 1,
                        "score": pop_pos,
                        "paired_score": pop_neg,
                        "pos_gt_neg": int(pop_pos > pop_neg),
                        "pos_minus_neg": pop_pos - pop_neg,
                        "pos_prevalence": float(prevalence[side_idx]),
                        "control_prevalence": float(prevalence[int(ctrl)]),
                        "prevalence_abs_diff": float(abs(prevalence[side_idx] - prevalence[int(ctrl)])),
                        "drug_name_external": pair.get("drug_name_external", ""),
                        "side_effect_external": pair.get("side_effect_external", ""),
                    }
                )
        coverage_rows.append(
            {
                "fold": fold,
                "external_positive_pairs": int(len(fold_pairs)),
                "positive_pairs_with_controls": int(n_with_controls),
                "positive_pairs_without_controls": int(n_without_controls),
                "mapped_drugs": int(fold_pairs["drug_index"].nunique()),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(coverage_rows)


def summarize_global(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, df in scored.groupby("model"):
        labels = np.r_[np.ones(len(df)), np.zeros(len(df))]
        scores = np.r_[df["score"].to_numpy(float), df["paired_score"].to_numpy(float)]
        rows.append(
            {
                "model": model,
                "n_matched_rows": int(len(df)),
                "n_drugs": int(df["drug_index"].nunique()),
                "n_positive_pairs": int(df[["drug_index", "side_index"]].drop_duplicates().shape[0]),
                "external_AUROC": safe_auroc(labels, scores),
                "external_AUPR": safe_aupr(labels, scores),
                "pos_gt_neg_rate": float(df["pos_gt_neg"].mean()),
                "mean_pos_minus_neg": float(df["pos_minus_neg"].mean()),
                "median_abs_prevalence_diff": float(df["prevalence_abs_diff"].median()),
                "mean_abs_prevalence_diff": float(df["prevalence_abs_diff"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("external_AUPR", ascending=False)


def summarize_per_drug(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, drug_idx), df in scored.groupby(["model", "drug_index"]):
        labels = np.r_[np.ones(len(df)), np.zeros(len(df))]
        scores = np.r_[df["score"].to_numpy(float), df["paired_score"].to_numpy(float)]
        rows.append(
            {
                "model": model,
                "drug_index": int(drug_idx),
                "n_rows": int(len(df)),
                "n_positive_pairs": int(df[["drug_index", "side_index"]].drop_duplicates().shape[0]),
                "external_AUROC": safe_auroc(labels, scores),
                "external_AUPR": safe_aupr(labels, scores),
                "pos_gt_neg_rate": float(df["pos_gt_neg"].mean()),
                "mean_pos_minus_neg": float(df["pos_minus_neg"].mean()),
            }
        )
    return pd.DataFrame(rows)


def holm(p_values: list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def paired_tests(per_drug: pd.DataFrame, target: str = "CAFNet-DG") -> pd.DataFrame:
    metrics = ["external_AUROC", "external_AUPR", "pos_gt_neg_rate", "mean_pos_minus_neg"]
    rows = []
    for baseline in sorted(set(per_drug["model"]) - {target}):
        merged = per_drug[per_drug["model"] == target].merge(
            per_drug[per_drug["model"] == baseline],
            on="drug_index",
            suffixes=("_target", "_baseline"),
        )
        for metric in metrics:
            diff = merged[f"{metric}_target"].to_numpy(float) - merged[f"{metric}_baseline"].to_numpy(float)
            diff = diff[np.isfinite(diff)]
            if len(diff) == 0 or np.allclose(diff, 0):
                p = 1.0
                stat = np.nan
            else:
                try:
                    res = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
                    p = float(res.pvalue)
                    stat = float(res.statistic)
                except ValueError:
                    p = 1.0
                    stat = np.nan
            rows.append(
                {
                    "comparison": f"{target} vs {baseline}",
                    "baseline": baseline,
                    "metric": metric,
                    "n_drugs": int(len(diff)),
                    "target_mean": float(np.nanmean(merged[f"{metric}_target"])),
                    "baseline_mean": float(np.nanmean(merged[f"{metric}_baseline"])),
                    "improvement_mean": float(np.nanmean(diff)) if len(diff) else np.nan,
                    "wilcoxon_stat": stat,
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


def write_report(global_summary: pd.DataFrame, per_drug: pd.DataFrame, paired: pd.DataFrame, coverage: pd.DataFrame) -> None:
    def md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
        cols = list(df.columns)
        out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in df.iterrows():
            vals = []
            for col in cols:
                value = row[col]
                if isinstance(value, (float, np.floating)):
                    vals.append(format(float(value), floatfmt))
                else:
                    vals.append(str(value))
            out.append("| " + " | ".join(vals) + " |")
        return "\n".join(out)

    lines = [
        "# CAFNet-DG OFFSIDES Matched-Control External Validation",
        "",
        "This analysis directly scores the current CAFNet-DG fixed residual-fusion predictions on OFFSIDES-derived external positive pairs mapped to the 750-drug/994-side-effect benchmark.",
        "Only OFFSIDES pairs not present in the SIDER-derived frequency matrix are used. For each external positive pair, up to five same-drug negative controls are sampled from matched or adjacent side-effect prevalence bins using the training fold only.",
        "",
        "## Coverage",
        "",
        f"- Positive pairs with matched controls: `{int(coverage['positive_pairs_with_controls'].sum())}`",
        f"- Positive pairs without controls: `{int(coverage['positive_pairs_without_controls'].sum())}`",
        f"- Mapped drugs: `{int(coverage['mapped_drugs'].sum())}` fold-counted; `{int(per_drug['drug_index'].nunique())}` unique drugs in scored output",
        "",
        "## Global matched-row summary",
        "",
        md_table(global_summary, ".4f"),
        "",
        "## Per-drug Wilcoxon tests",
        "",
        md_table(paired, ".4g"),
        "",
        "## Interpretation",
        "",
        "Use this as external matched-control evidence only if CAFNet-DG improves over CAFNet-D and global popularity on per-drug metrics. If gains are small or not significant, report it as a partial external robustness check rather than as definitive independent validation.",
    ]
    (OUT / "CAFNET_DG_OFFSIDES_EXTERNAL_VALIDATION_REPORT_20260701.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    scored, coverage = build_scored_pairs()
    scored.to_csv(OUT / "offsides_matched_control_scored_pairs.csv", index=False)
    coverage.to_csv(OUT / "offsides_matched_control_coverage.csv", index=False)
    global_summary = summarize_global(scored)
    global_summary.to_csv(OUT / "offsides_matched_control_global_summary.csv", index=False)
    per_drug = summarize_per_drug(scored)
    per_drug.to_csv(OUT / "offsides_matched_control_per_drug.csv", index=False)
    paired = paired_tests(per_drug)
    paired.to_csv(OUT / "offsides_matched_control_paired_tests.csv", index=False)
    write_report(global_summary, per_drug, paired, coverage)
    print(global_summary.to_string(index=False))
    print()
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
