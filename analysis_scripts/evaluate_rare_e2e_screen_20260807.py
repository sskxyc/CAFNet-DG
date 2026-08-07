from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
from sklearn.metrics import average_precision_score, ndcg_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs" / "scientific_gap_resolution_20260807" / "rare_e2e_screen"
RAW = sio.loadmat(ROOT / "data" / "raw_frequency_750.mat")["R"].astype(float)
LABEL = (RAW != 0).astype(int)
MASKS = sio.loadmat(ROOT / "data" / "blind_mask_mat_750.mat")
N_FOLDS = 10

MODEL_FILES = {
    "CAFNet": ROOT / "result_ICS" / (
        "10ICS_CAFNet_knn=5_wd=0.001_epoch=100_lamb=0.03_lr0.0004_dim=200_"
        "eps=0.5_DF=False_PCA=False_not-FC=False_cosine"
    ) / "blind_pred.csv",
    "CAFNet-D": ROOT / "result_ICS" / (
        "10cd3e100f10_CAFNetDecoupled_knn=5_wd=0.001_epoch=100_lamb=0.03_"
        "lr0.0004_dim=200_eps=0.5_DF=False_PCA=False_not-FC=False_cross=True_"
        "fusion=gate_gate=new_fa=0.5_gatdrop=0.0_mix=0.3_aw=1.0_fw=1.0_"
        "rw=0.05_popw=0.1_biasw=1.0_listw=0.1_abw=1.0_arw=1.0_cosine"
    ) / "blind_pred.csv",
    "CAFNet-DG": ROOT / "result_ICS" / "10cafnet_dg_ensemble06_cafnetd04_cafnet" / "blind_pred.csv",
    "Group-only": ROOT / "result_ICS" / "10R8GROUP_CAFNetDecoupled" / "blind_pred.csv",
    "Global gate": ROOT / "result_ICS" / "10R8GLOBAL_CAFNetDecoupled" / "blind_pred.csv",
    "Drug gate": ROOT / "result_ICS" / "10R8DRUG_CAFNetDecoupled" / "blind_pred.csv",
    "Drug-stratum gate": ROOT / "result_ICS" / "10R8STRAT_CAFNetDecoupled" / "blind_pred.csv",
}


def read_matrix(path: Path) -> np.ndarray:
    matrix = pd.read_csv(path, header=None).to_numpy(dtype=float)
    if matrix.shape[1] != RAW.shape[1]:
        raise ValueError(f"Unexpected prediction shape for {path}: {matrix.shape}")
    return matrix


def parts(matrix: np.ndarray) -> list[np.ndarray]:
    output = []
    start = 0
    for fold in range(N_FOLDS):
        ids = np.flatnonzero(MASKS[f"mask{fold}"][:, 0] == 0)
        output.append(matrix[start : start + len(ids)])
        start += len(ids)
    if len(matrix) not in {start, RAW.shape[0]}:
        raise ValueError(f"Prediction rows {len(matrix)} cannot be aligned to {N_FOLDS} folds ({start})")
    return output


def ap(y: np.ndarray, score: np.ndarray) -> float:
    if y.sum() in {0, len(y)}:
        return np.nan
    return float(average_precision_score(y, score))


def evaluate() -> pd.DataFrame:
    model_parts = {name: parts(read_matrix(path)) for name, path in MODEL_FILES.items()}
    rows = []
    for fold in range(N_FOLDS):
        test_ids = np.flatnonzero(MASKS[f"mask{fold}"][:, 0] == 0)
        train_ids = np.flatnonzero(MASKS[f"mask{fold}"][:, 0] != 0)
        prevalence = LABEL[train_ids].sum(axis=0)
        rare, middle, frequent = np.array_split(np.argsort(prevalence, kind="stable"), 3)
        for local, drug_id in enumerate(test_ids):
            y = LABEL[drug_id]
            if y.sum() == 0:
                continue
            for model, fold_parts in model_parts.items():
                score = fold_parts[fold][local]
                rows.append(
                    {
                        "fold": fold,
                        "drug_index": int(drug_id),
                        "model": model,
                        "macro_AP": ap(y, score),
                        "nDCG@10": float(ndcg_score(y[None, :], score[None, :], k=10)),
                        "rare_AP": ap(y[rare], score[rare]),
                        "middle_AP": ap(y[middle], score[middle]),
                        "frequent_AP": ap(y[frequent], score[frequent]),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    per_drug = evaluate()
    per_drug.to_csv(OUT / "screen_per_drug.csv", index=False)
    metrics = ["macro_AP", "nDCG@10", "rare_AP", "middle_AP", "frequent_AP"]
    summary = per_drug.groupby("model", sort=False)[metrics].agg(["mean", "std", "count"])
    summary.to_csv(OUT / "screen_summary.csv")

    test_rows = []
    for model in ["Group-only", "Global gate", "Drug gate", "Drug-stratum gate"]:
        target = per_drug[per_drug.model == model]
        for baseline_name in ["CAFNet", "CAFNet-DG"]:
            baseline = per_drug[per_drug.model == baseline_name]
            merged = target.merge(
                baseline, on=["fold", "drug_index"], suffixes=("_target", "_baseline"), validate="one_to_one"
            )
            start = len(test_rows)
            for metric in metrics:
                pair = merged[[f"{metric}_target", f"{metric}_baseline"]].dropna()
                delta = pair.iloc[:, 0].to_numpy() - pair.iloc[:, 1].to_numpy()
                if np.allclose(delta, 0):
                    statistic, p_value = 0.0, 1.0
                else:
                    test = stats.wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
                    statistic, p_value = float(test.statistic), float(test.pvalue)
                test_rows.append(
                    {
                        "comparison": f"{model} vs {baseline_name}",
                        "metric": metric,
                        "n_drugs": len(delta),
                        "mean_delta": float(delta.mean()),
                        "wilcoxon_statistic": statistic,
                        "wilcoxon_p": p_value,
                    }
                )
            p = np.asarray([row["wilcoxon_p"] for row in test_rows[start:]], dtype=float)
            order = np.argsort(p)
            adjusted = np.empty_like(p)
            running = 0.0
            for rank, position in enumerate(order):
                running = max(running, (len(p) - rank) * p[position])
                adjusted[position] = min(running, 1.0)
            for row, value in zip(test_rows[start:], adjusted):
                row["wilcoxon_p_holm"] = float(value)
                row["holm_significant"] = bool(value < 0.05)
    tests = pd.DataFrame(test_rows)
    tests.to_csv(OUT / "screen_paired_tests.csv", index=False)

    means = per_drug.groupby("model")[metrics].mean()
    decisions = []
    for model in ["Group-only", "Global gate", "Drug gate", "Drug-stratum gate"]:
        row = {"model": model}
        for baseline in ["CAFNet", "CAFNet-DG"]:
            row[f"rare_relative_vs_{baseline}"] = means.loc[model, "rare_AP"] / means.loc[baseline, "rare_AP"] - 1.0
            row[f"middle_delta_vs_{baseline}"] = means.loc[model, "middle_AP"] - means.loc[baseline, "middle_AP"]
            row[f"macro_AP_relative_vs_{baseline}"] = means.loc[model, "macro_AP"] / means.loc[baseline, "macro_AP"] - 1.0
            row[f"nDCG_relative_vs_{baseline}"] = means.loc[model, "nDCG@10"] / means.loc[baseline, "nDCG@10"] - 1.0
        row["metric_gate_pass"] = bool(
            row["rare_relative_vs_CAFNet"] >= 0.05
            and row["rare_relative_vs_CAFNet-DG"] >= 0.05
            and row["middle_delta_vs_CAFNet"] >= 0.0
            and row["middle_delta_vs_CAFNet-DG"] >= 0.0
            and row["macro_AP_relative_vs_CAFNet-DG"] >= -0.01
            and row["nDCG_relative_vs_CAFNet-DG"] >= -0.01
        )
        for baseline in ["CAFNet", "CAFNet-DG"]:
            rare_test = tests.loc[
                (tests["comparison"] == f"{model} vs {baseline}")
                & (tests["metric"] == "rare_AP")
            ]
            if len(rare_test) != 1:
                raise ValueError(f"Missing unique rare-AP test for {model} vs {baseline}")
            test_row = rare_test.iloc[0]
            row[f"rare_significant_vs_{baseline}"] = bool(
                test_row["mean_delta"] > 0 and test_row["wilcoxon_p_holm"] < 0.05
            )
        row["rare_inference_gate_pass"] = bool(
            row["rare_significant_vs_CAFNet"]
            and row["rare_significant_vs_CAFNet-DG"]
        )
        route_path = MODEL_FILES[model].parent / "routing_diagnostics.csv"
        if route_path.exists():
            route = pd.read_csv(route_path)
            saturation = np.maximum(route["gate_fraction_below_005"], route["gate_fraction_above_095"])
            row["max_single_expert_saturation"] = float(saturation.max())
            row["routing_gate_pass"] = bool(row["max_single_expert_saturation"] <= 0.90)
        else:
            row["max_single_expert_saturation"] = np.nan
            row["routing_gate_pass"] = bool(model == "Group-only")
        row["promote_to_10fold"] = bool(
            row["metric_gate_pass"]
            and row["rare_inference_gate_pass"]
            and row["routing_gate_pass"]
        )
        decisions.append(row)
    decision = pd.DataFrame(decisions)
    decision.to_csv(OUT / "screen_promotion_decision.csv", index=False)
    print(means.to_string())
    print(
        decision[
            ["model", "metric_gate_pass", "rare_inference_gate_pass", "routing_gate_pass", "promote_to_10fold"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
