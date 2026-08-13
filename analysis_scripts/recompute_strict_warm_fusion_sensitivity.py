from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from sklearn.metrics import average_precision_score

from evaluate_predictions_unified import evaluate_warm_association


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs" / "unified_evaluation_20260811"
RAW = scipy.io.loadmat(ROOT / "data" / "raw_frequency_750.mat")["R"].astype(np.float32)
MASKS = scipy.io.loadmat(ROOT / "data" / "mask_mat_750.mat")
RHOS = np.round(np.arange(0.0, 1.01, 0.1), 1)


def find_run(prefix: str) -> Path:
    matches = [
        path
        for path in (ROOT / "result_WS").glob(prefix + "*")
        if (path / "full_predictions").is_dir()
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one strict run for {prefix}, found {len(matches)}")
    return matches[0]


def load_prediction(run: Path, fold: int) -> np.ndarray:
    path = run / "full_predictions" / f"full_pred_fold{fold}.csv"
    prediction = pd.read_csv(path, header=None).values.astype(np.float32)
    if prediction.shape != RAW.shape or not np.all(np.isfinite(prediction)):
        raise ValueError(f"Invalid prediction matrix: {path}")
    return prediction


def macro_group_ap(
    prediction: np.ndarray,
    mask: np.ndarray,
    columns: np.ndarray,
) -> float:
    values = []
    original_zero = RAW == 0
    heldout_positive = (mask == 0) & (RAW > 0)
    for drug in range(RAW.shape[0]):
        candidate = columns[original_zero[drug, columns] | heldout_positive[drug, columns]]
        if candidate.size == 0:
            continue
        labels = heldout_positive[drug, candidate].astype(np.int8)
        if labels.sum() == 0 or labels.sum() == labels.size:
            continue
        values.append(average_precision_score(labels, prediction[drug, candidate]))
    return float(np.mean(values)) if values else float("nan")


def subgroup_metrics(prediction: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    visible_positive_count = ((RAW > 0) & (mask > 0)).sum(axis=0)
    positive_columns = np.flatnonzero(visible_positive_count > 0)
    q1, q2 = np.quantile(visible_positive_count[positive_columns], [1 / 3, 2 / 3])
    rare = np.flatnonzero(visible_positive_count <= q1)
    middle = np.flatnonzero((visible_positive_count > q1) & (visible_positive_count <= q2))
    frequent = np.flatnonzero(visible_positive_count > q2)
    hot100 = np.argsort(visible_positive_count)[::-1][:100]
    nonhot100 = np.setdiff1d(np.arange(RAW.shape[1]), hot100, assume_unique=False)
    return {
        "rare_AP": macro_group_ap(prediction, mask, rare),
        "middle_AP": macro_group_ap(prediction, mask, middle),
        "frequent_AP": macro_group_ap(prediction, mask, frequent),
        "nonhot100_AP": macro_group_ap(prediction, mask, nonhot100),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cafnet_run = find_run("10SC100_CAFNet_")
    cafnet_d_run = find_run("10SD100_CAFNetDecoupled_")
    rows = []
    for fold in range(10):
        mask = MASKS[f"mask{fold}"].astype(np.float32)
        cafnet = load_prediction(cafnet_run, fold)
        cafnet_d = load_prediction(cafnet_d_run, fold)
        for rho in RHOS:
            fused = rho * cafnet_d + (1.0 - rho) * cafnet
            rows.append(
                {
                    "fold": fold,
                    "rho": float(rho),
                    **evaluate_warm_association(fused, RAW, mask),
                    **subgroup_metrics(fused, mask),
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "warm_rho_sensitivity_strict_by_fold.csv", index=False)
    metrics = ["mAP", "AUROC", "AUPR", "nDCG@10", "nonhot100_AP", "rare_AP", "middle_AP"]
    summary = frame.groupby("rho")[metrics].agg(["mean", "std"])
    summary.to_csv(OUT / "warm_rho_sensitivity_strict_summary.csv")

    lines = []
    for rho in RHOS:
        row = frame[np.isclose(frame["rho"], rho)]
        cells = [f"${row[m].mean():.3f} \\pm {row[m].std(ddof=1):.3f}$" for m in metrics]
        lines.append(f"{rho:.1f} & " + " & ".join(cells) + " \\\\")
    (OUT / "warm_rho_sensitivity_strict_table_rows.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(summary.loc[0.6].to_string())


if __name__ == "__main__":
    main()
