from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from scipy import stats
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

from evaluate_predictions_unified import (  # noqa: E402
    RANKING_METRICS,
    evaluate_cold_association,
    evaluate_warm_association,
)


OUT = ROOT / "analysis_outputs" / "fusion_scale_audit_20260814"
RAW = scipy.io.loadmat(ROOT / "data" / "raw_frequency_750.mat")["R"].astype(np.float64)
WARM_MASKS = scipy.io.loadmat(ROOT / "data" / "mask_mat_750.mat")
COLD_MASKS = scipy.io.loadmat(ROOT / "data" / "blind_mask_mat_750.mat")
RHO = 0.6

COLD_CAFNET = ROOT / "result_ICS" / (
    "10ICS_CAFNet_knn=5_wd=0.001_epoch=100_lamb=0.03_lr0.0004_dim=200_"
    "eps=0.5_DF=False_PCA=False_not-FC=False_cosine"
) / "blind_pred.csv"
COLD_CAFNET_D = ROOT / "result_ICS" / (
    "10cd3e100f10_CAFNetDecoupled_knn=5_wd=0.001_epoch=100_lamb=0.03_"
    "lr0.0004_dim=200_eps=0.5_DF=False_PCA=False_not-FC=False_cross=True_"
    "fusion=gate_gate=new_fa=0.5_gatdrop=0.0_mix=0.3_aw=1.0_fw=1.0_"
    "rw=0.05_popw=0.1_biasw=1.0_listw=0.1_abw=1.0_arw=1.0_cosine"
) / "blind_pred.csv"


def find_warm_run(prefix: str) -> Path:
    matches = [
        path for path in (ROOT / "result_WS").glob(prefix + "*")
        if (path / "full_predictions").is_dir()
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one strict warm run for {prefix}, found {len(matches)}")
    return matches[0]


def load_csv(path: Path) -> np.ndarray:
    value = pd.read_csv(path, header=None).values.astype(np.float64)
    if value.shape != RAW.shape or not np.all(np.isfinite(value)):
        raise ValueError(f"Invalid prediction matrix: {path}")
    return value


def row_zscore(value: np.ndarray) -> np.ndarray:
    mean = value.mean(axis=1, keepdims=True)
    scale = value.std(axis=1, keepdims=True)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (value - mean) / scale


def row_percentile(value: np.ndarray) -> np.ndarray:
    denominator = max(1, value.shape[1] - 1)
    return np.apply_along_axis(
        lambda row: (rankdata(row, method="average") - 1.0) / denominator,
        axis=1,
        arr=value,
    )


def fuse(cafnet: np.ndarray, cafnet_d: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        left, right = cafnet_d, cafnet
    elif mode == "row_zscore":
        left, right = row_zscore(cafnet_d), row_zscore(cafnet)
    elif mode == "row_percentile":
        left, right = row_percentile(cafnet_d), row_percentile(cafnet)
    else:
        raise ValueError(mode)
    return RHO * left + (1.0 - RHO) * right


def scale_rows(scenario: str, fold: int, model: str, value: np.ndarray) -> dict[str, float | str | int]:
    row_std = value.std(axis=1)
    return {
        "scenario": scenario,
        "fold": fold,
        "model": model,
        "mean": float(value.mean()),
        "std": float(value.std()),
        "minimum": float(value.min()),
        "maximum": float(value.max()),
        "median_row_std": float(np.median(row_std)),
    }


def holm(values: list[float]) -> list[float]:
    result = np.full(len(values), np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return result.tolist()
    ordered = valid[np.argsort(np.asarray(values)[valid])]
    running = 0.0
    total = len(ordered)
    for position, index in enumerate(ordered):
        adjusted = min(1.0, (total - position) * values[index])
        running = max(running, adjusted)
        result[index] = running
    return result.tolist()


def paired_tests(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in sorted(frame["scenario"].unique()):
        base = frame[(frame["scenario"] == scenario) & (frame["mode"] == "raw")].set_index("fold")
        for mode in ["row_zscore", "row_percentile"]:
            comp = frame[(frame["scenario"] == scenario) & (frame["mode"] == mode)].set_index("fold")
            for metric in RANKING_METRICS:
                delta = comp[metric] - base[metric]
                try:
                    wilcoxon_p = float(stats.wilcoxon(delta, zero_method="wilcox").pvalue)
                except ValueError:
                    wilcoxon_p = 1.0
                rows.append({
                    "scenario": scenario,
                    "comparison": f"{mode} minus raw",
                    "metric": metric,
                    "mean_delta": float(delta.mean()),
                    "paired_t_p": float(stats.ttest_rel(comp[metric], base[metric]).pvalue),
                    "wilcoxon_p": wilcoxon_p,
                })
    result = pd.DataFrame(rows)
    for columns in [["paired_t_p"], ["wilcoxon_p"]]:
        source = columns[0]
        result[source + "_holm"] = np.nan
        for (_, _), indices in result.groupby(["scenario", "comparison"]).groups.items():
            result.loc[indices, source + "_holm"] = holm(result.loc[indices, source].tolist())
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    scales: list[dict] = []

    cold_c = load_csv(COLD_CAFNET)
    cold_d = load_csv(COLD_CAFNET_D)
    cold_truth_concat = load_csv(COLD_CAFNET.parent / "blind_raw.csv")
    cold_fusions = {
        mode: fuse(cold_c, cold_d, mode)
        for mode in ["raw", "row_zscore", "row_percentile"]
    }
    start = 0
    for fold in range(10):
        mask = COLD_MASKS[f"mask{fold}"].astype(float)
        test = mask[:, 0] == 0
        stop = start + int(test.sum())
        truth = RAW[test]
        if not np.array_equal(cold_truth_concat[start:stop], truth):
            raise ValueError(f"Cold concatenation order mismatch at fold {fold}")
        for model, value in [("CAFNet", cold_c[start:stop]), ("CAFNet-D", cold_d[start:stop])]:
            scales.append(scale_rows("cold", fold, model, value))
        for mode in ["raw", "row_zscore", "row_percentile"]:
            prediction = cold_fusions[mode][start:stop]
            records.append({
                "scenario": "cold", "fold": fold, "mode": mode,
                **evaluate_cold_association(prediction, truth),
            })
        start = stop

    warm_c_run = find_warm_run("10SC100_CAFNet_")
    warm_d_run = find_warm_run("10SD100_CAFNetDecoupled_")
    for fold in range(10):
        warm_c = load_csv(warm_c_run / "full_predictions" / f"full_pred_fold{fold}.csv")
        warm_d = load_csv(warm_d_run / "full_predictions" / f"full_pred_fold{fold}.csv")
        for model, value in [("CAFNet", warm_c), ("CAFNet-D", warm_d)]:
            scales.append(scale_rows("warm", fold, model, value))
        for mode in ["raw", "row_zscore", "row_percentile"]:
            prediction = fuse(warm_c, warm_d, mode)
            records.append({
                "scenario": "warm", "fold": fold, "mode": mode,
                **evaluate_warm_association(prediction, RAW, WARM_MASKS[f"mask{fold}"]),
            })

    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "fusion_scale_audit_by_fold.csv", index=False)
    summary = frame.groupby(["scenario", "mode"])[RANKING_METRICS].agg(["mean", "std"])
    summary.to_csv(OUT / "fusion_scale_audit_summary.csv")
    tests = paired_tests(frame)
    tests.to_csv(OUT / "fusion_scale_audit_paired_tests.csv", index=False)
    pd.DataFrame(scales).to_csv(OUT / "component_score_scale_by_fold.csv", index=False)

    lines = []
    for scenario in ["warm", "cold"]:
        for mode in ["raw", "row_zscore", "row_percentile"]:
            row = frame[(frame["scenario"] == scenario) & (frame["mode"] == mode)]
            cells = [f"${row[m].mean():.3f} \\pm {row[m].std(ddof=1):.3f}$" for m in RANKING_METRICS]
            lines.append(f"{scenario.capitalize()} & {mode.replace('_', ' ')} & " + " & ".join(cells) + " \\\\")
    (OUT / "fusion_scale_audit_table_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string())
    print("\nPaired comparisons with Holm p < 0.05:")
    print(tests[(tests.paired_t_p_holm < 0.05) | (tests.wilcoxon_p_holm < 0.05)].to_string(index=False))


if __name__ == "__main__":
    main()
