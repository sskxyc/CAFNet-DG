"""SIDER-only post-hoc calibration followed by untouched CT-ADE evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupKFold
from scipy.stats import wilcoxon

from ct_ade_frequency_validation_20260807 import LABELS, metrics, reconstruct_oof


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "analysis_outputs"
    / "scientific_gap_resolution_20260807"
    / "ct_ade_frequency_validation"
)
OUT = ROOT / "analysis_outputs" / "core_issue_resolution_20260814" / "frequency_calibration"
FREQ_PRED = ROOT / "result_ICS" / "10ORD8COLD2_CAFNetDecoupled" / "blind_freq_pred.csv"


def fit_calibrator(name: str, x: np.ndarray, y: np.ndarray):
    if name == "affine":
        model = LinearRegression().fit(x[:, None], y)
        return lambda values: np.clip(model.predict(np.asarray(values)[:, None]), 1.0, 5.0)
    if name == "isotonic":
        model = IsotonicRegression(y_min=1.0, y_max=5.0, out_of_bounds="clip").fit(x, y)
        return lambda values: model.predict(np.asarray(values))
    if name == "clipped_identity":
        return lambda values: np.clip(np.asarray(values), 1.0, 5.0)
    raise ValueError(name)


def compact_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    pred_class = np.floor(np.clip(pred, 1.0, 5.0) + 0.5).astype(int)
    return {
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "mae": float(np.mean(np.abs(pred - y))),
        "qwk": float(cohen_kappa_score(y.astype(int), pred_class, labels=LABELS, weights="quadratic")),
        "within_one_accuracy": float(np.mean(np.abs(pred_class - y) <= 1)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = sio.loadmat(ROOT / "data" / "raw_frequency_750.mat")["R"].astype(float)
    masks = sio.loadmat(ROOT / "data" / "blind_mask_mat_750.mat")
    oof = reconstruct_oof(FREQ_PRED, masks, raw.shape)

    observed = raw > 0
    drug_ids, _ = np.nonzero(observed)
    x = oof[observed].astype(float)
    y = raw[observed].astype(float)

    candidates = ["clipped_identity", "affine", "isotonic"]
    cv_rows = []
    cv_predictions = {name: np.full_like(y, np.nan, dtype=float) for name in candidates}
    splitter = GroupKFold(n_splits=5)
    for split, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups=drug_ids)):
        for name in candidates:
            calibrate = fit_calibrator(name, x[train_idx], y[train_idx])
            pred = calibrate(x[test_idx])
            cv_predictions[name][test_idx] = pred
            cv_rows.append({"split": split, "method": name, **compact_metrics(y[test_idx], pred)})
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(OUT / "sider_oof_calibration_by_split.csv", index=False)
    cv_summary = cv.groupby("method", as_index=False).agg(
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        qwk_mean=("qwk", "mean"),
        qwk_std=("qwk", "std"),
        within_one_mean=("within_one_accuracy", "mean"),
        within_one_std=("within_one_accuracy", "std"),
    )
    cv_summary.to_csv(OUT / "sider_oof_calibration_summary.csv", index=False)
    selected = str(cv_summary.sort_values(["rmse_mean", "mae_mean"]).iloc[0]["method"])

    external = pd.read_csv(SOURCE / "ct_ade_external_frequency_pairs.csv")
    external_y = external["ct_ade_ordinal_class"].to_numpy(int)
    external_raw = external["CAFNet-D_frequency"].to_numpy(float)
    external_baseline = external["Training-side-mean"].to_numpy(float)

    external_summary = [
        {"model": "CAFNet-D frequency (raw)", **metrics(external_y, external_raw)},
        {"model": "Training-side mean", **metrics(external_y, external_baseline)},
    ]
    for name in candidates:
        calibrate = fit_calibrator(name, x, y)
        pred = calibrate(external_raw)
        external[f"CAFNet-D_{name}"] = pred
        external_summary.append({"model": f"CAFNet-D + {name}", **metrics(external_y, pred)})
    external.to_csv(OUT / "ct_ade_calibrated_frequency_pairs.csv", index=False)
    pd.DataFrame(external_summary).to_csv(OUT / "ct_ade_calibration_summary.csv", index=False)

    selected_column = f"CAFNet-D_{selected}"
    per_drug_rows = []
    for drug_id, drug in external.groupby("drug_index"):
        if len(drug) < 3 or drug.ct_ade_ordinal_class.nunique() < 2:
            continue
        row = {"drug_index": int(drug_id), "drug_name": drug.drug_name.iloc[0], "n": int(len(drug))}
        for label, column in {
            "selected_calibration": selected_column,
            "raw_model": "CAFNet-D_frequency",
            "training_side_mean": "Training-side-mean",
        }.items():
            values = metrics(drug.ct_ade_ordinal_class.to_numpy(int), drug[column].to_numpy(float))
            for metric_name, value in values.items():
                row[f"{label}_{metric_name}"] = value
        per_drug_rows.append(row)
    per_drug = pd.DataFrame(per_drug_rows)
    per_drug.to_csv(OUT / "ct_ade_calibration_per_drug.csv", index=False)

    tests = []
    for comparator in ["raw_model", "training_side_mean"]:
        for metric_name in ["qwk", "within_one_accuracy", "exact_accuracy", "spearman", "rmse", "mae"]:
            delta = (
                per_drug[f"selected_calibration_{metric_name}"]
                - per_drug[f"{comparator}_{metric_name}"]
            ).dropna().to_numpy(float)
            if len(delta) == 0 or np.allclose(delta, 0):
                statistic, p_value = 0.0, 1.0
            else:
                result = wilcoxon(delta, alternative="two-sided")
                statistic, p_value = float(result.statistic), float(result.pvalue)
            tests.append(
                {
                    "comparator": comparator,
                    "metric": metric_name,
                    "n_drugs": int(len(delta)),
                    "mean_delta_selected_minus_comparator": float(delta.mean()) if len(delta) else np.nan,
                    "wilcoxon_statistic": statistic,
                    "p_raw": p_value,
                }
            )
    tests_df = pd.DataFrame(tests)
    tests_df["p_holm"] = np.nan
    for comparator, idx in tests_df.groupby("comparator").groups.items():
        positions = np.asarray(list(idx), dtype=int)
        p_values = tests_df.loc[positions, "p_raw"].to_numpy(float)
        order = np.argsort(p_values)
        adjusted = np.empty_like(p_values)
        running = 0.0
        for rank, local_idx in enumerate(order):
            running = max(running, min(1.0, (len(p_values) - rank) * p_values[local_idx]))
            adjusted[local_idx] = running
        tests_df.loc[positions, "p_holm"] = adjusted
    tests_df["significant_holm"] = tests_df.p_holm < 0.05
    tests_df.to_csv(OUT / "ct_ade_calibration_paired_tests.csv", index=False)

    protocol = {
        "calibration_source": "SIDER cold-start out-of-fold predictions on observed non-zero entries only",
        "calibration_selection": "Five-fold GroupKFold by drug; minimum mean RMSE, MAE tie-break",
        "candidate_methods": candidates,
        "selected_without_ct_ade_labels": selected,
        "external_test": "CT-ADE non-SIDER pairs remain untouched until final evaluation",
        "claim_boundary": "Post-hoc score calibration diagnostic, not retraining and not patient-level incidence estimation",
        "sider_calibration_pairs": int(len(y)),
        "ct_ade_test_pairs": int(len(external)),
    }
    (OUT / "calibration_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    report = [
        "# External frequency calibration audit",
        "",
        "The calibrator is selected using SIDER out-of-fold predictions only. CT-ADE labels are not used for fitting or method selection.",
        "",
        f"Selected method: **{selected}**",
        "",
        "## SIDER grouped calibration selection",
        "",
        "```text",
        cv_summary.to_string(index=False),
        "```",
        "",
        "## Untouched CT-ADE evaluation",
        "",
        "```text",
        pd.DataFrame(external_summary).to_string(index=False),
        "```",
        "",
        "## Per-drug paired tests for the SIDER-selected calibrator",
        "",
        "```text",
        tests_df.to_string(index=False),
        "```",
    ]
    (OUT / "frequency_calibration_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
