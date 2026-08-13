from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

from evaluate_predictions_unified import (  # noqa: E402
    ORDINAL_METRICS,
    RANKING_METRICS,
    evaluate_cold_association,
    evaluate_cold_ordinal,
)


OUT = ROOT / "analysis_outputs" / "eps_lambda_sensitivity_20260814"
RAW = scipy.io.loadmat(ROOT / "data" / "raw_frequency_750.mat")["R"].astype(float)
MASKS = scipy.io.loadmat(ROOT / "data" / "blind_mask_mat_750.mat")
REPORTED_BASE = ROOT / "result_ICS" / "10ORD8COLD2_CAFNetDecoupled"
RERUN_BASE = ROOT / "result_ICS" / "10SENS_BASE_CAFNetDecoupled"
REPORTED_ASSOCIATION = ROOT / "result_ICS" / (
    "10cd3e100f10_CAFNetDecoupled_knn=5_wd=0.001_epoch=100_lamb=0.03_"
    "lr0.0004_dim=200_eps=0.5_DF=False_PCA=False_not-FC=False_cross=True_"
    "fusion=gate_gate=new_fa=0.5_gatdrop=0.0_mix=0.3_aw=1.0_fw=1.0_"
    "rw=0.05_popw=0.1_biasw=1.0_listw=0.1_abw=1.0_arw=1.0_cosine"
)
SETTINGS = {}
if (REPORTED_ASSOCIATION / "blind_pred.csv").exists() and (REPORTED_BASE / "blind_freq_pred.csv").exists():
    SETTINGS["lambda0=0.03, epsilon=0.50 (reported)"] = {
        "association": REPORTED_ASSOCIATION,
        "frequency": REPORTED_BASE,
    }
SETTINGS.update({
    "lambda0=0.03, epsilon=0.50 (same-code rerun)": {
        "association": RERUN_BASE,
        "frequency": RERUN_BASE,
    },
    "lambda0=0.03, epsilon=0.00": {"association": ROOT / "result_ICS" / "10SENS_EPS0_CAFNetDecoupled", "frequency": ROOT / "result_ICS" / "10SENS_EPS0_CAFNetDecoupled"},
    "lambda0=0.03, epsilon=0.25": {"association": ROOT / "result_ICS" / "10SENS_EPS025_CAFNetDecoupled", "frequency": ROOT / "result_ICS" / "10SENS_EPS025_CAFNetDecoupled"},
    "lambda0=0.03, epsilon=0.75": {"association": ROOT / "result_ICS" / "10SENS_EPS075_CAFNetDecoupled", "frequency": ROOT / "result_ICS" / "10SENS_EPS075_CAFNetDecoupled"},
    "lambda0=0.00, epsilon=0.50": {"association": ROOT / "result_ICS" / "10SENS_LAM0_CAFNetDecoupled", "frequency": ROOT / "result_ICS" / "10SENS_LAM0_CAFNetDecoupled"},
    "lambda0=0.01, epsilon=0.50": {"association": ROOT / "result_ICS" / "10SENS_LAM001_CAFNetDecoupled", "frequency": ROOT / "result_ICS" / "10SENS_LAM001_CAFNetDecoupled"},
    "lambda0=0.10, epsilon=0.50": {"association": ROOT / "result_ICS" / "10SENS_LAM01_CAFNetDecoupled", "frequency": ROOT / "result_ICS" / "10SENS_LAM01_CAFNetDecoupled"},
})
ALL_METRICS = RANKING_METRICS + ORDINAL_METRICS


def load(path: Path) -> np.ndarray:
    value = pd.read_csv(path, header=None).values.astype(float)
    if value.shape != RAW.shape or not np.isfinite(value).all():
        raise ValueError(f"Invalid matrix {path}: {value.shape}")
    return value


def holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    result = np.empty(len(values), dtype=float)
    running = 0.0
    for position, index in enumerate(order):
        adjusted = min(1.0, (len(values) - position) * values[index])
        running = max(running, adjusted)
        result[index] = running
    return result.tolist()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for setting, sources in SETTINGS.items():
        association_prediction = load(sources["association"] / "blind_pred.csv")
        frequency_prediction = load(sources["frequency"] / "blind_freq_pred.csv")
        truth_saved = load(sources["frequency"] / "blind_raw.csv")
        start = 0
        for fold in range(10):
            test = MASKS[f"mask{fold}"][:, 0] == 0
            stop = start + int(test.sum())
            truth = RAW[test]
            if not np.array_equal(truth_saved[start:stop], truth):
                raise ValueError(f"Truth order mismatch for {setting}, fold {fold}")
            ranking = evaluate_cold_association(association_prediction[start:stop], truth)
            ordinal, _ = evaluate_cold_ordinal(frequency_prediction[start:stop], truth)
            records.append({
                "setting": setting,
                "fold": fold,
                **ranking,
                **ordinal,
            })
            start = stop

    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "eps_lambda_sensitivity_by_fold.csv", index=False)
    summary = frame.groupby("setting", sort=False)[ALL_METRICS].agg(["mean", "std"])
    summary.to_csv(OUT / "eps_lambda_sensitivity_summary.csv")

    base_name = "lambda0=0.03, epsilon=0.50 (same-code rerun)"
    base = frame[frame.setting == base_name].set_index("fold")
    tests = []
    for setting in SETTINGS:
        if setting == base_name:
            continue
        current = frame[frame.setting == setting].set_index("fold")
        for metric in ALL_METRICS:
            delta = current[metric] - base[metric]
            try:
                wp = float(stats.wilcoxon(delta, zero_method="wilcox").pvalue)
            except ValueError:
                wp = 1.0
            tests.append({
                "setting": setting,
                "family": "ranking" if metric in RANKING_METRICS else "ordinal",
                "metric": metric,
                "mean_delta_vs_default_rerun": float(delta.mean()),
                "paired_t_p": float(stats.ttest_rel(current[metric], base[metric]).pvalue),
                "wilcoxon_p": wp,
            })
    tests = pd.DataFrame(tests)
    for column in ["paired_t_p", "wilcoxon_p"]:
        tests[column + "_holm"] = np.nan
        for _, indices in tests.groupby(["setting", "family"]).groups.items():
            tests.loc[indices, column + "_holm"] = holm(tests.loc[indices, column].tolist())
    tests.to_csv(OUT / "eps_lambda_sensitivity_paired_tests.csv", index=False)

    rows = []
    for setting in SETTINGS:
        part = frame[frame.setting == setting]
        cells = [f"${part[m].mean():.3f} \\pm {part[m].std(ddof=1):.3f}$" for m in ORDINAL_METRICS]
        rows.append(setting.replace("lambda0", "$\\lambda_0$").replace("epsilon", "$\\epsilon$") + " & " + " & ".join(cells) + " \\\\")
    (OUT / "eps_lambda_sensitivity_table_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (OUT / "protocol.json").write_text(
        json.dumps(
            {
                "scenario": "drug-disjoint cold-start",
                "folds": 10,
                "epochs": 100,
                "mask_file": "data/blind_mask_mat_750.mat",
                "ordinal_contract": "nonzero 1-5 frequency labels only",
                "metrics": ORDINAL_METRICS,
                "ranking_metrics": RANKING_METRICS,
                "default_reported_source": str(REPORTED_BASE.relative_to(ROOT)),
                "default_same_code_rerun_source": str(RERUN_BASE.relative_to(ROOT)),
                "paired_test_reference": "lambda0=0.03, epsilon=0.50 (same-code rerun)",
                "multiplicity": "Holm correction separately across five ordinal metrics for each setting",
                "selection_rule": "sensitivity analysis only; no test-fold-based replacement of the reported model",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
