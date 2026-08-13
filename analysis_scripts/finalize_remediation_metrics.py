from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

from evaluate_predictions_unified import (
    ORDINAL_METRICS,
    RANKING_METRICS,
    evaluate_cold_association,
    evaluate_cold_ordinal,
    evaluate_warm_association,
    evaluate_warm_ordinal,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs" / "unified_evaluation_20260811"
RAW = scipy.io.loadmat(ROOT / "data" / "raw_frequency_750.mat")["R"].astype(np.float32)
COLD_MASKS = scipy.io.loadmat(ROOT / "data" / "blind_mask_mat_750.mat")
WARM_MASKS = scipy.io.loadmat(ROOT / "data" / "mask_mat_750.mat")


def load_csv_matrix(path: Path) -> np.ndarray:
    matrix = pd.read_csv(path, header=None).values.astype(np.float32)
    if matrix.shape != RAW.shape:
        raise ValueError(f"Expected {RAW.shape}, found {matrix.shape}: {path}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"Non-finite values: {path}")
    return matrix


def find_run(base: Path, prefix: str, required_dir: str) -> Path:
    matches = [path for path in base.glob(prefix + "*") if (path / required_dir).is_dir()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one completed run for {base / prefix}, found {len(matches)}")
    return matches[0]


def evaluate_warm_model(model: str, run: Path, *, has_frequency: bool = True) -> tuple[list[dict], list[np.ndarray]]:
    rows, confusions = [], []
    for fold in range(10):
        association = load_csv_matrix(run / "full_predictions" / f"full_pred_fold{fold}.csv")
        frequency = (
            load_csv_matrix(run / "full_freq_predictions" / f"full_freq_pred_fold{fold}.csv")
            if has_frequency and (run / "full_freq_predictions" / f"full_freq_pred_fold{fold}.csv").exists()
            else association if has_frequency else None
        )
        mask = WARM_MASKS[f"mask{fold}"]
        ranking = evaluate_warm_association(association, RAW, mask)
        ordinal, confusion = ({}, None) if frequency is None else evaluate_warm_ordinal(frequency, RAW, mask)
        rows.append({"scenario": "warm", "model": model, "fold": fold, **ranking, **ordinal})
        if confusion is not None:
            confusions.append(confusion)
    return rows, confusions


def evaluate_hstrans(scenario: str) -> tuple[list[dict], list[np.ndarray]]:
    base = ROOT / "result_baselines_hstrans_foldlocal_v3_compute_matched" / scenario / "HSTrans"
    rows, confusions = [], []
    for fold in range(10):
        provenance_path = base / "fold_provenance" / f"provenance_fold{fold}.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        expected_rows = 75 if scenario == "cold" else 750
        required = {
            "scenario": scenario,
            "fold": fold,
            "epochs": 20,
            "checkpoint_policy": "fixed_predeclared_epoch_budget",
            "token_source": "fold_training_labels_only",
            "prediction_rows": expected_rows,
        }
        mismatches = {
            key: {"expected": value, "observed": provenance.get(key)}
            for key, value in required.items()
            if provenance.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Invalid HSTrans provenance for {scenario} fold {fold}: {mismatches}")
        prediction = np.load(base / f"hstrans_pred_fold{fold}.npy").astype(np.float32)
        if scenario == "cold":
            truth = np.load(base / f"hstrans_true_fold{fold}.npy").astype(np.float32)
            ranking = evaluate_cold_association(prediction, truth)
            ordinal, confusion = evaluate_cold_ordinal(prediction, truth)
        else:
            mask = WARM_MASKS[f"mask{fold}"]
            if prediction.shape != RAW.shape:
                raise ValueError(f"HSTrans warm fold {fold} shape {prediction.shape}")
            ranking = evaluate_warm_association(prediction, RAW, mask)
            ordinal, confusion = evaluate_warm_ordinal(prediction, RAW, mask)
        rows.append({"scenario": scenario, "model": "HSTrans", "fold": fold, **ranking, **ordinal})
        confusions.append(confusion)
    return rows, confusions


def evaluate_dsgat_warm() -> tuple[list[dict], list[np.ndarray]]:
    base = find_run(
        ROOT / "baselines" / "DSGAT-master" / "DSGAT-master" / "result_WS" / "strict_v2",
        "10WS_GAT3",
        "full_predictions",
    )
    rows, confusions = [], []
    for fold in range(10):
        prediction = np.load(base / "full_predictions" / f"dsgat_warm_pred_fold{fold}.npy").astype(np.float32)
        mask = WARM_MASKS[f"mask{fold}"]
        ranking = evaluate_warm_association(prediction, RAW, mask)
        ordinal, confusion = evaluate_warm_ordinal(prediction, RAW, mask)
        rows.append({"scenario": "warm", "model": "DSGAT", "fold": fold, **ranking, **ordinal})
        confusions.append(confusion)
    return rows, confusions


def evaluate_hmmf_cold() -> tuple[list[dict], list[np.ndarray]]:
    base = ROOT / "result_baselines_hmmf_fixed_policy" / "cold" / "HMMF" / "predictions"
    status_base = base.parent
    rows, confusions = [], []
    for fold in range(10):
        status = json.loads((status_base / f"fold{fold}_status.json").read_text(encoding="utf-8"))
        required = {
            "scenario": "cold",
            "fold": fold,
            "epochs": 10,
            "learning_rate": 2e-4,
            "seed": 42,
            "selection_policy": "fixed_final_epoch",
            "state": "complete",
        }
        mismatches = {
            key: {"expected": value, "observed": status.get(key)}
            for key, value in required.items()
            if status.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Invalid HMMF policy status for fold {fold}: {mismatches}")
        payload = np.load(base / f"hmmf_fold{fold}_predictions.npz")
        test_rows = np.flatnonzero(COLD_MASKS[f"mask{fold}"][:, 0] == 0)
        local = {int(drug): idx for idx, drug in enumerate(test_rows)}
        drug = payload["drug"].astype(int)
        side = payload["side_effect"].astype(int)
        coordinate_ids = drug * RAW.shape[1] + side
        if len(coordinate_ids) != len(test_rows) * RAW.shape[1] or np.unique(coordinate_ids).size != len(coordinate_ids):
            raise ValueError(f"HMMF fold {fold} does not contain one prediction per cold-test coordinate")
        if not set(drug.tolist()).issubset(set(test_rows.tolist())):
            raise ValueError(f"HMMF fold {fold} contains a training-drug prediction")
        if not np.allclose(payload["true_frequency"], RAW[drug, side]):
            raise ValueError(f"HMMF fold {fold} truth payload does not match the benchmark matrix")
        association = np.full((len(test_rows), RAW.shape[1]), np.nan, dtype=np.float32)
        frequency = np.full_like(association, np.nan)
        for drug, side, assoc_score, freq_score in zip(
            drug, side, payload["association_score"], payload["frequency_score"]
        ):
            association[local[int(drug)], int(side)] = assoc_score
            frequency[local[int(drug)], int(side)] = freq_score
        if not np.all(np.isfinite(association)) or not np.all(np.isfinite(frequency)):
            raise ValueError(f"Incomplete HMMF prediction grid for fold {fold}")
        truth = RAW[test_rows]
        ranking = evaluate_cold_association(association, truth)
        ordinal, confusion = evaluate_cold_ordinal(frequency, truth)
        rows.append({"scenario": "cold", "model": "HMMF-regenerated", "fold": fold, **ranking, **ordinal})
        confusions.append(confusion)
    return rows, confusions


def append_confusions(model: str, scenario: str, confusions: list[np.ndarray]) -> None:
    base = OUT / "confusion"
    base.mkdir(exist_ok=True)
    for fold, matrix in enumerate(confusions):
        np.save(base / f"{scenario}_{model.replace(' ', '_')}_fold{fold}.npy", matrix)


def main() -> None:
    partial = pd.read_csv(OUT / "valid_saved_predictions_fold_metrics.csv")
    partial = partial[~((partial["scenario"] == "warm") & partial["model"].isin(["CAFNet", "CAFNet-D", "CAFNet-DG", "A3Net", "DSGAT", "HSTrans"]))]

    cafnet_run = find_run(ROOT / "result_WS", "10SC100_CAFNet_", "full_predictions")
    cafnet_d_run = find_run(ROOT / "result_WS", "10SD100_CAFNetDecoupled_", "full_predictions")
    a3net_run = find_run(ROOT / "result_WS", "10SA100_A3_Net_", "full_predictions")

    added_rows = []
    for model, run, has_frequency in [
        ("CAFNet", cafnet_run, True),
        ("CAFNet-D", cafnet_d_run, True),
        ("A3Net", a3net_run, True),
    ]:
        rows, confusions = evaluate_warm_model(model, run, has_frequency=has_frequency)
        added_rows.extend(rows)
        append_confusions(model, "warm", confusions)

    fusion_dir = OUT / "corrected_warm_cafnet_dg_predictions"
    fusion_dir.mkdir(exist_ok=True)
    fusion_rows = []
    for fold in range(10):
        cafnet = load_csv_matrix(cafnet_run / "full_predictions" / f"full_pred_fold{fold}.csv")
        cafnet_d = load_csv_matrix(cafnet_d_run / "full_predictions" / f"full_pred_fold{fold}.csv")
        fused = 0.6 * cafnet_d + 0.4 * cafnet
        np.save(fusion_dir / f"cafnet_dg_warm_pred_fold{fold}.npy", fused.astype(np.float32))
        ranking = evaluate_warm_association(fused, RAW, WARM_MASKS[f"mask{fold}"])
        fusion_rows.append({"scenario": "warm", "model": "CAFNet-DG", "fold": fold, **ranking})
    added_rows.extend(fusion_rows)

    for loader, model, scenario in [
        (lambda: evaluate_dsgat_warm(), "DSGAT", "warm"),
        (lambda: evaluate_hstrans("warm"), "HSTrans", "warm"),
        (lambda: evaluate_hstrans("cold"), "HSTrans", "cold"),
        (lambda: evaluate_hmmf_cold(), "HMMF-regenerated", "cold"),
    ]:
        rows, confusions = loader()
        added_rows.extend(rows)
        append_confusions(model, scenario, confusions)

    master = pd.concat([partial, pd.DataFrame(added_rows)], ignore_index=True, sort=False)
    master = master.sort_values(["scenario", "model", "fold"]).reset_index(drop=True)
    expected = master.groupby(["scenario", "model"])["fold"].nunique()
    if not np.all(expected == 10):
        raise ValueError(f"Incomplete model/fold groups:\n{expected[expected != 10]}")
    master.to_csv(OUT / "master_fold_metrics.csv", index=False)

    summary_rows = []
    metrics = [metric for metric in RANKING_METRICS + ORDINAL_METRICS if metric in master.columns]
    for (scenario, model), group in master.groupby(["scenario", "model"], sort=False):
        for metric in metrics:
            values = group[metric].dropna()
            summary_rows.append(
                {
                    "scenario": scenario,
                    "model": model,
                    "metric": metric,
                    "mean": values.mean() if len(values) else np.nan,
                    "std": values.std(ddof=1) if len(values) > 1 else np.nan,
                    "n": len(values),
                }
            )
    pd.DataFrame(summary_rows).to_csv(OUT / "master_summary.csv", index=False)
    (OUT / "finalization_contract.json").write_text(
        json.dumps(
            {
                "cafnet_dg_warm": "0.6 * strict CAFNet-D + 0.4 * strict CAFNet; fixed before outer evaluation",
                "hmmf": "extended regenerated-PubMedBERT comparison; fixed final epoch",
                "all_groups_have_ten_folds": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(expected.to_string())


if __name__ == "__main__":
    main()
