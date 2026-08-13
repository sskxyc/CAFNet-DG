from __future__ import annotations

import hashlib
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


def unique_glob(base: Path, pattern: str) -> Path:
    matches = list(base.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one match for {base / pattern}, found {len(matches)}")
    return matches[0]


def load_matrix(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        matrix = np.load(path)
    else:
        matrix = pd.read_csv(path, header=None).values
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape == (751, 994):
        matrix = matrix[1:]
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"Non-finite prediction in {path}")
    return matrix


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cold_specs() -> list[dict]:
    cafnet = unique_glob(ROOT / "result_ICS", "10ICS_CAFNet*/blind_pred.csv")
    a3net = unique_glob(ROOT / "result_ICS", "10ICS_A3_Net*/blind_pred.csv")
    cafnet_d = unique_glob(ROOT / "result_ICS", "10cd3e100f10*/blind_pred.csv")
    cafnet_d_freq = unique_glob(ROOT / "result_ICS", "10ORD8COLD2*/blind_freq_pred.csv")
    cafnet_dg = ROOT / "result_ICS" / "10cafnet_dg_ensemble06_cafnetd04_cafnet" / "blind_pred.csv"
    dsgat = unique_glob(
        ROOT / "baselines" / "DSGAT-master" / "DSGAT-master" / "result_ICS",
        "10ICS_GAT3*epoch=100*/blind_pred.csv",
    )
    return [
        {"model": "CAFNet", "kind": "shared", "association": cafnet, "frequency": cafnet},
        {"model": "A3Net", "kind": "shared", "association": a3net, "frequency": a3net},
        {"model": "CAFNet-D", "kind": "shared", "association": cafnet_d, "frequency": cafnet_d_freq},
        {"model": "CAFNet-DG", "kind": "shared", "association": cafnet_dg, "frequency": None},
        {"model": "DSGAT", "kind": "shared", "association": dsgat, "frequency": dsgat},
        {
            "model": "RF",
            "kind": "fold",
            "association": ROOT / "result_baselines_a3net_rdkit_cold_v1" / "cold" / "RF" / "rf_cold_pred_fold{fold}.npy",
            "frequency": ROOT / "result_baselines_a3net_rdkit_cold_v1" / "cold" / "RF" / "rf_cold_pred_fold{fold}.npy",
        },
        {
            "model": "XGBoost",
            "kind": "fold",
            "association": ROOT / "result_baselines_a3net_rdkit_cold_v1" / "cold" / "XGB" / "xgb_cold_pred_fold{fold}.npy",
            "frequency": ROOT / "result_baselines_a3net_rdkit_cold_v1" / "cold" / "XGB" / "xgb_cold_pred_fold{fold}.npy",
        },
        {
            "model": "Global popularity",
            "kind": "fold",
            "association": ROOT / "result_baselines_popularity" / "cold" / "GLOBAL_POPULARITY" / "global_popularity_cold_assoc_pred_fold{fold}.npy",
            "frequency": ROOT / "result_baselines_popularity" / "cold" / "GLOBAL_POPULARITY" / "global_popularity_cold_freq_pred_fold{fold}.npy",
        },
    ]


def warm_specs() -> list[dict]:
    base = ROOT / "result_baselines_a3net_rdkit_warm_v2_strictneg" / "warm"
    return [
        {
            "model": "RF",
            "kind": "fold",
            "association": base / "RF" / "rf_pred_fold{fold}.npy",
            "frequency": base / "RF" / "rf_pred_fold{fold}.npy",
        },
        {
            "model": "XGBoost",
            "kind": "fold",
            "association": base / "XGB" / "xgb_pred_fold{fold}.npy",
            "frequency": base / "XGB" / "xgb_pred_fold{fold}.npy",
        },
    ]


def resolve_path(value: Path | None, fold: int) -> Path | None:
    if value is None:
        return None
    return Path(str(value).format(fold=fold))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    provenance = []
    confusion_dir = OUT / "confusion"
    confusion_dir.mkdir(exist_ok=True)

    for scenario, specs, masks in [
        ("cold", cold_specs(), COLD_MASKS),
        ("warm", warm_specs(), WARM_MASKS),
    ]:
        for spec in specs:
            shared_assoc_full = load_matrix(spec["association"]) if spec["kind"] == "shared" else None
            shared_freq_full = (
                load_matrix(spec["frequency"])
                if spec["kind"] == "shared" and spec["frequency"] is not None
                else None
            )
            shared_truth_path = spec["association"].parent / "blind_raw.csv" if spec["kind"] == "shared" else None
            shared_truth_full = load_matrix(shared_truth_path) if shared_truth_path and shared_truth_path.exists() else None
            for fold in range(10):
                mask = masks[f"mask{fold}"]
                assoc_path = resolve_path(spec["association"], fold)
                freq_path = resolve_path(spec["frequency"], fold)
                association = shared_assoc_full if shared_assoc_full is not None else load_matrix(assoc_path)
                frequency = (
                    shared_freq_full
                    if shared_freq_full is not None
                    else (load_matrix(freq_path) if freq_path else None)
                )

                if scenario == "cold":
                    test_rows = np.flatnonzero(mask[:, 0] == 0)
                    truth = RAW[test_rows]
                    if spec["kind"] == "shared":
                        start = sum(int(np.sum(masks[f"mask{i}"][:, 0] == 0)) for i in range(fold))
                        stop = start + len(test_rows)
                        association = association[start:stop]
                        if frequency is not None:
                            frequency = frequency[start:stop]
                        if shared_truth_full is not None and not np.array_equal(shared_truth_full[start:stop], truth):
                            raise ValueError(
                                f"Cold concatenation order mismatch for {spec['model']} fold {fold}: {shared_truth_path}"
                            )
                    ranking = evaluate_cold_association(association, truth)
                    ordinal, confusion = ({}, None) if frequency is None else evaluate_cold_ordinal(frequency, truth)
                else:
                    if association.shape != RAW.shape:
                        raise ValueError(f"Warm prediction must be 750x994: {assoc_path}")
                    ranking = evaluate_warm_association(association, RAW, mask)
                    ordinal, confusion = ({}, None) if frequency is None else evaluate_warm_ordinal(frequency, RAW, mask)

                rows.append({"scenario": scenario, "model": spec["model"], "fold": fold, **ranking, **ordinal})
                for role, path in [("association", assoc_path), ("frequency", freq_path)]:
                    if path is not None:
                        provenance.append(
                            {
                                "scenario": scenario,
                                "model": spec["model"],
                                "fold": fold,
                                "role": role,
                                "path": str(path.relative_to(ROOT)),
                                "sha256": sha256(path),
                            }
                        )
                if confusion is not None:
                    np.save(confusion_dir / f"{scenario}_{spec['model'].replace(' ', '_')}_fold{fold}.npy", confusion)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "valid_saved_predictions_fold_metrics.csv", index=False)
    metric_columns = [column for column in RANKING_METRICS + ORDINAL_METRICS if column in frame.columns]
    summary_rows = []
    for (scenario, model), group in frame.groupby(["scenario", "model"], sort=False):
        for metric in metric_columns:
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
    pd.DataFrame(summary_rows).to_csv(OUT / "valid_saved_predictions_summary.csv", index=False)
    pd.DataFrame(provenance).drop_duplicates().to_csv(OUT / "prediction_provenance_sha256.csv", index=False)
    (OUT / "contract.json").write_text(
        json.dumps(
            {
                "ranking": "warm: heldout positives plus original-zero candidates; cold: all 994 ADRs",
                "ordinal": "true nonzero heldout frequency labels only",
                "excluded": [
                    "all legacy HSTrans outputs",
                    "legacy warm CAFNet/A3Net/CAFNet-D/CAFNet-DG/DSGAT outputs",
                    "outer-test-selected HMMF outputs",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(frame.groupby(["scenario", "model"]).size())


if __name__ == "__main__":
    main()
